"""
Interface de synchronisation Biopac MP150 <-> camera thermique Optris (PIX Connect).

Un seul bouton : demarre l'enregistrement PIX Connect ET envoie le trigger Biopac, puis
(second appui) envoie le trigger de fin, arrete et sauvegarde automatiquement le .ravi.
Le bouton n'est actif que lorsque tout est pret :
  - boitier trigger connecte (port COM) et canal present dans channel_map.json ;
  - PIX Connect ouvert, camera connectee, en direct (pas en lecture de fichier), sans dialogue ouvert ;
  - case "AcqKnowledge en acquisition" cochee (le Biopac est pilote depuis l'autre PC).

Pour chaque enregistrement : <nom>.ravi + <nom>_sync.json (instants des triggers en temps video)
et une ligne par evenement dans sync_log.csv.

Lancement :  python sync_app.py      (ou double-clic sur sync_app.pyw)
"""

import csv
import json
import queue
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, ttk

from serial.tools import list_ports

import pix_control as pc
from test_com_trigger import BAUDRATE, load_channel_map
from trigger_background import TriggerBox

HERE = Path(__file__).parent
LOG_CSV = HERE / "sync_log.csv"
BOX_VIDS = (0x0403, 0x2341, 0x2A03)


def box_present() -> str | None:
    for p in list_ports.comports():
        if p.vid in BOX_VIDS:
            return p.device
    return None


class SyncApp:
    def __init__(self, root: tk.Tk, parent: tk.Widget | None = None, log_height: int = 9):
        """parent : cadre ou integrer le panneau (demonstrateur) ; sinon fenetre autonome."""
        self.root = root
        self.pix = pc.PixConnect()
        self.box: TriggerBox | None = None
        self.events: queue.Queue = queue.Queue()
        self.busy = False
        self.recording = False
        self.t_video0 = None  # perf_counter de F2 = debut de la video
        self.session: dict = {}
        self.ready_parts = {"box": False, "pix": False}

        if parent is None:
            root.title("Synchro Biopac - Optris")
            root.minsize(560, 520)
        style = ttk.Style()
        style.configure("Big.TButton", font=("Segoe UI", 16, "bold"), padding=14)
        style.configure("Status.TLabel", font=("Segoe UI", 10))
        style.configure("Timer.TLabel", font=("Consolas", 22, "bold"))

        frm = ttk.Frame(parent or root, padding=14 if parent is None else 0)
        frm.pack(fill="both", expand=True)

        form = ttk.LabelFrame(frm, text="Enregistrement", padding=10)
        form.pack(fill="x")
        self.var_name = tk.StringVar(value="participant01")
        self.var_dir = tk.StringVar(value=str(HERE))
        self.var_channel = tk.IntVar(value=2)
        self.var_width = tk.DoubleVar(value=1.0)
        self.var_acq = tk.BooleanVar(value=False)
        ttk.Label(form, text="Nom :").grid(row=0, column=0, sticky="w")
        ttk.Entry(form, textvariable=self.var_name, width=30).grid(row=0, column=1, sticky="we", padx=6)
        ttk.Label(form, text="Dossier :").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(form, textvariable=self.var_dir, width=30).grid(row=1, column=1, sticky="we", padx=6)
        ttk.Button(form, text="...", width=3, command=self.choose_dir).grid(row=1, column=2)
        ttk.Label(form, text="Canal Biopac :").grid(row=2, column=0, sticky="w")
        ttk.Spinbox(form, from_=1, to=8, textvariable=self.var_channel, width=5).grid(row=2, column=1, sticky="w", padx=6)
        ttk.Label(form, text="Duree trigger (s) :").grid(row=3, column=0, sticky="w", pady=4)
        ttk.Spinbox(form, from_=0.05, to=5, increment=0.05, textvariable=self.var_width, width=6).grid(
            row=3, column=1, sticky="w", padx=6)
        form.columnconfigure(1, weight=1)

        checks = ttk.LabelFrame(frm, text="Etat", padding=10)
        checks.pack(fill="x", pady=10)
        self.lbl_box = ttk.Label(checks, style="Status.TLabel")
        self.lbl_pix = ttk.Label(checks, style="Status.TLabel")
        self.lbl_box.pack(anchor="w")
        self.lbl_pix.pack(anchor="w", pady=2)
        acq = ttk.Frame(checks)
        acq.pack(anchor="w", fill="x")
        ttk.Checkbutton(acq, text="AcqKnowledge est en acquisition sur le PC Biopac",
                        variable=self.var_acq).pack(side="left")
        self.btn_launch = ttk.Button(checks, text="Ouvrir PIX Connect", command=self.pix.launch)

        self.btn = ttk.Button(frm, text="DEMARRER", style="Big.TButton", command=self.on_button)
        self.btn.pack(fill="x", pady=4)
        self.lbl_reason = ttk.Label(frm, foreground="#a15c00")
        self.lbl_reason.pack(anchor="w")
        self.lbl_timer = ttk.Label(frm, text="00:00.0", style="Timer.TLabel")
        self.lbl_timer.pack(pady=6)

        self.log = tk.Text(frm, height=log_height, state="disabled", font=("Consolas", 9))
        self.log.pack(fill="both", expand=True)

        if parent is None:
            root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.read_form()
        threading.Thread(target=self.monitor, daemon=True).start()
        self.root.after(100, self.tick)

    # ---------- utilitaires interface ----------
    def choose_dir(self):
        d = filedialog.askdirectory(initialdir=self.var_dir.get())
        if d:
            self.var_dir.set(d)

    def write_log(self, text: str):
        line = f"{datetime.now():%H:%M:%S}  {text}\n"
        self.log.configure(state="normal")
        self.log.insert("end", line)
        self.log.see("end")
        self.log.configure(state="disabled")

    def post(self, kind: str, payload=None):
        self.events.put((kind, payload))

    def read_form(self):
        """Copie les champs Tk dans des attributs simples, lisibles depuis les threads."""
        try:
            self.channel = int(self.var_channel.get())
            self.width = float(self.var_width.get())
        except (tk.TclError, ValueError):
            pass  # saisie en cours : on garde les valeurs precedentes
        self.name = self.var_name.get().strip() or "enregistrement"
        self.out_dir = Path(self.var_dir.get())

    # ---------- surveillance (thread) ----------
    def monitor(self):
        while True:
            port = box_present()
            if self.box is None and port:
                try:
                    self.box = TriggerBox(port, BAUDRATE, self.width)
                    self.box.start()
                except Exception as exc:  # port occupe par un autre script, etc.
                    self.box = None
                    self.post("box", (False, f"Boitier trigger : {port} occupe par un autre programme "
                                             f"(autre fenetre sync_app/demo ouverte ?)"))
                    if not self.busy:
                        ok, msg = self.pix.status()
                        self.post("pix", (ok, f"PIX Connect : {msg}"))
                    time.sleep(1)
                    continue
            if port is None and self.box is not None and not self.recording:
                self.box.close()
                self.box = None
            channel = self.channel
            mapping = load_channel_map()
            if self.box is None:
                self.post("box", (False, "Boitier trigger : non branche"))
            elif channel not in mapping:
                self.post("box", (False, f"Boitier trigger : canal {channel} absent de channel_map.json"))
            else:
                self.post("box", (True, f"Boitier trigger : OK sur {self.box.port}, canal {channel}"))
            if not self.busy:
                ok, msg = self.pix.status()
                self.post("pix", (ok, f"PIX Connect : {msg}"))
            time.sleep(1.0)

    # ---------- boucle interface ----------
    def tick(self):
        if not self.busy:
            self.read_form()
        while not self.events.empty():
            kind, payload = self.events.get()
            if kind in ("box", "pix"):
                ok, msg = payload
                self.ready_parts[kind] = ok
                lbl = self.lbl_box if kind == "box" else self.lbl_pix
                lbl.configure(text=("[OK]  " if ok else "[--]  ") + msg, foreground="#1a7f37" if ok else "#b42318")
                if kind == "pix":
                    if "n'est pas ouvert" in msg:
                        self.btn_launch.pack(anchor="w", pady=4)
                    else:
                        self.btn_launch.pack_forget()
            elif kind == "log":
                self.write_log(payload)
            elif kind == "done":
                self.busy = False
        self.update_button()
        if self.recording and self.t_video0:
            el = time.perf_counter() - self.t_video0
            self.lbl_timer.configure(text=f"{int(el // 60):02d}:{el % 60:04.1f}", foreground="#b42318")
        else:
            self.lbl_timer.configure(foreground="")
        self.root.after(100, self.tick)

    def update_button(self):
        reasons = []
        if not self.ready_parts["box"]:
            reasons.append("boitier trigger")
        if not self.ready_parts["pix"]:
            reasons.append("PIX Connect / camera")
        if not self.var_acq.get() and not self.recording:
            reasons.append("cocher AcqKnowledge")
        if self.busy:
            self.btn.configure(state="disabled")
            self.lbl_reason.configure(text="Operation en cours...")
        elif self.recording:
            self.btn.configure(state="normal", text="ARRETER")
            self.lbl_reason.configure(text="")
        else:
            self.btn.configure(state="disabled" if reasons else "normal", text="DEMARRER")
            self.lbl_reason.configure(text=("En attente : " + ", ".join(reasons)) if reasons else "Pret a enregistrer")

    # ---------- actions ----------
    def on_button(self):
        if self.busy or str(self.btn["state"]) == "disabled":
            return
        self.busy = True
        self.update_button()
        target = self.stop_sequence if self.recording else self.start_sequence
        threading.Thread(target=target, daemon=True).start()

    def trigger(self, event: str) -> float:
        mapping = load_channel_map()
        channel = self.channel
        self.box.width = self.width
        t = time.perf_counter()
        self.box.send(1 << mapping[channel])
        self.log_event(event, t)
        return t

    def log_event(self, event: str, t_perf: float, extra: str = ""):
        now = datetime.now()
        video_t = "" if self.t_video0 is None else f"{t_perf - self.t_video0:.3f}"
        new = not LOG_CSV.exists()
        with LOG_CSV.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["session", "event", "timestamp_iso", "unix_time", "perf_counter", "video_time_s", "info"])
            w.writerow([self.session.get("name", ""), event, now.isoformat(timespec="microseconds"),
                        f"{now.timestamp():.6f}", f"{t_perf:.6f}", video_t, extra])

    def start_sequence(self):
        started = False
        try:
            if self.box is None or self.box.ser is None or not self.box.ser.is_open:
                raise RuntimeError("boitier trigger non connecte")
            if self.channel not in load_channel_map():
                raise RuntimeError(f"canal {self.channel} absent de channel_map.json")
            name = f"{self.name}_{datetime.now():%Y%m%d_%H%M%S}"
            self.session = {"name": name, "started": datetime.now().isoformat(timespec="seconds")}
            t_key, t_conf = self.pix.start()
            started = True
            self.t_video0 = t_key
            self.recording = True
            self.log_event("pix_record_start_key", t_key)
            self.log_event("pix_record_confirmed", t_conf, f"{1000 * (t_conf - t_key):.0f} ms")
            t_trig = self.trigger("trigger_start")
            self.session.update(pix_control=self.pix.mode, start_confirm_ms=round(1000 * (t_conf - t_key), 1),
                                trigger_start_video_s=round(t_trig - t_key, 3))
            self.post("log", f"Enregistrement demarre ({name}), trigger a t={t_trig - t_key:.3f} s dans la video")
        except Exception as exc:
            self.post("log", f"ERREUR demarrage : {exc}")
            if started:  # PIX enregistre deja : on arrete et on garde le fichier, marque en erreur
                try:
                    self.pix.stop()
                    path = self.pix.save(self.out_dir / f"{self.session['name']}_ERREUR.ravi")
                    self.post("log", f"Enregistrement arrete et sauve sous {path.name}")
                except Exception as exc2:
                    self.post("log", f"ERREUR arret de secours : {exc2} (verifier PIX Connect)")
            self.recording = False
            self.t_video0 = None
        finally:
            self.post("done")

    def stop_sequence(self):
        try:
            t_trig = self.trigger("trigger_stop")
            time.sleep(self.width + 0.3)  # le trigger de fin doit etre entierement dans la video
            t_key, t_last = self.pix.stop()
            self.recording = False
            self.log_event("pix_record_stop_key", t_key)
            self.session.update(trigger_stop_video_s=round(t_trig - self.t_video0, 3),
                                stop_key_video_s=round(t_key - self.t_video0, 3))
            self.post("log", f"Trigger de fin a t={t_trig - self.t_video0:.3f} s, arret, sauvegarde...")
            path = self.pix.save(self.out_dir / f"{self.session['name']}.ravi")
            frames, fps = pc.ravi_info(path)
            counters = pc.ravi_frame_counters(path)
            dropped = (counters[-1] - counters[0] + 1 - len(counters)) if counters else None
            if dropped:
                self.post("log", f"ATTENTION : {dropped} image(s) perdue(s), utiliser le compteur camera")
            self.session.update(file=str(path), frames=frames, fps=round(fps, 3),
                                video_duration_s=round(frames / fps, 3),
                                camera_counter_first=counters[0] if counters else None,
                                camera_counter_last=counters[-1] if counters else None,
                                frames_dropped=dropped,
                                channel=self.channel, trigger_width_s=self.width,
                                note="video_s = temps depuis la commande de demarrage (~1re image) ; decalage "
                                     "mesure < 1-2 images ; "
                                     "temps d'une image = (compteur camera - camera_counter_first) / fps")
            path.with_name(path.stem + "_sync.json").write_text(json.dumps(self.session, indent=2), encoding="utf-8")
            self.log_event("saved", time.perf_counter(), str(path))
            self.post("log", f"Sauvegarde : {path.name} ({frames} images, {frames / fps:.2f} s)")
        except Exception as exc:
            self.post("log", f"ERREUR arret/sauvegarde : {exc}")
        finally:
            self.t_video0 = None
            self.post("done")

    def on_close(self):
        if self.recording:
            self.write_log("Arret de l'enregistrement avant fermeture...")
            self.stop_sequence()
        if self.box is not None:
            self.box.close()
        self.root.destroy()


def main():
    root = tk.Tk()
    SyncApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
