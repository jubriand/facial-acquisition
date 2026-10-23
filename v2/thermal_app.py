"""
Demonstrateur v2 - thermique seule (camera Optris via PIX Connect, Connect SDK / IPC).

Affiche :
  - la video thermique avec deux ROI suivies : front et nez, temperature affichee
    directement sur l'image ;
  - la temperature globale du visage (moyenne et maximum des pixels de peau) ;
  - la frequence cardiaque (bpm, modulation pulsatile de la temperature du front) ;
  - la frequence respiratoire (flux optique narines/bouche, controle par la temperature sous les narines) ;
  - les courbes des deux frequences sur 2 minutes ;
  - une pastille REC / STOP : enregistre a chaque image les temperatures (visage, front, nez), la
    frequence respiratoire et les bpm dans un CSV (<nom>_<date>_<heure>.csv) ; a l'arret, une fenetre
    affiche les graphiques empiles sur le meme axe temporel (0 = lancement), enregistres aussi en PNG.

Pas de Biopac, pas de webcam, pas d'enregistrement video PIX Connect (voir la v1 pour cela).
Methodes et references : voir vitals.py.

Lancement :
    python thermal_app.py                      # direct (PIX Connect ouvert, Connect SDK (IPC) active)
    python thermal_app.py --ravi fichier.ravi  # relecture d'un enregistrement
"""

import argparse
import csv
import threading
import time
import tkinter as tk
from collections import deque
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

import vitals as vt

VIDEO_SCALE = 2
LIVE_SPAN = (24.0, 37.0)  # degC pour la palette en direct
COL_DIM = "#8a8a8a"  # valeur peu fiable (qualite insuffisante)
DEFAULT_DIR = Path(__file__).resolve().parent.parent / "acquisitions"
# pastille REC : (fond, texte, fond au survol)
REC_IDLE = ("#ecebe7", "#3a3936", "#e1e0d9")
REC_ON = ("#d03b3b", "#ffffff", "#b83232")
# pastille de phase : (libelle, fond, texte, fond au survol) ; bascule au clic ou avec la touche P
PHASES = {"flight": ("✈  FLIGHT PHASE", "#2a78d6", "#ffffff", "#256abf"),
          "landing": ("✈  LANDING PHASE", "#eb6834", "#ffffff", "#d95926")}


class Recorder:
    """Ecrit une ligne CSV par image pendant l'acquisition (appele depuis le thread video)."""

    HEADER = ["timestamp_iso", "elapsed_s", "camera_time_s", "unit", "face_tracked",
              "t_face", "t_face_max", "t_forehead", "t_nose",
              "rr_per_min", "rr_quality", "rr_nostril_temp_per_min", "hr_bpm", "hr_quality", "phase"]

    def __init__(self):
        self.lock = threading.Lock()
        self.file = self.writer = None
        self.path = None
        self.t0 = None
        self.rows = 0
        self.phase = "flight"  # phase courante, ecrite sur chaque ligne (modifiee depuis l'interface)

    @property
    def active(self) -> bool:
        return self.writer is not None

    def start(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            self.file = open(path, "w", newline="", encoding="utf-8")
            self.writer = csv.writer(self.file)
            self.writer.writerow(self.HEADER)
            self.path, self.t0, self.rows = path, time.perf_counter(), 0

    def write(self, cam_t: float, v: vt.Vitals, celsius: bool):
        with self.lock:
            if self.writer is None:
                return
            f = lambda x, n=3: "" if x is None else f"{x:.{n}f}"
            self.writer.writerow([
                datetime.now().isoformat(timespec="milliseconds"), f"{time.perf_counter() - self.t0:.3f}",
                f"{cam_t:.3f}", "degC" if celsius else "brut", int(v.tracking),
                f(v.t_face), f(v.t_face_max), f(v.t_forehead), f(v.t_nose),
                f(v.rr_flow, 2), f(v.rr_flow_q, 2), f(v.rr_temp, 2), f(v.hr, 2), f(v.hr_q, 2), self.phase])
            self.rows += 1
            if self.rows % 50 == 0:
                self.file.flush()  # limite les pertes si l'application est fermee brutalement

    def stop(self):
        with self.lock:
            if self.file is not None:
                self.file.close()
            self.file = self.writer = None
            return self.path, self.rows


class FrameSource(threading.Thread):
    """Images thermiques (IPC en direct ou .ravi en relecture) + calcul des signes vitaux."""

    def __init__(self, ravi: str | None):
        super().__init__(daemon=True)
        self.ravi = ravi
        self.lock = threading.Lock()
        self.latest = None  # (image BGR annotee, Vitals copie, degC ?, attente avant estimation)
        self.status = "Connexion..."
        self.stop_flag = False
        self.recorder = Recorder()

    def publish(self, img, proc, v, celsius, span, cam_t):
        frame = vt.render(img, proc, v, VIDEO_SCALE, span, celsius)
        snapshot = vt.Vitals(**{k: getattr(v, k) for k in vt.Vitals.__dataclass_fields__})
        self.recorder.write(cam_t, snapshot, celsius)
        with self.lock:
            self.latest = (frame, snapshot, celsius, proc.warmup_left())

    def run(self):
        try:
            self.run_ravi() if self.ravi else self.run_ipc()
        except Exception as exc:
            self.status = f"Erreur source video : {exc}"

    def run_ravi(self):
        rd = vt.RaviReader(self.ravi)
        proc = vt.VitalsProcessor(rd.fps, rd.width, rd.height)
        self.status = f"Relecture : {Path(self.ravi).name} (valeurs brutes camera, pas de degC)"
        t_start, span = time.perf_counter(), None
        for i, (img, _) in enumerate(rd.frames()):
            if self.stop_flag:
                break
            t = i / rd.fps
            wait = t_start + t - time.perf_counter()  # cadence temps reel
            if wait > 0:
                time.sleep(wait)
            if span is None:
                span = (float(np.percentile(img, 1)), float(np.percentile(img, 99.5)))
            self.publish(img, proc, proc.process(img, t), False, span, t)
        self.status = "Fin de la relecture"

    def run_ipc(self):
        from pix_ipc import get_hub
        hub = get_hub()
        state = {"proc": None}

        def on_frame(temps, meta, t_recv):
            if self.stop_flag:
                return
            h, w = temps.shape
            if state["proc"] is None or (state["proc"].w, state["proc"].h) != (w, h):
                state["proc"] = vt.VitalsProcessor(27.0, w, h)
            proc = state["proc"]
            cam_t = meta.Timestamp / 1e7  # horodatage camera (s)
            v = proc.process(temps, cam_t)
            self.publish(temps, proc, v, True, LIVE_SPAN, cam_t)

        hub.subscribe(on_frame)
        while not self.stop_flag:
            if hub.streaming():
                self.status = "Direct : PIX Connect via IPC"
            else:
                self.status = ("En attente des images PIX Connect (PIX Connect ouvert ? Connect SDK (IPC) active ? "
                               "Une autre application utilise peut-etre deja la connexion)" +
                               (f" - {hub.error}" if hub.error else ""))
            time.sleep(0.5)
        hub.stop_flag = True


class ThermalApp:
    PLOT_S = 120

    def __init__(self, root: tk.Tk, ravi: str | None):
        self.root = root
        root.title("Demonstrateur v2 - thermique")
        style = ttk.Style()
        style.configure("Big.TLabel", font=("Segoe UI", 24, "bold"))
        style.configure("Name.TLabel", font=("Segoe UI", 11))
        style.configure("Small.TLabel", font=("Segoe UI", 9))

        main = ttk.Frame(root, padding=8)
        main.pack(fill="both", expand=True)
        left = ttk.Frame(main)
        left.pack(side="left", fill="y")
        right = ttk.Frame(main, padding=(12, 0, 0, 0))
        right.pack(side="left", fill="both", expand=True)

        self.canvas = tk.Label(left, bg="black")
        self.canvas.pack()
        self.photo = ImageTk.PhotoImage(Image.new("RGB", (764, 576)))
        self.canvas.configure(image=self.photo)
        self.lbl_status = ttk.Label(left, text="", style="Small.TLabel", wraplength=760)  # ne pas elargir la fenetre
        self.lbl_status.pack(anchor="w", pady=2)

        # --- acquisition (enregistrement des valeurs en CSV) ---
        acq = ttk.Frame(left, padding=(0, 6, 0, 0))
        acq.pack(fill="x")
        self.var_name = tk.StringVar(value="participant01")
        self.var_dir = tk.StringVar(value=str(DEFAULT_DIR))
        # pastille d'enregistrement compacte : gris "REC" au repos, rouge "STOP" pendant l'acquisition
        self.btn_acq = tk.Button(acq, text="●  REC", command=self.toggle_acq, font=("Segoe UI", 9, "bold"),
                                 relief="flat", bd=0, padx=12, pady=3, cursor="hand2",
                                 bg=REC_IDLE[0], fg=REC_IDLE[1], activebackground=REC_IDLE[2],
                                 activeforeground=REC_IDLE[1])
        self.btn_acq.pack(side="left")
        self.btn_acq.bind("<Enter>", lambda e: self._rec_hover(True))
        self.btn_acq.bind("<Leave>", lambda e: self._rec_hover(False))
        self.lbl_timer = ttk.Label(acq, text="", font=("Consolas", 10, "bold"), width=6)
        self.lbl_timer.pack(side="left", padx=(6, 6))
        # pastille de phase de vol : Flight (bleu) <-> Landing (orange), clic ou touche P
        self.btn_phase = tk.Button(acq, command=self.toggle_phase, font=("Segoe UI", 9, "bold"), relief="flat",
                                   bd=0, padx=12, pady=3, cursor="hand2")
        self.btn_phase.pack(side="left", padx=(0, 10))
        self.btn_phase.bind("<Enter>", lambda e: self._phase_style(True))
        self.btn_phase.bind("<Leave>", lambda e: self._phase_style(False))
        root.bind("<KeyPress-p>", self._phase_key)
        root.bind("<KeyPress-P>", self._phase_key)
        root.bind("<KeyRelease-p>", self._phase_key_up)
        root.bind("<KeyRelease-P>", self._phase_key_up)
        self._p_down, self._p_last = False, 0.0
        ttk.Label(acq, text="Nom", style="Small.TLabel").pack(side="left")
        ttk.Entry(acq, textvariable=self.var_name, width=16).pack(side="left", padx=(4, 10))
        ttk.Label(acq, text="Dossier", style="Small.TLabel").pack(side="left")
        ttk.Entry(acq, textvariable=self.var_dir, width=30).pack(side="left", padx=4, fill="x", expand=True)
        ttk.Button(acq, text="…", width=2, command=self.choose_dir).pack(side="left")
        self.lbl_saved = ttk.Label(left, text="", style="Small.TLabel", wraplength=760)
        self.lbl_saved.pack(anchor="w", pady=(3, 0))

        # --- mesures ---
        self.tiles = {}
        for key, name in (("face", "Temperature du visage"), ("tf", "Front"), ("tn", "Nez"),
                          ("hr", "Frequence cardiaque"), ("rr", "Frequence respiratoire")):
            box = ttk.Frame(right, padding=6, relief="groove")
            box.pack(fill="x", pady=3)
            ttk.Label(box, text=name, style="Name.TLabel").pack(anchor="w")
            val = ttk.Label(box, text="--", style="Big.TLabel")
            val.pack(anchor="w")
            sub = ttk.Label(box, text="", style="Small.TLabel", wraplength=330)
            sub.pack(anchor="w")
            self.tiles[key] = (val, sub)

        # --- courbes ---
        self.plots = {}
        for key, title in (("hr", "Frequence cardiaque (bpm) - 2 min"), ("rr", "Frequence respiratoire (/min) - 2 min")):
            ttk.Label(right, text=title, style="Name.TLabel").pack(anchor="w", pady=(6, 0))
            cv = tk.Canvas(right, width=360, height=95, bg="white", highlightthickness=1, highlightbackground="#ccc")
            cv.pack(anchor="w")
            self.plots[key] = cv
        self.hist = deque()
        self._last_hist = 0.0

        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.source = FrameSource(ravi)
        self._phase_style(False)
        self.source.start()
        self._after_id = root.after(40, self.refresh)

    def choose_dir(self):
        d = filedialog.askdirectory(initialdir=self.var_dir.get())
        if d:
            self.var_dir.set(d)

    def _rec_hover(self, inside: bool):
        bg, fg, hover = REC_ON if self.source.recorder.active else REC_IDLE
        self.btn_acq.configure(bg=hover if inside else bg, fg=fg, activebackground=hover, activeforeground=fg)

    def _phase_style(self, hover: bool):
        label, bg, fg, bg_hover = PHASES[self.source.recorder.phase]
        self.btn_phase.configure(text=label, bg=bg_hover if hover else bg, fg=fg, activebackground=bg_hover,
                                 activeforeground=fg)

    def toggle_phase(self):
        rec = self.source.recorder
        rec.phase = "landing" if rec.phase == "flight" else "flight"
        self._phase_style(False)

    def _phase_key(self, event):
        # pas de bascule quand on tape un "p" dans le nom ou le dossier
        if isinstance(event.widget, (tk.Entry, ttk.Entry)):
            return
        # touche maintenue : Windows repete l'appui -> une seule bascule par appui, et au plus toutes les 0.4 s
        now = time.perf_counter()
        if self._p_down or now - self._p_last < 0.4:
            return
        self._p_down, self._p_last = True, now
        self.toggle_phase()

    def _phase_key_up(self, event):
        self._p_down = False

    def toggle_acq(self):
        rec = self.source.recorder
        if rec.active:
            path, rows = rec.stop()
            self.btn_acq.configure(text="●  REC")
            self._rec_hover(False)
            self.lbl_timer.configure(text="", foreground="")
            self.lbl_saved.configure(text=f"Enregistre : {path} ({rows} lignes)")
            if rows:
                from results_view import show_results
                png = show_results(self.root, path)
                if png:
                    self.lbl_saved.configure(text=f"Enregistre : {path} ({rows} lignes) + graphiques {png.name}")
        else:
            name = self.var_name.get().strip() or "acquisition"
            path = Path(self.var_dir.get()) / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.csv"
            try:
                rec.phase = "flight"  # chaque acquisition commence en phase de vol
                self._phase_style(False)
                rec.start(path)
            except OSError as exc:
                self.lbl_saved.configure(text=f"Impossible de creer le fichier : {exc}")
                return
            self.btn_acq.configure(text="■  STOP")
            self._rec_hover(False)
            self.lbl_saved.configure(text=f"Acquisition en cours -> {path}")

    def set_tile(self, key, text, sub="", color=None):
        val, lab = self.tiles[key]
        val.configure(text=text, foreground=color or "")
        lab.configure(text=sub)

    def draw_plot(self, key, idx, lo, hi, step):
        c = self.plots[key]
        c.delete("all")
        w, h = int(c["width"]), int(c["height"])
        y = lambda val: h - 10 - (min(max(val, lo), hi) - lo) / (hi - lo) * (h - 18)
        for val in range(lo, hi + 1, step):
            c.create_line(30, y(val), w - 4, y(val), fill="#eee")
            c.create_text(15, y(val), text=str(val), fill="#888", font=("Segoe UI", 7))
        now = time.perf_counter()
        x = lambda t: 30 + (t - (now - self.PLOT_S)) / self.PLOT_S * (w - 34)
        pts = [(p[0], x(p[0]), y(p[idx])) for p in self.hist if p[idx] is not None]
        for a, b in zip(pts, pts[1:]):
            if b[0] - a[0] < 2.5:
                c.create_line(a[1], a[2], b[1], b[2], fill="#e8590c", width=2)

    def refresh(self):
        with self.source.lock:
            latest = self.source.latest
        self.lbl_status.configure(text=self.source.status)
        v = None
        if latest is not None:
            frame, v, celsius, (wait_rr, wait_hr) = latest
            self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.configure(image=self.photo)
            fmt = lambda x: "--" if x is None else (f"{x:.1f} °C" if celsius else f"{x:.0f}")
            raw = "" if celsius else "relecture : valeurs brutes camera"
            self.set_tile("face", fmt(v.t_face), (f"maximum {fmt(v.t_face_max)}" if v.t_face_max else "") or raw)
            self.set_tile("tf", fmt(v.t_forehead), raw)
            self.set_tile("tn", fmt(v.t_nose), raw)
            if v.hr is None:
                self.set_tile("hr", "--", f"calcul dans {wait_hr:.0f} s" if wait_hr > 0 else "signal insuffisant")
            else:
                good = v.hr_q >= 0.30
                self.set_tile("hr", f"{v.hr:.0f} bpm", f"temperature du front, qualite {v.hr_q:.0%}" +
                              ("" if good else " - peu fiable"), None if good else COL_DIM)
            if v.rr_flow is None:
                self.set_tile("rr", "--", f"calcul dans {wait_rr:.0f} s" if wait_rr > 0 else "signal insuffisant")
            else:
                agree = v.rr_temp is not None and abs(v.rr_temp - v.rr_flow) <= 3
                cross = f"T° narines {v.rr_temp:.0f}" if v.rr_temp else "T° narines --"
                self.set_tile("rr", f"{v.rr_flow:.0f} /min",
                              f"flux optique, qualite {v.rr_flow_q:.0%} | {cross}" + (" (concordant)" if agree else ""),
                              None if v.rr_flow_q >= 0.5 else COL_DIM)
            if not v.tracking:
                for key in ("face", "tf", "tn"):
                    self.set_tile(key, "--", "visage non suivi")

        rec = self.source.recorder
        if rec.active:
            el = time.perf_counter() - rec.t0
            self.lbl_timer.configure(text=f"{int(el // 60):02d}:{int(el % 60):02d}", foreground="#d03b3b")
        now = time.perf_counter()
        if now - self._last_hist >= 1.0:
            self._last_hist = now
            self.hist.append((now, v.hr if v else None, v.rr_flow if v else None))
            while self.hist and self.hist[0][0] < now - self.PLOT_S:
                self.hist.popleft()
            self.draw_plot("hr", 1, 40, 140, 20)
            self.draw_plot("rr", 2, 0, 40, 10)
        self._after_id = self.root.after(40, self.refresh)

    def on_close(self):
        if self.source.recorder.active:  # ne pas perdre une acquisition en cours
            self.source.recorder.stop()
        self.source.stop_flag = True
        if getattr(self, "_after_id", None):
            self.root.after_cancel(self._after_id)
        try:  # fenetres de graphiques encore ouvertes : on libere matplotlib avant de quitter
            import matplotlib.pyplot as plt
            plt.close("all")
        except ImportError:
            pass
        self.root.after(200, self._quit)

    def _quit(self):
        self.root.quit()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ravi", help="Relire un enregistrement .ravi au lieu du direct")
    args = parser.parse_args()
    root = tk.Tk()
    ThermalApp(root, args.ravi)
    root.mainloop()


if __name__ == "__main__":
    main()
