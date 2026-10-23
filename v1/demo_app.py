"""
Demonstrateur temps reel : video thermique + signes vitaux pendant l'acquisition synchronisee.

  - Retour video (PIX Connect via Connect SDK/IPC) avec cercle-guide : le participant garde la tete
    dedans (vert = bonne position, rouge = a corriger, consigne affichee sur la video).
  - Duree d'acquisition, temperature du front et du bout du nez, frequence cardiaque (bpm) et
    frequence respiratoire (flux optique ; controle croise par la temperature sous les narines).
  - Panneau d'enregistrement (sync_app) : demarrage PIX Connect + trigger Biopac, arret, sauvegarde.
  - Webcam (RGB) a cote : frequence cardiaque par rPPG (methode POS) pour comparer avec le thermique ;
    courbe des deux frequences cardiaques sur 2 min.
  - Pendant l'enregistrement : <nom>_vitals.csv a cote du .ravi (temps video, valeurs par image).

Methodes et references : voir vitals.py (thermique) et rppg.py (webcam).

Lancement :
    python demo_app.py                    # direct (PIX Connect : Connect SDK (IPC) active)
    python demo_app.py --ravi fichier.ravi  # relecture d'un enregistrement (demo / test)
    python demo_app.py --no-webcam         # sans webcam
"""

import argparse
import csv
import threading
import time
import tkinter as tk
from collections import deque
from pathlib import Path
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

import vitals as vt
from sync_app import SyncApp

VIDEO_SCALE = 2
LIVE_SPAN = (24.0, 37.0)  # degC pour la palette en direct
COL_OK, COL_WARN, COL_BAD, COL_DIM = "#1a7f37", "#d9480f", "#b42318", "#8a8a8a"


class FrameSource(threading.Thread):
    """Lit les images (IPC en direct ou .ravi en relecture) et calcule les signes vitaux."""

    def __init__(self, ravi: str | None):
        super().__init__(daemon=True)
        self.ravi = ravi
        self.lock = threading.Lock()
        self.latest = None          # (image BGR, Vitals copie, unite)
        self.status = "Connexion..."
        self.stop_flag = False
        self.on_frame = None        # callback(t_perf, vitals, cam_timestamp) pour le journal

    def publish(self, img, proc, v, unit, span):
        frame = vt.render(img, proc, v, VIDEO_SCALE, span, celsius=(unit == "degC"))
        snapshot = vt.Vitals(**{k: getattr(v, k) for k in vt.Vitals.__dataclass_fields__})
        with self.lock:
            self.latest = (frame, snapshot, unit, proc.warmup_left())

    def run(self):
        try:
            self.run_ravi() if self.ravi else self.run_ipc()
        except Exception as exc:
            self.status = f"Erreur source video : {exc}"

    def run_ravi(self):
        rd = vt.RaviReader(self.ravi)
        proc = vt.VitalsProcessor(rd.fps, rd.width, rd.height)
        self.status = f"Relecture : {Path(self.ravi).name}"
        t_start = time.perf_counter()
        span = None
        for i, (img, counter) in enumerate(rd.frames()):
            if self.stop_flag:
                break
            t = i / rd.fps
            wait = t_start + t - time.perf_counter()  # cadence temps reel
            if wait > 0:
                time.sleep(wait)
            if span is None:
                span = (float(np.percentile(img, 1)), float(np.percentile(img, 99.5)))
            v = proc.process(img, t)
            if self.on_frame:
                self.on_frame(time.perf_counter(), v, None)
            self.publish(img, proc, v, "u.a.", span)
        self.status = "Fin de la relecture"

    def run_ipc(self):
        from pix_ipc import get_hub
        hub = get_hub()  # connexion IPC partagee avec la commande d'enregistrement (sync_app)
        state = {"proc": None}

        def on_ipc_frame(temps, meta, t_recv):
            if self.stop_flag:
                return
            h, w = temps.shape
            proc = state["proc"]
            if proc is None or (proc.w, proc.h) != (w, h):
                proc = state["proc"] = vt.VitalsProcessor(27.0, w, h)
            v = proc.process(temps, meta.Timestamp / 1e7)  # horodatage camera (s)
            if self.on_frame:
                self.on_frame(t_recv, v, meta.Timestamp)
            self.publish(temps, proc, v, "degC", LIVE_SPAN)

        hub.subscribe(on_ipc_frame)
        while not self.stop_flag:
            if hub.streaming():
                self.status = "Direct : PIX Connect via IPC"
            else:
                self.status = ("En attente des images PIX Connect (PIX Connect ouvert ? Tools > Configuration > "
                               "External Communication > Connect SDK (IPC))" + (f" - {hub.error}" if hub.error else ""))
            time.sleep(0.5)


class DemoApp:
    VIDEO_W, VIDEO_H = 440, 330
    PLOT_S = 120  # secondes affichees sur la courbe de comparaison des frequences cardiaques

    def __init__(self, root: tk.Tk, ravi: str | None, webcam: int | None = 0):
        self.root = root
        root.title("Demonstrateur - signes vitaux thermiques et webcam")
        style = ttk.Style()
        style.configure("Big.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure("Name.TLabel", font=("Segoe UI", 10))
        style.configure("Small.TLabel", font=("Segoe UI", 8))

        main = ttk.Frame(root, padding=8)
        main.pack(fill="both", expand=True)
        left = ttk.Frame(main)
        left.pack(side="left", fill="both")
        right = ttk.Frame(main, padding=(10, 0, 0, 0))
        right.pack(side="left", fill="both", expand=True)

        # --- videos : thermique (IR) et webcam (RGB) cote a cote ---
        vids = ttk.Frame(left)
        vids.pack()
        for col, title in enumerate(("Camera thermique (IR)", "Webcam (RGB)")):
            ttk.Label(vids, text=title, style="Name.TLabel").grid(row=0, column=col, sticky="w")
        self.canvas_ir = tk.Label(vids, bg="black")
        self.canvas_ir.grid(row=1, column=0, padx=(0, 6))
        self.canvas_rgb = tk.Label(vids, bg="black")
        self.canvas_rgb.grid(row=1, column=1)
        blank = ImageTk.PhotoImage(Image.new("RGB", (self.VIDEO_W, self.VIDEO_H)))
        self.photo_ir = self.photo_rgb = blank
        self.canvas_ir.configure(image=blank)
        self.canvas_rgb.configure(image=blank)
        self.lbl_msg = tk.Label(vids, text="", font=("Segoe UI", 16, "bold"), fg="white", bg="#333", pady=4)
        self.lbl_msg.grid(row=2, column=0, sticky="we", padx=(0, 6))
        self.lbl_msg_rgb = tk.Label(vids, text="", font=("Segoe UI", 16, "bold"), fg="white", bg="#333", pady=4)
        self.lbl_msg_rgb.grid(row=2, column=1, sticky="we")

        self.plots = {}
        for key, title in (("hr", "Frequence cardiaque (bpm) : thermique (orange) vs webcam (bleu)"),
                           ("rr", "Frequence respiratoire (/min) : thermique (orange) vs webcam (bleu)")):
            ttk.Label(left, text=title, style="Name.TLabel").pack(anchor="w", pady=(6, 0))
            cv = tk.Canvas(left, width=2 * self.VIDEO_W + 6, height=115, bg="white", highlightthickness=1,
                           highlightbackground="#ccc")
            cv.pack()
            self.plots[key] = cv
        self.lbl_status = ttk.Label(left, text="", style="Small.TLabel")
        self.lbl_status.pack(anchor="w", pady=2)

        # --- tuiles de mesures ---
        tiles = ttk.Frame(right)
        tiles.pack(fill="x")
        self.tiles = {}
        layout = [("dur", "Duree d'acquisition", 0, 0, 2),
                  ("rr", "Freq. respiratoire IR", 1, 0, 1), ("rr_rgb", "Freq. respiratoire webcam", 1, 1, 1),
                  ("hr", "Freq. cardiaque IR", 2, 0, 1), ("hr_rgb", "Freq. cardiaque webcam", 2, 1, 1),
                  ("tf", "Temperature front", 3, 0, 1), ("tn", "Temperature nez", 3, 1, 1)]
        for key, name, row, col, span in layout:
            box = ttk.Frame(tiles, padding=5, relief="groove")
            box.grid(row=row, column=col, columnspan=span, sticky="nsew", padx=3, pady=3)
            ttk.Label(box, text=name, style="Name.TLabel").pack(anchor="w")
            val = ttk.Label(box, text="--", style="Big.TLabel")
            val.pack(anchor="w")
            sub = ttk.Label(box, text="", style="Small.TLabel", wraplength=180)
            sub.pack(anchor="w")
            self.tiles[key] = (val, sub)
        tiles.columnconfigure(0, weight=1, uniform="t")
        tiles.columnconfigure(1, weight=1, uniform="t")
        self.lbl_cmp = ttk.Label(right, text="", style="Name.TLabel")
        self.lbl_cmp.pack(anchor="w", pady=(2, 0))

        # --- panneau d'enregistrement synchronise ---
        rec = ttk.LabelFrame(right, text="Enregistrement synchronise (PIX Connect + trigger Biopac)", padding=6)
        rec.pack(fill="both", expand=True, pady=(6, 0))
        self.sync = SyncApp(root, parent=rec, log_height=4)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.source = FrameSource(ravi)
        self.source.on_frame = self.log_frame
        self.webcam = None
        if webcam is not None:
            from rppg import WebcamSource
            self.webcam = WebcamSource(webcam)
            self.webcam.start()
        self.hist = deque()  # (perf_counter, hr_ir, hr_rgb) pour la courbe
        self._last_hist = 0.0
        self.rgb_state = None
        self.vitals_file = None
        self.vitals_writer = None
        self.source.start()
        root.after(40, self.refresh)

    # --- journal des signes vitaux pendant l'enregistrement ---
    def log_frame(self, t_perf, v: vt.Vitals, cam_ts):
        t0 = self.sync.t_video0
        if self.sync.recording and t0 is not None:
            if self.vitals_writer is None:
                path = self.sync.out_dir / f"{self.sync.session.get('name', 'session')}_vitals.csv"
                self.vitals_file = open(path, "w", newline="")
                self.vitals_writer = csv.writer(self.vitals_file)
                self.vitals_writer.writerow(["video_time_s", "camera_timestamp_100ns", "in_circle", "tracking",
                                             "t_forehead", "t_nose", "rr_flow_bpm", "rr_flow_q", "rr_temp_bpm",
                                             "rr_temp_q", "hr_ir_bpm", "hr_ir_q", "hr_webcam_bpm", "hr_webcam_q",
                                             "rr_webcam_bpm", "rr_webcam_q", "rr_webcam_source"])
            f = lambda x: "" if x is None else f"{x:.3f}"
            rs = self.rgb_state
            self.vitals_writer.writerow([f"{t_perf - t0:.3f}", cam_ts if cam_ts is not None else "", int(v.in_circle),
                                         int(v.tracking), f(v.t_forehead), f(v.t_nose), f(v.rr_flow),
                                         f"{v.rr_flow_q:.2f}", f(v.rr_temp), f"{v.rr_temp_q:.2f}", f(v.hr),
                                         f"{v.hr_q:.2f}", f(rs.hr if rs else None), f"{rs.hr_q:.2f}" if rs else "",
                                         f(rs.rr if rs else None), f"{rs.rr_q:.2f}" if rs else "",
                                         rs.rr_source if rs else ""])
        elif self.vitals_writer is not None:
            self.vitals_file.close()
            self.vitals_file = self.vitals_writer = None

    # --- rafraichissement de l'interface ---
    def set_tile(self, key, text, sub="", color=None):
        val, lab = self.tiles[key]
        val.configure(text=text, foreground=color or "")
        lab.configure(text=sub)

    def show(self, label, frame_bgr, attr):
        img = cv2.resize(frame_bgr, (self.VIDEO_W, self.VIDEO_H), interpolation=cv2.INTER_AREA)
        photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        setattr(self, attr, photo)  # garder une reference, sinon Tk efface l'image
        label.configure(image=photo)

    def draw_plot(self, key, lo, hi, step, i_ir, i_rgb):
        c = self.plots[key]
        c.delete("all")
        w, h = int(c["width"]), int(c["height"])
        y = lambda val: h - 12 - (val - lo) / (hi - lo) * (h - 20)
        for val in range(lo, hi + 1, step):
            c.create_line(35, y(val), w - 5, y(val), fill="#eee")
            c.create_text(18, y(val), text=str(val), fill="#888", font=("Segoe UI", 7))
        now = time.perf_counter()
        x = lambda t: 35 + (t - (now - self.PLOT_S)) / self.PLOT_S * (w - 40)
        for idx, color in ((i_ir, "#e8590c"), (i_rgb, "#1c7ed6")):
            pts = [(p[0], x(p[0]), y(min(max(p[idx], lo), hi))) for p in self.hist if p[idx] is not None]
            for a, b in zip(pts, pts[1:]):
                if b[0] - a[0] < 2.5:  # pas de trait au-dessus d'un trou
                    c.create_line(a[1], a[2], b[1], b[2], fill=color, width=2)
        c.create_text(w - 8, 8, anchor="ne", text=f"{self.PLOT_S} s", fill="#888", font=("Segoe UI", 7))

    def refresh(self):
        with self.source.lock:
            latest = self.source.latest
        status = self.source.status
        v = None
        if latest is not None:
            frame, v, unit, (wait_rr, wait_hr) = latest
            self.show(self.canvas_ir, frame, "photo_ir")
            self.lbl_msg.configure(text=v.message, bg=COL_OK if v.in_circle else (COL_WARN if v.tracking else COL_BAD))

            deg = unit == "degC"
            fmt_t = lambda x: "--" if x is None else (f"{x:.1f} °C" if deg else f"{x:.0f}")
            self.set_tile("tf", fmt_t(v.t_forehead), "" if deg else "relecture : valeurs brutes")
            self.set_tile("tn", fmt_t(v.t_nose), "" if deg else "relecture : valeurs brutes")
            if v.rr_flow is None:
                self.set_tile("rr", "--", f"calcul dans {wait_rr:.0f} s" if wait_rr > 0 else "signal insuffisant")
            else:
                agree = v.rr_temp is not None and abs(v.rr_temp - v.rr_flow) <= 3
                cross = f"T° narines {v.rr_temp:.0f}" if v.rr_temp else "T° narines --"
                self.set_tile("rr", f"{v.rr_flow:.0f} /min",
                              f"flux optique, q {v.rr_flow_q:.0%} | {cross}" + (" (concordant)" if agree else ""),
                              None if v.rr_flow_q >= 0.5 else COL_DIM)
            if v.hr is None:
                self.set_tile("hr", "--", f"calcul dans {wait_hr:.0f} s" if wait_hr > 0 else "signal insuffisant")
            else:
                good = v.hr_q >= 0.30
                self.set_tile("hr", f"{v.hr:.0f} bpm", f"T° front, q {v.hr_q:.0%}" + ("" if good else " - peu fiable"),
                              None if good else COL_DIM)

        rs = None
        if self.webcam is not None:
            with self.webcam.lock:
                wl = self.webcam.latest
            status += "  |  " + self.webcam.status
            if wl is not None:
                frame_rgb, rs = wl
                self.rgb_state = rs
                self.show(self.canvas_rgb, frame_rgb, "photo_rgb")
                txt = rs.message.replace("Webcam : ", "").capitalize()
                if rs.face and not rs.frontal:
                    txt = "Visage tourne"
                self.lbl_msg_rgb.configure(text=txt, bg=(COL_OK if rs.frontal else COL_WARN) if rs.face else COL_BAD)
                # la webcam voit le visage mais pas de face : la position thermique n'est pas ideale non plus
                if v is not None and v.in_circle and rs.face and not rs.frontal:
                    self.lbl_msg.configure(text="Mesure approchee - regardez la camera", bg=COL_WARN)
                if rs.hr is None:
                    self.set_tile("hr_rgb", "--", f"calcul dans {rs.warmup_left:.0f} s" if rs.face else "aucun visage")
                else:
                    good = rs.hr_q >= 0.30
                    self.set_tile("hr_rgb", f"{rs.hr:.0f} bpm", f"rPPG (POS), q {rs.hr_q:.0%}" +
                                  ("" if good else " - peu fiable"), None if good else COL_DIM)
                if rs.rr is None:
                    self.set_tile("rr_rgb", "--", f"calcul dans {rs.rr_warmup_left:.0f} s" if rs.face else "aucun visage")
                else:
                    good = rs.rr_q >= 0.5
                    self.set_tile("rr_rgb", f"{rs.rr:.0f} /min", f"flux optique {rs.rr_source}, q {rs.rr_q:.0%}" +
                                  ("" if good else " - peu fiable"), None if good else COL_DIM)
        else:
            self.set_tile("hr_rgb", "--", "webcam desactivee")
            self.set_tile("rr_rgb", "--", "webcam desactivee")

        hr_ir = v.hr if v is not None else None
        hr_rgb = rs.hr if rs is not None else None
        rr_ir = v.rr_flow if v is not None else None
        rr_rgb = rs.rr if rs is not None else None
        cmp = []
        if rr_ir is not None and rr_rgb is not None:
            d = rr_ir - rr_rgb
            cmp.append(f"FR IR - webcam : {d:+.0f} /min" + (" (concordant)" if abs(d) <= 3 else ""))
        if hr_ir is not None and hr_rgb is not None:
            d = hr_ir - hr_rgb
            cmp.append(f"FC IR - webcam : {d:+.0f} bpm" + (" (concordant)" if abs(d) <= 5 else ""))
        self.lbl_cmp.configure(text="   |   ".join(cmp))
        now = time.perf_counter()
        if now - self._last_hist >= 1.0:
            self._last_hist = now
            self.hist.append((now, hr_ir, hr_rgb, rr_ir, rr_rgb))
            while self.hist and self.hist[0][0] < now - self.PLOT_S:
                self.hist.popleft()
            self.draw_plot("hr", 40, 140, 20, 1, 2)
            self.draw_plot("rr", 0, 40, 10, 3, 4)

        self.lbl_status.configure(text=status)
        if self.sync.recording and self.sync.t_video0:
            el = time.perf_counter() - self.sync.t_video0
            self.set_tile("dur", f"{int(el // 60):02d}:{int(el % 60):02d}", "enregistrement en cours", COL_BAD)
        else:
            self.set_tile("dur", "--:--", "pas d'enregistrement")
        self.root.after(40, self.refresh)

    def on_close(self):
        self.source.stop_flag = True
        if self.webcam is not None:
            self.webcam.stop_flag = True
        if self.vitals_file:
            self.vitals_file.close()
        self.sync.on_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ravi", help="Relire un enregistrement .ravi au lieu du direct")
    parser.add_argument("--webcam", type=int, default=0, help="Index de la webcam (defaut 0)")
    parser.add_argument("--no-webcam", action="store_true", help="Desactiver la webcam")
    args = parser.parse_args()
    root = tk.Tk()
    DemoApp(root, args.ravi, None if args.no_webcam else args.webcam)
    root.mainloop()


if __name__ == "__main__":
    main()
