"""
Frequence cardiaque par webcam (photoplethysmographie a distance, rPPG).

  - Visage : MediaPipe FaceLandmarker (478 points, mode video, modele officiel dans models/) ; a defaut,
    detecteur de Haar d'OpenCV (boite).
  - ROI : front + deux joues (zones de peau peu mobiles, bien perfusees), en polygones qui suivent le
    visage (MediaPipe), pixels de peau seulement (seuil YCrCb), moyenne R, G, B par image.
  - Visage de face ou tourne : position du bout du nez par rapport au milieu des yeux (MediaPipe).
  - Signal de pouls : methode POS "Plane-Orthogonal-to-Skin" (Wang et al. 2016,
    doi:10.1109/TBME.2016.2609282) sur fenetres glissantes de 1.6 s, recombinees par recouvrement.
    Principe de la rPPG sous lumiere ambiante : Verkruysse et al. 2008 (doi:10.1364/oe.16.021434).
  - Frequence : pic spectral 0.75-3 Hz (45-180 bpm) sur 15 s, apres reechantillonnage regulier
    (la cadence webcam varie), avec indice de qualite (part de puissance autour du pic).

Frequence respiratoire par webcam :
  - Flux optique dense (Farneback) sur la poitrine/les epaules sous le visage : mouvement vertical
    moyen, corrige du mouvement de la tete, integre en deplacement ; pic spectral 0.1-0.75 Hz sur 30 s.
    Methode "flux optique" de Romano et al. 2021 (doi:10.3390/s21155126), biais -0.03 +/- 1.38 resp/min.
  - Si la poitrine n'est pas dans l'image : mouvement vertical de la tete, couple a la respiration
    (Schrumpf et al. 2019, doi:10.1109/EMBC.2019.8856387).
"""

import threading
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from vitals import (CARDIAC_BAND, MP_EYE_OUTER, MP_FACE_OVAL, MP_FOREHEAD, MP_NOSE_TIP, RESP_BAND, MPFace,
                    spectral_rate)

WINDOW_S, MIN_S, POS_WIN_S = 15.0, 8.0, 1.6
RESP_WINDOW_S, RESP_MIN_S = 30.0, 12.0
# joues (maillage MediaPipe) : zones classiques de la rPPG, peu de poils, bien perfusees
MP_LEFT_CHEEK = [50, 101, 118, 117, 123, 147, 187, 205]
MP_RIGHT_CHEEK = [280, 330, 347, 346, 352, 376, 411, 425]
YAW_FRONTAL = 0.18  # decalage max du bout du nez / milieu des yeux (en ecart entre les yeux) pour "de face"


class MPWebcam:
    """MediaPipe FaceLandmarker en mode video pour le flux webcam (suivi temporel interne)."""

    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self._mp = mp
        opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(MPFace.MODEL)),
                                            running_mode=vision.RunningMode.VIDEO, num_faces=1,
                                            min_face_detection_confidence=0.4, min_face_presence_confidence=0.4,
                                            min_tracking_confidence=0.4)
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)
        self._ts = 0

    def detect(self, frame_bgr: np.ndarray, t: float) -> np.ndarray | None:
        h, w = frame_bgr.shape[:2]
        self._ts = max(self._ts + 1, int(t * 1000))  # horodatages strictement croissants (ms)
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        res = self.landmarker.detect_for_video(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb),
                                               self._ts)
        if not res.face_landmarks:
            return None
        return np.array([(p.x * w, p.y * h) for p in res.face_landmarks[0]], dtype=np.float32)


@dataclass
class RppgState:
    face: bool = False
    frontal: bool = False   # visage vu de face par la webcam (sert aussi a valider la position cote thermique)
    hr: float | None = None
    hr_q: float = 0.0
    rr: float | None = None
    rr_q: float = 0.0
    rr_source: str = ""
    message: str = "Webcam : en attente"
    warmup_left: float = MIN_S
    rr_warmup_left: float = RESP_MIN_S


class RppgProcessor:
    def __init__(self):
        self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.profile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
        self.mp = MPWebcam() if MPFace.available() else None
        self.face_poly = None   # contour du visage (MediaPipe), pour l'affichage
        self.polys = None       # front + joues (MediaPipe) ; None -> rectangles depuis la boite (Haar)
        self.box = None
        self.lost = 0
        self.n = 0
        self.t = deque(maxlen=60 * 30)
        self.rgb = deque(maxlen=60 * 30)
        self.state = RppgState()
        self._last_est = -1e9
        self.rois = []
        # respiration (flux optique poitrine)
        self.chest_roi = None
        self.rt = deque(maxlen=40 * 30)
        self.rdisp = deque(maxlen=40 * 30)
        self._rdisp = 0.0
        self._prev_small = None
        self._rsource = ""
        self._switch = 0
        self._last_rest = -1e9

    def _reset_resp(self):
        self.rt.clear(); self.rdisp.clear()
        self._prev_small, self._rdisp, self.chest_roi = None, 0.0, None
        self.state.rr, self.state.rr_q, self.state.rr_warmup_left = None, 0.0, RESP_MIN_S

    def _respiration(self, frame: np.ndarray, t: float):
        """Mouvement vertical de la poitrine (flux optique), corrige du mouvement de la tete."""
        s = self.state
        H, W = frame.shape[:2]
        x, y, w, h = self.box
        clip = lambda a, b, c, d: (max(0, int(a)), max(0, int(b)), min(W, int(c)), min(H, int(d)))
        if self.polys is not None:  # boite = contour MediaPipe (front -> menton) : poitrine juste sous le menton
            chest = clip(x - 0.4 * w, y + 1.10 * h, x + 1.4 * w, y + 2.0 * h)
        else:                       # boite Haar (sourcils -> menton)
            chest = clip(x - 0.4 * w, y + 1.25 * h, x + 1.4 * w, y + 2.4 * h)
        face = clip(x + 0.2 * w, y + 0.1 * h, x + 0.8 * w, y + 0.9 * h)
        chest_visible = (chest[3] - chest[1]) >= 0.30 * h and (chest[2] - chest[0]) >= 0.8 * w
        # hysteresis : on ne change de zone (poitrine <-> tete) qu'apres ~2 s de condition stable,
        # sinon un visage pres du bord de l'image ferait repartir le calcul a zero en permanence
        want = "poitrine" if chest_visible else "tete"
        self._switch = self._switch + 1 if (want != self._rsource) else 0
        if not self._rsource or self._switch > 60:
            self._reset_resp()
            self._rsource, self._switch = want, 0
        source = self._rsource
        chest_ok = source == "poitrine" and (chest[3] - chest[1]) >= 8
        self.chest_roi = chest if chest_ok else None
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (W // 2, H // 2))
        if self._prev_small is not None:
            ux0, uy0 = min(face[0], chest[0] if chest_ok else W) // 2, face[1] // 2
            ux1 = max(face[2], chest[2] if chest_ok else 0) // 2
            uy1 = (chest[3] if chest_ok else face[3]) // 2
            if ux1 - ux0 > 8 and uy1 - uy0 > 8:
                flow = cv2.calcOpticalFlowFarneback(self._prev_small[uy0:uy1, ux0:ux1], small[uy0:uy1, ux0:ux1],
                                                    None, 0.5, 3, 15, 3, 5, 1.2, 0)
                vy = lambda r: float(flow[r[1] // 2 - uy0:r[3] // 2 - uy0, r[0] // 2 - ux0:r[2] // 2 - ux0, 1].mean())
                vy_face = vy(face)
                self._rdisp += (vy(chest) - vy_face) if chest_ok else vy_face
                self.rt.append(t)
                self.rdisp.append(self._rdisp)
        self._prev_small = small
        s.rr_source = source
        dur = (self.rt[-1] - self.rt[0]) if len(self.rt) > 1 else 0.0
        s.rr_warmup_left = max(0.0, RESP_MIN_S - dur)
        if dur >= RESP_MIN_S and t - self._last_rest >= 1.0:
            self._last_rest = t
            tt = np.asarray(self.rt)
            xx = np.asarray(self.rdisp, dtype=np.float64)
            keep = tt >= tt[-1] - RESP_WINDOW_S
            tt, xx = tt[keep], xx[keep]
            fs = 1.0 / np.median(np.diff(tt))
            grid = np.arange(tt[0], tt[-1], 1.0 / fs)
            s.rr, s.rr_q, _, _ = spectral_rate(np.interp(grid, tt, xx), fs, RESP_BAND)

    def _detect(self, frame):
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        faces = list(self.cascade.detectMultiScale(gray, 1.15, 5, minSize=(90, 90)))
        self.state.frontal = bool(faces)  # visage de face (detecteur frontal) ou non
        if not faces:  # tete tournee : detecteur de profil (et image miroir pour l'autre cote)
            faces = list(self.profile.detectMultiScale(gray, 1.15, 5, minSize=(90, 90)))
            W = gray.shape[1]
            faces += [(W - x - w, y, w, h) for x, y, w, h in
                      self.profile.detectMultiScale(cv2.flip(gray, 1), 1.15, 5, minSize=(90, 90))]
        if not faces:
            return None
        return np.array(max(faces, key=lambda f: f[2] * f[3]), dtype=np.float32)

    def _detect_mp(self, frame, t) -> bool:
        """Points MediaPipe -> boite, polygones front/joues, visage de face ou non. False si pas de visage."""
        P = self.mp.detect(frame, t)
        if P is None:
            return False
        oval = P[MP_FACE_OVAL]
        x0, y0 = oval.min(axis=0)
        x1, y1 = oval.max(axis=0)
        self.box = np.array([x0, y0, x1 - x0, y1 - y0], dtype=np.float32)
        self.face_poly = oval
        self.polys = [P[MP_FOREHEAD], cv2.convexHull(P[MP_LEFT_CHEEK]).reshape(-1, 2),
                      cv2.convexHull(P[MP_RIGHT_CHEEK]).reshape(-1, 2)]
        mid = (P[MP_EYE_OUTER[0]] + P[MP_EYE_OUTER[1]]) / 2
        iod = float(np.linalg.norm(P[MP_EYE_OUTER[0]] - P[MP_EYE_OUTER[1]])) or 1.0
        self.state.frontal = abs(float(P[MP_NOSE_TIP][0] - mid[0])) / iod < YAW_FRONTAL
        return True

    def process(self, frame: np.ndarray, t: float) -> RppgState:
        s = self.state
        if self.mp is not None and self._detect_mp(frame, t):
            self.lost = 0
        elif self.mp is not None and self.lost < 6 and self.box is not None:
            self.lost += 1  # quelques images manquees : on garde la derniere position
        elif self.box is None or self.n % 5 == 0:
            self.polys = self.face_poly = None  # repli : detecteur de Haar (boite)
            found = self._detect(frame)
            if found is not None:
                self.box = found if self.box is None else 0.7 * self.box + 0.3 * found
                self.lost = 0
            else:
                self.lost += 1
                if self.lost > 6:  # ~1 s sans visage
                    self.box = None
        self.n += 1
        if self.box is None:
            s.face, s.message, self.rois = False, "Webcam : aucun visage", []
            self.t.clear(); self.rgb.clear()
            s.hr, s.hr_q, s.warmup_left = None, 0.0, MIN_S
            self._reset_resp()
            return s
        x, y, w, h = self.box
        if self.polys is None:  # repli Haar : rectangles places par proportions de la boite
            rect = lambda fx0, fy0, fx1, fy1: np.array([(x + fx0 * w, y + fy0 * h), (x + fx1 * w, y + fy0 * h),
                                                        (x + fx1 * w, y + fy1 * h), (x + fx0 * w, y + fy1 * h)])
            self.rois = [rect(0.30, 0.08, 0.70, 0.25), rect(0.15, 0.50, 0.38, 0.72), rect(0.62, 0.50, 0.85, 0.72)]
        else:
            self.rois = self.polys
        self._respiration(frame, t)
        pix = []
        H, W = frame.shape[:2]
        for poly in self.rois:
            x0, y0 = np.maximum(np.floor(poly.min(axis=0)).astype(int), 0)
            x1, y1 = np.minimum(np.ceil(poly.max(axis=0)).astype(int) + 1, [W, H])
            patch = frame[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            inside = np.zeros(patch.shape[:2], np.uint8)
            cv2.fillPoly(inside, [(poly - [x0, y0]).astype(np.int32)], 1)
            ycc = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)
            skin = (ycc[..., 1] > 133) & (ycc[..., 1] < 173) & (ycc[..., 2] > 77) & (ycc[..., 2] < 127) & inside.astype(bool)
            if skin.sum() > 20:
                pix.append(patch[skin].reshape(-1, 3))
        if not pix:
            s.face, s.message = True, "Webcam : peau non visible (eclairage ?)"
            return s
        b, g, r = np.concatenate(pix).mean(axis=0)
        self.t.append(t)
        self.rgb.append((r, g, b))
        s.face, s.message = True, "Webcam : visage suivi"
        dur = self.t[-1] - self.t[0]
        s.warmup_left = max(0.0, MIN_S - dur)
        if dur >= MIN_S and t - self._last_est >= 1.0:
            self._last_est = t
            s.hr, s.hr_q = self._estimate()
        return s

    def _estimate(self):
        t = np.asarray(self.t)
        x = np.asarray(self.rgb, dtype=np.float64)
        keep = t >= t[-1] - WINDOW_S
        t, x = t[keep], x[keep]
        fs = 1.0 / np.median(np.diff(t))
        grid = np.arange(t[0], t[-1], 1.0 / fs)
        c = np.column_stack([np.interp(grid, t, x[:, k]) for k in range(3)])  # R, G, B
        # POS (Wang et al. 2016) : projection orthogonale a la teinte de peau, fenetres de 1.6 s
        L = max(4, int(POS_WIN_S * fs))
        H = np.zeros(len(c))
        for n in range(L, len(c) + 1):
            cn = c[n - L:n] / c[n - L:n].mean(axis=0)
            s1 = cn[:, 1] - cn[:, 2]
            s2 = cn[:, 1] + cn[:, 2] - 2 * cn[:, 0]
            h = s1 + (s1.std() / (s2.std() + 1e-9)) * s2
            H[n - L:n] += h - h.mean()
        hr, q, _, _ = spectral_rate(H, fs, CARDIAC_BAND)
        return hr, q


class WebcamSource(threading.Thread):
    """Capture webcam + rPPG en tache de fond ; derniere image annotee dans .latest."""

    def __init__(self, index: int = 0):
        super().__init__(daemon=True)
        self.index = index
        self.proc = RppgProcessor()
        self.lock = threading.Lock()
        self.latest = None   # (image BGR annotee, RppgState copie)
        self.status = "Webcam : ouverture..."
        self.stop_flag = False

    def run(self):
        cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self.status = "Webcam introuvable"
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        self.status = "Webcam : active"
        while not self.stop_flag:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            st = self.proc.process(frame, time.perf_counter())
            out = frame.copy()
            if self.proc.box is not None:
                if self.proc.face_poly is not None:  # contour du visage MediaPipe
                    cv2.polylines(out, [self.proc.face_poly.astype(np.int32)], True, (80, 200, 60), 2, cv2.LINE_AA)
                else:
                    x, y, w, h = map(int, self.proc.box)
                    cv2.rectangle(out, (x, y), (x + w, y + h), (80, 200, 60), 2)
                for poly in self.proc.rois:  # front + joues
                    cv2.polylines(out, [np.asarray(poly).astype(np.int32)], True, (255, 200, 0), 1, cv2.LINE_AA)
                if self.proc.chest_roi is not None:
                    x0, y0, x1, y1 = self.proc.chest_roi
                    cv2.rectangle(out, (x0, y0), (x1, y1), (200, 120, 255), 1)
            with self.lock:
                self.latest = (out, RppgState(**st.__dict__))
        cap.release()
