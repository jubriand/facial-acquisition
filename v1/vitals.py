"""
Module thermique de la v1, aligne sur le suivi des boites de la v2 (MediaPipe + flux optique + One Euro).
Differences avec la v2 : cercle-guide et consigne de position affiches (render, Vitals.message).


Estimation des signes vitaux sur images thermiques (temps reel ou relecture .ravi) - version 2.

v2 : thermique seule. Ajoute la temperature globale du visage (moyenne et maximum des pixels de peau
du visage) et un rendu avec deux ROI suivies et etiquetees (front, nez).

Methodes (references, voir README du demonstrateur) :
  - Visage : segmentation par seuil adaptatif (Otsu) dans un cercle-guide ; regions d'interet (ROI)
    placees par proportions anthropometriques du visage, bout du nez = minimum thermique local
    (le nez est plus froid que les joues), comme dans le suivi du nez de Pereira et al. 2015.
  - Temperatures : moyenne de la ROI front et de la ROI bout du nez.
  - Frequence respiratoire : flux optique dense (Farneback) dans la region narines/bouche ; la
    composante verticale moyenne, corrigee du mouvement rigide de la tete (flux du front), est
    integree en deplacement, filtree 0.1-0.75 Hz (6-45 resp/min), pic spectral (Welch) sur 30 s.
    Controle croise : temperature sous les narines (Pereira et al. 2015 ; Negishi et al. 2020).
  - Frequence cardiaque : modulation pulsatile de la temperature cutanee (Garbey et al. 2007),
    ROI front ; filtre 0.75-3 Hz (45-180 bpm), pic spectral sur 20 s avec indice de qualite
    (rapport signal/bruit). Mesure fragile en thermique : afficher seulement si la qualite suffit.
"""

import mmap
import struct
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy import signal

RESP_BAND = (0.10, 0.75)   # Hz -> 6 a 45 respirations/min
CARDIAC_BAND = (0.75, 3.0)  # Hz -> 45 a 180 bpm
RESP_WINDOW_S, CARDIAC_WINDOW_S = 30.0, 20.0
HOLD_S = 4.0         # duree pendant laquelle on garde les reperes du visage s'ils ne sont plus detectes
MP_EVERY = 2         # MediaPipe toutes les 2 images (~20 ms) ; flux optique entre deux detections
MP_HOLD_S = 1.0      # sans detection MediaPipe pendant 1 s : repli sur la detection thermique par regles
DET_EVERY = 6       # detection anatomique toutes les 6 images (~0.25 s) pour recaler le suivi
YAW_MAX = 0.12      # decalage max (en rayons du cercle) de l'axe yeux/nez / centre de la silhouette pour "de face"
MIN_RESP_S, MIN_CARDIAC_S = 12.0, 10.0


# ----------------------------------------------------------------------------------------------
# Lecture d'un .ravi (relecture hors ligne : valeurs brutes camera, pas des degC)
# ----------------------------------------------------------------------------------------------
class RaviReader:
    def __init__(self, path: str | Path):
        self.file = open(path, "rb")
        self.mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        i = self.mm.find(b"strf")
        _, self.width, height, _, _, _ = struct.unpack_from("<IiiHHI", self.mm, i + 8)
        self.height = abs(height) - 1  # 1re ligne = metadonnees
        j = self.mm.find(b"strh")
        scale, rate = struct.unpack_from("<II", self.mm, j + 8 + 20)
        self.fps = rate / scale

    def frames(self):
        """Genere (image float32 hauteur x largeur, compteur camera)."""
        pos = self.mm.find(b"movi") + 4
        row = self.width * 2
        while 0 < pos < len(self.mm) - 8:
            chunk, size = self.mm[pos:pos + 4], struct.unpack_from("<I", self.mm, pos + 4)[0]
            if chunk == b"idx1":
                break
            if chunk == b"LIST":
                pos += 12
                continue
            if chunk in (b"00db", b"00dc"):
                counter = struct.unpack_from("<I", self.mm, pos + 8 + 72)[0]
                img = np.frombuffer(self.mm, dtype="<u2", count=self.width * self.height,
                                    offset=pos + 8 + row).reshape(self.height, self.width)
                yield img.astype(np.float32), counter
            pos += 8 + size + (size & 1)

    def close(self):
        self.mm.close()
        self.file.close()


# ----------------------------------------------------------------------------------------------
# Outils signal
# ----------------------------------------------------------------------------------------------
def spectral_rate(x: np.ndarray, fs: float, band: tuple[float, float]) -> tuple[float | None, float, np.ndarray, np.ndarray]:
    """Frequence dominante (par minute) dans la bande, qualite (part de puissance autour du pic)."""
    if len(x) < fs * 4 or not np.isfinite(x).all() or np.std(x) == 0:
        return None, 0.0, np.array([]), np.array([])
    x = signal.detrend(x)
    sos = signal.butter(3, band, btype="bandpass", fs=fs, output="sos")
    y = signal.sosfiltfilt(sos, x)
    nfft = max(4096, 1 << int(np.ceil(np.log2(len(y)))))
    f, p = signal.periodogram(y * np.hanning(len(y)), fs=fs, nfft=nfft)
    sel = (f >= band[0]) & (f <= band[1])
    if not sel.any() or p[sel].sum() == 0:
        return None, 0.0, f, p
    fb, pb = f[sel], p[sel]
    k = int(np.argmax(pb))
    peak = fb[k]
    near = np.abs(fb - peak) <= 0.05 * max(1.0, peak) + 0.02
    quality = float(pb[near].sum() / pb.sum())
    return float(peak * 60.0), quality, fb, pb


class OneEuro:
    """Filtre One Euro (Casiez, Roussel & Vogel, CHI 2012) : passe-bas dont la coupure augmente avec la
    vitesse -> lisse les petits tremblements au repos, suit sans retard les mouvements rapides."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.05, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self.x is None or t <= self.t:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x.astype(np.float32)
        dt = t - self.t
        dx = (x - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * np.abs(self.dx)
        a = self._alpha(cutoff, dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x.astype(np.float32)


class MPFace:
    """Points du visage MediaPipe FaceLandmarker (478 points, modele officiel Google) sur image thermique.

    Le modele est entraine sur des images couleur : on lui presente l'image thermique en niveaux de gris,
    en gris inverse ou avec contraste renforce (le rendu qui marche depend de la personne et de la scene ;
    le dernier rendu efficace est essaye en premier)."""

    MODEL = Path(__file__).with_name("models") / "face_landmarker.task"
    RENDERS = ("gris", "gris_inverse", "clahe")

    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self._mp = mp
        opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(self.MODEL)),
                                            running_mode=vision.RunningMode.IMAGE, num_faces=1,
                                            min_face_detection_confidence=0.3, min_face_presence_confidence=0.3)
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)
        self.clahe = cv2.createCLAHE(3.0, (8, 8))
        self.order = list(self.RENDERS)

    @classmethod
    def available(cls) -> bool:
        try:
            import mediapipe  # noqa: F401
            return cls.MODEL.exists()
        except ImportError:
            return False

    def detect(self, img: np.ndarray, scale) -> np.ndarray | None:
        """Renvoie les 478 points (x, y) en pixels de l'image thermique, ou None."""
        lo, hi = scale
        g = np.clip((img - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        h, w = g.shape
        for key in list(self.order):
            gg = {"gris": g, "gris_inverse": 255 - g, "clahe": self.clahe.apply(g)}[key]
            rgb = np.ascontiguousarray(cv2.cvtColor(cv2.resize(gg, (2 * w, 2 * h)), cv2.COLOR_GRAY2RGB))
            res = self.landmarker.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb))
            if res.face_landmarks:
                self.order.remove(key)
                self.order.insert(0, key)  # rendu efficace en premier la prochaine fois
                return np.array([(p.x * w, p.y * h) for p in res.face_landmarks[0]], dtype=np.float32)
        return None


# indices du maillage MediaPipe (topologie FaceMesh)
MP_FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
                176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
MP_FOREHEAD = [109, 10, 338, 337, 336, 9, 107, 108]  # haut du front -> au-dessus des sourcils
MP_NOSE_TIP, MP_SUBNASALE = 1, 2
MP_EYE_INNER = (133, 362)
MP_EYE_OUTER = (33, 263)


@dataclass
class Rois:
    face: tuple[int, int, int, int] | None = None      # x0, y0, x1, y1
    face_poly: np.ndarray | None = None                # contour du visage (MediaPipe)
    forehead_poly: np.ndarray | None = None            # polygone du front (MediaPipe)
    forehead: tuple[int, int, int, int] | None = None
    nose: tuple[int, int, int, int] | None = None
    nostril: tuple[int, int, int, int] | None = None
    resp: tuple[int, int, int, int] | None = None


@dataclass
class Vitals:
    in_circle: bool = False   # position ideale (vert)
    tracking: bool = False    # visage suivi, mesures en cours (orange si pas en position ideale)
    message: str = "Placez la tete dans le cercle"
    t_forehead: float | None = None
    t_nose: float | None = None
    t_face: float | None = None      # temperature moyenne de la peau du visage
    t_face_max: float | None = None  # maximum (coins internes des yeux, proche de la temperature centrale)
    rr_flow: float | None = None
    rr_flow_q: float = 0.0
    rr_temp: float | None = None
    rr_temp_q: float = 0.0
    hr: float | None = None
    hr_q: float = 0.0
    rois: Rois = field(default_factory=Rois)


class VitalsProcessor:
    def __init__(self, fps: float, width: int, height: int, use_mediapipe: bool = True):
        self.fps = fps
        self.w, self.h = width, height
        # v2 : plus de cercle-guide. r reste l'echelle de reference (taille typique d'une tete dans l'image) ;
        # la recherche du visage se fait sur toute l'image (marge de 2 px pour les bords du capteur)
        self.cx, self.cy, self.r = width // 2, height // 2, int(0.44 * height)
        self.circle = np.zeros((height, width), bool)
        self.circle[2:-2, 2:-2] = True
        n = int(fps * RESP_WINDOW_S) + 10
        self.t = deque(maxlen=n)
        self.resp_disp = deque(maxlen=n)
        self.nostril_temp = deque(maxlen=n)
        self.forehead_temp = deque(maxlen=n)
        self.forehead_vy = deque(maxlen=n)
        self._disp = 0.0
        self._prev8 = None
        self._lm = None
        self._avg = None
        self._fail = ""
        self._s = None
        self.mode = None
        self._lock_mode = None
        self.glasses = False
        self._disagree = 0
        self._skin_thr = None
        self._off_skin = 0
        self._cur = None
        self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self._pts = None
        self._prev_g = None
        self._speed = 0.0
        self._lm_f = None
        self._euro = OneEuro(min_cutoff=1.0, beta=0.05)
        # MediaPipe (si le modele est present dans v2/models) ; sinon detection thermique par regles seule
        self.mp = MPFace() if use_mediapipe and MPFace.available() else None
        self._mp_pts = None
        self._last_mp_t = -1e9
        self._euro_mp = OneEuro(min_cutoff=1.0, beta=0.05)
        self._glasses_top = 0
        self._glasses_bottom = 0
        self._glasses_off = 0.0
        self._last_det_t = -1e9
        self.confirmed = False
        self.yaw = 0.0
        self._frame_idx = 0
        self._last_lm_t = -1e9
        self._scale = None
        self._last_estimate = -1e9
        self.vitals = Vitals()

    # --- geometrie du visage ---
    # Reperes thermiques : coins internes des yeux = points les plus chauds du visage ; narines =
    # petites zones froides sous le nez. L'echelle s = distance yeux -> narines (independante des
    # cheveux, souvent aussi chauds que la peau) dimensionne toutes les ROI.
    def _skin_mask(self, img: np.ndarray):
        vals = img[self.circle]
        lo, hi = np.percentile(vals, 1), np.percentile(vals, 99.5)
        if hi - lo < 1e-6:
            return None
        norm = np.clip((img - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        thr, _ = cv2.threshold(norm[self.circle], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self._skin_thr = lo + thr / 255.0 * (hi - lo)  # seuil peau/fond en unites de l'image (controle du suivi)
        mask = ((norm > thr) & self.circle).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if n <= 1:
            return None
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[k, cv2.CC_STAT_AREA] < 0.08 * self.circle.sum():
            return None
        return labels == k

    @staticmethod
    def _argmax_in(img, x0, y0, x1, y1):
        win = img[y0:y1, x0:x1]
        if win.size == 0:
            return None
        iy, ix = np.unravel_index(int(np.argmax(cv2.GaussianBlur(win, (5, 5), 0))), win.shape)
        return x0 + ix, y0 + iy

    def _glasses(self, mask: np.ndarray, fcx: float, img: np.ndarray | None = None, band=None):
        """Verres de lunettes = opaques a l'infrarouge -> deux 'trous' froids dans le masque de peau,
        a la meme hauteur, de part et d'autre du centre du visage. Renvoie (ex, ey, ecart) ou None."""
        r = self.r
        # zones froides a l'interieur de l'enveloppe convexe de la tete (un verre peut toucher le bord du
        # visage : ce n'est alors pas un 'trou' du masque de peau). Enveloppe calculee sur la tete seule
        # (sans le buste, qui l'elargirait jusqu'aux epaules).
        y_lo, y_hi = int(band[0] - 0.30 * r), int(band[1] + 0.15 * r)
        head = mask.copy()
        head[max(0, y_hi):] = False
        cnts, _ = cv2.findContours(head.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        hull = np.zeros((self.h, self.w), np.uint8)
        cv2.fillPoly(hull, [cv2.convexHull(max(cnts, key=cv2.contourArea))], 1)
        holes = (hull.astype(bool) & ~mask).astype(np.uint8)
        holes = cv2.morphologyEx(holes, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        holes[: max(0, y_lo)] = 0
        holes[max(0, y_hi):] = 0
        n, labels, stats, cents = cv2.connectedComponentsWithStats(holes)
        area_min, area_max = 0.004 * np.pi * r * r, 0.08 * np.pi * r * r
        # un verre (opaque a l'IR) est nettement plus froid que la peau, proche du fond ; une orbite ou un
        # sourcil n'est que legerement plus froid -> seuil a mi-chemin entre peau et fond
        if img is not None:
            skin_med = float(np.median(img[mask]))
            bg = img[self.circle & ~mask]
            cold_thr = skin_med - 0.45 * (skin_med - float(np.median(bg))) if bg.size else -np.inf
            is_cold = lambda lab: float(np.median(img[labels == lab])) < cold_thr
        else:
            is_cold = lambda lab: True
        blobs = []
        for i in range(1, n):
            area, bw, bh = stats[i, cv2.CC_STAT_AREA], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            if not is_cold(i):
                continue
            # centre des verres a hauteur des yeux (une meche froide sur le front est plus haut)
            if band is not None and not band[0] <= cents[i][1] <= band[1]:
                continue
            # verres + monture d'un seul tenant : une zone froide large, centree sur le visage
            # (ecart entre centres des verres ~ 0.47 x largeur totale des lunettes)
            if (2.0 < bw / max(bh, 1) < 7 and 0.40 * r < bw < 1.5 * r and area < 2.5 * area_max
                    and abs(cents[i][0] - fcx) < 0.25 * r):
                self._glasses_top = stats[i, cv2.CC_STAT_TOP]
                self._glasses_bottom = stats[i, cv2.CC_STAT_TOP] + bh
                return cents[i][0], cents[i][1], 0.47 * bw
            if not area_min < area < area_max or not 0.8 < bw / max(bh, 1) < 2.8:
                continue
            cnts, _ = cv2.findContours((labels == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            hull = cv2.contourArea(cv2.convexHull(cnts[0])) if cnts else 0
            if hull and area / hull > 0.8:  # verre = forme pleine et convexe (les trous entre meches ne le sont pas)
                blobs.append((cents[i][0], cents[i][1], area, stats[i, cv2.CC_STAT_TOP], i,
                              stats[i, cv2.CC_STAT_TOP] + bh))
        best = None
        for i, a in enumerate(blobs):
            for b in blobs[i + 1:]:
                (lx, ly, la, lt, _, lb), (rx, ry, ra, rt, _, rb) = sorted((a, b))
                gap = rx - lx
                if (abs(ly - ry) < 0.12 * r and 0.25 * r < gap < 0.95 * r and max(la, ra) / min(la, ra) < 2.0
                        and abs((lx + rx) / 2 - fcx) < 0.20 * r):
                    score = abs((lx + rx) / 2 - fcx)
                    if best is None or score < best[0]:
                        best = (score, (lx + rx) / 2, (ly + ry) / 2, gap, min(lt, rt), max(lb, rb))
        if best is not None:
            self._glasses_top, self._glasses_bottom = best[4], best[5]
            return best[1:4]
        if img is None or band is None:
            return None
        # un seul verre net (l'autre se confond avec le fond au bord du visage) : on le valide par la
        # symetrie -> la position miroir par rapport a l'axe du visage doit aussi etre froide (2e verre)
        skin_med = float(np.median(img[mask]))
        for bx, by, area, top, lab, bottom in sorted(blobs, key=lambda b: -b[2]):
            if not band[0] <= by <= band[1] or not 0.10 * r < abs(bx - fcx) < 0.45 * r:
                continue
            mx = int(2 * fcx - bx)
            hw = int(0.06 * r)
            patch = img[max(0, int(by) - hw):int(by) + hw, max(0, mx - hw):mx + hw]
            lens = img[labels == lab]
            if patch.size and np.median(patch) < skin_med - 0.5 * (skin_med - float(np.median(lens))):
                self._glasses_top, self._glasses_bottom = top, bottom
                return (bx + mx) / 2, by, abs(bx - mx)
        return None

    def _canthi(self, skin: np.ndarray, fcx: float, y0: int, y1: int):
        """Coins internes des yeux (points les plus chauds) : (ex, ey, ecart) ou None."""
        r = self.r
        clipx = lambda x: int(min(max(x, 0), self.w))
        clipy = lambda y: int(min(max(y, 0), self.h))
        first = self._argmax_in(skin, clipx(fcx - 0.45 * r), y0, clipx(fcx + 0.45 * r), y1)
        if first is None:
            return None
        fy0, fy1 = clipy(first[1] - 0.10 * r), clipy(first[1] + 0.10 * r)
        if first[0] < fcx:
            left, right = first, self._argmax_in(skin, clipx(first[0] + 0.08 * r), fy0, clipx(first[0] + 0.60 * r), fy1)
        else:
            right, left = first, self._argmax_in(skin, clipx(first[0] - 0.60 * r), fy0, clipx(first[0] - 0.08 * r), fy1)
        if left is None or right is None or not 0.08 * r < right[0] - left[0] < 0.70 * r:
            return None
        return (left[0] + right[0]) / 2, (left[1] + right[1]) / 2, right[0] - left[0]

    def _nostrils(self, img, x0, y0, x1, y1, inside=None, require_pair=False):
        """Narines : filtre 'black-hat' = creux froids plus petits que le noyau (insensible au nez entier).
        inside : masque (interieur du visage) hors duquel les reponses sont ignorees.
        require_pair : n'accepter que deux creux cote a cote (recherche sans a priori)."""
        clipx = lambda x: int(min(max(x, 0), self.w))
        clipy = lambda y: int(min(max(y, 0), self.h))
        sx0, sx1, sy0, sy1 = clipx(x0), clipx(x1), clipy(y0), clipy(y1)
        win = img[sy0:sy1, sx0:sx1]
        if win.shape[0] < 5 or win.shape[1] < 5:
            return None
        k = max(3, int(0.10 * self.r) | 1)
        lo, hi = float(win.min()), float(win.max())
        win8 = ((win - lo) / max(hi - lo, 1e-6) * 255).astype(np.uint8)
        bh = cv2.GaussianBlur(cv2.morphologyEx(cv2.GaussianBlur(win8, (3, 3), 0), cv2.MORPH_BLACKHAT,
                                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))), (5, 5), 0)
        if inside is not None:
            bh = np.where(inside[sy0:sy1, sx0:sx1], bh, 0)
        if bh.max() == 0:
            return None
        r = self.r
        if require_pair:
            # meilleur couple parmi les creux les plus marques : meme hauteur, cote a cote, centre dans la fenetre
            # (un bord de verre de lunettes, plus froid, ne forme pas de paire et n'est donc pas retenu)
            work, peaks = bh.astype(np.float32).copy(), []
            sup = max(2, int(0.04 * r))
            for _ in range(8):
                py, px = np.unravel_index(int(np.argmax(work)), work.shape)
                if work[py, px] <= 0:
                    break
                peaks.append((px, py, float(work[py, px])))
                work[max(0, py - sup):py + sup + 1, max(0, px - sup):px + sup + 1] = 0
            best, cxw = None, bh.shape[1] / 2
            for i, (ax, ay, av) in enumerate(peaks):
                for bx, by, bv in peaks[i + 1:]:
                    # ecart entre narines jusqu'a 0.32 r : visage proche de la camera
                    if abs(ay - by) < 0.06 * r and 0.05 * r < abs(ax - bx) < 0.32 * r:
                        score = min(av, bv) - 0.15 * abs((ax + bx) / 2 - cxw)
                        if best is None or score > best[0]:
                            best = (score, (ax + bx) / 2, (ay + by) / 2)
            return None if best is None else (sx0 + best[1], sy0 + best[2])
        iy, ix = np.unravel_index(int(np.argmax(bh)), bh.shape)
        # 2e narine : autre creux a la meme hauteur, a cote -> on renvoie le milieu des deux narines
        yy, xx = np.mgrid[0:bh.shape[0], 0:bh.shape[1]]
        side = (np.abs(yy - iy) < 0.06 * r) & (np.abs(xx - ix) > 0.05 * r) & (np.abs(xx - ix) < 0.20 * r)
        cand = np.where(side, bh, 0)
        jy, jx = np.unravel_index(int(np.argmax(cand)), cand.shape)
        if cand[jy, jx] > 0.35 * bh[iy, ix]:
            return sx0 + (ix + jx) / 2, sy0 + (iy + jy) / 2
        if require_pair:
            return None
        return sx0 + ix, sy0 + iy

    def _face_interior(self, mask: np.ndarray) -> np.ndarray:
        """Masque de peau sans trous (narines, verres) et erode : evite les bords du visage et les meches."""
        m8 = mask.astype(np.uint8)
        flood = m8.copy()
        cv2.floodFill(flood, np.zeros((self.h + 2, self.w + 2), np.uint8), (0, 0), 1)
        filled = (m8 | (flood == 0)).astype(np.uint8)
        k = max(3, int(0.08 * self.r))
        return cv2.erode(filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool)

    def _landmarks(self, img: np.ndarray, mask: np.ndarray):
        """Reperes : narines (toujours) + ligne des yeux (coins internes ou verres de lunettes) si coherente.

        Mode 'complet' : yeux et narines coherents (de face, d/s ~0.70-0.76 et narine a ~0.45 d de l'axe
        des yeux sur les enregistrements). Sinon mode 'narines' : ligne des yeux deduite des narines avec la
        derniere echelle apprise (tete tournee, lunettes non reconnues, yeux masques) -> mesure approchee.
        """
        r = self.r
        clipy = lambda y: int(min(max(y, 0), self.h))
        rows = mask.any(axis=1)
        if not rows.any():
            return None
        # haut de la tete = premiere ligne de peau ; axe du visage = centre de la peau dans la hauteur de la tete
        # (le buste, plus bas, n'influence pas l'axe)
        head_top = int(np.argmax(rows))
        band = mask[head_top:clipy(head_top + 1.0 * r)]
        fcx = float(np.average(np.arange(self.w), weights=band.sum(axis=0)))
        cy = head_top + 0.8 * r  # centre approximatif du visage (remplace le centre du cercle)
        # bande des yeux relative au haut de la tete : sinon le cou (zone la plus chaude) est pris pour les yeux
        cols = mask[:, max(0, int(fcx - 0.1 * r)):int(fcx + 0.1 * r)].any(axis=1)
        top = int(np.argmax(cols)) if cols.any() else head_top
        y0, y1 = clipy(top + 0.30 * r), clipy(top + 1.20 * r)
        if y1 - y0 < 5:
            return None
        inside = self._face_interior(mask)

        # 1) ligne des yeux : verres de lunettes, puis coins internes des yeux (si les 'lunettes' detectees ne
        #    donnent pas un visage coherent, ce n'en etaient probablement pas : meches de cheveux, etc.)
        candidates = []
        # bande large : un visage proche de la camera place les yeux plus bas sous le haut de la tete
        g = self._glasses(mask, fcx, img, (top + 0.42 * r, top + 1.40 * r))
        if g is not None:
            candidates.append((True, (g[0], g[1], g[2] / 2.0)))  # ecart verres ~ 2 x ecart coins des yeux
        c = self._canthi(np.where(mask, img, img.min()), fcx, y0, y1)
        if c is not None:
            candidates.append((False, c))

        # 2) mode complet : narines cherchees sous les yeux, puis coherence yeux / narines
        for is_glasses, eyes in candidates:
            self.glasses = is_glasses
            if is_glasses:
                self._glasses_off = eyes[1] - self._glasses_top  # haut des lunettes au-dessus de ey
            ex, ey, d = eyes
            if self.glasses:
                # avec lunettes : narines cherchees SOUS la monture (les coins interieurs des verres sont froids
                # et ressemblent a des narines), en exigeant une paire ; l'axe du visage est alors donne par les
                # narines, plus fiable que la symetrie de la silhouette (cheveux asymetriques)
                gb = self._glasses_bottom
                # +0.08 r : le bord inferieur des verres est une ligne froide qui donne de fausses 'paires'
                nos = self._nostrils(img, ex - 0.30 * r, gb + 0.08 * r, ex + 0.30 * r, gb + 0.50 * r, inside,
                                     require_pair=True)
            else:
                nos = self._nostrils(img, ex - 0.20 * r, ey + 0.15 * r, ex + 0.20 * r, ey + 0.60 * r, inside)
            if nos is not None:
                nx, ny = nos
                if self.glasses:
                    yaw = (ex - nx) / r  # tete tournee : narines decalees par rapport au centre des lunettes
                    ex = nx
                else:
                    yaw = ((ex + nx) / 2 - fcx) / r  # axe yeux/nez decale par rapport au centre de la silhouette
                s = ny - ey
                # lunettes : l'ecart entre centres des verres depend de la monture -> plage plus large
                if s > 0.08 * r and abs(nx - ex) <= 0.75 * d and 0.55 < d / s < (1.6 if self.glasses else 1.10):
                    self._s = s if self._s is None else 0.9 * self._s + 0.1 * s  # echelle apprise
                    self.yaw = yaw
                    self.mode = "complet"
                    return np.array([ex, ey, nx, ny], dtype=np.float32)

        # 3) mode narines : paire de narines dans le haut du visage (sous le front, au-dessus du cou/col)
        self.glasses = bool(candidates and candidates[0][0])
        nos = None
        if self._lm is not None:
            px, py = self._lm[2], self._lm[3]
            nos = self._nostrils(img, px - 0.12 * r, py - 0.10 * r, px + 0.12 * r, py + 0.10 * r, inside)
        if nos is None:
            # avec lunettes, les narines sont forcement sous la monture (pas dans les cheveux au-dessus)
            y_start = max(top + 0.45 * r, self._glasses_bottom + 0.08 * r) if self.glasses else top + 0.45 * r
            nos = self._nostrils(img, fcx - 0.25 * r, y_start, fcx + 0.25 * r, y_start + 0.70 * r, inside,
                                 require_pair=True)
        if nos is None:
            return None
        nx, ny = nos
        s = self._s if self._s is not None else 0.22 * r  # ~0.22 r mesure sur les enregistrements de face
        self.mode = "narines"
        self._fail = "Tournez un peu la tete vers la camera"
        return np.array([nx, ny - s, nx, ny], dtype=np.float32)

    def _rois(self) -> Rois:
        return self._rois_for(self._lm)

    def _rois_for(self, lm) -> Rois:
        ex, ey, nx, ny = lm
        s = max(ny - ey, 4.0)
        rect = lambda x, y, hw, hh: (max(0, int(x - hw)), max(0, int(y - hh)),
                                     min(self.w, int(x + hw + 1)), min(self.h, int(y + hh + 1)))
        # front : milieu du front (entre sourcils et racine des cheveux) ; avec lunettes, au-dessus de la monture
        fy = ey - (self._glasses_off + 0.55 * s) if self.glasses else ey - 0.60 * s
        return Rois(face=rect(ex, (ey + ny) / 2, 1.7 * s, 2.2 * s),
                    forehead=rect(ex, fy, 0.60 * s, 0.22 * s),             # front, au-dessus des sourcils
                    nose=rect(nx, ny - 0.28 * s, 0.15 * s, 0.13 * s),      # bout du nez
                    nostril=rect(nx, ny + 0.08 * s, 0.45 * s, 0.20 * s),   # flux d'air sous les narines
                    resp=rect(nx, ny + 0.35 * s, 0.85 * s, 0.55 * s))      # narines + levre + bouche

    @staticmethod
    def _mean(img, r):
        if r is None:
            return None
        patch = img[r[1]:r[3], r[0]:r[2]]
        return float(patch.mean()) if patch.size else None

    def _face_temp(self, img, r, poly=None):
        """Temperature globale du visage : pixels de peau (seuil d'Otsu dans le contour du visage, ce qui
        exclut fond, cheveux froids et verres de lunettes). Renvoie (moyenne, 99e percentile)."""
        if r is None:
            return None, None
        patch = img[r[1]:r[3], r[0]:r[2]]
        if patch.size < 20:
            return None, None
        if poly is not None:
            patch = patch[self._poly_mask(r, poly)]
            if patch.size < 20:
                return None, None
        lo, hi = float(patch.min()), float(patch.max())
        p8 = ((patch - lo) / max(hi - lo, 1e-6) * 255).astype(np.uint8)
        thr, _ = cv2.threshold(p8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        skin = patch[p8 > thr]
        if skin.size < 10:
            return None, None
        return float(skin.mean()), float(np.percentile(skin, 99))

    def _mean_warm(self, img, r, poly=None):
        """Moyenne de la moitie la plus chaude de la ROI : ignore meches, monture de lunettes, sourcils."""
        if r is None:
            return None
        patch = img[r[1]:r[3], r[0]:r[2]]
        if poly is not None and patch.size:
            patch = patch[self._poly_mask(r, poly)]
        if not patch.size:
            return None
        return float(patch[patch >= np.median(patch)].mean())

    # --- suivi reactif ---------------------------------------------------------------------------
    # 1) a chaque image : points caracteristiques du visage suivis par flux optique pyramidal
    #    (Lucas-Kanade, controle aller-retour) -> transformation de similitude (translation, rotation,
    #    echelle ; RANSAC) appliquee aux reperes. Aucun retard, suit sursauts et rotations de la tete.
    # 2) toutes les DET_EVERY images : detection anatomique (yeux/lunettes + narines) pour recaler le suivi
    #    (derive), seulement quand la tete est quasi immobile ; reamorce les points.
    # 3) affichage et mesures : reperes filtres par un filtre One Euro (Casiez et al. 2012) -> stable au
    #    repos, quasi sans retard en mouvement rapide.
    def _img8(self, img):
        lo, hi = self._scale
        return np.clip((img - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)

    def _track8(self, img):
        """Image 8 bits a contraste local renforce (CLAHE) : plus de details a suivre dans un visage thermique."""
        return self._clahe.apply(self._img8(img))

    def _seed_points(self, g, lm):
        x0, y0, x1, y1 = self._rois_for(lm).face
        m = np.zeros_like(g)
        m[max(0, y0):y1, max(0, x0):x1] = 255
        pts = cv2.goodFeaturesToTrack(g, maxCorners=80, qualityLevel=0.005, minDistance=4, mask=m, blockSize=5)
        self._pts = pts if pts is not None and len(pts) >= 8 else None

    def _lk_step(self, g):
        """Deplace les reperes selon le mouvement des points suivis. Renvoie (reperes, deplacement px) ou None."""
        M = self._lk_matrix(g)
        if M is None:
            return None
        new = self._apply(M, self._lm.reshape(2, 2)).reshape(-1)
        return new, float(np.abs(new - self._lm).max())

    @staticmethod
    def _apply(M, pts):
        return (pts @ M[:, :2].T + M[:, 2]).astype(np.float32)

    def _lk_matrix(self, g):
        """Mouvement du visage entre l'image precedente et l'image courante (similitude 2x3), ou None."""
        if self._pts is None or self._prev_g is None:
            return None
        p1, st, _ = cv2.calcOpticalFlowPyrLK(self._prev_g, g, self._pts, None, winSize=(15, 15), maxLevel=3,
                                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
        if p1 is None:
            return None
        p0r, st2, _ = cv2.calcOpticalFlowPyrLK(g, self._prev_g, p1, None, winSize=(15, 15), maxLevel=3)
        fb = np.linalg.norm((self._pts - p0r).reshape(-1, 2), axis=1)
        good = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 1.0)
        if good.sum() < 6:
            return None
        a, b = self._pts[good].reshape(-1, 2), p1[good].reshape(-1, 2)
        M, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=2.0)
        if M is None or inl is None or inl.sum() < 6:
            return None
        self._pts = b[inl.ravel() == 1].reshape(-1, 1, 2).astype(np.float32)
        return M

    # --- suivi MediaPipe : detection toutes les MP_EVERY images, flux optique entre deux detections ---
    def _seed_points_poly(self, g, poly):
        m = np.zeros_like(g)
        cv2.fillPoly(m, [poly.astype(np.int32)], 255)
        pts = cv2.goodFeaturesToTrack(g, maxCorners=80, qualityLevel=0.005, minDistance=4, mask=m, blockSize=5)
        self._pts = pts if pts is not None and len(pts) >= 8 else None

    def _update_mp(self, t, v, g) -> bool:
        """Met a jour les 478 points MediaPipe. Renvoie False si MediaPipe n'a plus le visage (repli)."""
        if self._mp_pts is not None:
            M = self._lk_matrix(g)
            if M is not None:
                self._mp_pts = self._apply(M, self._mp_pts)
        if self._mp_pts is None or self._frame_idx % MP_EVERY == 0:
            pts = self.mp.detect(self._cur, self._scale)
            if pts is not None:
                if self._mp_pts is None:
                    self._euro_mp.reset()
                self._mp_pts, self._last_mp_t = pts, t
                self._seed_points_poly(g, pts[MP_FACE_OVAL])
        if self._mp_pts is not None and t - self._last_mp_t > MP_HOLD_S:
            self._mp_pts = None  # MediaPipe ne retrouve plus le visage : repli sur la detection thermique
        self._prev_g = g
        if self._mp_pts is None:
            return False
        P = self._mp_pts
        # reperes au format commun (milieu des coins internes des yeux, point sous le nez) : utilises par les
        # ROI respiratoires et par le repli si MediaPipe perd le visage
        self._lm = np.array([*(P[MP_EYE_INNER[0]] + P[MP_EYE_INNER[1]]) / 2, *P[MP_SUBNASALE]], dtype=np.float32)
        self._last_lm_t = self._last_det_t = t
        self.mode = self._lock_mode = "mediapipe"
        self.confirmed = True
        self.glasses = False
        self.yaw = 0.0
        return True

    def _rois_mp(self, P) -> Rois:
        """ROI a partir des points MediaPipe filtres : front et visage en polygones, bout du nez carre."""
        clip = lambda r: (max(0, int(r[0])), max(0, int(r[1])), min(self.w, int(r[2]) + 1), min(self.h, int(r[3]) + 1))
        bbox = lambda poly: clip((poly[:, 0].min(), poly[:, 1].min(), poly[:, 0].max(), poly[:, 1].max()))
        face_poly, fh_poly = P[MP_FACE_OVAL], P[MP_FOREHEAD]
        iod = float(np.linalg.norm(P[MP_EYE_OUTER[0]] - P[MP_EYE_OUTER[1]]))
        h = max(2.0, 0.07 * iod)
        nx, ny = P[MP_NOSE_TIP]
        lm = np.array([*(P[MP_EYE_INNER[0]] + P[MP_EYE_INNER[1]]) / 2, *P[MP_SUBNASALE]], dtype=np.float32)
        base = self._rois_for(lm)  # zones narines / respiration (meme geometrie que la detection thermique)
        return Rois(face=bbox(face_poly), face_poly=face_poly, forehead_poly=fh_poly, forehead=bbox(fh_poly),
                    nose=clip((nx - h, ny - h, nx + h, ny + h)), nostril=base.nostril, resp=base.resp)

    def _poly_mask(self, rect, poly):
        x0, y0, x1, y1 = rect
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(m, [(poly - [x0, y0]).astype(np.int32)], 1)
        return m.astype(bool)

    def _detect(self):
        mask = self._skin_mask(self._avg)
        det = self._landmarks(self._avg, mask) if mask is not None else None
        return det, mask

    def _update_tracking(self, t, v, g):
        r = self.r
        moved = 0.0
        if self._lm is not None:
            step = self._lk_step(g)
            if step is not None:
                self._lm, moved = step
                self._last_lm_t = t
        self._speed = 0.7 * self._speed + 0.3 * moved  # px/image, lisse
        few_points = self._pts is None or len(self._pts) < 15
        if self._lm is None or few_points or self._frame_idx % DET_EVERY == 0:
            self._fail = "Placez le visage dans le cercle"
            det, mask = self._detect()
            still = self._speed < 0.6  # recalage seulement tete quasi immobile (detection sur image moyennee)
            if det is not None and self._lm is None:
                self._lm, self._lock_mode = det, self.mode
                self._last_det_t = self._last_lm_t = t
                self._euro.reset()
                self._seed_points(g, self._lm)
            elif det is not None:
                close = np.hypot(*(det[2:] - self._lm[2:])) < 0.12 * r and np.hypot(*(det[:2] - self._lm[:2])) < 0.20 * r
                self._disagree = 0 if close else self._disagree + (self.mode == "complet")
                if close and still:
                    self._lm = self._lm + 0.5 * (det - self._lm)  # corrige la derive en douceur
                    self._lock_mode = self.mode if self.mode == "complet" else self._lock_mode
                    self._last_det_t = t
                    self._seed_points(g, self._lm)
                elif still and self.mode == "complet" and (self._lock_mode != "complet" or self._disagree >= 3):
                    self._lm, self._lock_mode = det, "complet"  # le suivi s'etait accroche ailleurs
                    self._disagree = 0
                    self._last_det_t = t
                    self._seed_points(g, self._lm)
                elif few_points:
                    self._seed_points(g, self._lm)
            elif self._lm is not None and few_points:
                self._seed_points(g, self._lm)
            if self._lm is None:
                v.message = "Aucun visage dans le cercle" if mask is None else self._fail
        if self._lm is not None and t - self._last_lm_t > HOLD_S:
            self._lm, self._pts = None, None  # plus aucun suivi depuis HOLD_S secondes
        # controle a chaque image : la zone suivie doit rester de la peau (chaude). Sinon la personne est sortie
        # du champ ou les points ont glisse sur le fond -> suivi perdu, nouvelle detection
        if self._lm is not None and self._skin_thr is not None:
            x0, y0, x1, y1 = self._rois_for(self._lm).face
            patch = self._cur[max(0, y0):y1, max(0, x0):x1]
            skin = float((patch > self._skin_thr).mean()) if patch.size else 0.0
            self._off_skin = self._off_skin + 1 if skin < 0.35 else 0
            if self._off_skin >= 3:
                self._lm, self._pts, self._off_skin = None, None, 0
        self.mode = self._lock_mode
        self.confirmed = t - self._last_det_t <= 1.5
        self._prev_g = g

    def display_lm(self):
        """Reperes filtres (One Euro) utilises pour l'affichage et les mesures."""
        return self._lm_f

    # --- traitement d'une image ---
    def process(self, img: np.ndarray, t: float) -> Vitals:
        v = self.vitals
        # detection sur une image legerement moyennee (~0.15 s) : les narines changent d'aspect au cours du
        # cycle respiratoire ; le suivi image par image, lui, travaille sur l'image courante (aucun retard)
        self._avg = img.copy() if self._avg is None else 0.6 * self._avg + 0.4 * img
        self._frame_idx += 1
        if self._scale is None:
            self._scale = (np.percentile(img, 1), np.percentile(img, 99.5))
        self._cur = img
        g = self._track8(img)
        # 1) MediaPipe (points du visage appris) ; 2) a defaut, detection thermique par regles
        if self._skin_thr is None:
            self._skin_mask(img)  # fixe le seuil peau/fond utilise par les controles
        use_mp = self.mp is not None and self._update_mp(t, v, g)
        if not use_mp:
            self._update_tracking(t, v, g)
        if self._lm is None:
            v.in_circle = v.tracking = False
            v.rois = Rois()
            self._lm_f = None
            self._prev8 = None
            self._reset_buffers()
            return v
        v.tracking = True
        # in_circle = reperes fiables (MediaPipe ou detection complete confirmee, tete de face)
        v.in_circle = bool(self.mode in ("complet", "mediapipe") and self.confirmed and abs(self.yaw) < YAW_MAX)
        # v1 : consigne affichee sous la video (cercle-guide vert / orange / rouge)
        v.message = ("Position correcte" + (" (lunettes)" if self.glasses else "")) if v.in_circle \
            else "Mesure approchee - regardez la camera"
        if use_mp:
            Pf = self._euro_mp(self._mp_pts.reshape(-1), t).reshape(-1, 2)
            rois = self._rois_mp(Pf)
            self._lm_f = np.array([*(Pf[MP_EYE_INNER[0]] + Pf[MP_EYE_INNER[1]]) / 2, *Pf[MP_SUBNASALE]],
                                  dtype=np.float32)
            self._euro(self._lm_f, t)  # garde le filtre des reperes a jour pour un repli sans saut
        else:
            self._lm_f = self._euro(self._lm, t)
            rois = self._rois_for(self._lm_f)
        v.rois = rois
        v.t_forehead = self._mean_warm(img, rois.forehead, rois.forehead_poly)
        v.t_nose = self._mean(img, rois.nose)
        v.t_face, v.t_face_max = self._face_temp(img, rois.face, rois.face_poly)

        # flux optique sur image 8 bits a echelle fixe (fixee a la 1re image pour rester stable)
        img8 = self._img8(img)
        vy_resp = vy_head = 0.0
        if self._prev8 is not None:
            fx0, fy0, fx1, fy1 = rois.face
            fx0, fy0 = max(0, fx0 - 4), max(0, fy0 - 4)
            fx1, fy1 = min(self.w, fx1 + 4), min(self.h, fy1 + 4)
            flow = cv2.calcOpticalFlowFarneback(self._prev8[fy0:fy1, fx0:fx1], img8[fy0:fy1, fx0:fx1], None,
                                                0.5, 2, 9, 3, 5, 1.1, 0)
            def roi_flow(r):
                patch = flow[r[1] - fy0:r[3] - fy0, r[0] - fx0:r[2] - fx0, 1]
                return float(patch.mean()) if patch.size else 0.0
            vy_resp, vy_head = roi_flow(rois.resp), roi_flow(rois.forehead)
        self._prev8 = img8
        self._disp += vy_resp - vy_head  # deplacement respiratoire, mouvement rigide de la tete retire

        # on accumule tant que le visage est suivi ; la position (cercle) ne sert qu'a guider le participant
        self.t.append(t)
        self.resp_disp.append(self._disp)
        self.nostril_temp.append(self._mean(img, rois.nostril))
        self.forehead_temp.append(v.t_forehead)
        self.forehead_vy.append(vy_head)

        if t - self._last_estimate >= 1.0:
            self._last_estimate = t
            self._estimate()
        return v

    def _reset_buffers(self):
        for d in (self.t, self.resp_disp, self.nostril_temp, self.forehead_temp, self.forehead_vy):
            d.clear()

    def _window(self, buf, seconds) -> tuple[np.ndarray, float]:
        """Derniere fenetre reechantillonnee sur une grille reguliere (le flux IPC saute des images)."""
        t = np.asarray(self.t, dtype=np.float64)
        x = np.asarray(buf, dtype=np.float64)
        keep = t >= t[-1] - seconds
        t, x = t[keep], x[keep]
        dt = np.median(np.diff(t)) if len(t) > 1 else 1.0 / self.fps
        fs = 1.0 / dt if dt > 0 else self.fps
        grid = np.arange(t[0], t[-1], 1.0 / fs)
        return np.interp(grid, t, x), fs

    def _estimate(self):
        v = self.vitals
        dur = (self.t[-1] - self.t[0]) if len(self.t) > 1 else 0.0
        if dur >= MIN_RESP_S:
            v.rr_flow, v.rr_flow_q, _, _ = spectral_rate(*self._window(self.resp_disp, RESP_WINDOW_S), RESP_BAND)
            v.rr_temp, v.rr_temp_q, _, _ = spectral_rate(*self._window(self.nostril_temp, RESP_WINDOW_S), RESP_BAND)
        else:
            v.rr_flow = v.rr_temp = None
            v.rr_flow_q = v.rr_temp_q = 0.0
        if dur >= MIN_CARDIAC_S:
            v.hr, v.hr_q, _, _ = spectral_rate(*self._window(self.forehead_temp, CARDIAC_WINDOW_S), CARDIAC_BAND)
        else:
            v.hr, v.hr_q = None, 0.0

    def warmup_left(self) -> tuple[float, float]:
        """Secondes restantes avant la 1re estimation (respiration, pouls)."""
        dur = (self.t[-1] - self.t[0]) if len(self.t) > 1 else 0.0
        return max(0.0, MIN_RESP_S - dur), max(0.0, MIN_CARDIAC_S - dur)


def render(img: np.ndarray, proc: VitalsProcessor, v: Vitals, scale: int = 2,
           span: tuple[float, float] | None = None, celsius: bool = True) -> np.ndarray:
    """Image couleur (BGR) avec les ROI front/nez etiquetees, agrandie d'un facteur scale."""
    lo, hi = span if span else (np.percentile(img, 1), np.percentile(img, 99.5))
    g = np.clip((img - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    col = cv2.applyColorMap(g, cv2.COLORMAP_INFERNO)
    col = cv2.resize(col, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    # v1 : cercle-guide (vert = position correcte, orange = mesure approchee, rouge = visage perdu)
    color = (80, 200, 60) if v.in_circle else ((0, 165, 255) if v.tracking else (40, 40, 230))
    cv2.circle(col, (proc.cx * scale, proc.cy * scale), proc.r * scale, color, 2, cv2.LINE_AA)
    if not v.tracking:
        return col
    # en relecture .ravi : valeurs brutes camera (pas de degC)
    fmt = lambda x: "--" if x is None else (f"{x:.1f} C" if celsius else f"{x:.0f}")
    r = v.rois
    if r.face_poly is not None:  # contour du visage (zone de la temperature globale), discret
        cv2.polylines(col, [(r.face_poly * scale).astype(np.int32)], True, (90, 200, 90), 1, cv2.LINE_AA)
    for roi, poly, c, label, val, pad in ((r.forehead, r.forehead_poly, (255, 220, 60), "Front", v.t_forehead, 0),
                                          (r.nose, None, (255, 255, 255), "Nez", v.t_nose, 4)):
        if not roi:
            continue
        x0, y0, x1, y1 = (roi[0] - pad) * scale, (roi[1] - pad) * scale, (roi[2] + pad) * scale, (roi[3] + pad) * scale
        if poly is not None:  # ROI MediaPipe : polygone qui suit la forme (et l'inclinaison) du front
            cv2.polylines(col, [(poly * scale).astype(np.int32)], True, c, 2, cv2.LINE_AA)
        else:  # la ROI du nez est petite : on l'affiche un peu agrandie autour du point mesure
            cv2.rectangle(col, (x0, y0), (x1, y1), c, 2, cv2.LINE_AA)
        text = f"{label} {fmt(val)}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        tx, ty = (x0, y0 - 8) if label == "Front" else (x1 + 8, (y0 + y1) // 2 + th // 2)
        tx = min(max(tx, 2), col.shape[1] - tw - 2)
        ty = min(max(ty, th + 2), col.shape[0] - 4)
        cv2.rectangle(col, (tx - 3, ty - th - 4), (tx + tw + 3, ty + 4), (0, 0, 0), -1)
        cv2.putText(col, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)
    return col
