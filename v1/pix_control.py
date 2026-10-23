"""
Pilotage d'Optris PIX Connect sans droits admin.

  - etat "pret" : PIX Connect ouvert, camera connectee (Event.log), pas d'enregistrement ni de
    dialogue en cours ;
  - commande par le Connect SDK (IPC : FileRecord/FileStop, sauvegarde automatique) si l'option
    est activee dans PIX Connect, sinon par les raccourcis F2/F3 + boite "Saving radiometric file" ;
  - demarrage/arret confirmes par le fichier temporaire d'enregistrement (temp.~avi).

Mesures sur Xi400 (27 Hz) : temp.~avi cree ~30 ms apres FileRecord (IPC), ~50-200 ms apres F2 ;
la video couvre l'intervalle demarrage -> arret a une ou deux images pres.
"""

import ctypes
import ctypes.wintypes as wt
import os
import shutil
import struct
import time
from pathlib import Path

PIX_EXE_PATH = Path(r"C:\Program Files (x86)\Optris GmbH\PIX Connect\Imager.exe")
IMAGER_DIR = Path(os.path.expandvars(r"%APPDATA%\Imager"))
TEMP_FILE = IMAGER_DIR / "temp.~avi"
EVENT_LOG = IMAGER_DIR / "Event.log"
PIX_EXE = "imager.exe"

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32
user32.GetForegroundWindow.restype = wt.HWND
user32.GetDlgItem.restype = wt.HWND

INPUT_KEYBOARD, KEYEVENTF_KEYUP = 1, 0x0002
VK_F2, VK_F3, VK_MENU = 0x71, 0x72, 0x12
WM_SETTEXT, BM_CLICK = 0x000C, 0x00F5
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
SW_RESTORE = 9


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wt.WORD), ("wScan", wt.WORD), ("dwFlags", wt.DWORD), ("time", wt.DWORD),
                ("dwExtraInfo", ctypes.c_size_t)]


class INPUT(ctypes.Structure):
    class _U(ctypes.Union):
        _fields_ = [("ki", KEYBDINPUT), ("_pad", ctypes.c_byte * 32)]
    _anonymous_ = ("u",)
    _fields_ = [("type", wt.DWORD), ("u", _U)]


def send_key(vk: int) -> None:
    events = (INPUT * 2)()
    for ev, flags in zip(events, (0, KEYEVENTF_KEYUP)):
        ev.type = INPUT_KEYBOARD
        ev.ki = KEYBDINPUT(vk, 0, flags, 0, 0)
    user32.SendInput(2, events, ctypes.sizeof(INPUT))


def process_name(hwnd: int) -> str:
    pid = wt.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
    if not handle:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(1024)
        size = wt.DWORD(len(buf))
        kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size))
        return Path(buf.value).name.lower()
    finally:
        kernel32.CloseHandle(handle)


def window_class(hwnd: int) -> str:
    buf = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, buf, 256)
    return buf.value


def window_title(hwnd: int) -> str:
    buf = ctypes.create_unicode_buffer(256)
    user32.GetWindowTextW(hwnd, buf, 256)
    return buf.value


def pix_windows() -> list[int]:
    """Fenetres visibles de premier niveau appartenant a PIX Connect."""
    found = []

    @ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
    def callback(hwnd, _):
        if user32.IsWindowVisible(hwnd) and process_name(hwnd) == PIX_EXE:
            found.append(hwnd)
        return True

    user32.EnumWindows(callback, 0)
    return found


def main_window() -> int | None:
    for hwnd in pix_windows():
        if window_title(hwnd).startswith("Optris PIX Connect"):
            return hwnd
    return None


def dialogs() -> list[int]:
    """Boites de dialogue ouvertes par PIX Connect (ex. sauvegarde)."""
    return [h for h in pix_windows() if window_class(h) == "#32770"]


def find_child(parent: int, cls: str, ctrl_id: int) -> int | None:
    found = []

    @ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
    def callback(hwnd, _):
        if window_class(hwnd) == cls and user32.GetDlgCtrlID(hwnd) == ctrl_id:
            found.append(hwnd)
        return True

    user32.EnumChildWindows(parent, callback, 0)
    return found[0] if found else None


def camera_connected() -> tuple[bool, str]:
    """Lit Event.log : camera connectee depuis le dernier demarrage de PIX Connect ?"""
    try:
        lines = EVENT_LOG.read_text(encoding="utf-8", errors="replace").splitlines()[-200:]
    except OSError:
        return False, "Event.log illisible"
    state, device = False, ""
    for line in lines:
        parts = [p.strip() for p in line.split(";")]
        if len(parts) < 4:
            continue
        source, message = parts[1], parts[3]
        if source == "Appl" and message in ("Started", "Closed"):
            state, device = False, ""
        elif source == "Dev" and message.startswith("Connected"):
            state, device = True, message.split(":", 1)[-1].strip()
        elif source == "Dev" and "isconnect" in message:
            state, device = False, ""
    return state, device


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wt.DWORD), ("biWidth", wt.LONG), ("biHeight", wt.LONG), ("biPlanes", wt.WORD),
                ("biBitCount", wt.WORD), ("biCompression", wt.DWORD), ("biSizeImage", wt.DWORD),
                ("biXPelsPerMeter", wt.LONG), ("biYPelsPerMeter", wt.LONG), ("biClrUsed", wt.DWORD),
                ("biClrImportant", wt.DWORD)]


def capture_bottom_left(hwnd: int, width: int = 260, height: int = 40) -> bytes | None:
    """Pixels BGRA du coin bas-gauche de la fenetre (barre d'etat), meme si elle est masquee."""
    gdi32 = ctypes.windll.gdi32
    rect = wt.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    w, h = rect.right - rect.left, rect.bottom - rect.top
    if w <= 0 or h <= 0:
        return None
    hdc = user32.GetDC(hwnd)
    mem = gdi32.CreateCompatibleDC(hdc)
    bmp = gdi32.CreateCompatibleBitmap(hdc, w, h)
    old = gdi32.SelectObject(mem, bmp)
    try:
        user32.PrintWindow(hwnd, mem, 2)  # PW_RENDERFULLCONTENT
        info = BITMAPINFOHEADER(ctypes.sizeof(BITMAPINFOHEADER), w, -h, 1, 32, 0, 0, 0, 0, 0, 0)
        buf = ctypes.create_string_buffer(w * h * 4)
        gdi32.GetDIBits(mem, bmp, 0, h, buf, ctypes.byref(info), 0)
        rows = [buf.raw[(y * w) * 4:(y * w + min(width, w)) * 4] for y in range(max(0, h - height), h)]
        return b"".join(rows)
    finally:
        gdi32.SelectObject(mem, old)
        gdi32.DeleteObject(bmp)
        gdi32.DeleteDC(mem)
        user32.ReleaseDC(hwnd, hdc)


def playback_mode(hwnd: int | None = None) -> bool:
    """PIX Connect affiche un fichier (et non la camera) : case verte avec le chemin dans la barre d'etat."""
    hwnd = hwnd or main_window()
    pixels = capture_bottom_left(hwnd) if hwnd else None
    if not pixels:
        return False
    green = sum(1 for i in range(0, len(pixels), 4)
                if pixels[i + 1] > 100 and pixels[i + 1] > pixels[i + 2] + 50 and pixels[i + 1] > pixels[i] + 50)
    return green > 500


def temp_size() -> int | None:
    try:
        return TEMP_FILE.stat().st_size
    except FileNotFoundError:
        return None


def bring_to_front(hwnd: int) -> bool:
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, SW_RESTORE)
    user32.SetForegroundWindow(hwnd)
    if user32.GetForegroundWindow() != hwnd:
        send_key(VK_MENU)  # un appui Alt debloque le changement de premier plan
        user32.SetForegroundWindow(hwnd)
    time.sleep(0.05)
    return user32.GetForegroundWindow() == hwnd


class PixConnect:
    """Commande de l'enregistrement PIX Connect.

    Par defaut via le Connect SDK (IPC : FileRecord/FileStop) : demarrage confirme en ~30 ms, pas de
    changement de fenetre, sauvegarde automatique (deplacee ensuite vers le nom voulu). Si l'IPC n'est
    pas disponible (option non activee dans PIX Connect), repli sur les raccourcis F2/F3.
    """

    def __init__(self, use_ipc: bool = True):
        self.recording = False
        self.t_key_start = self.t_key_stop = None
        self.hub = None
        if use_ipc:
            from pix_ipc import get_hub
            self.hub = get_hub()
        self.mode = None           # "ipc" ou "touches" pour l'enregistrement en cours
        self._path_before = ""

    def ipc_ready(self) -> bool:
        return self.hub is not None and self.hub.streaming()

    def status(self) -> tuple[bool, str]:
        """(pret, message) pour l'interface."""
        if main_window() is None:
            return False, "PIX Connect n'est pas ouvert"
        ok, device = camera_connected()
        if not ok:
            return False, "Camera non connectee dans PIX Connect"
        if dialogs():
            return False, "Une boite de dialogue PIX Connect est ouverte"
        via = "IPC" if self.ipc_ready() else "raccourcis F2/F3 (activer Connect SDK (IPC))"
        if self.recording:
            return True, f"Enregistrement en cours ({device}, {self.mode})"
        if playback_mode():
            return False, "PIX Connect affiche un fichier (mode lecture) au lieu de la camera"
        return True, f"Pret : {device} - commande par {via}"

    def back_to_live(self, timeout: float = 5.0) -> bool:
        """Apres une sauvegarde, PIX Connect rouvre le fichier : Stop (F3) revient a la camera."""
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            if not playback_mode():
                return True
            self._press(VK_F3)
            time.sleep(1.0)
        return not playback_mode()

    def launch(self) -> None:
        if main_window() is None and PIX_EXE_PATH.exists():
            os.startfile(PIX_EXE_PATH)

    def _press(self, vk: int) -> float:
        hwnd = main_window()
        if hwnd is None:
            raise RuntimeError("PIX Connect n'est pas ouvert")
        previous = user32.GetForegroundWindow()
        if not bring_to_front(hwnd):
            raise RuntimeError("Impossible de mettre PIX Connect au premier plan")
        t_key = time.perf_counter()
        send_key(vk)
        time.sleep(0.05)
        if previous and previous != hwnd:
            user32.SetForegroundWindow(previous)
        return t_key

    def start(self, timeout: float = 3.0) -> tuple[float, float]:
        """Demarre puis attend la creation de temp.~avi. Renvoie (t_commande, t_confirmation) en perf_counter."""
        before = temp_size()
        if self.ipc_ready():
            self.mode = "ipc"
            self._path_before, _ = self.hub.call("stored_path")
            _, self.t_key_start = self.hub.call("record")
        else:
            self.mode = "touches"
            self.t_key_start = self._press(VK_F2)
        while time.perf_counter() - self.t_key_start < timeout:
            size = temp_size()
            if size is not None and (before is None or size != before):
                self.recording = True
                return self.t_key_start, time.perf_counter()
            time.sleep(0.005)
        raise RuntimeError("PIX Connect n'a pas demarre l'enregistrement (temp.~avi inchange)")

    def stop(self, settle: float = 1.5) -> tuple[float, float]:
        """Arrete puis attend la fin des ecritures. Renvoie (t_commande, t_derniere_ecriture)."""
        if self.mode == "ipc":
            _, self.t_key_stop = self.hub.call("stop")
        else:
            self.t_key_stop = self._press(VK_F3)
        last_size, last_change = temp_size(), time.perf_counter()
        while time.perf_counter() - last_change < settle:
            size = temp_size()
            if size != last_size:
                last_size, last_change = size, time.perf_counter()
            time.sleep(0.005)
        self.recording = False
        return self.t_key_stop, last_change

    def save(self, target: Path, timeout: float = 30.0) -> Path:
        """Range le .ravi sous le nom voulu (IPC : deplacement du fichier auto-sauve ; touches : boite de dialogue)."""
        target = Path(target).with_suffix(".ravi")
        target.parent.mkdir(parents=True, exist_ok=True)
        if self.mode == "ipc":
            return self._save_ipc(target, timeout)
        return self._save_dialog(target, timeout)

    def _save_ipc(self, target: Path, timeout: float) -> Path:
        t0 = time.perf_counter()
        src = ""
        while time.perf_counter() - t0 < timeout:
            path, _ = self.hub.call("stored_path")
            if path and path != self._path_before and Path(path).exists():
                src = path
                break
            time.sleep(0.2)
        if not src:
            raise RuntimeError("PIX Connect n'a pas indique le fichier enregistre")
        while time.perf_counter() - t0 < timeout:  # PIX Connect peut garder le fichier ouvert un instant
            try:
                shutil.move(src, target)
                return target
            except PermissionError:
                time.sleep(0.5)
        raise RuntimeError(f"Impossible de deplacer {src} vers {target}")

    def _save_dialog(self, target: Path, timeout: float) -> Path:
        """Remplit la boite 'Saving radiometric file' et attend l'ecriture du .ravi."""
        t0 = time.perf_counter()
        dlg = None
        while time.perf_counter() - t0 < timeout and dlg is None:
            found = dialogs()
            dlg = found[0] if found else None
            time.sleep(0.1)
        if dlg is None:
            raise RuntimeError("Boite de sauvegarde PIX Connect introuvable")
        edit = find_child(dlg, "Edit", 1001)
        if edit is None:
            raise RuntimeError("Champ 'Nom du fichier' introuvable dans la boite de sauvegarde")
        user32.SendMessageW(edit, WM_SETTEXT, 0, ctypes.c_wchar_p(str(target)))
        user32.PostMessageW(user32.GetDlgItem(dlg, 1), BM_CLICK, 0, 0)
        while time.perf_counter() - t0 < timeout:
            if target.exists() and not dialogs():
                time.sleep(1.0)  # laisse PIX Connect rouvrir le fichier avant de revenir au direct
                self.back_to_live()
                return target
            time.sleep(0.1)
        raise RuntimeError(f"Fichier non ecrit : {target}")


def ravi_frame_counters(path: Path) -> list[int]:
    """Compteur d'images de la camera pour chaque image du .ravi.

    La 1re ligne de chaque image est une ligne de metadonnees ; l'octet 72 y contient un compteur
    uint32 incremente de 1 par image par la camera (continu d'un fichier a l'autre). Il n'y a pas
    d'horodatage : le temps d'une image se deduit de (compteur - compteur_0) / fps.
    """
    data = Path(path).read_bytes()
    pos = data.find(b"movi") + 4
    counters = []
    while 0 < pos < len(data) - 8:
        chunk, size = data[pos:pos + 4], struct.unpack_from("<I", data, pos + 4)[0]
        if chunk == b"idx1":
            break
        if chunk == b"LIST":
            pos += 12
            continue
        if chunk in (b"00db", b"00dc"):
            counters.append(struct.unpack_from("<I", data, pos + 8 + 72)[0])
        pos += 8 + size + (size & 1)
    return counters


def ravi_info(path: Path) -> tuple[int, float]:
    """(nombre d'images, fps) lus dans l'en-tete AVI du .ravi."""
    head = Path(path).read_bytes()[:65536]
    i = head.find(b"strh")
    scale, rate, _, length = struct.unpack_from("<IIII", head, i + 8 + 20)
    return length, rate / scale
