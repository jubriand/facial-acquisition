"""
Demarre/arrete l'enregistrement Optris PIX Connect ET envoie un trigger Biopac en meme temps.

Base : trigger_background.py (raccourcis globaux, boitier trigger, log CSV).

Deux facons de piloter PIX Connect :
  - Par defaut, sans droits admin : le script commande directement la fenetre de PIX Connect
    (commandes "Record" = F2 et "Stop" = F3 de PIX Connect). Il cherche d'abord l'entree du
    menu correspondante et l'active sans toucher au focus ; a defaut, il met PIX Connect au
    premier plan, simule F2/F3, puis rend le focus a la fenetre precedente.
  - Avec --pix-port COMx : interface "COM-Port" de PIX Connect (!RecordStart / !RecordStop),
    necessite une paire de ports virtuels com0com (installation admin). PIX Connect ecoute sur un
    port de la paire (Tools > Configuration > External Communication > COM-Port), le script sur l'autre.

Raccourcis (globaux, actifs meme si PIX Connect a le focus) :
    F9          : demarre l'enregistrement PIX Connect, puis trigger Biopac
    F10         : trigger Biopac, puis arrete l'enregistrement (2e trigger = fin, corrige la derive)
    F8          : trigger Biopac seul (marqueur pendant l'enregistrement)
    Ctrl+Alt+Q  : quitter (arrete l'enregistrement en cours, avec trigger de fin)

Usage :
    python trigger_record.py                       # pilotage de la fenetre PIX Connect
    python trigger_record.py --pix-port COM11      # pilotage par port COM (com0com)
    python trigger_record.py --channel 2 --width 1 --start-delay 0.5

Chaque evenement est horodate dans trigger_record_log.csv.
"""

import argparse
import csv
import ctypes
import ctypes.wintypes as wt
import logging
import time
from datetime import datetime
from pathlib import Path

import serial

from test_com_trigger import BAUDRATE, load_channel_map
from trigger_background import (HOTKEY_QUIT, PM_REMOVE, QS_ALLINPUT, WM_HOTKEY, TriggerBox, parse_hotkey,
                                user32)

LOG_CSV = Path(__file__).with_name("trigger_record_log.csv")
LOG_TXT = Path(__file__).with_name("trigger_record.log")
HOTKEY_START, HOTKEY_STOP, HOTKEY_MARK = 10, 11, 12
PIX_EXE = "imager.exe"

kernel32 = ctypes.windll.kernel32
WM_COMMAND = 0x0111
MF_BYPOSITION, MF_POPUP, MF_GRAYED, MF_DISABLED = 0x0400, 0x0010, 0x0001, 0x0002
INPUT_KEYBOARD, KEYEVENTF_KEYUP = 1, 0x0002
VK_F2, VK_F3, VK_MENU = 0x71, 0x72, 0x12
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

user32.GetMenuItemID.restype = wt.UINT
user32.GetSubMenu.restype = wt.HMENU
user32.GetMenu.restype = wt.HMENU
user32.GetForegroundWindow.restype = wt.HWND


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


def find_pix_window() -> int | None:
    found = []

    @ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
    def callback(hwnd, _):
        if user32.IsWindowVisible(hwnd) and user32.GetWindowTextLengthW(hwnd) and process_name(hwnd) == PIX_EXE:
            found.append(hwnd)
        return True

    user32.EnumWindows(callback, 0)
    return found[0] if found else None


def menu_items(hmenu: int, path: str = ""):
    """Parcourt le menu : renvoie (chemin texte, id, desactive) pour chaque commande."""
    for pos in range(user32.GetMenuItemCount(hmenu)):
        buf = ctypes.create_unicode_buffer(256)
        user32.GetMenuStringW(hmenu, pos, buf, len(buf), MF_BYPOSITION)
        text = f"{path} > {buf.value.replace('&', '')}" if path else buf.value.replace("&", "")
        sub = user32.GetSubMenu(hmenu, pos)
        if sub:
            yield from menu_items(sub, text)
        else:
            state = user32.GetMenuState(hmenu, pos, MF_BYPOSITION)
            yield text, user32.GetMenuItemID(hmenu, pos), bool(state & (MF_GRAYED | MF_DISABLED))


class PixWindow:
    """Pilote la fenetre PIX Connect : commande de menu (sans focus) ou touche F2/F3 simulee."""

    def __init__(self, method: str):
        self.method = method

    def _hwnd(self) -> int:
        hwnd = find_pix_window()
        if not hwnd:
            raise RuntimeError("Fenetre PIX Connect introuvable (Imager.exe lance ?)")
        return hwnd

    def _menu_id(self, hwnd: int, accel: str) -> int | None:
        """Entree de menu dont le raccourci affiche est accel (ex. 'F2'), independant de la langue."""
        hmenu = user32.GetMenu(hwnd)
        if not hmenu:
            return None
        for text, item_id, _ in menu_items(hmenu):
            if text.split("\t")[-1].strip().upper() == accel and "\t" in text:
                return item_id
        return None

    def describe(self) -> str:
        hwnd = find_pix_window()
        if not hwnd:
            return "PIX Connect n'est pas ouvert (le lancer avant d'appuyer sur F9)"
        found = {k: self._menu_id(hwnd, k) for k in ("F2", "F3")}
        return f"fenetre PIX Connect trouvee, entrees de menu Record/Stop : {found}"

    def _send(self, accel: str, vk: int) -> str:
        hwnd = self._hwnd()
        if self.method in ("auto", "menu"):
            item_id = self._menu_id(hwnd, accel)
            if item_id:
                user32.PostMessageW(hwnd, WM_COMMAND, item_id, 0)
                return f"menu id {item_id}"
            if self.method == "menu":
                raise RuntimeError(f"Entree de menu {accel} introuvable dans PIX Connect")
        previous = user32.GetForegroundWindow()
        send_key(VK_MENU)  # un appui Alt autorise le changement de premier plan
        user32.SetForegroundWindow(hwnd)
        time.sleep(0.05)
        send_key(vk)
        time.sleep(0.05)
        if previous and previous != hwnd:
            user32.SetForegroundWindow(previous)
        return f"touche {accel}"

    def start(self) -> tuple[str, float]:
        t0 = time.perf_counter()
        how = self._send("F2", VK_F2)
        return how, time.perf_counter() - t0

    def stop(self) -> tuple[str, float]:
        t0 = time.perf_counter()
        how = self._send("F3", VK_F3)
        return how, time.perf_counter() - t0

    def close(self) -> None:
        pass


class PixSerial:
    """Commandes texte vers PIX Connect via un port COM (virtuel, com0com)."""

    def __init__(self, port: str, baud: int, address: int | None, timeout: float):
        self.prefix = f"{address:03d}" if address else ""
        self.ser = serial.Serial(port=port, baudrate=baud, bytesize=8, parity="N", stopbits=1,
                                 timeout=timeout, write_timeout=1)

    def command(self, cmd: str) -> tuple[str, float]:
        """Envoie la commande (terminee par CR/LF) et renvoie (reponse, duree en s). Reponse vide = timeout."""
        self.ser.reset_input_buffer()
        t0 = time.perf_counter()
        self.ser.write(f"{self.prefix}{cmd}\r\n".encode("latin-1"))
        self.ser.flush()
        answer = self.ser.read_until(b"\n").decode("latin-1").strip()
        return answer, time.perf_counter() - t0

    def describe(self) -> str:
        version, _ = self.command("?VAppl")
        return f"PIX Connect repond : {version}" if version else \
            "AUCUNE reponse de PIX Connect : verifier External Communication > COM-Port (port, baudrate)"

    def start(self) -> tuple[str, float]:
        return self.command("!RecordStart")

    def stop(self) -> tuple[str, float]:
        return self.command("!RecordStop")

    def close(self) -> None:
        self.ser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pix-method", choices=["auto", "menu", "keys"], default="auto",
                        help="Pilotage fenetre : menu, touche simulee, ou auto (menu puis touche)")
    parser.add_argument("--pix-port", help="Port COM relie a PIX Connect via com0com (sinon pilotage fenetre)")
    parser.add_argument("--pix-baud", type=int, default=115200, help="Baudrate regle dans PIX Connect (defaut 115200)")
    parser.add_argument("--pix-address", type=int, help="Adresse bus PIX Connect (1..999), si configuree")
    parser.add_argument("--pix-timeout", type=float, default=2.0, help="Attente max de la reponse PIX en s (defaut 2)")
    parser.add_argument("--start-delay", type=float, default=0.5,
                        help="Attente entre le demarrage PIX et le trigger, pour qu'il tombe dans la video (defaut 0.5)")
    parser.add_argument("--start-key", default="f9")
    parser.add_argument("--stop-key", default="f10")
    parser.add_argument("--mark-key", default="f8")
    parser.add_argument("--quit-key", default="ctrl+alt+q")
    parser.add_argument("--channel", type=int, default=2, help="Canal AcqKnowledge (defaut 2)")
    parser.add_argument("--width", type=float, default=1.0, help="Duree du trigger en s (defaut 1)")
    parser.add_argument("--port", help="Port COM du boitier trigger (defaut : detection automatique)")
    parser.add_argument("--baud", type=int, default=BAUDRATE)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(LOG_TXT, encoding="utf-8"), logging.StreamHandler()])

    channel_map = load_channel_map()
    if args.channel not in channel_map:
        raise SystemExit(f"Canal {args.channel} absent de channel_map.json (canaux : {sorted(channel_map)})")
    code = 1 << channel_map[args.channel]

    if args.pix_port:
        try:
            pix = PixSerial(args.pix_port, args.pix_baud, args.pix_address, args.pix_timeout)
        except serial.SerialException as exc:
            raise SystemExit(f"Impossible d'ouvrir {args.pix_port} ({exc}). La paire com0com est-elle installee ?")
    else:
        pix = PixWindow(args.pix_method)
    logging.info("PIX Connect : %s", pix.describe())

    box = TriggerBox(args.port, args.baud, args.width)
    box.start()

    hotkeys = {HOTKEY_START: args.start_key, HOTKEY_STOP: args.stop_key,
               HOTKEY_MARK: args.mark_key, HOTKEY_QUIT: args.quit_key}
    for hk_id, spec in hotkeys.items():
        mods, vk = parse_hotkey(spec)
        if not user32.RegisterHotKey(None, hk_id, mods, vk):
            for other in hotkeys:
                user32.UnregisterHotKey(None, other)
            box.close()
            pix.close()
            raise SystemExit(f"Raccourci '{spec}' deja utilise par une autre application, choisis-en un autre.")

    new_file = not LOG_CSV.exists()
    f = LOG_CSV.open("a", newline="")
    writer = csv.writer(f)
    if new_file:
        writer.writerow(["event", "timestamp_iso", "unix_time", "perf_counter", "channel", "pix_answer", "pix_delay_s"])

    def log_event(event: str, answer: str = "", delay: float | None = None) -> None:
        t = datetime.now()
        writer.writerow([event, t.isoformat(timespec="microseconds"), f"{t.timestamp():.6f}",
                         f"{time.perf_counter():.6f}", args.channel, answer,
                         "" if delay is None else f"{delay:.4f}"])
        f.flush()

    def trigger(event: str) -> None:
        box.send(code)
        log_event(event)

    recording = False

    def start_recording() -> None:
        nonlocal recording
        if recording:
            logging.info("Enregistrement deja en cours, demarrage ignore")
            return
        try:
            answer, delay = pix.start()
        except RuntimeError as exc:
            logging.error("%s : pas d'enregistrement ni de trigger", exc)
            return
        log_event("pix_record_start", answer, delay)
        if args.pix_port and not answer:
            logging.warning("Pas de confirmation de PIX Connect (%.2f s)", delay)
        time.sleep(args.start_delay)
        trigger("trigger_start")
        recording = True
        logging.info("Enregistrement demarre (%s) + trigger canal %d", answer, args.channel)

    def stop_recording() -> None:
        nonlocal recording
        if not recording:
            logging.info("Aucun enregistrement en cours, arret ignore")
            return
        trigger("trigger_stop")
        time.sleep(args.width + 0.2)  # le trigger de fin doit tomber dans la video
        try:
            answer, delay = pix.stop()
        except RuntimeError as exc:
            logging.error("%s : arrete l'enregistrement a la main (F3 dans PIX Connect)", exc)
            answer, delay = "echec", None
        log_event("pix_record_stop", answer, delay)
        recording = False
        logging.info("Trigger de fin + enregistrement arrete (%s)", answer)

    logging.info("Pret : %s = start enregistrement + trigger | %s = trigger + stop | %s = trigger seul | %s = quitter",
                 args.start_key, args.stop_key, args.mark_key, args.quit_key)

    msg = wt.MSG()
    try:
        while True:
            user32.MsgWaitForMultipleObjects(0, None, False, 100, QS_ALLINPUT)
            while user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, PM_REMOVE):
                if msg.message != WM_HOTKEY:
                    continue
                if msg.wParam == HOTKEY_QUIT:
                    raise KeyboardInterrupt
                if msg.wParam == HOTKEY_START:
                    start_recording()
                elif msg.wParam == HOTKEY_STOP:
                    stop_recording()
                elif msg.wParam == HOTKEY_MARK:
                    trigger("trigger_mark")
                    logging.info("Trigger marqueur canal %d", args.channel)
    except KeyboardInterrupt:
        if recording:
            stop_recording()
        logging.info("Arret")
    finally:
        for hk_id in hotkeys:
            user32.UnregisterHotKey(None, hk_id)
        box.close()
        pix.close()
        f.close()


if __name__ == "__main__":
    main()
