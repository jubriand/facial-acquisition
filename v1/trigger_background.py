"""
Trigger en arriere-plan pour la synchro Biopac MP150 <-> camera thermique.

Tourne en fond : un raccourci clavier GLOBAL (actif meme si une autre fenetre a le
focus, ex. PIX Connect) envoie un trigger sur le canal AcqKnowledge choisi via le
boitier USB (COM). Reutilise la logique de test_com_trigger.py : inversion des bits
et correspondance canal -> bit de channel_map.json (creee par 'calibrate').

Usage :
    python trigger_background.py                         # F9 = trigger canal 2, Ctrl+Alt+Q = quitter
    python trigger_background.py --key enter             # Entree comme declencheur (voir remarque)
    python trigger_background.py --key ctrl+alt+t --channel 2 --width 1
    pythonw trigger_background.py                        # sans fenetre console (tout va dans le log)

Remarque : Windows reserve la touche enregistree pour ce script. Avec --key enter,
Entree ne fonctionne plus dans les autres applications tant que le script tourne.

Chaque trigger est horodate dans trigger_background_log.csv (heure systeme a la
microseconde + compteur haute resolution) pour recaler ensuite la video thermique.
"""

import argparse
import csv
import ctypes
import ctypes.wintypes as wt
import logging
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import serial

from test_com_trigger import BAUDRATE, encode, find_box_port, load_channel_map, open_port

LOG_CSV = Path(__file__).with_name("trigger_background_log.csv")
LOG_TXT = Path(__file__).with_name("trigger_background.log")

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

MODIFIERS = {"alt": 0x0001, "ctrl": 0x0002, "shift": 0x0004, "win": 0x0008}
MOD_NOREPEAT = 0x4000  # pas de triggers en rafale si la touche reste enfoncee
VK_NAMES = {"enter": 0x0D, "space": 0x20, "pause": 0x13, "insert": 0x2D, "home": 0x24, "end": 0x23,
            **{f"f{i}": 0x70 + i - 1 for i in range(1, 25)}}
WM_HOTKEY = 0x0312
PM_REMOVE = 0x0001
QS_ALLINPUT = 0x04FF
HOTKEY_TRIGGER, HOTKEY_QUIT = 1, 2


def parse_hotkey(spec: str) -> tuple[int, int]:
    """'ctrl+alt+t' -> (modificateurs, code touche virtuelle)."""
    *mods, key = spec.lower().replace(" ", "").split("+")
    try:
        mod_flags = sum(MODIFIERS[m] for m in mods)
    except KeyError as exc:
        raise ValueError(f"Modificateur inconnu {exc} (possibles : {', '.join(MODIFIERS)})") from None
    if key in VK_NAMES:
        vk = VK_NAMES[key]
    elif len(key) == 1 and key.isalnum():
        vk = ord(key.upper())
    else:
        raise ValueError(f"Touche inconnue '{key}' (ex. f9, enter, space, t, ctrl+alt+t)")
    return mod_flags | MOD_NOREPEAT, vk


class TriggerBox:
    """Garde le port ouvert et envoie les pulses dans un thread dedie (le raccourci ne bloque jamais)."""

    def __init__(self, port: str | None, baud: int, width: float):
        self.port, self.baud, self.width = port, baud, width
        self.ser: serial.Serial | None = None
        self.jobs: queue.Queue = queue.Queue()
        self.worker = threading.Thread(target=self._run, daemon=True)

    def connect(self) -> None:
        port = self.port or find_box_port()
        self.ser = open_port(port, self.baud)
        self.port = port
        logging.info("Boitier connecte sur %s", port)

    def start(self) -> None:
        self.connect()
        self.worker.start()

    def send(self, code: int) -> None:
        self.jobs.put(code)

    def _write(self, code: int) -> None:
        self.ser.write(encode(code))
        self.ser.flush()

    def _run(self) -> None:
        while True:
            code = self.jobs.get()
            if code is None:
                break
            try:
                if self.ser is None or not self.ser.is_open:
                    self.connect()
                self._write(code)
                time.sleep(self.width)
                self._write(0)
            except (serial.SerialException, OSError) as exc:
                logging.error("Envoi impossible (%s) : reconnexion au prochain trigger", exc)
                if self.ser is not None:
                    self.ser.close()
                self.ser = None

    def close(self) -> None:
        self.jobs.put(None)
        self.worker.join(timeout=self.width + 1)
        if self.ser is not None and self.ser.is_open:
            self._write(0)
            self.ser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--key", default="f9", help="Raccourci de trigger (defaut f9 ; ex. enter, ctrl+alt+t)")
    parser.add_argument("--quit-key", default="ctrl+alt+q", help="Raccourci pour quitter (defaut ctrl+alt+q)")
    parser.add_argument("--channel", type=int, default=2, help="Canal AcqKnowledge (defaut 2)")
    parser.add_argument("--width", type=float, default=1.0, help="Duree du trigger en s (defaut 1)")
    parser.add_argument("--min-interval", type=float, default=0.5,
                        help="Ignore les appuis plus rapproches que cet intervalle en s (defaut 0.5)")
    parser.add_argument("--port", help="Port COM (defaut : detection automatique)")
    parser.add_argument("--baud", type=int, default=BAUDRATE)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(LOG_TXT, encoding="utf-8"), logging.StreamHandler()])

    channel_map = load_channel_map()
    if args.channel not in channel_map:
        raise SystemExit(f"Canal {args.channel} absent de channel_map.json (canaux : {sorted(channel_map)})")
    code = 1 << channel_map[args.channel]

    box = TriggerBox(args.port, args.baud, args.width)
    box.start()

    for hk_id, spec in ((HOTKEY_TRIGGER, args.key), (HOTKEY_QUIT, args.quit_key)):
        mods, vk = parse_hotkey(spec)
        if not user32.RegisterHotKey(None, hk_id, mods, vk):
            box.close()
            raise SystemExit(f"Raccourci '{spec}' deja utilise par une autre application, choisis-en un autre.")

    new_file = not LOG_CSV.exists()
    f = LOG_CSV.open("a", newline="")
    writer = csv.writer(f)
    if new_file:
        writer.writerow(["n", "timestamp_iso", "unix_time", "perf_counter", "port", "channel", "code"])

    logging.info("Pret : %s = trigger canal %d (bit %d, %.3f s) | %s = quitter",
                 args.key, args.channel, channel_map[args.channel], args.width, args.quit_key)

    msg = wt.MSG()
    n, last = 0, -float("inf")
    try:
        while True:
            # Attente d'un message avec timeout court : Ctrl+C reste actif dans la console
            user32.MsgWaitForMultipleObjects(0, None, False, 100, QS_ALLINPUT)
            while user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, PM_REMOVE):
                if msg.message != WM_HOTKEY:
                    continue
                if msg.wParam == HOTKEY_QUIT:
                    raise KeyboardInterrupt
                t_perf = time.perf_counter()
                if t_perf - last < args.min_interval:
                    continue
                last = t_perf
                t = datetime.now()
                box.send(code)
                n += 1
                writer.writerow([n, t.isoformat(timespec="microseconds"), f"{t.timestamp():.6f}",
                                 f"{t_perf:.6f}", box.port, args.channel, code])
                f.flush()
                logging.info("Trigger #%d canal %d", n, args.channel)
    except KeyboardInterrupt:
        logging.info("Arret (%d trigger(s) envoye(s))", n)
    finally:
        for hk_id in (HOTKEY_TRIGGER, HOTKEY_QUIT):
            user32.UnregisterHotKey(None, hk_id)
        box.close()
        f.close()


if __name__ == "__main__":
    main()
