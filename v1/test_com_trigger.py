"""
Test des ports COM et envoi de triggers vers le Biopac MP150 via le boitier
USB -> TTL (NeuroSpec MMBT-S) branche sur le PC recent.

Principe : chaque octet envoye sur le port COM est recopie sur les lignes TTL du
boitier. Sur ce montage la logique est inversee (un bit a 1 fait descendre la ligne),
le script envoie donc le complement (desactivable avec --no-invert, place AVANT la
commande). L'ordre des fils ne suit pas l'ordre des bits : 'calibrate' enregistre la
correspondance canal AcqKnowledge -> bit dans channel_map.json, utilisee par 'trigger'.

Le script envoie toujours le code, attend --width secondes, puis envoie 0.
Cela marche quel que soit le mode du boitier (pas besoin de selecteur) :
  - si le boitier fait des pulses automatiques (~8 ms), le 0 est sans effet ;
  - si la ligne reste HIGH (mode "simple"), le 0 la remet a zero apres --width.
La duree du pulse vue dans AcqKnowledge indique donc le mode : ~8 ms ou ~--width.
En pulse de 8 ms, echantillonner AcqKnowledge a >= 500 Hz pour ne rien rater.

Usage :
    python test_com_trigger.py list                       # liste les ports COM
    python test_com_trigger.py scan                       # pulse sur fil 1 via CHAQUE port -> lequel reagit ?
    python test_com_trigger.py lines --port COM3          # fils 1 a 7 l'un apres l'autre -> mapping D1..D7
    python test_com_trigger.py calibrate                  # 1 fois : associe chaque canal AcqKnowledge a son bit
    python test_com_trigger.py trigger                    # trigger manuel : touche N = canal N, log CSV horodate
    python test_com_trigger.py hold                       # tous les fils a 1 jusqu'a une touche (diagnostic)
    python test_com_trigger.py probe                      # essaie des formats de commande (firmware inconnu)

Options communes : --width (duree avant remise a 0, en s), --baud.
"""

import argparse
import csv
import json
import msvcrt
import sys
import time
from datetime import datetime
from pathlib import Path

import serial
from serial.tools import list_ports

BAUDRATE = 115200  # le MMBT-S est un port COM virtuel : le baudrate n'a pas d'effet reel
N_LINES = 7
LOG_FILE = Path(__file__).with_name("trigger_log.csv")
MAP_FILE = Path(__file__).with_name("channel_map.json")  # canal AcqKnowledge -> bit, cree par 'calibrate'

# Logique inversee (observee sur ce montage) : un bit a 1 fait descendre la ligne.
# On envoie donc le complement : repos = 0xFF, trigger = tous les bits sauf celui du canal.
INVERT = True


def line_to_code(line: int) -> int:
    if not 1 <= line <= 8:
        raise ValueError("Le numero de fil doit etre entre 1 et 8")
    return 1 << (line - 1)


def encode(code: int) -> bytes:
    return bytes([(~code & 0xFF) if INVERT else code])


def open_port(port: str, baud: int) -> serial.Serial:
    ser = serial.Serial(port=port, baudrate=baud, bytesize=8, parity="N", stopbits=1, timeout=1, write_timeout=1)
    ser.write(encode(0))  # etat initial connu : aucun trigger
    ser.flush()
    time.sleep(0.05)
    return ser


def pulse(ser: serial.Serial, code: int, width: float) -> datetime:
    t = datetime.now()
    ser.write(encode(code))
    ser.flush()
    time.sleep(width)
    ser.write(encode(0))  # retour au repos
    ser.flush()
    return t


def load_channel_map() -> dict[int, int]:
    """{canal AcqKnowledge: bit}. Sans fichier de calibration : canal N -> bit N-1."""
    if MAP_FILE.exists():
        return {int(ch): bit for ch, bit in json.loads(MAP_FILE.read_text()).items()}
    return {line: line - 1 for line in range(1, N_LINES + 1)}


def cmd_list(_args) -> None:
    ports = sorted(list_ports.comports(), key=lambda p: p.device)
    if not ports:
        print("Aucun port COM detecte. Le boitier est-il branche ? (voir Gestionnaire de peripheriques)")
        return
    for p in ports:
        vidpid = f"VID:PID={p.vid:04X}:{p.pid:04X}" if p.vid is not None else ""
        hint = "  <-- probablement le boitier" if p.vid in (0x0403, 0x2341, 0x2A03) else ""
        print(f"{p.device:6s}  {p.description}  {vidpid}  {p.manufacturer or ''}{hint}")


def cmd_scan(args) -> None:
    ports = sorted(p.device for p in list_ports.comports())
    if not ports:
        print("Aucun port COM detecte.")
        return
    print("Surveille AcqKnowledge (canal D1) : un pulse sera envoye sur le fil 1 via chaque port.\n")
    for port in ports:
        input(f"Entree pour tester {port}...")
        try:
            with open_port(port, args.baud) as ser:
                t = pulse(ser, line_to_code(1), args.width)
            print(f"  {port} : trigger envoye a {t:%H:%M:%S.%f}. Pulse visible sur D1 ? -> c'est ce port.")
        except serial.SerialException as exc:
            print(f"  {port} : ouverture impossible ({exc})")


def cmd_lines(args) -> None:
    with open_port(args.port, args.baud) as ser:
        print(f"Port {args.port} ouvert. Envoi d'un pulse sur les fils 1 a {N_LINES}, "
              f"{args.interval} s d'ecart, {args.repeat} fois.")
        print("Dans AcqKnowledge, les pulses doivent apparaitre en escalier D1 -> D7.\n")
        for r in range(args.repeat):
            for line in range(1, N_LINES + 1):
                code = line_to_code(line)
                t = pulse(ser, code, args.width)
                print(f"  [{t:%H:%M:%S.%f}] fil {line} (code {code:3d}, bits {code:08b})")
                time.sleep(args.interval)
        # Tous les fils ensemble : verifie qu'aucun fil n'est mort
        t = pulse(ser, (1 << N_LINES) - 1, args.width)
        print(f"  [{t:%H:%M:%S.%f}] tous les fils (code {(1 << N_LINES) - 1})")


def cmd_hold(args) -> None:
    port = args.port or find_box_port()
    code = line_to_code(args.line) if args.line else args.code
    with open_port(port, args.baud) as ser:
        print(f"Port {port} : code {code} (bits {code:08b}) maintenu tant qu'aucune touche n'est pressee.")
        print("Verifie AcqKnowledge (en acquisition) ou mesure au multimetre : ~5 V sur les fils concernes.")
        print("Si le niveau retombe tout seul apres ~8 ms, le boitier est en mode pulse.\n")
        ser.write(encode(code))
        ser.flush()
        msvcrt.getwch()
        ser.write(encode(0))
        ser.flush()
        print("Lignes remises au repos.")


def cmd_calibrate(args) -> None:
    """Active chaque bit seul et demande quel canal AcqKnowledge a reagi -> channel_map.json."""
    port = args.port or find_box_port()
    mapping = {}
    with open_port(port, args.baud) as ser:
        print(f"Port {port}. AcqKnowledge doit etre en acquisition.")
        print("Pour chaque bit, un trigger est maintenu : tape le numero du canal qui a monte (1..9),")
        print("n si aucun canal ne reagit, q pour arreter.\n")
        for bit in range(8):
            ser.write(encode(1 << bit))
            ser.flush()
            print(f"  bit {bit} actif -> canal ? ", end="", flush=True)
            key = msvcrt.getwch().lower()
            print(key)
            ser.write(encode(0))
            ser.flush()
            time.sleep(0.3)
            if key == "q":
                break
            if key.isdigit() and key != "0":
                mapping[int(key)] = bit
    if not mapping:
        print("\nAucune correspondance enregistree.")
        return
    MAP_FILE.write_text(json.dumps({str(ch): bit for ch, bit in sorted(mapping.items())}, indent=2))
    print(f"\nCorrespondance enregistree dans {MAP_FILE} :")
    for ch, bit in sorted(mapping.items()):
        print(f"  touche {ch} -> canal {ch} (bit {bit})")


# Formats de commande courants pour un Arduino de trigger : (description, envoi "ON", envoi "OFF")
PROBE_CANDIDATES = [
    ("octet brut 255 (tous les bits)", b"\xff", b"\x00"),
    ("octet brut 1", b"\x01", b"\x00"),
    ("octet brut 2", b"\x02", b"\x00"),
    ("texte '1'", b"1", b"0"),
    ("texte '1' + fin de ligne \\n", b"1\n", b"0\n"),
    ("texte '1' + fin de ligne \\r\\n", b"1\r\n", b"0\r\n"),
    ("texte '255' + \\n", b"255\n", b"0\n"),
    ("texte '2' + \\n", b"2\n", b"0\n"),
    ("texte '7' + \\n", b"7\n", b"0\n"),
    ("lettre 'T'", b"T", b""),
    ("lettre 't'", b"t", b""),
    ("lettre 'H' / 'L'", b"H", b"L"),
    ("lettre 'a'", b"a", b""),
    ("texte 'ON' / 'OFF' + \\n", b"ON\n", b"OFF\n"),
    ("texte 'trigger' + \\n", b"trigger\n", b""),
]


def cmd_probe(args) -> None:
    port = args.port or find_box_port()
    results = []
    with serial.Serial(port=port, baudrate=args.baud, timeout=0.2, write_timeout=1) as ser:
        time.sleep(1.0)  # certains Arduino attendent l'ouverture du port avant de traiter les commandes
        print(f"Port {port}. Pour chaque format, observe la LED de l'Arduino, le multimetre ou AcqKnowledge.")
        print("Reponds o (reaction), n (rien) ou q (arreter).\n")
        for i, (desc, on, off) in enumerate(PROBE_CANDIDATES, 1):
            ser.reset_input_buffer()
            ser.write(on)
            ser.flush()
            time.sleep(0.3)
            echo = ser.read(200)
            extra = f"  [reponse du boitier : {echo!r}]" if echo else ""
            print(f"{i:2d}. {desc:35s} envoye {on!r}{extra}  -> reaction ? ", end="", flush=True)
            key = msvcrt.getwch().lower()
            print(key)
            if off:
                ser.write(off)
                ser.flush()
            time.sleep(0.5)
            if key == "q":
                break
            results.append((desc, on, key == "o", echo))
    ok = [r for r in results if r[2]]
    print("\nFormats qui ont fait reagir le boitier :" if ok else "\nAucun format n'a fait reagir le boitier.")
    for desc, on, _, echo in ok:
        print(f"  - {desc} ({on!r})")
    answered = [r for r in results if r[3]]
    for desc, on, _, echo in answered:
        print(f"  (le boitier a repondu a {on!r} : {echo!r})")


def find_box_port() -> str:
    ports = list_ports.comports()
    # 0x0403 = FTDI ; 0x2341 / 0x2A03 = Arduino (le MMBT-S est a base d'ATmega32U4)
    ftdi = [p.device for p in ports if p.vid in (0x0403, 0x2341, 0x2A03)]
    if len(ftdi) == 1:
        return ftdi[0]
    if len(ports) == 1:
        return ports[0].device
    raise serial.SerialException(
        f"Port du boitier introuvable automatiquement ({len(ports)} port(s) COM). "
        "Precise-le avec --port (voir la commande 'list')."
    )


def cmd_trigger(args) -> None:
    port = args.port or find_box_port()
    channel_map = load_channel_map()
    new_file = not LOG_FILE.exists()
    with open_port(port, args.baud) as ser, LOG_FILE.open("a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["timestamp_iso", "unix_time", "port", "line", "code"])
        source = MAP_FILE.name if MAP_FILE.exists() else "defaut, lancer 'calibrate' pour corriger"
        print(f"Port {port} ouvert. Log : {LOG_FILE}")
        print(f"  Touches {', '.join(map(str, sorted(channel_map)))} : trigger sur ce canal AcqKnowledge ({source})")
        print(f"  Espace / Entree : trigger sur le canal {args.line} (--line)")
        print("  q / Echap       : quitter\n")
        n = 0
        while True:
            key = msvcrt.getwch()  # attend un appui, sans avoir besoin de Entree
            if key in ("q", "Q", "\x1b"):
                break
            if key in (" ", "\r"):
                channel = args.line
            elif key.isdigit():
                channel = int(key)
            else:
                continue
            if channel not in channel_map:
                print(f"  canal {channel} absent de la correspondance")
                continue
            code = 1 << channel_map[channel]
            t = pulse(ser, code, args.width)
            n += 1
            writer.writerow([t.isoformat(timespec="microseconds"), f"{t.timestamp():.6f}", port, channel, code])
            f.flush()
            print(f"  #{n} [{t:%H:%M:%S.%f}] trigger canal {channel} (bit {channel_map[channel]})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--baud", type=int, default=BAUDRATE)
    common.add_argument("--width", type=float, default=0.1, help="Duree avant remise a 0, en s (defaut 0.1)")

    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="Liste les ports COM")
    sub.add_parser("scan", parents=[common], help="Pulse sur le fil 1 via chaque port COM")
    p_lines = sub.add_parser("lines", parents=[common], help="Pulse sur chaque fil 1..7 successivement")
    p_lines.add_argument("--port", required=True)
    p_lines.add_argument("--interval", type=float, default=1.0, help="Ecart entre fils en s (defaut 1)")
    p_lines.add_argument("--repeat", type=int, default=1)
    p_trig = sub.add_parser("trigger", parents=[common], help="Trigger manuel au clavier (touches 1..7), log CSV")
    p_trig.add_argument("--port", help="Port COM (defaut : detection automatique du boitier FTDI)")
    p_trig.add_argument("--line", type=int, default=1, help="Fil envoye par Espace/Entree, 1..7 (defaut 1)")

    p_hold = sub.add_parser("hold", parents=[common], help="Maintient un niveau fixe (diagnostic cablage)")
    p_hold.add_argument("--port", help="Port COM (defaut : detection automatique)")
    p_hold.add_argument("--code", type=int, default=255, help="Octet a maintenir (defaut 255 = tous les fils)")
    p_hold.add_argument("--line", type=int, help="Maintenir un seul fil 1..8 (prioritaire sur --code)")

    p_probe = sub.add_parser("probe", parents=[common], help="Essaie des formats de commande courants")
    p_probe.add_argument("--port", help="Port COM (defaut : detection automatique)")

    p_cal = sub.add_parser("calibrate", parents=[common], help="Associe chaque canal AcqKnowledge a son bit")
    p_cal.add_argument("--port", help="Port COM (defaut : detection automatique)")

    parser.add_argument("--no-invert", action="store_true", help="Logique directe (bit a 1 = ligne haute)")

    args = parser.parse_args()
    global INVERT
    INVERT = not args.no_invert
    handlers = {"list": cmd_list, "scan": cmd_scan, "lines": cmd_lines, "trigger": cmd_trigger, "hold": cmd_hold,
                "probe": cmd_probe, "calibrate": cmd_calibrate}
    try:
        handlers[args.cmd](args)
    except serial.SerialException as exc:
        print(f"Erreur port serie : {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrompu.")


if __name__ == "__main__":
    main()
