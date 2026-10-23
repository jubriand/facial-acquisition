"""
Images thermiques en direct depuis PIX Connect via le Connect SDK (IPC) d'Optris.

PIX Connect garde la camera (et peut enregistrer) ; ce module recoit en parallele chaque image
en temperatures (degC) avec ses metadonnees, dont l'horodatage camera (unites de 100 ns).
Prerequis dans PIX Connect : Tools > Configuration > External Communication > Connect SDK (IPC).
"""

import ctypes
import ctypes.wintypes as wt
import queue
import threading
import time
from pathlib import Path

import numpy as np

DLL_PATH = Path(r"C:\Program Files (x86)\Optris GmbH\PIX Connect\SDK\Connect SDK\Lib\v120\ImagerIPC2x64.dll")
S_OK = 0
STATE_INIT_COMPLETED, STATE_SERVER_STOPPED, STATE_FRAME_INIT = 0x0001, 0x0002, 0x0010
IPC_MODE_TEMPS = 1


class FCOORD(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]


class FrameMetadata2(ctypes.Structure):
    _pack_ = 2  # #pragma pack(2) dans ImagerIPC2.h
    _fields_ = [("Size", ctypes.c_ushort), ("Counter", ctypes.c_uint), ("CounterHW", ctypes.c_uint),
                ("Timestamp", ctypes.c_longlong), ("TimestampMedia", ctypes.c_longlong),
                ("FlagState", ctypes.c_int), ("IsSameFrame", wt.BOOL), ("TempChip", ctypes.c_float),
                ("TempFlag", ctypes.c_float), ("TempBox", ctypes.c_float), ("TempOptics", ctypes.c_float),
                ("LensPosition", ctypes.c_ushort), ("HotSpot", FCOORD), ("ColdSpot", FCOORD),
                ("PIFnDI", ctypes.c_ushort), ("PIFDI", wt.DWORD), ("PIFnAI", ctypes.c_ushort),
                ("PIFAI", ctypes.c_ushort * 30)]


class PixIPC:
    def __init__(self, index: int = 0):
        self.index = index
        self.dll = ctypes.WinDLL(str(DLL_PATH))
        self.dll.GetIPCState.restype = ctypes.c_ushort
        self.dll.GetFrameQueue.restype = ctypes.c_ushort
        self.dll.GetTempRangeDecimal.restype = ctypes.c_ushort
        self.dll.GetSerialNumber.restype = ctypes.c_ulong
        self.dll.GetRecordingStartTime.restype = ctypes.c_longlong
        self.running = self.connected = False
        self.width = self.height = 0
        self.decimals = 1
        self.buf = None
        self.meta = FrameMetadata2()

    def start(self) -> None:
        hr = self.dll.InitImagerIPC(self.index)
        if hr != S_OK:
            raise RuntimeError(f"InitImagerIPC a echoue ({hr:#x}) : activer 'Connect SDK (IPC)' dans PIX Connect")
        hr = self.dll.RunImagerIPC(self.index)
        if hr != S_OK:
            raise RuntimeError(f"RunImagerIPC a echoue ({hr:#x})")
        self.running = True

    def poll_state(self) -> None:
        self.dll.ImagerIPCProcessMessages(self.index)  # indispensable en mode "polling" (cf. exemple C++ Optris)
        state = self.dll.GetIPCState(self.index, True)
        if state & STATE_SERVER_STOPPED:
            self.connected = False
            self.width = 0
        if state & STATE_INIT_COMPLETED:
            self.connected = True
            self.dll.SetIPCMode(self.index, IPC_MODE_TEMPS)
            self.decimals = self.dll.GetTempRangeDecimal(self.index, True)
        if state & STATE_FRAME_INIT or (self.connected and not self.width):
            # l'evenement FrameInit peut etre emis avant la connexion : on demande le format directement
            w, h, d = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
            if self.dll.GetFrameConfig(self.index, ctypes.byref(w), ctypes.byref(h), ctypes.byref(d)) == S_OK \
                    and w.value > 0 and h.value > 0:
                self.width, self.height = w.value, h.value
                self.buf = (ctypes.c_ushort * (self.width * self.height))()
                size = ctypes.c_int()
                self.dll.GetFrameMetadataSize(self.index, ctypes.byref(size))
                # taille exacte attendue par PIX Connect (78 octets pour la Xi400), sinon GetFrame2 echoue
                self.meta_size = min(size.value, ctypes.sizeof(self.meta)) or ctypes.sizeof(self.meta)

    def get_frame(self, timeout_ms: int = 100) -> tuple[np.ndarray, FrameMetadata2] | None:
        """Prochaine image en degC (float32, hauteur x largeur) avec ses metadonnees, ou None."""
        self.poll_state()
        if not self.width or self.dll.GetFrameQueue(self.index) == 0:
            return None
        hr = self.dll.GetFrame2(self.index, timeout_ms, self.buf, ctypes.sizeof(self.buf),
                                ctypes.byref(self.meta), self.meta_size)
        if hr != S_OK:
            return None
        raw = np.frombuffer(self.buf, dtype=np.uint16).reshape(self.height, self.width).astype(np.float32)
        temps = (raw - 1000.0) / 10.0 if self.decimals == 1 else raw / 100.0
        return temps, self.meta

    def serial(self) -> int:
        return self.dll.GetSerialNumber(self.index)

    def record(self) -> None:
        self.dll.FileRecord(self.index)

    def stop(self) -> None:
        self.dll.FileStop(self.index)

    def stored_path(self) -> str:
        """Chemin du dernier fichier enregistre par PIX Connect (sauvegarde automatique en mode IPC)."""
        buf = ctypes.create_unicode_buffer(1024)
        return buf.value if self.dll.GetPathOfStoredFile(self.index, buf, len(buf)) == S_OK else ""

    def release(self) -> None:
        if self.running:
            self.dll.ReleaseImagerIPC(self.index)
            self.running = False


class IPCHub(threading.Thread):
    """Connexion IPC unique, partagee : un seul thread appelle la DLL.

    - diffuse chaque image aux abonnes : callback(temps degC, metadonnees, perf_counter de reception) ;
    - execute les commandes (record/stop/stored_path) dans ce thread via call() ;
    - se (re)connecte automatiquement si PIX Connect est lance plus tard ou redemarre.
    """

    def __init__(self):
        super().__init__(daemon=True)
        self.listeners = []
        self.commands: queue.Queue = queue.Queue()
        self.ipc: PixIPC | None = None
        self.connected = False
        self.last_frame_t = 0.0
        self.error = ""
        self.stop_flag = False

    def subscribe(self, callback) -> None:
        self.listeners.append(callback)

    def call(self, fn_name: str, timeout: float = 3.0):
        """Execute ipc.<fn_name>() dans le thread IPC ; renvoie (resultat, perf_counter de l'appel)."""
        done, box = threading.Event(), {}
        self.commands.put((fn_name, done, box))
        if not done.wait(timeout):
            raise RuntimeError(f"IPC : pas de reponse a {fn_name}")
        if "error" in box:
            raise RuntimeError(box["error"])
        return box["result"], box["t"]

    def streaming(self, max_age: float = 2.0) -> bool:
        return self.connected and time.perf_counter() - self.last_frame_t < max_age

    def _run_commands(self):
        while not self.commands.empty():
            fn_name, done, box = self.commands.get()
            try:
                if self.ipc is None or not self.connected:
                    raise RuntimeError("IPC non connecte a PIX Connect")
                box["t"] = time.perf_counter()
                box["result"] = getattr(self.ipc, fn_name)()
            except Exception as exc:
                box["error"] = str(exc)
            done.set()

    def run(self):
        while not self.stop_flag:
            if self.ipc is None:
                try:
                    self.ipc = PixIPC()
                    self.ipc.start()
                    self.error = ""
                    self.last_frame_t = time.perf_counter()
                except Exception as exc:
                    self.error = str(exc)
                    self.ipc = None
                    self._run_commands()
                    time.sleep(2.0)
                    continue
            res = self.ipc.get_frame(10)
            self.connected = self.ipc.connected
            if res is not None:
                temps, meta = res
                self.last_frame_t = time.perf_counter()
                for cb in list(self.listeners):
                    cb(temps, meta, self.last_frame_t)
            else:
                time.sleep(0.003)
            self._run_commands()
            if time.perf_counter() - self.last_frame_t > 10:
                self.ipc.release()  # plus d'images depuis 10 s (PIX Connect ferme/redemarre) : reconnexion
                self.ipc, self.connected = None, False
        if self.ipc is not None:
            self.ipc.release()


_hub: IPCHub | None = None


def get_hub() -> IPCHub:
    """Connexion IPC partagee du processus (demarree au premier appel)."""
    global _hub
    if _hub is None:
        _hub = IPCHub()
        _hub.start()
    return _hub


if __name__ == "__main__":
    ipc = PixIPC()
    ipc.start()
    t0 = time.perf_counter()
    n, stamps = 0, []
    while time.perf_counter() - t0 < 5:
        res = ipc.get_frame()
        if res is None:
            time.sleep(0.005)
            continue
        temps, meta = res
        n += 1
        stamps.append((meta.Counter, meta.CounterHW, meta.Timestamp, time.perf_counter()))
        if n == 1:
            print(f"connecte camera #{ipc.serial()} : {ipc.width}x{ipc.height}, decimales {ipc.decimals}")
            print(f"temp min/max/centre : {temps.min():.1f} / {temps.max():.1f} / {temps[ipc.height // 2, ipc.width // 2]:.1f} degC")
    print(f"{n} images en 5 s, connecte={ipc.connected}")
    if len(stamps) > 2:
        ts = np.array([s[2] for s in stamps], dtype=np.float64) / 1e7
        print("compteurs", stamps[0][:2], "->", stamps[-1][:2], "| periode horodatage moy %.2f ms" % (1000 * np.diff(ts).mean()))
    ipc.release()
