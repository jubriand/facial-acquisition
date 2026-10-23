"""Telecharge le modele MediaPipe Face Landmarker (Apache 2.0) dans v1/models et v2/models."""
import shutil
import urllib.request
from pathlib import Path

URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
       "face_landmarker/float16/1/face_landmarker.task")
HERE = Path(__file__).resolve().parent

targets = [HERE / v / "models" / "face_landmarker.task" for v in ("v1", "v2")]
first = targets[0]
first.parent.mkdir(exist_ok=True)
if not first.exists():
    print("Telechargement :", URL)
    urllib.request.urlretrieve(URL, first)
for t in targets[1:]:
    t.parent.mkdir(exist_ok=True)
    shutil.copyfile(first, t)
print("Modele installe :", ", ".join(str(t.relative_to(HERE)) for t in targets))
