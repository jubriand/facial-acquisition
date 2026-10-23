# Contactless physiological monitoring with thermal and RGB imaging: a proof-of-concept interface

This repository holds the acquisition and real-time monitoring software I developed during my PhD for a
neuroscience project in aeronautics. The project studies how pilots respond physiologically to the different
phases of a flight, especially high-workload periods such as landing. The aim is to observe these responses
**without placing anything on the participant**. Belts, electrodes and finger clips are awkward in a cockpit or a
flight simulator. They can also interfere with the task we want to study.

The code is a **proof of concept**. It shows that we can, on standard lab hardware:

1. synchronize a thermal camera with a reference physiological recording system (Biopac MP150) to within a few
   milliseconds, using a hardware TTL trigger;
2. track the face of a moving participant in real time on thermal video, even when the participant startles or
   turns their head;
3. estimate breathing rate, pulse rate and facial skin temperatures from thermal imaging (and, in v1, from an
   ordinary RGB webcam), live and with a quality index;
4. tag the experimental phases (*flight* / *landing*) during the acquisition and compare them immediately
   afterwards.

> **This is research software, not a medical device.** The estimates have not yet been validated against the
> reference sensors on a participant sample. They must not be used for any clinical or operational
> decision. See [Limitations](#limitations).

---

## Repository layout

| Folder | Content |
|---|---|
| [`v1/`](v1) | Full multimodal setup: TTL trigger box → Biopac, automated thermal recording with synchronization, and a live demonstrator that combines **thermal + RGB webcam** estimates. |
| [`v2/`](v2) | Lightweight **thermal-only** interface for the experiments: forehead/nose ROI tracking, face temperature, breathing and pulse rate, CSV recording, flight/landing phase marker and an automatic results window. |

Both versions share the same face-tracking and signal-processing core (`vitals.py`). v1 also displays a
positioning guide for the participant.

---

## Hardware and software setup

| Component | Model used in the lab |
|---|---|
| Thermal camera | Optris Xi 400 (382 × 288 px, ~27 Hz), driven by **Optris PIX Connect** |
| Reference physiology | Biopac MP150 with GSR100C (electrodermal activity) and RSP100C (respiratory belt), AcqKnowledge 4.2 |
| Trigger | USB → TTL box (serial COM port, Arduino-based), wired to the MP150 digital inputs |
| RGB camera (v1) | Any UVC webcam |
| Acquisition PC | Windows 10/11, Python 3.10+ (developed on 3.14), **no administrator rights required** |

The Biopac and the camera run on two different computers. The only link between them is the TTL line, which
writes a marker into the AcqKnowledge file exactly when the thermal recording starts and stops.

### Interfacing with PIX Connect

The camera is controlled through the **Connect SDK IPC interface** that ships with PIX Connect
(`ImagerIPC2x64.dll`, not redistributed here). Enable it once in PIX Connect: *Tools → Configuration → External
communication → Inter Process Communication (IPC)*. Through IPC the software can:

- read the radiometric frames live (temperatures in °C) together with their metadata (frame counter, camera
  timestamp);
- start and stop recordings (`FileRecord` / `FileStop`), so they no longer depend on keyboard shortcuts or on
  the window state.

Only one IPC client can be connected to PIX Connect at a time, so close the other interfaces before starting
one.

---

## Installation

```bash
git clone https://github.com/<user>/aero-physio-sensing.git
cd aero-physio-sensing
python -m pip install --user -r requirements.txt
```

Face tracking uses the **MediaPipe Face Landmarker** model (Apache 2.0, not included in this repository).
Download it into the `models/` folder of each version:

```bash
python download_model.py
```

If the model is missing, the thermal module falls back to the rule-based detector described below. It is less
robust to large head rotations.

---

## v1: synchronization and multimodal demonstrator

### Trigger box (`test_com_trigger.py`, `trigger_background.py`)

The trigger box receives one byte over a serial port, and each bit drives one TTL line. The box we used has
**inverted logic**: it outputs `~code`, and `0xFF` is the idle state. Its wiring to the MP150 digital channels is
calibrated once and stored in `channel_map.json`.

```bash
python v1/test_com_trigger.py scan                # list the serial ports and find the box
python v1/test_com_trigger.py calibrate           # map each bit to an AcqKnowledge channel
python v1/test_com_trigger.py trigger             # Space/Enter = one pulse
python v1/trigger_background.py --key f9          # global hotkey running in the background
```

### Synchronized recording (`sync_app.py`)

A single button starts the thermal recording through IPC and sends the TTL trigger to the Biopac. The button is
only enabled once the camera stream, the trigger box and PIX Connect are all ready. The latencies we measured
during development are:

- IPC confirmation of the recording start: **4–8 ms** after the command;
- duration of the recorded `.ravi` file versus the interval between the start and stop commands: **+1 to
  +34 ms**.

Each session saves the `.ravi` file, the trigger timestamps and the measured offsets, so the thermal and Biopac
timelines can be realigned offline.

### Live demonstrator (`demo_app.py`)

Shows the thermal video and the webcam video side by side. On each one it displays the tracked regions of
interest, the forehead and nose temperatures, and breathing and pulse rate estimates with their quality index.
It can also replay a `.ravi` file.

```bash
python v1/demo_app.py                 # live (PIX Connect + webcam 0)
python v1/demo_app.py --no-webcam     # thermal only
python v1/demo_app.py --ravi session.ravi
```

---

## v2: thermal-only acquisition interface

`thermal_app.py` is the interface used during the experimental sessions.

- Live thermal video showing the **face contour**, the **forehead ROI** and the **nose ROI**, each with its
  temperature.
- Tiles for the global face temperature (mean and maximum of the skin pixels), forehead, nose, **breathing rate**
  and **pulse rate**, with 2-minute trends.
- A compact **REC** button that writes every value to a CSV file, together with timestamps (wall clock, elapsed
  time, camera time), tracking status, quality indices and the current phase.
- A **phase marker**: the *✈ FLIGHT PHASE* pill switches to *✈ LANDING PHASE* on click or with the **P** key.
  Each recording starts in the flight phase.
- When recording stops, a **results window** opens with two tabs:
  - **Evolution**: temperatures, breathing rate and pulse rate on a shared time axis, with landing periods
    shaded and low-quality segments faded;
  - **Flight vs Landing**: per-metric distributions (box plots and means) of the valid samples in each phase,
    with a summary table. It is exported as `<name>_phases.csv` and `<name>_phases.png`.

```bash
python v2/thermal_app.py                      # live
python v2/thermal_app.py --ravi session.ravi  # replay
```

CSV columns: `timestamp_iso, elapsed_s, camera_time_s, unit, face_tracked, t_face, t_face_max, t_forehead,
t_nose, rr_per_min, rr_quality, rr_nostril_temp_per_min, hr_bpm, hr_quality, phase`.

---

## Methods

### Face tracking on thermal video

Thermal images have no color and little texture, and glasses appear as cold, opaque patches. The tracker has
three layers, so that it stays responsive when the participant moves suddenly:

1. **Landmark detection.** MediaPipe Face Landmarker [1] runs on a contrast-normalized rendering of the thermal
   frame (inverted grey levels or CLAHE, whichever works), every other frame. The forehead, nose and face-oval
   polygons come from the 478-point mesh.
2. **Between detections.** Sparse pyramidal Lucas–Kanade optical flow propagates the regions [2]. A robust
   similarity transform is estimated with RANSAC, which follows translations, rotations and scale changes at
   frame rate.
3. **Smoothing.** A One Euro filter [3] removes jitter when the head is still, while keeping latency low during
   fast movements.

If MediaPipe loses the face for more than about one second, a **rule-based detector** takes over. It segments
warm skin, finds the inner eye corners (or the cold patches of glasses) and the nostrils, then re-anchors the
tracker. On our test recordings, the tracker recovered from a startle in about 40 ms and processed frames in
15–18 ms.

### Breathing rate

- **Primary signal: optical flow** [4] in the nostril/mouth region. The global head motion, measured on the
  forehead, is subtracted to isolate the breathing movement.
- **Secondary signal: nostril temperature.** Air flowing in and out cools and warms the skin under the nostrils
  periodically [5, 6]. Its rate is recorded separately (`rr_nostril_temp_per_min`) as a cross-check.
- Signals are resampled on the camera timestamps and band-passed to 0.1–0.75 Hz (6–45 breaths/min). The rate is
  taken as the spectral peak over a 30 s sliding window, with a first estimate after 12 s.

### Pulse rate (thermal)

The mean forehead temperature carries a small pulsatile component caused by superficial blood flow [7]. The
signal is band-passed to 0.75–3 Hz (45–180 bpm) and analyzed on a 20 s window.

### RGB webcam (v1)

- **Pulse:** remote photoplethysmography with the **POS** (Plane-Orthogonal-to-Skin) algorithm [8], applied to the
  forehead and cheeks given by MediaPipe and restricted to skin pixels in the YCrCb color space. A frontal-pose
  check rejects frames where the head is turned away.
- **Breathing:** optical flow on a chest region located below the face [9], with a fallback on subtle head
  movements [10].

### Quality index

Each rate comes with a quality index in [0, 1]: the fraction of in-band spectral power concentrated around the
detected peak. Values below a threshold (breathing 0.5, pulse 0.3) are kept in the CSV but displayed as
unreliable and excluded from the phase comparison. These thresholds are **provisional**. They still have to be
calibrated against the Biopac respiratory belt.

---

## Limitations

- **No formal validation yet.** Agreement with the reference sensors (RSP100C belt, ECG/PPG) has not been
  quantified on a participant sample. This is the next step of the project.
- **Thermal pulse rate is the weakest measure.** Its quality index is often low, which is consistent with the
  literature: the pulse is best measured on a superficial vessel rather than averaged over the forehead.
- **Cadence.** IPC delivers about 20 frames/s out of the camera's 27 Hz.
- **Constraints.** Performance degrades with extreme head poses, a partly framed face, airflow on the face and
  short phases (under about 40–60 s, too few breathing estimates can be made).
- **Platform.** The software is Windows-only (Win32 APIs, Optris SDK) and was tested with one camera model and one
  trigger box.
- The user interface text is partly in French.

---

## References

1. Kartynnik Y., Ablavatski A., Grishchenko I., Grundmann M. *Real-time facial surface geometry from monocular
   video on mobile GPUs.* CVPR Workshops, 2019.
2. Bouguet J.-Y. *Pyramidal implementation of the Lucas Kanade feature tracker.* Intel Corporation, 2000.
3. Casiez G., Roussel N., Vogel D. *1€ filter: a simple speed-based low-pass filter for noisy input in
   interactive systems.* CHI, 2012.
4. Farnebäck G. *Two-frame motion estimation based on polynomial expansion.* SCIA, 2003.
5. Murthy R., Pavlidis I. *Noncontact measurement of breathing function.* IEEE Engineering in Medicine and Biology
   Magazine, 25(3), 2006.
6. Cho Y., Julier S. J., Marquardt N., Bianchi-Berthouze N. *Robust tracking of respiratory rate in high-dynamic
   range scenes using mobile thermal imaging.* Biomedical Optics Express, 8(10), 2017.
7. Garbey M., Sun N., Merla A., Pavlidis I. *Contact-free measurement of cardiac pulse based on the analysis of
   thermal imagery.* IEEE Transactions on Biomedical Engineering, 54(8), 2007.
8. Wang W., den Brinker A. C., Stuijk S., de Haan G. *Algorithmic principles of remote PPG.* IEEE Transactions on
   Biomedical Engineering, 64(7), 2017.
9. Romano C., Schena E., Silvestri S., Massaroni C. *Non-contact respiratory monitoring using an RGB camera for
   real-world applications.* Sensors, 21(15), 2021.
10. Schrumpf F., Mönch C., Bausch G., Fuchs M. *Exploiting weak head movements for camera-based respiration
    detection.* IEEE EMBC, 2019.

---

## License

The code is released under the MIT License (see [`LICENSE`](LICENSE)). Optris PIX Connect, its SDK and the
MediaPipe model are third-party components covered by their own licenses and are not distributed here.

If you use this work, please cite this repository. A paper describing the validation study is in preparation.
