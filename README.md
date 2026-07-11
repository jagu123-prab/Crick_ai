# Cricket AI Coach (prototype)

Takes a video of a batter playing a shot, overlays a pose skeleton, detects
rough shot phases (stance → backlift → downswing → impact → follow-through),
and burns in rule-based coaching tips at the point in the clip they apply to.
Also writes a plain-text summary report.

**This is a prototype.** Shot-phase detection and coaching feedback are
heuristics based on joint angles and wrist trajectory — not a trained
classifier and not a substitute for a real coach. Treat the output as a
first-pass technique check.

## How it works

1. **Pose estimation** — [MediaPipe Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
   extracts 33 body landmarks per frame.
2. **Biomechanics** (`biomechanics.py`) — turns raw landmarks into joint
   angles (elbow, knee), spine lean, and key positions.
3. **Shot analysis** (`shot_analyzer.py`) — tracks the bat-side wrist's
   vertical trajectory to segment the swing into phases, then runs a small
   rule engine at each phase (backlift height, front-knee bend, head-over-knee
   balance, leading-arm extension, follow-through height).
4. **Overlay** (`overlay.py`) — draws the skeleton and a caption banner with
   the current phase + tips onto each frame.
5. **`main.py`** ties it together: pass 1 runs pose estimation once and
   caches results, pass 2 renders the annotated output video.

## Setup (VS Code / local machine)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Open the folder in VS Code, select the `.venv` interpreter (Cmd/Ctrl+Shift+P →
"Python: Select Interpreter"), and you're set.

**First run only:** the pose model file (~5–30MB) is downloaded automatically
into `./models/` — you need internet access once for that.

## Usage

```bash
python main.py --input path/to/shot.mp4
```

Optional flags:

| Flag | Default | Notes |
|---|---|---|
| `--output` | `<input>_coached.mp4` | Output video path |
| `--handedness` | `right` | `right` or `left` — which hand is the batter's bottom hand. Determines which side is treated as the "front" leg/arm for the drills below, since this can't be reliably inferred from the video alone. |
| `--model` | `full` | `lite` (fastest), `full` (balanced), `heavy` (most accurate, slowest) |
| `--report` | `<output>_report.txt` | Where to save the text report |

Example:

```bash
python main.py -i clips/cover_drive.mp4 --handedness right --model full
```

This produces:
- `clips/cover_drive_coached.mp4` — original video with skeleton overlay + tip banners
- `clips/cover_drive_coached_report.txt` — text summary of phases, joint angles at impact, and all coaching notes

## What it checks

- **Backlift** — is the bat hand raised well above hip height before the downswing?
- **Front-knee bend at impact** — locked straight vs. over-collapsed vs. solid base
- **Head-over-front-knee balance at impact**
- **Leading-arm extension at impact** — collapsed ("chicken-wing") vs. extended
- **Spine lean at impact** — falling away from the shot
- **Follow-through height** — finishing low vs. finishing high

## Known limitations (it's a prototype)

- Assumes a single batter, reasonably visible, roughly side-on or front-on to the camera.
- Doesn't track the bat itself — the bat-side wrist is used as a proxy, so bat-face angle and swing path aren't directly measured.
- Front leg/arm is assigned from the `--handedness` flag rather than detected, since camera angle changes what's visible.
- Phase detection is a simple heuristic (wrist-height local min/max) — fast or unconventional shots (e.g. sweeps, ramps) may segment poorly.
- Thresholds (knee angle ranges, lean angle, etc.) are general coaching rules of thumb, not calibrated against real player data.
- No true 3D — angles are computed from 2D pixel projections, so camera angle affects accuracy. A side-on camera works best for elbow/knee flexion; a front-on camera works best for head/knee alignment.

## Extending it

- Swap the rule engine in `shot_analyzer.py` for a trained shot classifier (e.g. an LSTM over landmark sequences) if you want automatic shot-type detection (drive/cut/pull/sweep) instead of generic phase-based rules.
- Add bat tracking (a small object detector) to get real swing-path and bat-face-angle feedback.
- Support multiple camera angles / multi-view triangulation for true 3D joint angles.
