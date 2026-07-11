"""
Turns a sequence of per-frame biomechanical metrics into:
  1. A rough phase segmentation of the shot (stance / backlift / downswing /
     impact / follow-through), detected heuristically from wrist trajectory.
  2. A list of coaching Tips, each tagged to the frame range it applies to.

This is a prototype rule engine, not a trained shot classifier. The
thresholds below are reasonable coaching rules-of-thumb, not hard science —
treat the output as a first-pass technique check, not a verdict.
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Tip:
    frame_start: int
    frame_end: int
    phase: str
    category: str
    severity: str  # "good" | "tip"
    message: str


@dataclass
class ShotReport:
    phases: dict  # phase_name -> (start_frame, end_frame)
    tips: list = field(default_factory=list)
    swing_side: str = "right"  # which wrist we tracked as the bat-side hand
    metrics_at_impact: dict = field(default_factory=dict)


def _smooth(values, window=5):
    """Simple moving average that tolerates None gaps by forward-filling."""
    filled = []
    last = None
    for v in values:
        if v is None:
            filled.append(last)
        else:
            filled.append(v)
            last = v
    arr = np.array([v if v is not None else np.nan for v in filled], dtype=np.float64)
    if np.all(np.isnan(arr)):
        return arr
    # fill remaining leading NaNs with first valid value
    first_valid = np.argmax(~np.isnan(arr))
    arr[:first_valid] = arr[first_valid]
    # forward fill any interior NaNs
    for i in range(1, len(arr)):
        if np.isnan(arr[i]):
            arr[i] = arr[i - 1]
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def _pick_swing_wrist(metrics_list):
    left_ys = [m["left_wrist"][1] if m and m.get("left_wrist") is not None else None for m in metrics_list]
    right_ys = [m["right_wrist"][1] if m and m.get("right_wrist") is not None else None for m in metrics_list]

    def valid_range(ys):
        vals = [y for y in ys if y is not None]
        return (max(vals) - min(vals)) if len(vals) >= 2 else 0.0

    return "left" if valid_range(left_ys) > valid_range(right_ys) else "right"


def detect_phases(metrics_list, fps):
    """
    Returns (phases_dict, swing_side).
    phases_dict maps phase name -> (start_frame_idx, end_frame_idx) inclusive.
    """
    n = len(metrics_list)
    swing_side = _pick_swing_wrist(metrics_list)
    wrist_key = f"{swing_side}_wrist"

    ys = [m[wrist_key][1] if m and m.get(wrist_key) is not None else None for m in metrics_list]
    ys_smooth = _smooth(ys, window=max(3, int(fps // 6) or 3))

    if len(ys_smooth) == 0 or np.all(np.isnan(ys_smooth)):
        # No usable wrist data at all — treat whole clip as one phase.
        return {"full_clip": (0, max(n - 1, 0))}, swing_side

    backlift_frame = int(np.argmin(ys_smooth))  # highest point on screen = smallest y

    search_start = min(backlift_frame + 1, n - 1)
    tail = ys_smooth[search_start:]
    impact_frame = search_start + int(np.argmax(tail)) if len(tail) > 0 else search_start

    follow_search_start = min(impact_frame + 1, n - 1)
    tail2 = ys_smooth[follow_search_start:]
    follow_frame = follow_search_start + int(np.argmin(tail2)) if len(tail2) > 0 else n - 1

    stance_end = max(backlift_frame - 1, 0)
    impact_window = max(1, int(fps // 15) or 1)  # a small window of frames around true impact

    phases = {
        "stance": (0, stance_end),
        "backlift": (stance_end, backlift_frame),
        "downswing": (backlift_frame, max(impact_frame - impact_window, backlift_frame)),
        "impact": (max(impact_frame - impact_window, 0), min(impact_frame + impact_window, n - 1)),
        "follow_through": (min(impact_frame + impact_window, n - 1), follow_frame),
    }
    return phases, swing_side


def _avg_metric(metrics_list, key, start, end):
    vals = [metrics_list[i][key] for i in range(start, end + 1)
            if metrics_list[i] and metrics_list[i].get(key) is not None]
    return float(np.mean(vals)) if vals else None


def analyze_shot(metrics_list, fps, handedness="right"):
    """
    handedness: "right" or "left" — which hand is the batter's *bottom* hand.
    For a right-handed batter the front leg/arm (leading side facing the
    bowler) is conventionally the LEFT side; for left-handed it's the RIGHT.
    This is a simplifying convention for the prototype, not detected from
    video, since front-on vs side-on camera angle changes what's visible.
    """
    front_side = "left" if handedness == "right" else "right"
    phases, swing_side = detect_phases(metrics_list, fps)
    n = len(metrics_list)

    tips = []

    def add(phase, category, severity, message):
        s, e = phases.get(phase, (0, n - 1))
        tips.append(Tip(frame_start=s, frame_end=e, phase=phase, category=category,
                         severity=severity, message=message))

    # ---- Backlift check ----
    if "backlift" in phases:
        s, e = phases["backlift"]
        wrist_key = f"{swing_side}_wrist"
        shoulder_key = "left_hip" if False else None  # placeholder not used
        wrist_y = _avg_metric(metrics_list, wrist_key, s, min(e, n - 1))
        # compare backlift wrist height to hip height as a scale-free reference
        hip_y = None
        hip_vals = [metrics_list[i]["hip_mid"][1] for i in range(s, min(e, n - 1) + 1)
                    if metrics_list[i] and metrics_list[i].get("hip_mid") is not None]
        if hip_vals:
            hip_y = float(np.mean(hip_vals))
        if wrist_y is not None and hip_y is not None:
            if wrist_y > hip_y:  # wrist below hip height on screen (y grows downward)
                add("backlift", "backlift", "tip",
                    "Backlift looks low — the bat hand barely rises above hip height. "
                    "A higher, straighter backlift (hands to around shoulder height) gives you "
                    "more time and a freer swing through the ball.")
            else:
                add("backlift", "backlift", "good",
                    "Good backlift height — hands are rising well above the hips before the downswing.")

    # ---- Impact-frame checks ----
    if "impact" in phases:
        s, e = phases["impact"]
        mid = (s + e) // 2
        impact_metrics = metrics_list[mid] if mid < n else None

        # Front knee bend
        knee_angle = _avg_metric(metrics_list, f"{front_side}_knee_angle", s, e)
        if knee_angle is not None:
            if knee_angle > 165:
                add("impact", "footwork", "tip",
                    f"Front ({front_side}) knee looks almost locked straight (~{knee_angle:.0f}°) at impact. "
                    "A little more knee flex helps you get into the shot and transfer weight forward.")
            elif knee_angle < 110:
                add("impact", "footwork", "tip",
                    f"Front knee is quite collapsed (~{knee_angle:.0f}°) at impact — watch that you're not "
                    "over-bending and losing height over the ball.")
            else:
                add("impact", "footwork", "good",
                    f"Front knee flex at impact looks solid (~{knee_angle:.0f}°) — good base for weight transfer.")

        # Head over front knee
        nose = impact_metrics.get("nose") if impact_metrics else None
        front_knee = impact_metrics.get(f"{front_side}_knee") if impact_metrics else None
        shoulder_mid = impact_metrics.get("shoulder_mid") if impact_metrics else None
        hip_mid = impact_metrics.get("hip_mid") if impact_metrics else None
        if nose is not None and front_knee is not None and shoulder_mid is not None and hip_mid is not None:
            torso_width = max(abs(shoulder_mid[1] - hip_mid[1]), 1.0)
            horiz_offset = abs(nose[0] - front_knee[0])
            ratio = horiz_offset / torso_width
            if ratio > 0.6:
                add("impact", "balance", "tip",
                    "Head is drifting away from the front knee at impact — try to get your head "
                    "stacked over the front leg for better balance and control.")
            else:
                add("impact", "balance", "good",
                    "Head position over the front knee looks well balanced at impact.")

        # Leading (front-side) arm/elbow
        elbow_key = f"{front_side}_elbow_angle"
        elbow_angle = _avg_metric(metrics_list, elbow_key, s, e)
        shoulder = impact_metrics.get(f"{front_side}_ankle") if impact_metrics else None  # not used
        if elbow_angle is not None:
            if elbow_angle < 100:
                add("impact", "arms", "tip",
                    f"Leading elbow is quite bent (~{elbow_angle:.0f}°) at impact — a higher, straighter "
                    "leading arm usually gives cleaner bat swing direction through the shot.")
            else:
                add("impact", "arms", "good",
                    f"Leading arm extension at impact looks good (~{elbow_angle:.0f}°).")

        # Spine lean
        lean = _avg_metric(metrics_list, "spine_lean_deg", s, e)
        if lean is not None:
            if lean > 25:
                add("impact", "balance", "tip",
                    f"Noticeable backward/side lean of the upper body (~{lean:.0f}° from vertical) at impact — "
                    "falling away can cost you control on the shot.")

    # ---- Follow-through check ----
    if "follow_through" in phases:
        s, e = phases["follow_through"]
        wrist_key = f"{swing_side}_wrist"
        end_y = None
        for i in range(e, max(s - 1, -1), -1):
            if metrics_list[i] and metrics_list[i].get(wrist_key) is not None:
                end_y = metrics_list[i][wrist_key][1]
                break
        hip_vals = [metrics_list[i]["hip_mid"][1] for i in range(s, e + 1)
                    if metrics_list[i] and metrics_list[i].get("hip_mid") is not None]
        hip_y = float(np.mean(hip_vals)) if hip_vals else None
        if end_y is not None and hip_y is not None:
            if end_y > hip_y * 0.95:
                add("follow_through", "follow_through", "tip",
                    "Follow-through finishes low — try to let the bat swing all the way through and "
                    "up, rather than stopping at the ball.")
            else:
                add("follow_through", "follow_through", "good",
                    "Nice full follow-through, finishing high.")

    metrics_at_impact = {}
    if "impact" in phases:
        s, e = phases["impact"]
        for key in [f"{front_side}_knee_angle", f"{front_side}_elbow_angle", "spine_lean_deg"]:
            metrics_at_impact[key] = _avg_metric(metrics_list, key, s, e)

    return ShotReport(phases=phases, tips=tips, swing_side=swing_side, metrics_at_impact=metrics_at_impact)
