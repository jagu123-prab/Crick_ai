"""
Cricket AI Coach — prototype

Usage:
    python main.py --input path/to/shot.mp4 [--output out.mp4]
                    [--handedness right|left] [--model lite|full|heavy]

Reads a video of a batter playing a shot, overlays a pose skeleton, detects
rough shot phases (stance / backlift / downswing / impact / follow-through)
from wrist trajectory, and burns in rule-based coaching tips at the relevant
points in the clip. Also prints/saves a text summary report.

This is a prototype: the phase detection and coaching rules are heuristics,
not a trained model, and assume a single batter roughly facing the camera.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

import biomechanics as bio
import overlay as ov
from pose_estimator import PoseEstimator
from shot_analyzer import analyze_shot


def parse_args():
    p = argparse.ArgumentParser(description="Cricket AI Coach prototype")
    p.add_argument("--input", "-i", required=True, help="Path to input video")
    p.add_argument("--output", "-o", default=None, help="Path to output video (default: <input>_coached.mp4)")
    p.add_argument("--handedness", choices=["right", "left"], default="right",
                    help="Batter's dominant (bottom) hand — affects which side is treated as the front leg/arm")
    p.add_argument("--model", choices=["lite", "full", "heavy"], default="full",
                    help="MediaPipe pose model variant (lite=fastest, heavy=most accurate)")
    p.add_argument("--report", default=None, help="Path to save a text coaching report (default: alongside output video)")
    return p.parse_args()


def phase_for_frame(phases, idx):
    for name, (s, e) in phases.items():
        if s <= idx <= e:
            return name
    return None


def tips_for_frame(tips, idx):
    good, warn = [], []
    for t in tips:
        if t.frame_start <= idx <= t.frame_end:
            (good if t.severity == "good" else warn).append(t.message)
    return good, warn


def build_report_text(report, fps, handedness):
    lines = []
    lines.append("CRICKET AI COACH — SHOT REPORT (prototype)")
    lines.append("=" * 50)
    lines.append(f"Assumed batting hand: {handedness}-handed  |  Bat-side wrist tracked: {report.swing_side}")
    lines.append("")
    lines.append("Detected phases (frame ranges):")
    for name, (s, e) in report.phases.items():
        lines.append(f"  - {name:<15} frames {s:>5} - {e:<5} ({(e - s + 1) / fps:.2f}s)")
    lines.append("")
    lines.append("Metrics at impact:")
    for k, v in report.metrics_at_impact.items():
        lines.append(f"  - {k}: {v:.1f} deg" if v is not None else f"  - {k}: n/a")
    lines.append("")
    lines.append("Coaching notes:")
    for t in report.tips:
        tag = "GOOD" if t.severity == "good" else "TIP "
        lines.append(f"  [{tag}] ({t.phase}) {t.message}")
    lines.append("")
    lines.append("Note: this is an automated, rule-based heuristic check on a single clip,")
    lines.append("not a substitute for a qualified coach's eye.")
    return "\n".join(lines)


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"Input video not found: {args.input}")
        sys.exit(1)

    output_path = args.output or (os.path.splitext(args.input)[0] + "_coached.mp4")
    report_path = args.report or (os.path.splitext(output_path)[0] + "_report.txt")

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Could not open video: {args.input}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[main] Input: {args.input} ({width}x{height} @ {fps:.1f}fps, ~{total_frames} frames)")
    print(f"[main] Loading pose model ({args.model}) — first run downloads the model file...")

    landmarks_by_frame = []
    metrics_by_frame = []

    with PoseEstimator(model_variant=args.model) as estimator:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp_ms = int((idx / fps) * 1000)
            lm = estimator.process_frame(frame, timestamp_ms)
            landmarks_by_frame.append(lm)
            metrics_by_frame.append(bio.frame_metrics(lm) if lm is not None else None)
            idx += 1
            if idx % 30 == 0:
                print(f"[main] Pose pass: {idx}/{total_frames} frames processed", end="\r")
    cap.release()
    print(f"\n[main] Pose estimation complete on {len(landmarks_by_frame)} frames.")

    detected = sum(1 for m in metrics_by_frame if m is not None)
    if detected == 0:
        print("[main] No person detected in this video — check that the batter is clearly visible.")
        sys.exit(1)
    print(f"[main] Person detected in {detected}/{len(metrics_by_frame)} frames.")

    report = analyze_shot(metrics_by_frame, fps, handedness=args.handedness)

    report_text = build_report_text(report, fps, args.handedness)
    print("\n" + report_text)
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\n[main] Saved report to {report_path}")

    # ---- Pass 2: render annotated video ----
    cap = cv2.VideoCapture(args.input)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        lm = landmarks_by_frame[idx] if idx < len(landmarks_by_frame) else None
        frame = ov.draw_skeleton(frame, lm)

        phase = phase_for_frame(report.phases, idx)
        good, warn = tips_for_frame(report.tips, idx)
        frame = ov.draw_tip_banner(frame, phase, warn, good)

        writer.write(frame)
        idx += 1
        if idx % 30 == 0:
            print(f"[main] Render pass: {idx}/{total_frames} frames written", end="\r")

    cap.release()
    writer.release()
    print(f"\n[main] Saved annotated video to {output_path}")


if __name__ == "__main__":
    main()
