"""
Drawing helpers for rendering the skeleton and coaching-tip captions onto
video frames with OpenCV.
"""
import cv2
import numpy as np
import landmarks as L

SKELETON_COLOR = (60, 220, 60)      # green bones (BGR)
JOINT_COLOR = (40, 180, 255)        # orange joints (BGR)
GOOD_COLOR = (80, 220, 80)          # green text
TIP_COLOR = (60, 170, 255)          # amber text
TEXT_COLOR = (255, 255, 255)


def draw_skeleton(frame, lm, threshold=L.VISIBILITY_THRESHOLD):
    if lm is None:
        return frame
    for a, b in L.SKELETON_BONES:
        if lm[a][3] >= threshold and lm[b][3] >= threshold:
            pa = (int(lm[a][0]), int(lm[a][1]))
            pb = (int(lm[b][0]), int(lm[b][1]))
            cv2.line(frame, pa, pb, SKELETON_COLOR, 2, cv2.LINE_AA)
    for i in range(lm.shape[0]):
        if lm[i][3] >= threshold:
            p = (int(lm[i][0]), int(lm[i][1]))
            cv2.circle(frame, p, 3, JOINT_COLOR, -1, cv2.LINE_AA)
    return frame


def _wrap_text(text, font, scale, thickness, max_width):
    words = text.split(" ")
    lines, current = [], ""
    for w in words:
        candidate = (current + " " + w).strip()
        size = cv2.getTextSize(candidate, font, scale, thickness)[0]
        if size[0] > max_width and current:
            lines.append(current)
            current = w
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def draw_tip_banner(frame, phase_label, tip_messages, good_messages):
    """
    Draws a semi-transparent caption box at the bottom of the frame with the
    current shot phase and any active coaching messages.
    """
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    pad = 14
    max_text_width = w - 2 * pad

    lines = []
    if phase_label:
        lines.append((f"PHASE: {phase_label.upper()}", TEXT_COLOR, 0.62, 2))
    for msg in good_messages:
        for wline in _wrap_text("+ " + msg, font, 0.55, 1, max_text_width):
            lines.append((wline, GOOD_COLOR, 0.55, 1))
    for msg in tip_messages:
        for wline in _wrap_text("! " + msg, font, 0.55, 1, max_text_width):
            lines.append((wline, TIP_COLOR, 0.55, 1))

    if not lines:
        return frame

    line_height = 24
    box_h = pad * 2 + line_height * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - box_h), (w, h), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    y = h - box_h + pad + 14
    for text, color, scale, thickness in lines:
        cv2.putText(frame, text, (pad, y), font, scale, color, thickness, cv2.LINE_AA)
        y += line_height

    return frame
