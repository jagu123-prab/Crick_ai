"""
Geometry / biomechanics helpers. Everything here works on the (33, 4)
landmark array produced by PoseEstimator: columns are [x_px, y_px, z, visibility].
"""
import numpy as np
import landmarks as L


def angle_deg(a, b, c):
    """
    Angle at point b (in degrees), formed by rays b->a and b->c.
    a, b, c are [x, y] pixel coordinates.
    """
    a, b, c = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), np.asarray(c, dtype=np.float64)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom < 1e-6:
        return None
    cosang = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def visible(lm_row, threshold=L.VISIBILITY_THRESHOLD):
    return lm_row is not None and lm_row[3] >= threshold


def xy(lm, idx):
    return lm[idx][:2]


def frame_metrics(lm):
    """
    Given one frame's landmark array, compute a dict of biomechanical
    metrics used later for phase detection and coaching rules.
    Any metric that can't be computed (low visibility) is set to None.
    """
    m = {}

    def safe_angle(a_idx, b_idx, c_idx):
        if visible(lm[a_idx]) and visible(lm[b_idx]) and visible(lm[c_idx]):
            return angle_deg(xy(lm, a_idx), xy(lm, b_idx), xy(lm, c_idx))
        return None

    # Elbow angles (shoulder-elbow-wrist)
    m["left_elbow_angle"] = safe_angle(L.LEFT_SHOULDER, L.LEFT_ELBOW, L.LEFT_WRIST)
    m["right_elbow_angle"] = safe_angle(L.RIGHT_SHOULDER, L.RIGHT_ELBOW, L.RIGHT_WRIST)

    # Knee angles (hip-knee-ankle)
    m["left_knee_angle"] = safe_angle(L.LEFT_HIP, L.LEFT_KNEE, L.LEFT_ANKLE)
    m["right_knee_angle"] = safe_angle(L.RIGHT_HIP, L.RIGHT_KNEE, L.RIGHT_ANKLE)

    # Spine lean: angle of shoulder-midpoint -> hip-midpoint line vs vertical
    if visible(lm[L.LEFT_SHOULDER]) and visible(lm[L.RIGHT_SHOULDER]) and \
       visible(lm[L.LEFT_HIP]) and visible(lm[L.RIGHT_HIP]):
        shoulder_mid = (xy(lm, L.LEFT_SHOULDER) + xy(lm, L.RIGHT_SHOULDER)) / 2
        hip_mid = (xy(lm, L.LEFT_HIP) + xy(lm, L.RIGHT_HIP)) / 2
        vec = shoulder_mid - hip_mid
        # angle from vertical (0 = perfectly upright)
        vertical = np.array([0.0, -1.0])
        denom = np.linalg.norm(vec)
        if denom > 1e-6:
            cosang = np.clip(np.dot(vec, vertical) / denom, -1.0, 1.0)
            m["spine_lean_deg"] = float(np.degrees(np.arccos(cosang)))
        else:
            m["spine_lean_deg"] = None
        m["hip_mid"] = hip_mid
        m["shoulder_mid"] = shoulder_mid
    else:
        m["spine_lean_deg"] = None
        m["hip_mid"] = None
        m["shoulder_mid"] = None

    # Head (nose) position — used later for head-over-front-knee alignment
    m["nose"] = xy(lm, L.NOSE) if visible(lm[L.NOSE]) else None

    # Wrist heights (used for backlift / downswing / follow-through phase detection)
    m["left_wrist"] = xy(lm, L.LEFT_WRIST) if visible(lm[L.LEFT_WRIST]) else None
    m["right_wrist"] = xy(lm, L.RIGHT_WRIST) if visible(lm[L.RIGHT_WRIST]) else None

    # Ankles / knees / hips (positions, for front-leg / weight-transfer detection)
    for name, idx in [("left_ankle", L.LEFT_ANKLE), ("right_ankle", L.RIGHT_ANKLE),
                       ("left_knee", L.LEFT_KNEE), ("right_knee", L.RIGHT_KNEE),
                       ("left_hip", L.LEFT_HIP), ("right_hip", L.RIGHT_HIP)]:
        m[name] = xy(lm, idx) if visible(lm[idx]) else None

    return m
