"""Identify one object in the source bin and estimate its grasp pose from one RGB-D frame.

Uses camera calibration, the known bin layout and a table of known object dimensions; never reads
simulator object state. Replaces milestone 2's colour threshold (perception.py, kept for milestones
2-3) with depth segmentation, so it works for any packaging.

Method:
1. Back-project every pixel to a world point.
2. Segment: points inside the source bin's inner walls and above its floor, largest connected region in
   the image. Restricting to the bin interior also drops the wall-edge pixels milestone 2 had to reject.
3. Identify the object from its measured height and footprint (OBJECTS table).
4. Estimate the pose from the top surface: the points within TOP_MARGIN of the highest point.
   Centre = centroid of top points after median-based outlier rejection.
   Orientation, only for objects whose grasp depends on it:
   - cereal box: the top face is a 30 x 100 mm rectangle; fingers close across its short side.
   - milk carton: the top is a ridge (gable roof) parallel to two faces; fingers close along the ridge,
     i.e. across the two faces it runs between.
   Can and bread need no orientation: round, or small enough to fit the gripper (80 mm) at any rotation.
   (The grasp planner may still rotate the gripper for bread, to keep the hand clear of the bin walls.)
"""
from dataclasses import dataclass, field
import time

import cv2
import numpy as np

# Known scene layout (world frame, metres): source bin interior with a 1 cm margin from the walls.
BIN_X = (-0.08, 0.28)
BIN_Y = (-0.48, -0.02)
BIN_FLOOR_Z = 0.82
MIN_ABOVE_FLOOR = 0.005
MAX_ABOVE_FLOOR = 0.20  # excludes the arm, parked above the bin

# Known object dimensions (collision meshes): height and footprint (short, long), metres.
OBJECTS = {
    "bread": dict(height=0.048, footprint=(0.040, 0.048), align="free"),
    "can": dict(height=0.080, footprint=(0.050, 0.050), align=None),
    "milk": dict(height=0.144, footprint=(0.040, 0.040), align="faces"),
    "cereal": dict(height=0.150, footprint=(0.030, 0.100), align="short_side"),
}
TOP_MARGIN = 0.004
HEIGHT_TOL = 0.012  # measured height must match the identified object's within this
MIN_POINTS = 150
MIN_TOP_POINTS = 10


@dataclass
class ObjectEstimate:
    label: str | None = None
    center_xy: np.ndarray | None = None  # world frame, metres
    top_z: float | None = None
    height: float | None = None  # known height of the identified object
    closing_dir: np.ndarray | None = None  # horizontal direction the fingers should close along
    measured_height: float | None = None
    measured_footprint: tuple | None = None  # (short, long) of all object points, metres
    valid: bool = False
    reason: str = ""
    mask: np.ndarray | None = field(default=None, repr=False)
    latency_s: float = 0.0


def classify(height, long_side):
    if height < 0.064:
        return "bread"
    if height < 0.112:
        return "can"
    return "cereal" if long_side > 0.07 else "milk"


def backproject_frame(depth_normalized, camera):
    """World point for every pixel, shape (H, W, 3)."""
    H, W = camera.height, camera.width
    v, u = np.mgrid[0:H, 0:W]
    Z = camera.metric_depth(depth_normalized)
    return camera.backproject(np.stack([u.ravel(), v.ravel()], axis=1) + 0.5, Z.ravel()).reshape(H, W, 3)


def above_bin_floor(pts):
    """Pixels whose points lie inside the source bin's walls and above its floor."""
    x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
    return ((x > BIN_X[0]) & (x < BIN_X[1]) & (y > BIN_Y[0]) & (y < BIN_Y[1])
            & (z > BIN_FLOOR_Z + MIN_ABOVE_FLOOR) & (z < BIN_FLOOR_Z + MAX_ABOVE_FLOOR))


def fit_region(est, P, label=None):
    """Identify (unless `label` is given) and fit the grasp pose of one object from its points P (N, 3).
    Fills `est`; returns the failure reason, or None if the estimate is valid."""
    est.label = label  # the network's label, if given, even when the fit below fails
    if len(P) < MIN_POINTS:
        return f"object region too small ({len(P)} px)"
    z_top = np.percentile(P[:, 2], 99)
    (_, _), (w, h), _ = cv2.minAreaRect((P[:, :2] * 1000).astype(np.float32))
    est.measured_height = z_top - BIN_FLOOR_Z
    est.measured_footprint = (min(w, h) / 1000, max(w, h) / 1000)
    label = label or classify(est.measured_height, est.measured_footprint[1])
    spec = OBJECTS[label]
    est.label, est.top_z, est.height = label, z_top, spec["height"]
    if abs(est.measured_height - spec["height"]) > HEIGHT_TOL:
        return f"height {est.measured_height * 1000:.0f} mm does not match {label}"

    top = P[P[:, 2] > z_top - TOP_MARGIN]
    if len(top) < MIN_TOP_POINTS:
        return f"too few top points ({len(top)})"
    radius = 0.5 * np.hypot(*spec["footprint"]) + 0.004
    median = np.median(top[:, :2], axis=0)
    top = top[np.linalg.norm(top[:, :2] - median, axis=1) < radius]
    est.center_xy = top[:, :2].mean(axis=0)

    if spec["align"] == "short_side":
        (_, _), (w, h), ang = cv2.minAreaRect((top[:, :2] * 1000).astype(np.float32))
        a = np.radians(ang)
        along_w = np.array([np.cos(a), np.sin(a)])
        est.closing_dir = along_w if w <= h else np.array([-along_w[1], along_w[0]])
    elif spec["align"] == "faces":
        centred = top[:, :2] - top[:, :2].mean(axis=0)
        est.closing_dir = np.linalg.svd(centred, full_matrices=False)[2][0]  # principal (ridge) direction
    return None


def estimate_object(rgb, depth_normalized, camera):
    """Milestone 4: the one object in the bin. rgb (H, W, 3) and depth (H, W[, 1]) in image orientation;
    camera is a geometry.CameraModel. rgb is unused: segmentation and identification are geometric."""
    t0 = time.perf_counter()
    est = ObjectEstimate()

    def done(reason, valid=False):
        est.reason, est.valid, est.latency_s = reason, valid, time.perf_counter() - t0
        return est

    pts = backproject_frame(depth_normalized, camera)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(above_bin_floor(pts).astype(np.uint8), connectivity=8)
    if n <= 1:
        return done("nothing above the bin floor")
    est.mask = labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    reason = fit_region(est, pts[est.mask])
    return done(reason or "ok", valid=reason is None)


def estimate_objects(rgb, depth_normalized, camera, masks=None, mask_labels=None):
    """Milestone 5: every object in the bin. Without `masks`, segmentation is geometric: each connected
    region of points above the bin floor is taken to be one object, so touching objects merge into one
    region. With `masks` (boolean (H, W) arrays, e.g. from a learned instance segmenter) and optional
    `mask_labels`, each mask's pixels are used instead, still intersected with the above-floor region
    so that mask pixels on the bin floor or walls do not enter the fit."""
    t0 = time.perf_counter()
    pts = backproject_frame(depth_normalized, camera)
    above = above_bin_floor(pts)
    if masks is None:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(above.astype(np.uint8), connectivity=8)
        masks = [labels == i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= MIN_POINTS]
        mask_labels = [None] * len(masks)
    elif mask_labels is None:
        mask_labels = [None] * len(masks)
    estimates = []
    for mask, label in zip(masks, mask_labels):
        est = ObjectEstimate(mask=mask)
        reason = fit_region(est, pts[mask & above], label)
        est.reason, est.valid = reason or "ok", reason is None
        estimates.append(est)
    latency = time.perf_counter() - t0
    for est in estimates:
        est.latency_s = latency
    return estimates
