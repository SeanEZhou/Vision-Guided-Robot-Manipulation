"""Locate the can from one RGB-D frame. Uses camera calibration and known object geometry only;
never reads simulator object state.

Method:
1. Segment: HSV threshold for the can's red body, keep the largest connected component.
2. Back-project every mask pixel to a world point using metric depth.
3. Height: the camera looks down on the can, so its lid is always visible while the bottom can be
   hidden by the bin wall. Take the top from the highest points and subtract the known half-height.
4. Axis position: the lid is a horizontal disk centred on the can axis, so the xy centroid of the
   lid points is the axis. From this camera the lid is most of the visible can; the side is a thin
   strip partly covered by a grey logo, so a cylinder fit to side points was unreliable.

Edge pixels: the colour image blends can and background along the can's outline, but each depth
pixel holds a single surface. Edge pixels red enough to pass the threshold therefore back-project to
the background. Usually that is the bin floor, which fails the lid height test. The exception is the
source bin's front wall: when it hides the can's base, its top edge is at lid height, so those
points pass the height test and are removed by the outlier step instead.
"""
from dataclasses import dataclass, field
import time

import cv2
import numpy as np

# Known object geometry (robosuite CanObject collision mesh). The mesh is centred on the body
# origin, which is the point we estimate.
CAN_RADIUS = 0.025
CAN_HALF_HEIGHT = 0.040
LID_MARGIN = 0.004  # points within this height of the top are treated as the lid

MIN_MASK_PIXELS = 80
MIN_LID_PIXELS = 40
MAX_LID_SPREAD = CAN_RADIUS + 0.004  # lid points farther than this from the median are outliers


@dataclass
class GraspEstimate:
    position: np.ndarray | None  # world frame, metres; None when no estimate could be formed
    frame: str = "world"
    valid: bool = False
    reason: str = ""
    mask: np.ndarray | None = field(default=None, repr=False)
    n_pixels: int = 0
    latency_s: float = 0.0


def segment_red(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = ((h < 8) | (h > 172)) & (s > 120) & (v > 60)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(red.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(red)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return labels == largest


def estimate_can_position(rgb, depth_normalized, camera):
    """rgb (H, W, 3) uint8 and depth (H, W[, 1]) in image orientation; camera is a CameraModel."""
    t0 = time.perf_counter()

    def result(position, valid, reason):
        return GraspEstimate(position, valid=valid, reason=reason, mask=mask, n_pixels=n,
                             latency_s=time.perf_counter() - t0)

    mask = segment_red(rgb)
    n = int(mask.sum())
    if n < MIN_MASK_PIXELS:
        return result(None, False, f"mask too small ({n} px)")

    v, u = np.nonzero(mask)
    Z = camera.metric_depth(depth_normalized)[v, u]
    pts = camera.backproject(np.stack([u, v], axis=1) + 0.5, Z)  # +0.5: pixel centres

    z_top = np.percentile(pts[:, 2], 99)
    lid = pts[pts[:, 2] > z_top - LID_MARGIN]
    if len(lid) < MIN_LID_PIXELS:
        return result(None, False, f"too few lid points ({len(lid)})")

    # Where the source bin's front wall hides the can's base, edge pixels along that boundary are red
    # enough to pass the mask but take their depth from the wall's top edge, which is at lid height.
    # Keep only points within one can radius of the median, which those few outliers cannot drag.
    median_xy = np.median(lid[:, :2], axis=0)
    inliers = lid[np.linalg.norm(lid[:, :2] - median_xy, axis=1) < MAX_LID_SPREAD]
    if len(inliers) < MIN_LID_PIXELS:
        return result(None, False, f"too few lid inliers ({len(inliers)} of {len(lid)})")

    center_xy = inliers[:, :2].mean(axis=0)
    return result(np.array([center_xy[0], center_xy[1], z_top - CAN_HALF_HEIGHT]), True, "ok")
