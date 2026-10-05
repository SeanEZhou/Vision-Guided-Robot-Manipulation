"""Milestone 2: locate the can from one RGB-D frame and compare against simulator truth.

For each trial: robosuite places the can at a random pose in the source bin, the arm stays still,
and perception estimates the can's body position from the agentview camera. Only this evaluator
reads the true pose; perception sees RGB, depth and camera calibration.

Usage: python scripts/localize_eval.py [--trials 100] [--seed 0] [--out results/step2]
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

if os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "glx")  # see scripts/pick_place_scripted.py

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import robosuite as suite

from geometry import CameraModel
from perception import estimate_can_position

CAMERA = "agentview"
H, W = 480, 640
SETTLE_STEPS = 10
GRASP_TOLERANCE_XY = 0.015  # rough lateral slack for the Panda gripper on a 5 cm can


def check_round_trip(camera, rng):
    """Project random world points in front of the camera and back-project them."""
    p_cam = np.c_[rng.uniform(-0.3, 0.3, (100, 2)), rng.uniform(0.5, 2.0, 100)]
    p_world = p_cam @ camera.T_world_cam[:3, :3].T + camera.T_world_cam[:3, 3]
    uv, Z = camera.project(p_world)
    return np.abs(camera.backproject(uv, Z) - p_world).max()


def overlay(rgb, est, true_pos, camera):
    img = rgb.copy()
    if est.mask is not None:
        img[est.mask] = (0.5 * img[est.mask] + [0, 127, 0]).astype(np.uint8)
    (gu, gv), = camera.project(true_pos[None])[0]
    cv2.drawMarker(img, (round(gu), round(gv)), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
    if est.position is not None:
        (eu, ev), = camera.project(est.position[None])[0]
        cv2.drawMarker(img, (round(eu), round(ev)), (255, 255, 0), cv2.MARKER_TILTED_CROSS, 14, 2)
    return img


def crop(img, center_uv, half=70, scale=3):
    u, v = (int(round(c)) for c in center_uv)
    c = img[max(0, v - half):v + half, max(0, u - half):u + half]
    return cv2.resize(c, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/step2")
    args = parser.parse_args()
    out = Path(args.out)
    (out / "overlays").mkdir(parents=True, exist_ok=True)

    env = suite.make(
        "PickPlaceCan", robots="Panda", has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=True, use_object_obs=False, camera_names=CAMERA, camera_heights=H,
        camera_widths=W, camera_depths=True, control_freq=20, ignore_done=True, hard_reset=False, seed=args.seed,
    )
    rng = np.random.default_rng(args.seed)
    can_body = env.obj_body_id[env.objects[env.object_id].name]
    hold = np.r_[np.zeros(6), -1.0]

    rows, overlays = [], []
    try:
        for trial in range(args.trials):
            obs = env.reset()
            for _ in range(SETTLE_STEPS):
                obs, *_ = env.step(hold)
            camera = CameraModel.from_sim(env.sim, CAMERA, H, W)
            rgb = obs[f"{CAMERA}_image"][::-1]
            depth = obs[f"{CAMERA}_depth"][::-1]

            if trial == 0:
                estimate_can_position(rgb, depth, camera)  # warm-up, excluded from timing
                rt = check_round_trip(camera, rng)
                print(f"Projection round-trip max error: {rt:.2e} m")
                assert rt < 1e-9, "projection and back-projection disagree"

            est = estimate_can_position(rgb, depth, camera)
            truth = env.sim.data.body_xpos[can_body].copy()  # evaluator only

            row = dict(trial=trial, valid=est.valid, reason=est.reason, n_pixels=est.n_pixels,
                       latency_ms=est.latency_s * 1000, true_x=truth[0], true_y=truth[1], true_z=truth[2])
            if est.position is not None:
                err = est.position - truth
                row.update(est_x=est.position[0], est_y=est.position[1], est_z=est.position[2],
                           err_x_mm=err[0] * 1000, err_y_mm=err[1] * 1000, err_z_mm=err[2] * 1000,
                           err_xy_mm=np.linalg.norm(err[:2]) * 1000, err_3d_mm=np.linalg.norm(err) * 1000)
            rows.append(row)
            overlays.append((trial, overlay(rgb, est, truth, camera), camera.project(truth[None])[0][0]))
            status = f"{row.get('err_xy_mm', float('nan')):5.1f} mm xy" if est.position is not None else "-"
            print(f"trial {trial:3d}  {status}  {est.reason}")
    finally:
        env.close()

    fields = ["trial", "valid", "reason", "n_pixels", "latency_ms", "true_x", "true_y", "true_z", "est_x",
              "est_y", "est_z", "err_x_mm", "err_y_mm", "err_z_mm", "err_xy_mm", "err_3d_mm"]
    with open(out / "trials.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    valid = [r for r in rows if r["valid"]]
    def stats(key):
        a = np.array([r[key] for r in valid])
        return dict(median=float(np.median(a)), p95=float(np.percentile(a, 95)), max=float(np.max(a)))
    summary = dict(
        trials=len(rows), valid=len(valid), camera=CAMERA, resolution=[W, H], seed=args.seed,
        err_xy_mm=stats("err_xy_mm"), err_z_mm_abs=dict(stats("err_z_mm"), median=float(np.median(np.abs([r["err_z_mm"] for r in valid])))),
        err_3d_mm=stats("err_3d_mm"),
        mean_err_mm=dict(x=float(np.mean([r["err_x_mm"] for r in valid])), y=float(np.mean([r["err_y_mm"] for r in valid])),
                         z=float(np.mean([r["err_z_mm"] for r in valid]))),
        latency_ms=stats("latency_ms"),
        within_grasp_tolerance=int(sum(r["err_xy_mm"] < GRASP_TOLERANCE_XY * 1000 for r in valid)),
        invalid_reasons=[(r["trial"], r["reason"]) for r in rows if not r["valid"]],
    )
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save the 3 best, 3 median and 3 worst trials as zoomed overlays for review.
    ranked = sorted(valid, key=lambda r: r["err_xy_mm"])
    picks = ranked[:3] + ranked[len(ranked) // 2 - 1:len(ranked) // 2 + 2] + ranked[-3:]
    by_trial = {t: (img, uv) for t, img, uv in overlays}
    for r in picks:
        img, uv = by_trial[r["trial"]]
        cv2.imwrite(str(out / "overlays" / f"trial{r['trial']:03d}_{r['err_xy_mm']:.1f}mm.png"),
                    cv2.cvtColor(crop(img, uv), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / "overlays" / "full_frame_trial000.png"), cv2.cvtColor(by_trial[0][0], cv2.COLOR_RGB2BGR))

    print(json.dumps({k: v for k, v in summary.items() if k != "invalid_reasons"}, indent=2))
    print(f"Invalid: {summary['invalid_reasons']}")


if __name__ == "__main__":
    main()
