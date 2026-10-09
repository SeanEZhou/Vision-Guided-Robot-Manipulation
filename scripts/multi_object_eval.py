"""Milestone 4: pick-and-place of four object types with random position and rotation, comparing
grasp targets from simulator truth (reference) and from perception, on the same placements.

Each trial: robosuite's PickPlace places one object (milk, bread, cereal or can) at a random position
and rotation about the vertical axis in the source bin. The grasp target (position, top height, size,
and the direction of the object's narrow side) comes from simulator truth ("truth") or from one RGB-D
frame ("vision"); in the vision configuration the object's identity, which selects its target
compartment, also comes from perception. Simulator object state is read only by this evaluator.

Success: as in milestone 3 (scripts/pick_place_eval.py): robosuite's task check holds, the gripper is
open, and the object moves less than STABLE_TOL over 0.5 s, within a 30 s simulation budget.

Rotation sweep: config "yawN" uses the true target with the finger-closing direction rotated N degrees
(random sign per trial); "noalign" keeps the gripper's start orientation, as milestone 3 did. Both only
change objects whose grasp depends on rotation (cereal box, milk carton).

Seeds 2000-2003 were used during development; the default 3000+ are held out.

Usage: python scripts/multi_object_eval.py [--trials 50] [--seed 3000] [--out results/step4]
       python scripts/multi_object_eval.py --objects cereal --configs yaw10 yaw20 yaw30 yaw45 noalign \
           --out results/step4_yaw_sweep
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

if os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "glx")  # see scripts/pick_place_scripted.py

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import imageio
import numpy as np
import robosuite as suite

from geometry import CameraModel
from object_perception import OBJECTS as SPECS, estimate_object
from pick_place_eval import CONTROL_FREQ, SETTLE_STEPS, STABLE_STEPS, STABLE_TOL, TIME_BUDGET_STEPS, LIFT_MIN, \
    spread_out, wilson
from policy import OPEN, grasp_candidates, make_object_phases, move_action, run_phases

OBJECTS = ["milk", "bread", "cereal", "can"]
CAMERA = "agentview"
VIDEO_CAMERAS = ["frontview", "agentview"]
H, W = 480, 640


def make_env(obj, seed):
    return suite.make(
        "PickPlace", robots="Panda", single_object_mode=2, object_type=obj, has_renderer=False,
        has_offscreen_renderer=True, use_camera_obs=False, use_object_obs=False, control_freq=CONTROL_FREQ,
        ignore_done=True, hard_reset=False, seed=seed,
    )


def object_truth(env, name=None):
    """True geometry of an object (default: the active one) from its collision mesh (evaluator only):
    centre, top and bottom height, footprint (short, long), the horizontal direction of its short side,
    and tilt."""
    m, d = env.sim.model, env.sim.data
    body = env.obj_body_id[name or env.objects[env.object_id].name]
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    corners, axes = [], []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != body or not m.geom_contype[g]:
            continue
        R = d.geom_xmat[g].reshape(3, 3)
        c, h = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
        corners.append((signs * h + c) @ R.T + d.geom_xpos[g])
        axes.append((R, h))
    corners = np.concatenate(corners)
    R, h = axes[0]
    vertical = int(np.argmax(np.abs(R[2])))
    horiz = [i for i in range(3) if i != vertical]
    short = min(horiz, key=lambda i: h[i])
    long_ = max(horiz, key=lambda i: h[i])
    short_dir = R[:2, short] / np.linalg.norm(R[:2, short])
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    face_dir = R[:2, horiz[0]] / np.linalg.norm(R[:2, horiz[0]])
    return dict(center=(lo + hi) / 2, top=hi[2], bottom=lo[2], height=hi[2] - lo[2],
                footprint=(2 * h[short], 2 * h[long_]), short_dir=short_dir, face_dir=face_dir,
                tilt_deg=np.degrees(np.arccos(min(1.0, abs(R[2, vertical])))))


def video_frame(env, label):
    views = [env.sim.render(width=W // 2, height=H // 2, camera_name=cam)[::-1] for cam in VIDEO_CAMERAS]
    frame = np.ascontiguousarray(np.concatenate(views, axis=1))
    cv2.putText(frame, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    return frame


def run_trial(env, obj, config, trial, record=False):
    obs = env.reset()
    for _ in range(SETTLE_STEPS):
        obs, *_ = env.step(np.r_[np.zeros(6), OPEN])
    truth = object_truth(env)
    row = dict(object=obj, config=config, trial=trial, true_x=truth["center"][0], true_y=truth["center"][1],
               true_z=truth["center"][2], true_top=truth["top"], start_tilt_deg=truth["tilt_deg"])

    # Grasp alignment is set per object type, identically for both configurations.
    align = SPECS[obj]["align"]
    true_closing = {"short_side": truth["short_dir"], "faces": truth["face_dir"]}.get(align)
    camera = CameraModel.from_sim(env.sim, CAMERA, H, W)
    rgb, depth = env.sim.render(width=W, height=H, camera_name=CAMERA, depth=True)
    if trial == 0:
        estimate_object(rgb[::-1], depth[::-1], camera)  # warm-up, excluded from timing
    est = estimate_object(rgb[::-1], depth[::-1], camera)
    row.update(perceived=est.label, label_correct=est.label == obj, perception_valid=est.valid,
               perception_reason=est.reason, latency_ms=est.latency_s * 1000)
    if est.center_xy is not None:
        row.update(err_xy_mm=np.linalg.norm(est.center_xy - truth["center"][:2]) * 1000,
                   err_top_mm=(est.top_z - truth["top"]) * 1000)
    if est.closing_dir is not None and true_closing is not None:
        period = np.pi / 2 if align == "faces" else np.pi  # square carton vs. rectangular box
        diff = np.arctan2(est.closing_dir[1], est.closing_dir[0]) - np.arctan2(true_closing[1], true_closing[0])
        row["err_yaw_deg"] = abs(np.degrees((diff + period / 2) % period - period / 2))

    if config == "truth":
        target = dict(xy=truth["center"][:2], top=truth["top"], height=truth["height"],
                      closing=grasp_candidates(align, true_closing), label=obj)
    elif config == "noalign":
        target = dict(xy=truth["center"][:2], top=truth["top"], height=truth["height"], closing=None, label=obj)
    elif config.startswith("yaw"):
        # Controlled rotation error: the only allowed closing direction is the true one rotated N degrees.
        a = np.radians(float(config[3:])) * np.random.default_rng(trial).choice([-1, 1])
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        closing = None if true_closing is None else [rot @ true_closing]
        target = dict(xy=truth["center"][:2], top=truth["top"], height=truth["height"], closing=closing, label=obj)
    elif est.valid:
        target = dict(xy=est.center_xy, top=est.top_z, height=est.height,
                      closing=grasp_candidates(SPECS[est.label]["align"], est.closing_dir), label=est.label)
    else:
        row.update(outcome="perception_rejected", success=False, robosuite_success=False, sim_time_s=0.0)
        return row, []
    place_xy = env.target_bin_placements[env.object_to_id[target["label"]]][:2]
    phases = make_object_phases(target["xy"], target["top"], target["height"], target["closing"], place_xy)

    names = [p[0] for p in phases] + ["settle"]
    phase_name = [names[0]]
    frames = [video_frame(env, f"{config} | {obj} | trial {trial} | start")] if record else []
    after = {}
    body = env.obj_body_id[env.objects[env.object_id].name]
    obj_pos = lambda: env.sim.data.body_xpos[body].copy()
    z0 = obj_pos()[2]

    def on_step(o):
        if record:
            frames.append(video_frame(env, f"{config} | {obj} | trial {trial} | {phase_name[0]}"))

    def on_phase_end(entry, o):
        after[entry["phase"]] = obj_pos()
        phase_name[0] = names[names.index(entry["phase"]) + 1]

    t0 = time.perf_counter()
    obs, log = run_phases(env, obs, phases, on_step, on_phase_end)
    positions, checks = [], []
    for _ in range(STABLE_STEPS):
        obs, *_ = env.step(move_action(obs["robot0_eef_pos"], phases[-1][1], OPEN, obs["robot0_eef_quat"],
                                       phases[-1][5]))
        on_step(obs)
        positions.append(obj_pos())
        checks.append(bool(env._check_success()))
    wall = time.perf_counter() - t0

    steps = sum(e["steps"] for e in log) + STABLE_STEPS
    drift = np.linalg.norm(np.ptp(np.array(positions), axis=0))
    final = object_truth(env)
    lifted = after["lift"][2] - z0 > LIFT_MIN
    held_to_release = after["lower"][2] - z0 > LIFT_MIN
    success = all(checks) and drift < STABLE_TOL and steps <= TIME_BUDGET_STEPS
    if success:
        outcome = "success"
    elif not lifted:
        outcome = "missed_grasp"
    elif not held_to_release:
        outcome = "dropped_in_transit"
    elif steps > TIME_BUDGET_STEPS:
        outcome = "timeout"
    elif not checks[-1]:
        outcome = "misplaced"
    else:
        outcome = "unstable"
    row.update(outcome=outcome, success=success, robosuite_success=checks[-1], sim_time_s=steps / CONTROL_FREQ,
               wall_time_s=wall, phase_timeouts=";".join(e["phase"] for e in log if e["timed_out"]),
               final_x=final["center"][0], final_y=final["center"][1], final_z=final["center"][2],
               final_tilt_deg=final["tilt_deg"], stable_drift_mm=drift * 1000)
    return row, frames


def save_cases(obj, seed, rows, out, n_success=1, n_failure=3):
    """Replay chosen vision trials with video (deterministic given the seed and the number of resets),
    asserting each replay reproduces the logged outcome and final position."""
    picks = (spread_out([r for r in rows if r["success"]], n_success)
             + [r["trial"] for r in rows if not r["success"]][:n_failure])
    if not picks:
        return
    by_trial = {r["trial"]: r for r in rows}
    env = make_env(obj, seed)
    try:
        for trial in range(max(picks) + 1):
            if trial not in picks:
                env.reset()
                continue
            row, frames = run_trial(env, obj, "vision", trial, record=True)
            ref = by_trial[trial]
            assert row["outcome"] == ref["outcome"] and ("final_x" not in ref or np.allclose(
                [row[k] for k in ("final_x", "final_y", "final_z")],
                [ref[k] for k in ("final_x", "final_y", "final_z")], atol=1e-6)), f"replay of {obj} {trial} diverged"
            stem = f"vision_{obj}_trial{trial:03d}_{row['outcome']}"
            if frames:
                imageio.mimsave(out / "cases" / f"{stem}.mp4", frames, fps=CONTROL_FREQ, macro_block_size=1)
            print(f"saved {stem} (replay matched)", flush=True)
    finally:
        env.close()


def stats(values):
    a = np.array([v for v in values if v is not None and not np.isnan(v)])
    return dict(n=len(a), median=float(np.median(a)), p95=float(np.percentile(a, 95)), max=float(a.max())) if len(a) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50, help="trials per object and configuration")
    parser.add_argument("--seed", type=int, default=3000, help="object k uses seed + k")
    parser.add_argument("--objects", nargs="+", default=OBJECTS, choices=OBJECTS)
    parser.add_argument("--configs", nargs="+", default=["truth", "vision"],
                        help="truth, vision, noalign, or yawN (true target, closing direction rotated N degrees)")
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument("--out", default="results/step4")
    args = parser.parse_args()
    for c in args.configs:
        if c not in ("truth", "vision", "noalign") and not (c.startswith("yaw") and c[3:].isdigit()):
            parser.error(f"unknown config {c}")
    out = Path(args.out)
    (out / "cases").mkdir(parents=True, exist_ok=True)

    rows = []
    for k, obj in enumerate(args.objects):
        seed = args.seed + k
        for config in args.configs:
            env = make_env(obj, seed)
            cfg_rows = []
            try:
                for trial in range(args.trials):
                    row, _ = run_trial(env, obj, config, trial)
                    cfg_rows.append(row)
                    err = row.get("err_xy_mm", float("nan"))
                    print(f"{obj:6s} {config:7s} trial {trial:3d}  {row['outcome']:20s} seen as {row['perceived']!s:6s} "
                          f"xy err {err:4.1f} mm  timeouts [{row.get('phase_timeouts', '')}]", flush=True)
            finally:
                env.close()
            rows += cfg_rows
            if config == "vision" and not args.no_videos:  # demo videos only for the deployed configuration
                save_cases(obj, seed, cfg_rows, out)

    # All configurations of an object must have seen identical placements.
    for obj in args.objects:
        per_cfg = [[r for r in rows if r["object"] == obj and r["config"] == c] for c in args.configs]
        for other in per_cfg[1:]:
            gap = max(abs(a[key] - b[key]) for a, b in zip(per_cfg[0], other) for key in ("true_x", "true_y", "true_z"))
            assert gap < 1e-6, f"{obj}: configurations saw different placements (max gap {gap})"

    with open(out / "trials.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)

    summary = dict(seed=args.seed, trials_per_object=args.trials)
    for config in args.configs:
        for obj in args.objects + ["all"]:
            rs = [r for r in rows if r["config"] == config and obj in (r["object"], "all")]
            k = int(sum(r["success"] for r in rs))
            outcomes = {}
            for r in rs:
                outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
            entry = dict(success=k, attempted=len(rs), success_rate=k / len(rs), ci95=wilson(k, len(rs)),
                         outcomes=outcomes, sim_time_s=stats([r.get("sim_time_s") for r in rs if r["outcome"] != "perception_rejected"]))
            if config == "vision":
                entry.update(label_accuracy=sum(r["label_correct"] for r in rs) / len(rs),
                             err_xy_mm=stats([r.get("err_xy_mm") for r in rs]),
                             err_top_mm_abs=stats([abs(r["err_top_mm"]) for r in rs if "err_top_mm" in r]),
                             err_yaw_deg=stats([r.get("err_yaw_deg") for r in rs]),
                             latency_ms=stats([r["latency_ms"] for r in rs]))
            summary[f"{config}/{obj}"] = entry
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    for key, e in summary.items():
        if isinstance(e, dict):
            extra = f"  label acc {e['label_accuracy']:.2f}" if "label_accuracy" in e else ""
            print(f"{key:16s} {e['success']:3d}/{e['attempted']:3d}  {e['outcomes']}{extra}")


if __name__ == "__main__":
    main()
