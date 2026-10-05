"""Milestone 3: pick-and-place from a randomly placed can, comparing grasp targets from simulator
truth (reference) and from perception, on the same placements.

Each trial: robosuite places the can at a random pose in the source bin. The grasp target comes
either from simulator truth ("truth") or from one RGB-D frame of the agentview camera ("vision").
The scripted policy (policy.py) then picks the can and places it in its target compartment.
Simulator object state is read only by this evaluator, to score and classify outcomes.

Camera frames are rendered on demand: one RGB-D frame per trial for perception, and small RGB
frames every step for review videos (excluded from the reported wall time).

Success: robosuite's task check holds, the gripper is open, and the can moves less than
STABLE_TOL over STABLE_STEPS (0.5 s) of simulation, all within a 30 s simulation budget.

A sensitivity sweep replaces perception with a controlled error: config "offsetN" shifts the true
position N mm in a random horizontal direction, to measure how much localization error grasping tolerates.

Usage: python scripts/pick_place_eval.py [--trials 100] [--seed 1000] [--out results/step3]
       python scripts/pick_place_eval.py --configs offset5 offset10 offset15 offset20 --out results/step3_sweep
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
from localize_eval import crop, overlay
from perception import estimate_can_position
from policy import OPEN, make_phases, move_action, run_phases

CAMERA = "agentview"
VIDEO_CAMERAS = ["frontview", "agentview"]
H, W = 480, 640
CONTROL_FREQ = 20
SETTLE_STEPS = 10
STABLE_STEPS = 10  # 0.5 s at 20 Hz
STABLE_TOL = 0.005  # metres the can may move during the stability window
TIME_BUDGET_STEPS = 30 * CONTROL_FREQ
LIFT_MIN = 0.05  # can must rise this far during "lift" to count as grasped


def wilson(k, n, z=1.96):
    """95% Wilson score interval for a success rate."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def video_frame(env, label):
    # Rendered on demand at reduced size; flipped from OpenGL's bottom-up rows like camera observations.
    views = [env.sim.render(width=W // 2, height=H // 2, camera_name=cam)[::-1] for cam in VIDEO_CAMERAS]
    frame = np.ascontiguousarray(np.concatenate(views, axis=1))
    cv2.putText(frame, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    return frame


def run_trial(env, config, trial, can_body, place_xy, bin_z, record=False):
    hold = np.r_[np.zeros(6), OPEN]
    obs = env.reset()
    for _ in range(SETTLE_STEPS):
        obs, *_ = env.step(hold)
    truth = env.sim.data.body_xpos[can_body].copy()  # evaluator only
    can_pos = lambda: env.sim.data.body_xpos[can_body].copy()

    row = dict(config=config, trial=trial, true_x=truth[0], true_y=truth[1], true_z=truth[2])
    camera = CameraModel.from_sim(env.sim, CAMERA, H, W)
    rgb, depth = env.sim.render(width=W, height=H, camera_name=CAMERA, depth=True)
    rgb, depth = rgb[::-1], depth[::-1]
    if trial == 0:
        estimate_can_position(rgb, depth, camera)  # warm-up, excluded from timing
    est = estimate_can_position(rgb, depth, camera)
    row.update(perception_valid=est.valid, perception_reason=est.reason, latency_ms=est.latency_s * 1000)
    if est.position is not None:
        row["target_err_xy_mm"] = np.linalg.norm(est.position[:2] - truth[:2]) * 1000
    vis = dict(overlay=overlay(rgb, est, truth, camera), uv=camera.project(truth[None])[0][0])

    if config == "truth":
        target = truth
    elif config.startswith("offset"):
        # Controlled perception error: truth shifted a fixed distance in a per-trial random
        # horizontal direction (the same direction for every offset size).
        angle = np.random.default_rng(trial).uniform(0, 2 * np.pi)
        target = truth + float(config[len("offset"):]) / 1000 * np.array([np.cos(angle), np.sin(angle), 0.0])
    elif est.valid:
        target = est.position
    else:
        row.update(outcome="perception_rejected", success=False, robosuite_success=False, sim_time_s=0.0)
        return row, [], vis
    row.update(target_x=target[0], target_y=target[1], target_z=target[2])

    phases = make_phases(target, place_xy, bin_z)
    names = [p[0] for p in phases] + ["settle"]
    phase_name = [names[0]]  # label each video frame with the phase in progress
    frames = [video_frame(env, f"{config} | trial {trial} | start")] if record else []
    after = {}
    render_s = [0.0]

    def on_step(o):
        if not record:
            return
        t = time.perf_counter()
        frames.append(video_frame(env, f"{config} | trial {trial} | {phase_name[0]}"))
        render_s[0] += time.perf_counter() - t

    def on_phase_end(entry, o):
        after[entry["phase"]] = can_pos()
        phase_name[0] = names[names.index(entry["phase"]) + 1]

    t0 = time.perf_counter()
    obs, log = run_phases(env, obs, phases, on_step, on_phase_end)

    # Stability window: gripper open and still, the can must stay put and in the correct bin.
    positions, checks = [], []
    for _ in range(STABLE_STEPS):
        obs, *_ = env.step(move_action(obs["robot0_eef_pos"], phases[-1][1], OPEN))
        on_step(obs)
        positions.append(can_pos())
        checks.append(bool(env._check_success()))
    wall = time.perf_counter() - t0 - render_s[0]  # excludes review-video rendering

    steps = sum(e["steps"] for e in log) + STABLE_STEPS
    drift = np.linalg.norm(np.ptp(np.array(positions), axis=0))
    final = positions[-1]
    lifted = after["lift"][2] - truth[2] > LIFT_MIN
    held_to_release = after["lower"][2] - truth[2] > LIFT_MIN
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
               final_x=final[0], final_y=final[1], final_z=final[2], stable_drift_mm=drift * 1000)
    return row, frames, vis


def make_env(seed):
    return suite.make(
        "PickPlaceCan", robots="Panda", has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=False, use_object_obs=False, control_freq=CONTROL_FREQ, ignore_done=True,
        hard_reset=False, seed=seed,
    )


def task_info(env):
    can_body = env.obj_body_id[env.objects[env.object_id].name]
    return can_body, env.target_bin_placements[env.object_id][:2], env.bin2_pos[2]


def spread_out(rows, k):
    """Greedy farthest-point choice of k trials, so demo videos show the can at distinct positions."""
    if not rows:
        return []
    xy = np.array([[r["true_x"], r["true_y"]] for r in rows])
    chosen = [int(np.argmax(np.linalg.norm(xy - xy.mean(axis=0), axis=1)))]
    while len(chosen) < min(k, len(rows)):
        dist = np.min(np.linalg.norm(xy[:, None] - xy[chosen][None], axis=2), axis=1)
        chosen.append(int(np.argmax(dist)))
    return [rows[i]["trial"] for i in chosen]


def save_cases(config, rows, args, out):
    """Replay the selected trials with video: successes spread across the bin, plus the first
    failures. The simulation is deterministic given the seed and the number of resets, so a fresh env
    reset the same number of times reproduces each trial exactly."""
    picks = (spread_out([r for r in rows if r["success"]], args.save_success)
             + [r["trial"] for r in rows if not r["success"]][:args.save_failure])
    if not picks:
        return
    by_trial = {r["trial"]: r for r in rows}
    env = make_env(args.seed)
    info = task_info(env)
    try:
        for trial in range(max(picks) + 1):
            if trial not in picks:
                env.reset()
                continue
            row, frames, vis = run_trial(env, config, trial, *info, record=True)
            ref = by_trial[trial]
            same = row["outcome"] == ref["outcome"] and (
                "final_x" not in ref or np.allclose([row[k] for k in ("final_x", "final_y", "final_z")],
                                                    [ref[k] for k in ("final_x", "final_y", "final_z")], atol=1e-6))
            assert same, f"replay of {config} trial {trial} diverged: {row['outcome']} vs {ref['outcome']}"
            stem = f"{config}_trial{trial:03d}_{row['outcome']}"
            if frames:
                imageio.mimsave(out / "cases" / f"{stem}.mp4", frames, fps=CONTROL_FREQ, macro_block_size=1)
            cv2.imwrite(str(out / "cases" / f"{stem}_perception.png"),
                        cv2.cvtColor(crop(vis["overlay"], vis["uv"]), cv2.COLOR_RGB2BGR))
            print(f"saved {stem} (replay matched)", flush=True)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--configs", nargs="+", default=["truth", "vision"],
                        help="truth, vision, or offsetN: truth shifted N mm horizontally (sensitivity sweep)")
    parser.add_argument("--save-success", type=int, default=3,
                        help="vision success videos to keep, chosen to show distinct can positions")
    parser.add_argument("--save-failure", type=int, default=3, help="vision failure videos to keep, if any")
    parser.add_argument("--out", default="results/step3")
    args = parser.parse_args()
    for c in args.configs:
        if c not in ("truth", "vision") and not (c.startswith("offset") and c[len("offset"):].isdigit()):
            parser.error(f"unknown config {c}")
    out = Path(args.out)
    (out / "cases").mkdir(parents=True, exist_ok=True)

    rows = []
    for config in args.configs:
        env = make_env(args.seed)
        info = task_info(env)
        cfg_rows = []
        try:
            for trial in range(args.trials):
                row, _, _ = run_trial(env, config, trial, *info)
                cfg_rows.append(row)
                err = row.get("target_err_xy_mm", float("nan"))
                print(f"{config:6s} trial {trial:3d}  {row['outcome']:20s}  target err {err:4.1f} mm  "
                      f"sim {row['sim_time_s']:5.1f} s", flush=True)
        finally:
            env.close()
        rows += cfg_rows
        if config == "vision":  # videos only for the deployed configuration
            save_cases(config, cfg_rows, args, out)

    fields = ["config", "trial", "outcome", "success", "robosuite_success", "perception_valid", "perception_reason",
              "latency_ms", "target_err_xy_mm", "true_x", "true_y", "true_z", "target_x", "target_y", "target_z",
              "final_x", "final_y", "final_z", "stable_drift_mm", "phase_timeouts", "sim_time_s", "wall_time_s"]
    with open(out / "trials.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # Both configs must have seen identical placements for the comparison to be paired.
    by_cfg = {c: [r for r in rows if r["config"] == c] for c in args.configs}
    first, *others = by_cfg.values()
    for other in others:
        gap = max(np.linalg.norm([ra[k] - rb[k] for k in ("true_x", "true_y", "true_z")]) for ra, rb in zip(first, other))
        assert gap < 1e-6, f"configs saw different placements (max gap {gap})"

    summary = dict(seed=args.seed, trials=args.trials)
    for config, rs in by_cfg.items():
        k, n = sum(r["success"] for r in rs), len(rs)
        ran = [r for r in rs if r["outcome"] != "perception_rejected"]
        outcomes = {}
        for r in rs:
            outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
        def stat(key, src):
            a = np.array([r[key] for r in src if key in r])
            return dict(median=float(np.median(a)), p95=float(np.percentile(a, 95)), max=float(np.max(a))) if len(a) else None
        summary[config] = dict(
            success=int(k), attempted=n, success_rate=k / n, success_rate_ci95=wilson(k, n), outcomes=outcomes,
            robosuite_check_agrees=int(sum(r["robosuite_success"] == r["success"] for r in ran)),
            sim_time_s=stat("sim_time_s", ran), wall_time_s=stat("wall_time_s", ran),
            perception_latency_ms=stat("latency_ms", rs), target_err_xy_mm=stat("target_err_xy_mm", rs),
        )
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
