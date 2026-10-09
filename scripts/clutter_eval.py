"""Milestone 5: sort a cluttered bin. All four objects (milk, bread, cereal, can) start together in
the source bin at random positions and rotations; the robot picks them one at a time into their own
compartments, re-observing the scene before every pick.

Configurations (grasp targets and identities for each pick):
  truth      simulator truth (reference: isolates motion and grasp failures)
  geometric  milestone 4 perception generalized to several objects: each connected region of points
             above the bin floor is one object, identified and fitted by shape
  learned    Mask R-CNN instance masks and classes (segmenter.py) replace geometric segmentation and
             shape-based identification; the same back-projection and pose fitting follow
Simulator object state and the rendered ground-truth object masks are read only by this evaluator
(and by the truth configuration's targets). Each object gets one attempt per scene; retries are a
later milestone.

Scoring, per object at the end of the scene: "sorted" if it rests in its own compartment and moves
less than STABLE_TOL over 0.5 s. Otherwise: still in the source bin, in a wrong compartment, or
elsewhere (dropped). A scene succeeds when all four objects are sorted.

Usage: python scripts/clutter_eval.py [--scenes 50] [--seed 5000] [--out results/step5]
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
from scipy.spatial.transform import Rotation

from geometry import CameraModel
from multi_object_eval import object_truth, video_frame
from object_perception import OBJECTS as SPECS, above_bin_floor, backproject_frame, estimate_objects
from pick_place_eval import CONTROL_FREQ, SETTLE_STEPS, STABLE_STEPS, STABLE_TOL, spread_out, wilson
from policy import OPEN, SOURCE_BIN_INNER_X, SOURCE_BIN_INNER_Y, grasp_candidates, make_object_phases, \
    move_action, run_phases

LABELS = ["milk", "bread", "cereal", "can"]
NAMES = {"milk": "Milk", "bread": "Bread", "cereal": "Cereal", "can": "Can"}
MAX_PICKS = 4
CAMERA, H, W = "agentview", 480, 640
MERGE_FRACTION = 0.3  # a predicted mask holding this share of two objects' visible pixels merged them
TIME_BUDGET_STEPS = MAX_PICKS * 30 * CONTROL_FREQ


def make_env(seed):
    return suite.make(
        "PickPlace", robots="Panda", single_object_mode=0, has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=False, use_object_obs=False, control_freq=CONTROL_FREQ, ignore_done=True,
        hard_reset=False, seed=seed,
    )


def in_source_bin(pos):
    return (SOURCE_BIN_INNER_X[0] < pos[0] < SOURCE_BIN_INNER_X[1]
            and SOURCE_BIN_INNER_Y[0] < pos[1] < SOURCE_BIN_INNER_Y[1] and pos[2] < 1.0)


ROBOT_BASE_XY = np.array([-0.5, -0.1])
PLACE_MARGIN = 0.035  # clearance from a released object's footprint to its compartment's walls; covers the
                      # closed fingers, which stick out beyond the held object and otherwise hit the walls


def release_xy(env, label):
    """Release point inside the object's compartment closest to the robot. The compartment centre of the
    can, the farthest corner, is 0.86 m from the robot base: at the edge of the Panda's reach, where
    moves to and from it pinned the elbow at its joint limit. Any point inside the compartment scores."""
    i = env.object_to_id[label]
    x_lo = env.bin2_pos[0] - (env.bin_size[0] / 2 if i in (0, 2) else 0)
    y_lo = env.bin2_pos[1] - (env.bin_size[1] / 2 if i < 2 else 0)
    x_hi, y_hi = x_lo + env.bin_size[0] / 2, y_lo + env.bin_size[1] / 2
    m = 0.5 * np.hypot(*SPECS[label]["footprint"]) + PLACE_MARGIN
    return np.clip(ROBOT_BASE_XY, [x_lo + m, y_lo + m], [x_hi - m, y_hi - m])


def observe_truth(env):
    """Grasp targets for every object still in the source bin, from simulator truth."""
    targets = []
    for label in LABELS:
        t = object_truth(env, NAMES[label])
        if not in_source_bin(t["center"]):
            continue
        align = SPECS[label]["align"]
        closing = {"short_side": t["short_dir"], "faces": t["face_dir"]}.get(align)
        targets.append(dict(label=label, xy=t["center"][:2], top=t["top"], height=t["height"],
                            closing=grasp_candidates(align, closing)))
    for t in targets:  # obstacles the carried object must clear: the other objects still in the bin
        t["others_top"] = max([o["top"] for o in targets if o is not t], default=0.0)
    return targets


def truth_masks(env):
    """Visible pixels of each object in the perception camera (evaluator only), image orientation."""
    seg = env.sim.render(width=W, height=H, camera_name=CAMERA, segmentation=True)[::-1]
    geom = np.where(seg[..., 0] == 5, seg[..., 1], -1)  # 5 = mjOBJ_GEOM
    m = env.sim.model
    out = {}
    for label in LABELS:
        body = env.obj_body_id[NAMES[label]]
        ids = [g for g in range(m.ngeom) if m.geom_bodyid[g] == body]
        out[label] = np.isin(geom, ids)
    return out


def segmentation_scores(estimates, true_masks):
    """Per true object: best IoU with any predicted mask, that mask's label and validity, and whether
    that mask also covers a large share of another object (merged)."""
    scores = {}
    for label, tm in true_masks.items():
        n = tm.sum()
        if n == 0 or not estimates:
            scores[label] = dict(visible_px=int(n), iou=0.0, pred_label=None, valid=False, merged=False)
            continue
        ious = [(e.mask & tm).sum() / max((e.mask | tm).sum(), 1) for e in estimates]
        best = estimates[int(np.argmax(ious))]
        merged = any((best.mask & om).sum() >= MERGE_FRACTION * om.sum() > 0
                     for ol, om in true_masks.items() if ol != label)
        scores[label] = dict(visible_px=int(n), iou=float(max(ious)), pred_label=best.label, valid=best.valid,
                             merged=bool(merged))
    return scores


def observe_vision(env, segmenter=None):
    """Grasp targets from one RGB-D frame. segmenter(rgb) -> (masks, labels) replaces geometric
    segmentation when given."""
    camera = CameraModel.from_sim(env.sim, CAMERA, H, W)
    rgb, depth = env.sim.render(width=W, height=H, camera_name=CAMERA, depth=True)
    rgb, depth = rgb[::-1].copy(), depth[::-1]
    # Region of interest for the network: pixels on objects inside the source bin (above its floor).
    roi = above_bin_floor(backproject_frame(depth, camera)) if segmenter else None
    masks, labels = segmenter(rgb, roi) if segmenter else (None, None)
    estimates = estimate_objects(rgb, depth, camera, masks, labels)
    targets = [dict(label=e.label, xy=e.center_xy, top=e.top_z, height=e.height,
                    closing=grasp_candidates(SPECS[e.label]["align"], e.closing_dir), est=e,
                    others_top=max([o.top_z for o in estimates if o is not e and o.top_z is not None], default=0.0))
               for e in estimates if e.valid]
    return targets, estimates


def run_scene(env, config, scene, record=False, segmenter=None):
    obs = env.reset()
    for _ in range(SETTLE_STEPS):
        obs, *_ = env.step(np.r_[np.zeros(6), OPEN])
    start = {label: object_truth(env, NAMES[label]) for label in LABELS}
    start_rot = Rotation.from_quat(obs["robot0_eef_quat"])  # round objects are grasped in the start pose
    # After each pick the arm returns to its start pose: every pick then begins from the same
    # well-conditioned configuration (going straight from the far compartment to the next object can
    # drive the elbow into its joint limit), and the arm is out of the camera's view for re-observation.
    home = [("home", obs["robot0_eef_pos"].copy(), OPEN, 0, 150, start_rot)]
    row = dict(config=config, scene=scene)
    frames = [video_frame(env, f"{config} | scene {scene} | start")] if record else []
    steps, picks, attempted, pick_errors, latencies = 0, [], set(), [], []
    t0 = time.perf_counter()

    for k in range(MAX_PICKS):
        if config == "truth":
            targets = observe_truth(env)
        else:
            targets, estimates = observe_vision(env, segmenter if config == "learned" else None)
            if estimates or config == "learned":
                net = segmenter.last_latency_s if config == "learned" else 0.0
                latencies.append((net + (estimates[0].latency_s if estimates else 0.0)) * 1000)
            if k == 0:  # segmentation quality on the full cluttered scene
                for label, sc in segmentation_scores(estimates, truth_masks(env)).items():
                    for key, val in sc.items():
                        row[f"{label}_seg_{key}"] = val
                row["n_regions"] = len(estimates)
                row["n_valid"] = sum(e.valid for e in estimates)
        targets = [t for t in targets if t["label"] not in attempted]  # one attempt per object
        if not targets:
            break
        target = max(targets, key=lambda t: t["top"])  # tallest first: the hand stays above the rest
        attempted.add(target["label"])
        if config != "truth":
            # Which object is really at the estimated position, and how far off is the estimate?
            actual = min(LABELS, key=lambda l: np.linalg.norm(object_truth(env, NAMES[l])["center"][:2] - target["xy"]))
            err = np.linalg.norm(object_truth(env, NAMES[actual])["center"][:2] - target["xy"]) * 1000
            pick_errors.append(err)
            row[f"pick{k + 1}_label_correct"] = actual == target["label"]
        place_xy = release_xy(env, target["label"])
        phases = make_object_phases(target["xy"], target["top"], target["height"], target["closing"], place_xy,
                                    hold_rot=start_rot, clear_z=target["others_top"])
        phase_name = [phases[0][0]]
        names = [p[0] for p in phases] + ["home", "next"]

        def on_step(o):
            if record:
                frames.append(video_frame(env, f"{config} | scene {scene} | pick {k + 1}: {target['label']} | "
                                               f"{phase_name[0]}"))

        def on_phase_end(entry, o):
            phase_name[0] = names[names.index(entry["phase"]) + 1]

        obs, log = run_phases(env, obs, phases + home, on_step, on_phase_end)
        steps += sum(e["steps"] for e in log)
        picks.append(f"{target['label']}" + ("(timeout:" + "/".join(e["phase"] for e in log if e["timed_out"]) + ")"
                                             if any(e["timed_out"] for e in log) else ""))

    positions = []
    for _ in range(STABLE_STEPS):
        obs, *_ = env.step(np.r_[np.zeros(6), OPEN])
        if record:
            frames.append(video_frame(env, f"{config} | scene {scene} | done"))
        positions.append({label: object_truth(env, NAMES[label])["center"] for label in LABELS})
    steps += STABLE_STEPS

    sorted_count = 0
    for label in LABELS:
        final = positions[-1][label]
        drift = max(np.linalg.norm(p[label] - final) for p in positions)
        in_own = not env.not_in_bin(final, env.object_to_id[label])
        in_other = any(not env.not_in_bin(final, i) for i in range(4) if i != env.object_to_id[label])
        if in_own and drift < STABLE_TOL:
            outcome = "sorted"
        elif in_source_bin(final):
            outcome = "left_in_source_bin"
        elif in_own:
            outcome = "unstable"
        elif in_other:
            outcome = "wrong_compartment"
        else:
            outcome = "dropped_elsewhere"
        sorted_count += outcome == "sorted"
        row[f"{label}_outcome"] = outcome
        row[f"{label}_start_x"], row[f"{label}_start_y"] = start[label]["center"][:2]
    if pick_errors:
        row.update(pick_err_xy_mm_max=max(pick_errors), pick_err_xy_mm_mean=float(np.mean(pick_errors)))
    if latencies:
        row["latency_ms_mean"] = float(np.mean(latencies))
    row.update(sorted=sorted_count, success=sorted_count == 4 and steps <= TIME_BUDGET_STEPS,
               picks=" > ".join(picks), sim_time_s=steps / CONTROL_FREQ, wall_time_s=time.perf_counter() - t0)
    return row, frames


def save_cases(rows, args, out, segmenter, n_success=2, n_failure=3):
    """Replay chosen learned-configuration scenes with video. The simulation is deterministic given the seed
    and the number of resets, so a fresh env reproduces each scene; each replay is checked against the log."""
    picks = ([r["scene"] for r in rows if r["success"]][:n_success]
             + [r["scene"] for r in rows if not r["success"]][:n_failure])
    if not picks:
        return
    by_scene = {r["scene"]: r for r in rows}
    env = make_env(args.seed)
    try:
        for scene in range(max(picks) + 1):
            if scene not in picks:
                env.reset()
                continue
            row, frames = run_scene(env, "learned", scene, record=True, segmenter=segmenter)
            ref = by_scene[scene]
            assert all(row[f"{l}_outcome"] == ref[f"{l}_outcome"] for l in LABELS), f"replay of scene {scene} diverged"
            stem = f"learned_scene{scene:03d}_sorted{row['sorted']}of4"
            imageio.mimsave(out / "cases" / f"{stem}.mp4", frames, fps=CONTROL_FREQ, macro_block_size=1)
            print(f"saved {stem} (replay matched)", flush=True)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=5000)
    parser.add_argument("--configs", nargs="+", default=["truth", "geometric", "learned"],
                        choices=["truth", "geometric", "learned"])
    parser.add_argument("--model", default="models/maskrcnn_seg.pt")
    parser.add_argument("--out", default="results/step5")
    args = parser.parse_args()
    segmenter = None
    if "learned" in args.configs:
        from segmenter import Segmenter
        segmenter = Segmenter(args.model)
    out = Path(args.out)
    (out / "cases").mkdir(parents=True, exist_ok=True)

    rows = []
    for config in args.configs:
        env = make_env(args.seed)
        if config == "learned":  # warm up on a real frame (no reset, so the scene sequence is unchanged):
            frame = env.sim.render(width=W, height=H, camera_name=CAMERA)[::-1].copy()  # the mask branch only
            for _ in range(3):                                                           # runs on detections
                segmenter(frame)
        try:
            for scene in range(args.scenes):
                row, _ = run_scene(env, config, scene, segmenter=segmenter)
                rows.append(row)
                detail = ", ".join(f"{l}:{row[f'{l}_outcome']}" for l in LABELS if row[f"{l}_outcome"] != "sorted")
                print(f"{config:7s} scene {scene:3d}  sorted {row['sorted']}/4  picks {row['picks']}  {detail}",
                      flush=True)
        finally:
            env.close()
        if config == "learned":  # demo videos only for the deployed configuration
            save_cases([r for r in rows if r["config"] == "learned"], args, out, segmenter)

    with open(out / "trials.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(seed=args.seed, scenes=args.scenes)
    for config in args.configs:
        rs = [r for r in rows if r["config"] == config]
        k = int(sum(r["success"] for r in rs))
        per_object = {l: {} for l in LABELS}
        for r in rs:
            for l in LABELS:
                per_object[l][r[f"{l}_outcome"]] = per_object[l].get(r[f"{l}_outcome"], 0) + 1
        n_obj = int(sum(r["sorted"] for r in rs))
        summary[config] = dict(scenes_fully_sorted=k, scenes=len(rs), scene_ci95=wilson(k, len(rs)),
                               objects_sorted=n_obj, objects=4 * len(rs), object_ci95=wilson(n_obj, 4 * len(rs)),
                               per_object=per_object)
        if config != "truth":
            seg = [(r[f"{l}_seg_iou"], r[f"{l}_seg_pred_label"] == l, r[f"{l}_seg_merged"], r[f"{l}_seg_valid"])
                   for r in rs for l in LABELS]
            summary[config].update(
                mask_iou_median=float(np.median([s[0] for s in seg])),
                objects_detected_iou50=int(sum(s[0] >= 0.5 for s in seg)),
                objects_detected_iou50_correct_label=int(sum(s[0] >= 0.5 and s[1] for s in seg)),
                objects_merged=int(sum(s[2] for s in seg)),
                objects_detected_iou50_valid_fit=int(sum(s[0] >= 0.5 and s[1] and s[3] for s in seg)),
                picks_with_wrong_label=int(sum(r.get(f"pick{i}_label_correct") is False for r in rs for i in range(1, 5))),
                latency_ms_mean=float(np.mean([r["latency_ms_mean"] for r in rs if "latency_ms_mean" in r])))
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
