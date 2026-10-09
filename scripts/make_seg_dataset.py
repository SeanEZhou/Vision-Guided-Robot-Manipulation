"""Generate a labeled instance-segmentation dataset from the perception camera.

Each image shows the source bin with a random non-empty subset of the four objects at random positions
and rotations (the rest are moved out of the scene, as during sorting), under randomized lighting.
Labels come from the simulator's segmentation render and cover only visible pixels. Each object type
appears at most once per image, so one label image encodes the instances: pixel value = class id
(1 milk, 2 bread, 3 cereal, 4 can; 0 background).

Simulator labels are used for training and evaluation only, never as inputs to the deployed pipeline.
Seeds: train 10000+, val 20000+ (development used 100, evaluation uses 5000+).

Usage: python scripts/make_seg_dataset.py [--train 3000] [--val 300] [--out data/seg]
"""
import argparse
import os
import sys
from pathlib import Path

if os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "glx")  # see scripts/pick_place_scripted.py

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from clutter_eval import CAMERA, H, LABELS, NAMES, W, make_env, truth_masks

CLASS_ID = {label: i + 1 for i, label in enumerate(LABELS)}
MIN_VISIBLE_PX = 30
SCENES_PER_ENV = 50  # images per environment instance before re-creating it with the next seed


def randomize_lighting(env, rng, base):
    m = env.sim.model
    m.light_pos[:] = base["pos"] + rng.uniform(-0.6, 0.6, base["pos"].shape)
    scale = rng.uniform(0.5, 1.4)
    m.light_diffuse[:] = np.clip(base["diffuse"] * scale * rng.uniform(0.85, 1.15, base["diffuse"].shape), 0, 1)
    m.light_specular[:] = np.clip(base["specular"] * rng.uniform(0.5, 1.5), 0, 1)


def hide(env, label):
    joint = env.objects[env.object_to_id[label]].joints[0]
    qpos = env.sim.data.get_joint_qpos(joint).copy()
    qpos[:3] = [5.0 + env.object_to_id[label], 5.0, 1.0]  # far outside the camera view
    env.sim.data.set_joint_qpos(joint, qpos)


def generate(split, count, seed0, out):
    (out / split).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed0)
    written, env_index = 0, 0
    while written < count:
        env = make_env(seed0 + env_index)
        env_index += 1
        m = env.sim.model
        base = dict(pos=m.light_pos.copy(), diffuse=m.light_diffuse.copy(), specular=m.light_specular.copy())
        try:
            for _ in range(SCENES_PER_ENV):
                if written >= count:
                    break
                env.reset()
                keep = [l for l in LABELS if rng.random() < 0.75] or [rng.choice(LABELS)]
                for label in LABELS:
                    if label not in keep:
                        hide(env, label)
                env.sim.forward()
                for _ in range(5):
                    env.step(np.r_[np.zeros(6), -1.0])
                randomize_lighting(env, rng, base)
                env.sim.forward()
                rgb = env.sim.render(width=W, height=H, camera_name=CAMERA)[::-1]
                label_img = np.zeros((H, W), np.uint8)
                for label, mask in truth_masks(env).items():
                    if mask.sum() >= MIN_VISIBLE_PX:
                        label_img[mask] = CLASS_ID[label]
                stem = out / split / f"{written:05d}"
                cv2.imwrite(f"{stem}_rgb.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                cv2.imwrite(f"{stem}_label.png", label_img)
                written += 1
        finally:
            env.close()
        print(f"{split}: {written}/{count}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=3000)
    parser.add_argument("--val", type=int, default=300)
    parser.add_argument("--out", default="data/seg")
    args = parser.parse_args()
    out = Path(args.out)
    generate("val", args.val, 20000, out)
    generate("train", args.train, 10000, out)


if __name__ == "__main__":
    main()
