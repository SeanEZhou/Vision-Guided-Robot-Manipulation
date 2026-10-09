"""README demo GIFs, rendered from deterministic replays of evaluated held-out trials (perception camera).

- docs/images/demo_multi_object.gif (milestone 4): 2x2 grid, one vision-guided success per object type.
- docs/images/demo_clutter.gif (milestone 5): one cluttered scene sorted with learned segmentation.

Before each pick the GIF shows what perception found. Each replay is checked against the logged outcome
in results/step4/trials.csv and results/step5/trials.csv, so the GIFs show the evaluated runs.

Usage: python scripts/make_demo_gifs.py   (needs the milestone 4/5 results and models/maskrcnn_seg.pt)
"""
import csv
import os
import sys
from pathlib import Path

if os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "glx")  # see scripts/pick_place_scripted.py

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

import clutter_eval as C
import multi_object_eval as M
from geometry import CameraModel
from object_perception import OBJECTS as SPECS, estimate_object
from policy import OPEN, grasp_candidates, make_object_phases, move_action, run_phases

CAM, H, W = "agentview", 480, 640
CROP = (40, 400)  # image rows kept: arm, both bins
FPS = 20
HOLD = 20  # frames showing the perception result before each pick (1 s)
NAMES = {"milk": "milk carton", "bread": "bread", "cereal": "cereal box", "can": "can"}
COLORS = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48)]
MARKER, LINE = (255, 230, 0), (0, 220, 255)
RESERVED = [MARKER, LINE, (255, 255, 255), (0, 0, 0), (30, 30, 30)]


def render(env):
    return env.sim.render(width=W, height=H, camera_name=CAM)[::-1].copy()


def caption(img, text, scale=0.6):
    bar = np.full((32, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, scale, (30, 30, 30), 1, cv2.LINE_AA)
    return np.vstack([img, bar])


def crop(img):
    return img[CROP[0]:CROP[1]]


def draw_estimate(img, est, cam, color=(60, 200, 60), text=None):
    img[est.mask] = (0.45 * img[est.mask] + 0.55 * np.array(color)).astype(np.uint8)
    cnts, _ = cv2.findContours(est.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cnts, -1, (255, 255, 255), 1)
    if est.center_xy is not None:
        c3 = np.array([est.center_xy[0], est.center_xy[1], est.top_z])
        if est.closing_dir is not None:
            d = np.r_[est.closing_dir, 0] * 0.045
            (a, b) = cam.project(np.array([c3 - d, c3 + d]))[0]
            cv2.line(img, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), LINE, 2, cv2.LINE_AA)
        (u, v), = cam.project(c3[None])[0]
        cv2.drawMarker(img, (round(u), round(v)), MARKER, cv2.MARKER_TILTED_CROSS, 12, 2)
    if text:
        ys, xs = np.nonzero(est.mask)
        org = (int(xs.min()), max(int(ys.min()) - 6, 12))
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def write_gif(frames, path, colors=128):
    """Simplest GIF structure, so every decoder shows the same thing: one global palette shared by all
    frames (built from a sample of them), full frames, no transparency. (ffmpeg's GIF encoder, even with
    its optimizations disabled, produced files that Pillow decoded with speckles of wrong colors.)"""
    sample = np.vstack(frames[:: max(1, len(frames) // 12)])
    # Median cut allots colors by pixel count, so the small, saturated product packaging (red can, yellow
    # cereal box) was mapped to the floor's browns. Over-weight saturated pixels so they get their own colors.
    hsv = cv2.cvtColor(sample, cv2.COLOR_RGB2HSV)
    saturated = sample[(hsv[..., 1] > 120) & (hsv[..., 2] > 60)]
    reps = max(1, int(0.15 * sample.shape[0] * sample.shape[1] / max(len(saturated), 1)))
    pixels = np.vstack([sample.reshape(-1, 3), np.repeat(saturated, reps, axis=0)])
    pixels = pixels[: len(pixels) // 1000 * 1000].reshape(-1, 1000, 3)
    palette = Image.fromarray(pixels).quantize(colors=colors - len(RESERVED), method=Image.Quantize.MEDIANCUT)
    # Exact entries for the overlay graphics: thin markers and lines are too few pixels to earn a color.
    entries = palette.getpalette()[: 3 * (colors - len(RESERVED))] + [c for rgb in RESERVED for c in rgb]
    palette.putpalette(entries + [0] * (768 - len(entries)))
    images = [Image.fromarray(f).quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=int(1000 / FPS), loop=0,
                   optimize=False, disposal=1)
    print(f"wrote {path} ({Path(path).stat().st_size / 1e6:.1f} MB, {len(frames)} frames)")


def multi_object_tile(obj, trial, every=4, size=(320, 180)):
    """Replay milestone 4's vision trial for one object; returns frames (perception, then the pick)."""
    k = M.OBJECTS.index(obj)
    env = M.make_env(obj, 3000 + k)
    try:
        for _ in range(trial):
            env.reset()
        obs = env.reset()
        for _ in range(M.SETTLE_STEPS):
            obs, *_ = env.step(np.r_[np.zeros(6), OPEN])
        cam = CameraModel.from_sim(env.sim, CAM, H, W)
        rgb, depth = env.sim.render(width=W, height=H, camera_name=CAM, depth=True)
        rgb, depth = rgb[::-1].copy(), depth[::-1]
        est = estimate_object(rgb, depth, cam)
        tile = lambda img, text: caption(cv2.resize(crop(img), size, interpolation=cv2.INTER_AREA), text, 0.5)
        frames = [tile(draw_estimate(rgb.copy(), est, cam), f"sees: {NAMES[est.label]}")] * HOLD
        place = env.target_bin_placements[env.object_to_id[est.label]][:2]
        phases = make_object_phases(est.center_xy, est.top_z, est.height,
                                    grasp_candidates(SPECS[est.label]["align"], est.closing_dir), place)
        step = [0]

        def on_step(o):
            step[0] += 1
            if step[0] % every == 0:
                frames.append(tile(render(env), f"picking: {NAMES[est.label]}"))

        obs, _ = run_phases(env, obs, phases, on_step)
        for _ in range(M.STABLE_STEPS):
            obs, *_ = env.step(move_action(obs["robot0_eef_pos"], phases[-1][1], OPEN, obs["robot0_eef_quat"],
                                           phases[-1][5]))
        final = M.object_truth(env)["center"]
        ref = next(r for r in csv.DictReader(open("results/step4/trials.csv"))
                   if r["object"] == obj and r["config"] == "vision" and int(r["trial"]) == trial)
        assert np.allclose(final, [float(ref[f]) for f in ("final_x", "final_y", "final_z")], atol=1e-6), obj
        frames.append(tile(render(env), f"done: {NAMES[est.label]} sorted"))
        return frames
    finally:
        env.close()


def clutter_scene(scene, segmenter, every=5, size=(480, 270)):
    """Replay milestone 5's learned-configuration scene; returns frames."""
    env = C.make_env(5000)
    try:
        for _ in range(scene):
            env.reset()
        obs = env.reset()
        for _ in range(C.SETTLE_STEPS):
            obs, *_ = env.step(np.r_[np.zeros(6), OPEN])
        frame = render(env)
        for _ in range(3):
            segmenter(frame)  # warm-up; does not change the simulation
        start_rot = Rotation.from_quat(obs["robot0_eef_quat"])
        home = [("home", obs["robot0_eef_pos"].copy(), OPEN, 0, 150, start_rot)]
        view = lambda img, text: caption(cv2.resize(crop(img), size, interpolation=cv2.INTER_AREA), text, 0.5)
        frames, attempted = [], set()
        for k in range(C.MAX_PICKS):
            targets, estimates = C.observe_vision(env, segmenter)
            cam = CameraModel.from_sim(env.sim, CAM, H, W)
            img = render(env)
            for i, e in enumerate(estimates):
                draw_estimate(img, e, cam, COLORS[i % 4], NAMES.get(e.label, "?") + ("" if e.valid else " (skipped)"))
            targets = [t for t in targets if t["label"] not in attempted]
            if not targets:
                break
            target = max(targets, key=lambda t: t["top"])
            attempted.add(target["label"])
            frames += [view(img, f"Mask R-CNN sees {len(estimates)}; next: {NAMES[target['label']]} (tallest)")] * HOLD
            phases = make_object_phases(target["xy"], target["top"], target["height"], target["closing"],
                                        C.release_xy(env, target["label"]), hold_rot=start_rot,
                                        clear_z=target["others_top"])
            step = [0]

            def on_step(o, label=target["label"]):
                step[0] += 1
                if step[0] % every == 0:
                    frames.append(view(render(env), f"pick {k + 1}: {NAMES[label]}"))

            obs, _ = run_phases(env, obs, phases + home, on_step)
        for _ in range(C.STABLE_STEPS):
            obs, *_ = env.step(np.r_[np.zeros(6), OPEN])
        ref = next(r for r in csv.DictReader(open("results/step5/trials.csv"))
                   if r["config"] == "learned" and int(r["scene"]) == scene)
        for label in C.LABELS:
            pos = M.object_truth(env, C.NAMES[label])["center"]
            sorted_now = not env.not_in_bin(pos, env.object_to_id[label])
            assert sorted_now == (ref[f"{label}_outcome"] == "sorted"), f"scene {scene} {label} diverged"
        frames += [view(render(env), f"done: {ref['sorted']} of 4 objects sorted")] * HOLD
        return frames
    finally:
        env.close()


def main():
    out = Path("docs/images")
    picks = {}  # the saved milestone 4 demo trial per object
    for f in Path("results/step4/cases").glob("vision_*_success.mp4"):
        _, obj, trial, _ = f.stem.split("_")
        picks[obj] = int(trial.replace("trial", ""))
    tiles = [multi_object_tile(obj, picks[obj]) for obj in M.OBJECTS]
    n = max(len(t) for t in tiles)
    tiles = [t + [t[-1]] * (n - len(t)) for t in tiles]  # hold each tile's last frame until all finish
    gap = lambda a: np.full((a.shape[0], 6, 3), 255, np.uint8)
    grid = []
    for i in range(n):
        top = np.hstack([tiles[0][i], gap(tiles[0][i]), tiles[1][i]])
        bottom = np.hstack([tiles[2][i], gap(tiles[2][i]), tiles[3][i]])
        grid.append(np.vstack([top, np.full((6, top.shape[1], 3), 255, np.uint8), bottom]))
    grid += [grid[-1]] * HOLD
    write_gif(grid, out / "demo_multi_object.gif")

    from segmenter import Segmenter
    write_gif(clutter_scene(0, Segmenter("models/maskrcnn_seg.pt")), out / "demo_clutter.gif")


if __name__ == "__main__":
    main()
