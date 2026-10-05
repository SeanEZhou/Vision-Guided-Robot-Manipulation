"""Milestone 1: scripted pick-and-place with known object and bin locations (no vision).

Loads robosuite's PickPlaceCan (Panda arm, one can, source bin, target bin), forces the can
to a fixed known pose, then runs a phase-based scripted policy that drives the default
OSC_POSE controller (delta position inputs) to hard-coded waypoints. Records a video and
keyframes for review, and saves one RGB-D frame from the agentview camera.

Usage: python scripts/pick_place_scripted.py [--out results/step1]
"""
import argparse
import os

# On WSL, EGL needs access to /dev/dri/renderD128 (the `render` group) and otherwise falls back to a
# software path that can return garbage frames. robosuite forces "egl" unless MUJOCO_GL is "glx" or
# "osmesa"; "glx" makes it use its GLFW context, which renders through WSLg.
if os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "glx")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import imageio
import numpy as np
import robosuite as suite
from PIL import Image

from policy import OPEN, make_phases, move_action, run_phases

# ---- Known scene layout (world frame, metres) ---------------------------------------------
SOURCE_BIN_POS = np.array([0.1, -0.25, 0.8])  # robosuite default bin1_pos
TARGET_BIN_POS = np.array([0.1, 0.28, 0.8])  # robosuite default bin2_pos
CAN_XY = np.array([0.1, -0.25])  # place the can at the centre of the source bin
CAN_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0])  # upright

CAMERAS = ["frontview", "agentview"]
H, W = 480, 640


def make_env():
    return suite.make(
        "PickPlaceCan",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        use_object_obs=True,
        camera_names=CAMERAS,
        camera_heights=H,
        camera_widths=W,
        camera_depths=True,
        control_freq=20,
        horizon=1000,
        ignore_done=True,
        bin1_pos=SOURCE_BIN_POS,
        bin2_pos=TARGET_BIN_POS,
        seed=0,
    )


def place_can_at_known_pose(env):
    """Override the randomly sampled can pose with a fixed, known one."""
    joint = env.objects[env.object_id].joints[0]
    sampled = env.sim.data.get_joint_qpos(joint)
    qpos = np.concatenate([[CAN_XY[0], CAN_XY[1], sampled[2]], CAN_QUAT_WXYZ])
    env.sim.data.set_joint_qpos(joint, qpos)
    env.sim.forward()


def frame_from_obs(obs):
    # robosuite renders with OpenGL's bottom-left origin; flip to image convention.
    views = [obs[f"{cam}_image"][::-1] for cam in CAMERAS]
    return np.concatenate(views, axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/step1")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    env = make_env()
    try:
        obs = env.reset()
        place_can_at_known_pose(env)
        # settle physics with zero motion, gripper open
        for _ in range(10):
            obs, *_ = env.step(np.concatenate([np.zeros(6), [-1.0]]))

        obj_name = env.objects[env.object_id].name
        can_body = env.sim.data.body_xpos[env.obj_body_id[obj_name]].copy()
        place_xy = env.target_bin_placements[env.object_id][:2]

        # Waypoints come from the known layout; can_body is only used for the can's resting height.
        phases = make_phases([CAN_XY[0], CAN_XY[1], can_body[2]], place_xy, TARGET_BIN_POS[2])

        print(f"Object: {obj_name}  known xy={CAN_XY}  true body pos={np.round(can_body, 4)}")
        print(f"Target placement xy={np.round(place_xy, 4)}  (target bin at {TARGET_BIN_POS})")

        Image.fromarray(obs["agentview_image"][::-1]).save(out / "agentview_rgb.png")
        np.save(out / "agentview_depth_normalized.npy", obs["agentview_depth"][::-1])

        frames = [frame_from_obs(obs)]

        def on_phase_end(entry, obs):
            status = "TIMEOUT" if entry["timed_out"] else "ok"
            print(f"  {entry['phase']:9s} steps={entry['steps']:4d}  pos_err={entry['pos_err'] * 1000:6.1f} mm  {status}")
            Image.fromarray(frames[-1]).save(out / f"key_{len(frames_logged) + 1:02d}_{entry['phase']}.png")
            frames_logged.append(entry["phase"])

        frames_logged = []
        obs, _ = run_phases(env, obs, phases, on_step=lambda o: frames.append(frame_from_obs(o)),
                            on_phase_end=on_phase_end)

        # let the can settle, then score with robosuite's own success check
        for _ in range(20):
            obs, *_ = env.step(move_action(obs["robot0_eef_pos"], phases[-1][1], OPEN))
            frames.append(frame_from_obs(obs))
        final_can = env.sim.data.body_xpos[env.obj_body_id[obj_name]]
        success = bool(env._check_success())
        print(f"Final can pos={np.round(final_can, 4)}  robosuite success={success}")

        imageio.mimsave(out / "pick_place.mp4", frames, fps=20, macro_block_size=1)
        print(f"Wrote {len(frames)} frames to {out / 'pick_place.mp4'}")
        print("Action bounds:", env.action_spec)
        return 0 if success else 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
