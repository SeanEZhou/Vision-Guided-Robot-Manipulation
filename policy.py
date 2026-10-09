"""Scripted pick-and-place policy: fixed phase sequence driving robosuite's OSC_POSE controller.

The policy is given a grasp target and a placement point; where those come from (simulator truth or
perception) is the caller's choice. It uses only the robot's own end-effector feedback to advance
phases, never the object's state.

Two phase builders share one executor:
- make_phases: milestones 1-3 (can only). Position control; the gripper keeps its start orientation.
- make_object_phases: milestone 4. Also turns the gripper about the vertical axis so the fingers close
  across the object's narrow side, and sets grasp, carry and release heights from the object's size.
"""
import numpy as np
from scipy.spatial.transform import Rotation

HOVER_DZ = 0.12  # approach height above grasp point
# The palm's collision surface sits ~2 cm above the grip site and the finger pads span 5.5 mm below to
# 12.8 mm above it, so the grip site cannot go much more than 2.5 cm below an object's top: the palm
# lands on the object first. For the can, 1.5 cm above its centre is 2.5 cm below its top.
GRASP_DZ = 0.015  # milestones 1-3: grasp height above the can's centre
GRASP_BELOW_TOP = 0.025  # milestone 4: grasp height below the object's top, for any object
CARRY_DZ = 0.25  # transfer height above the target bin, clears both bin walls
RELEASE_DZ = 0.10  # release height above the target placement point
POS_TOL = 0.01  # waypoint reached tolerance (m)
ROT_TOL = np.deg2rad(3.0)  # orientation reached tolerance, where a phase has an orientation target
KP = 10.0  # proportional gain from position error (m) to normalized action
KR = 2.0  # proportional gain from orientation error (rad) to normalized action (1.0 = 0.5 rad step)

BIN_RIM_Z = 0.90  # top of the bin walls (bins at z=0.8 in robosuite's PickPlace layout)
BIN_FLOOR_Z = 0.82  # bin floor surface
CARRY_CLEARANCE = 0.04  # gap between a held object's bottom and the bin rims during transfer
RELEASE_CLEARANCE = 0.02  # gap between a held object's bottom and the target bin floor at release
MIN_CARRY_Z = 1.05  # milestones 1-3 carry height (bin z + CARRY_DZ); taller objects carry higher

# Panda hand (collision mesh, gripper frame): 20.4 cm long along the finger-closing axis, 6.4 cm wide,
# lowest point 28.5 mm above the grip site. Grasping a short object puts the hand below the bin rim,
# where its ends can hit the walls; the grasp rotation is then chosen to keep it clear.
HAND_HALF_LENGTH = 0.104
HAND_HALF_WIDTH = 0.032
HAND_BOTTOM_ABOVE_GRIP = 0.0285
SOURCE_BIN_INNER_X = (-0.09, 0.29)
SOURCE_BIN_INNER_Y = (-0.49, -0.01)

OPEN, CLOSE = -1.0, 1.0


def make_phases(can_center, place_xy, bin_z):
    """Waypoints for one pick-and-place. Each phase: (name, target, gripper, hold steps, timeout)."""
    grasp = np.asarray(can_center, dtype=float) + [0, 0, GRASP_DZ]
    carry_z = bin_z + CARRY_DZ
    release = np.array([place_xy[0], place_xy[1], bin_z + RELEASE_DZ])
    return [
        ("approach", grasp + [0, 0, HOVER_DZ], OPEN, 0, 150),
        ("descend", grasp, OPEN, 0, 120),
        ("close", grasp, CLOSE, 15, 30),
        ("lift", np.array([grasp[0], grasp[1], carry_z]), CLOSE, 0, 150),
        ("transfer", np.array([release[0], release[1], carry_z]), CLOSE, 0, 200),
        ("lower", release, CLOSE, 0, 120),
        ("release", release, OPEN, 15, 30),
        ("retreat", np.array([release[0], release[1], carry_z]), OPEN, 0, 100),
    ]


# A parallel gripper is symmetric under a half turn, so each grasp has two equivalent yaws 180 degrees
# apart. Wrist joint 7 sits near +95 degrees in the start pose and turns opposite to yaw; yaws near -80
# drive it into its +166 degree limit during transfer. Choosing the yaw in [-45, 135) keeps it >25 deg clear.
YAW_WINDOW_START = np.radians(-45.0)


def gripper_down(closing_dir_xy):
    """Vertical gripper orientation whose fingers close along the given horizontal direction."""
    n = np.asarray(closing_dir_xy, dtype=float)
    yaw = np.arctan2(n[0], -n[1])  # finger axis is the gripper's local y, which points along -y at yaw 0
    yaw = (yaw - YAW_WINDOW_START) % np.pi + YAW_WINDOW_START
    return Rotation.from_euler("z", yaw) * Rotation.from_matrix(np.diag([1.0, -1.0, -1.0])), yaw


def grasp_candidates(align, direction=None):
    """Finger-closing directions an object allows, given how it must be gripped:
    "short_side": across the narrow side only; "faces": across either pair of faces of a square box;
    "free": any rotation (the object fits the 80 mm opening every way); None: keep the start pose."""
    if align is None:
        return None
    if align == "free":
        return [np.array([np.cos(a), np.sin(a)]) for a in np.radians(np.arange(0, 180, 15))]
    d = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
    return [d] if align == "short_side" else [d, np.array([-d[1], d[0]])]


def hand_clearance(center_xy, closing_dir):
    """Smallest distance from the hand's footprint to the source bin's inner walls (negative = overlap)."""
    d = np.asarray(closing_dir, dtype=float) / np.linalg.norm(closing_dir)
    perp = np.array([-d[1], d[0]])
    corners = [np.asarray(center_xy) + a * HAND_HALF_LENGTH * d + b * HAND_HALF_WIDTH * perp
               for a in (-1, 1) for b in (-1, 1)]
    return min(min(c[0] - SOURCE_BIN_INNER_X[0], SOURCE_BIN_INNER_X[1] - c[0],
                   c[1] - SOURCE_BIN_INNER_Y[0], SOURCE_BIN_INNER_Y[1] - c[1]) for c in corners)


def choose_closing_dir(center_xy, grip_z, candidates):
    """First candidate if the hand stays above the bin rim; otherwise the one with most wall clearance."""
    if grip_z + HAND_BOTTOM_ABOVE_GRIP > BIN_RIM_Z:
        return candidates[0]
    return max(candidates, key=lambda c: hand_clearance(center_xy, c))


def make_object_phases(grasp_xy, top_z, height, closing_candidates, place_xy, hold_rot=None, clear_z=BIN_RIM_Z):
    """Waypoints for one object. Each phase: (name, target, gripper, hold, timeout, orientation).
    closing_candidates (from grasp_candidates) lists the allowed finger-closing directions; None (a
    round object) keeps the gripper's start orientation, as in milestones 1-3: there is no side to align
    with, and the start pose's slight tilt lets the arm reach the target bin's far compartment, which a
    strictly vertical gripper cannot. With several picks per scene, pass that start orientation as
    hold_rot: it is held while grasping (otherwise the gripper would keep whatever orientation the previous
    pick left it in), then released after the lift so the arm can tilt as needed to reach the far
    compartment. clear_z is the highest obstacle the carried object must pass over: the bin rims, or
    in a cluttered bin the tallest remaining object."""
    grasp = np.array([grasp_xy[0], grasp_xy[1], top_z - GRASP_BELOW_TOP])
    rot = hold_rot if closing_candidates is None else \
        gripper_down(choose_closing_dir(grasp_xy, grasp[2], closing_candidates))[0]
    hang = grasp[2] - (top_z - height)  # how far the held object's bottom hangs below the grip site
    carry_z = max(max(clear_z, BIN_RIM_Z) + CARRY_CLEARANCE + hang, MIN_CARRY_Z)
    # Release low enough for a short drop, but never with the hand below the target bin's rim, where it
    # can hit the compartment walls (a short object would otherwise be lowered that far).
    release_z = max(BIN_FLOOR_Z + RELEASE_CLEARANCE + hang, BIN_RIM_Z - HAND_BOTTOM_ABOVE_GRIP + 0.005)
    release = np.array([place_xy[0], place_xy[1], release_z])
    carry_rot = None if closing_candidates is None else rot
    return [
        ("approach", grasp + [0, 0, HOVER_DZ], OPEN, 0, 150, rot),
        ("descend", grasp, OPEN, 0, 120, rot),
        ("close", grasp, CLOSE, 15, 30, rot),
        ("lift", np.array([grasp[0], grasp[1], carry_z]), CLOSE, 0, 150, rot),
        ("transfer", np.array([release[0], release[1], carry_z]), CLOSE, 0, 200, carry_rot),
        ("lower", release, CLOSE, 0, 120, carry_rot),
        ("release", release, OPEN, 15, 30, carry_rot),
        ("retreat", np.array([release[0], release[1], carry_z]), OPEN, 0, 100, carry_rot),
    ]


def rotation_error(eef_quat_xyzw, target_rot):
    """Axis-angle rotation (world frame) taking the current gripper orientation to the target."""
    return (target_rot * Rotation.from_quat(eef_quat_xyzw).inv()).as_rotvec()


def move_action(eef_pos, target, gripper, eef_quat=None, target_rot=None):
    # action 1.0 commands a 5 cm step (default_panda.json output_max), so KP=10 saturates beyond 10 cm
    delta = np.clip(KP * (target - eef_pos), -1.0, 1.0)
    # [dx, dy, dz, droll, dpitch, dyaw, gripper]. Without an orientation target, zero rotation keeps
    # the gripper's current orientation (milestones 1-3).
    rot = np.zeros(3) if target_rot is None else np.clip(KR * rotation_error(eef_quat, target_rot), -1.0, 1.0)
    return np.concatenate([delta, rot, [gripper]])


def run_phases(env, obs, phases, on_step=None, on_phase_end=None):
    """Execute phases in order. Advances when the end effector is within POS_TOL of the target (and
    within ROT_TOL of its orientation, if the phase has one), then holds `hold` steps; or when the phase
    times out. Returns final obs and a per-phase log."""
    log = []
    for name, target, grip, hold, timeout, *rot in phases:
        rot = rot[0] if rot else None
        reached_at = None
        for t in range(timeout):
            obs, *_ = env.step(move_action(obs["robot0_eef_pos"], target, grip, obs["robot0_eef_quat"], rot))
            if on_step:
                on_step(obs)
            err = np.linalg.norm(target - obs["robot0_eef_pos"])
            rot_err = 0.0 if rot is None else np.linalg.norm(rotation_error(obs["robot0_eef_quat"], rot))
            reached = err < POS_TOL and rot_err < ROT_TOL
            if reached_at is None and (reached or hold):
                reached_at = t
            if reached_at is not None and t - reached_at >= hold:
                break
        entry = dict(phase=name, steps=t + 1, pos_err=err, rot_err_deg=np.rad2deg(rot_err),
                     timed_out=not (reached or hold))
        log.append(entry)
        if on_phase_end:
            on_phase_end(entry, obs)
    return obs, log
