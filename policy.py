"""Scripted pick-and-place policy: fixed phase sequence driving robosuite's OSC_POSE controller.

The policy is given a grasp target (the can's estimated body centre) and a placement point; where
those come from (simulator truth or perception) is the caller's choice. It uses only the robot's
own end-effector feedback to advance phases, never the object's state.
"""
import numpy as np

HOVER_DZ = 0.12  # approach height above grasp point
GRASP_DZ = 0.015  # fingertips extend ~2 cm below the grip site; lower and they hit the bin floor
CARRY_DZ = 0.25  # transfer height above the target bin, clears both bin walls
RELEASE_DZ = 0.10  # release height above the target placement point
POS_TOL = 0.01  # waypoint reached tolerance (m)
KP = 10.0  # proportional gain from position error (m) to normalized action

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


def move_action(eef_pos, target, gripper):
    # action 1.0 commands a 5 cm step (default_panda.json output_max), so KP=10 saturates beyond 10 cm
    delta = np.clip(KP * (target - eef_pos), -1.0, 1.0)
    # [dx, dy, dz, droll, dpitch, dyaw, gripper]; zero rotation keeps the gripper pointing down
    return np.concatenate([delta, np.zeros(3), [gripper]])


def run_phases(env, obs, phases, on_step=None, on_phase_end=None):
    """Execute phases in order. Advances when the end effector is within POS_TOL of the target
    (then holds `hold` steps), or when the phase times out. Returns final obs and a per-phase log."""
    log = []
    for name, target, grip, hold, timeout in phases:
        reached_at = None
        for t in range(timeout):
            obs, *_ = env.step(move_action(obs["robot0_eef_pos"], target, grip))
            if on_step:
                on_step(obs)
            err = np.linalg.norm(target - obs["robot0_eef_pos"])
            if reached_at is None and (err < POS_TOL or hold):
                reached_at = t
            if reached_at is not None and t - reached_at >= hold:
                break
        entry = dict(phase=name, steps=t + 1, pos_err=err, timed_out=not (err < POS_TOL or hold))
        log.append(entry)
        if on_phase_end:
            on_phase_end(entry, obs)
    return obs, log
