# Vision-Guided Robot Manipulation

A simulated robot arm that uses an RGB-D camera to locate an object, pick it up, and place it in a bin. The project then measures how perception errors and processing delays affect task success, and adds recovery behavior for missed grasps and stale targets.

Built on [MuJoCo](https://mujoco.org/) physics and [robosuite](https://robosuite.ai/) task assets. The setup uses a Franka Panda arm with a parallel-jaw gripper, one fixed RGB-D camera, and one rigid object (robosuite's `PickPlaceCan` task).

## Status

| Milestone | State |
|---|---|
| 1. Scripted pick-and-place from known object and bin poses | **Done** |
| 2. Localize the object from RGB-D images; report position error against ground truth | Next |
| 3. Vision-driven pick-and-place | Planned |
| 4. Timestamps, stale-target rejection, timeouts, single retry | Planned |
| 5. Held-out evaluation across ground-truth, vision, and vision + recovery configurations | Planned |

Results and a demo video will be added as each milestone is evaluated. No performance numbers are reported until they have been measured.

### Milestone 1: scripted baseline

The arm picks the can from the source bin and places it in its target compartment, using known world-frame positions instead of perception. This baseline separates motion and grasp problems from vision problems, so later failures can be attributed to perception or timing.

- The can is placed at a fixed, known pose after each reset; robosuite's default random placement is overridden.
- A phase-based policy runs `approach → descend → close → lift → transfer → lower → release → retreat`. Each phase advances when the end effector is within 1 cm of its waypoint or after a per-phase timeout.
- Motion uses robosuite's `OSC_POSE` operational-space controller with delta position inputs (1.0 action = 5 cm step) and zero rotation deltas, which keeps the gripper pointing down.
- Success is scored with robosuite's task check. The run is deterministic: every phase reaches its tolerance, and the can ends in the target compartment.
- The script also saves an RGB frame and a depth frame from the `agentview` camera as inputs for milestone 2.

## System design

The target architecture keeps perception, geometry, and task logic as separate components with small interfaces. In the final vision configuration, the policy cannot access simulator object poses or segmentation IDs; only the evaluator uses ground truth.

| Component | Input | Output |
|---|---|---|
| Simulator adapter | Scene, robot actions | RGB, depth, timestamps, robot state |
| Perception | RGB and depth | Object mask, grasp point, validity flag |
| Geometry | Pixel, metric depth, camera calibration | Grasp position in the robot base frame |
| Task logic | Grasp target, robot feedback | Phase, motion target, gripper command |
| Controller adapter | Target and current end-effector pose | robosuite action vector |
| Evaluator | Simulator ground truth, episode log | Success, position error, latency, failure category |

## Quick start

Requires Linux (WSL2 works) and Python 3.10.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
python scripts/pick_place_scripted.py          # writes results/step1/
```

The script exits with status 0 on task success. Outputs:

- `results/step1/pick_place.mp4`: front and agentview cameras side by side
- `results/step1/key_*.png`: final frame of each phase
- `results/step1/agentview_rgb.png`, `agentview_depth_normalized.npy`: one RGB-D frame

## Engineering notes

- **MuJoCo version pin.** robosuite 1.5.2 fails on startup with MuJoCo 3.14 because newer MuJoCo returns joint types that no longer compare equal to robosuite's enum check. `requirements-lock.txt` pins MuJoCo 3.3.7.
- **Rendering backend.** robosuite forces `MUJOCO_GL=egl` unless the variable is `glx` or `osmesa`. On WSL without access to `/dev/dri/renderD128`, EGL falls back to a software path that can return corrupted frames. When a display is available, the script sets `MUJOCO_GL=glx`, which makes robosuite render through its GLFW context.
- **Image orientation.** robosuite camera images use OpenGL's bottom-left origin. RGB and depth are both flipped vertically before saving so that pixel coordinates match standard image convention for later back-projection.
- **Depth units.** Saved depth is MuJoCo's normalized `[0, 1]` buffer, not metres. Milestone 2 converts it to metric depth using the near and far clip planes before back-projection.
- **Grasp height.** The fingertips extend about 2 cm below the controller's grip site. A grasp target at the can's center drove the fingers into the bin floor, so the grasp point is set slightly above center.

## Repository layout

```
scripts/pick_place_scripted.py   Milestone 1: scripted pick-and-place with known poses
requirements-lock.txt            Exact package versions
results/                         Generated outputs (not committed)
```

As the vision pipeline is added, the script will be split into `sim_adapter.py`, `perception.py`, `geometry.py`, `policy.py`, and `evaluate.py`.

## Evaluation plan

The final evaluation runs the same held-out scene seeds under three configurations: ground-truth motion, vision baseline, and vision with recovery. Tuning uses a separate development seed set. A trial succeeds when the object is inside the target bin, the gripper has released it, and the object is stable for 0.5 s of simulation time, all within a 30 s simulation timeout. Reported metrics:

- Task success rate, with confidence intervals
- Median and p95 position error between the estimated and true object position
- p95 perception latency
- Target age at decision time
- Completion time in both simulation and wall-clock time
- Failure counts by category: missed grasp, wrong target, stale frame, timeout, prohibited contact
