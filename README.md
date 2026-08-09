# IsaacLab_UR5e_reach_Sim2Real
 
Deploying an [Isaac Lab](https://isaac-sim.github.io/IsaacLab/)-trained **PPO reach policy** for the **Universal Robots UR5e** onto a real (or driver-controlled simulated) robot through **ROS 2**.
 
The reinforcement-learning policy is trained in NVIDIA Isaac Lab, exported, and then executed **outside** the training runtime by a standalone ROS 2 inference node. Observations are built from live joint states, the policy computes actions, and joint-position commands are streamed back to the robot via the [Universal Robots ROS 2 driver](https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver).
 
> **Status:** The **Sim2Sim** stage is complete and documented in the blog post below. The **Sim2Real** stage (execution on physical hardware) is in progress. The control code in this repository is written to drive the real robot as well.
 
📝 **Blog post (Sim2Sim):** [Sim2Sim Transfer of a PPO Reach Policy via ROS2 and UR Driver](https://farkasbalintkaroly.github.io/Balint-Farkas.github.io//posts/2025/04/blog-post-6/)
 
---
 
## Overview
 
The pipeline separates *training* from *deployment* so the same policy can run against different back-ends (Isaac Lab, a UR5e in a virtual machine, or the physical robot) without retraining:
 
```
┌──────────────────┐     export      ┌─────────────────────────┐     ROS 2 topics     ┌──────────────────────┐
│   Isaac Lab      │  ────────────▶ │   run_task.py           │  ◀────────────────▶  │  UR5e                │
│   PPO training   │  policy (.pt)   │   ROS 2 inference node  │   state / commands   │  (real or VM sim)    │
│   (skrl)         │  + env (.yaml)  │   URReachPolicy         │                      │  ur_robot_driver     │
└──────────────────┘                 └─────────────────────────┘                      └──────────────────────┘
```
 
The inference node:
 
1. Subscribes to the controller state (`/scaled_joint_trajectory_controller/state`).
2. Builds the 43-dimensional observation vector from joint positions, velocities, the target command, and the previous action.
3. Runs a forward pass through the trained PPO policy (exploration noise disabled → deterministic control).
4. Maps the resulting joint targets to servo-safe ranges and publishes a `JointTrajectory` command at **100 Hz**.
---
 
## Repository structure
 
```
IsaacLab_UR5e_reach_Sim2Real/
├── run_task.py                     # ROS 2 node: control loop, observation→action→command at 100 Hz
├── checking.py                     # Small utility to inspect a checkpoint's keys
├── controllers/
│   └── policy_controller.py        # PolicyController / SKRLPolicy / PolicyControllerSKRL (policy loading + inference)
├── robots/
│   └── ur.py                       # URReachPolicy: UR5e joint model, observation/action construction
└── utils/
    └── config_loader.py            # Parses the Isaac Lab env.yaml (physics + joint properties). Adapted from NVIDIA Isaac Lab.
```
 
Not tracked in the repo (you must provide them):
 
```
sample/ur_reach/
├── ur5e_reach_policy.pt            # Trained PPO checkpoint exported from Isaac Lab
└── ur5e_reach_env.yaml             # Environment configuration used during training
```
 
---
 
## Requirements
 
- **Ubuntu 22.04**
- **ROS 2 Humble**
- **[Universal Robots ROS 2 Driver](https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver)** (`ur_robot_driver`)
- **Python 3.10** with:
  - `torch`
  - `skrl`
  - `numpy`
  - `pyyaml`
  - `rclpy`, `control_msgs`, `trajectory_msgs`, `builtin_interfaces` (provided by the ROS 2 environment)
- A **UR5e** — physical, or simulated in a separate machine/VM controlled via `ur_robot_driver`
---
 
## Setup
 
1. **Clone the repository**
```bash
   git clone https://github.com/FarkasBalintKaroly/IsaacLab_UR5e_reach_Sim2Real.git
   cd IsaacLab_UR5e_reach_Sim2Real
```
 
2. **Install the Python dependencies** (inside your ROS 2 / Python environment)
```bash
   pip install torch skrl numpy pyyaml
```
 
3. **Provide your trained policy** — copy the exported checkpoint and the training environment config into a location of your choice, e.g. `sample/ur_reach/`.
4. **Configure paths and IPs** (see below).
---
 
## Configuration
 
A few values are currently hard-coded and need to be adjusted to your setup:
 
| What | Where | Default |
|------|-------|---------|
| Policy checkpoint path | `robots/ur.py` → `load_policy(...)` | `/home/robotlab/isaaclab_ur_reach_sim2real/sample/ur_reach/ur5e_reach_policy.pt` |
| Environment config path | `robots/ur.py` → `load_policy(...)` | `/home/robotlab/isaaclab_ur_reach_sim2real/sample/ur_reach/ur5e_reach_env.yaml` |
| Robot IP | driver launch command | `192.168.56.101` (VM in the Sim2Sim setup) |
| Reach target (pos + quat) | `run_task.py` → `step_callback()` → `self.target_command` | `[0.5, 0.0, 0.2, 0.7071, 0.0, 0.7071, 0.0]` |
 
> The reach target is currently a **fixed command** set in `run_task.py`. Change `self.target_command` (3 position + 4 quaternion values) to send the end-effector to a different pose.
 
---
 
## Usage
 
### 1. Launch the UR driver
 
Point the driver at your robot (physical robot IP, or the VM/sim IP):
 
```bash
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5e robot_ip:=192.168.56.101
```
 
Make sure the **`scaled_joint_trajectory_controller`** is active — the node subscribes to its `/state` topic and publishes to its `/joint_trajectory` topic.
 
### 2. Run the policy inference node
 
```bash
python3 run_task.py
```
 
The node initializes `URReachPolicy`, loads the policy, and begins the closed-loop control at 100 Hz.
 
### 3. Inspect a checkpoint (optional)
 
```bash
python3 checking.py   # prints the top-level keys of the .pt checkpoint (adjust the path inside first)
```
 
---
 
## How it works
 
### Observation space (43-dim)
 
Built in `robots/ur.py → _compute_observation()`:
 
| Indices | Content |
|---------|---------|
| `0:6`   | Joint positions minus default positions (Δ from the training default pose) |
| `6:12`  | Joint velocities |
| `12:19` | Target command — 3 position + 4 quaternion |
| `19:31` | Previous action (full 12-dim) |
| `31:43` | Padding / reserved |
 
### Action space (12-dim)
 
- The **first 6** values are scaled (`action_scale = 0.5`) and added to the default joint positions to form joint-position targets.
- The remaining values are not applied to the arm in this reach task.
### Joint ordering
 
The UR ROS 2 driver reports joints in a different order than the simulation's DOF order. `run_task.py` handles this explicitly via `JOINT_NAMES` and `JOINT_NAME_TO_IDX`, so simulation-order actions map correctly onto the driver's expected joint order.
 
### Safety mapping
 
`map_joint_angle()` maps each simulation joint angle into the servo angle range, clipping out-of-range values and warning instead of sending unsafe commands.
 
---
 
## Policy model
 
- Trained with **PPO** using the **[skrl](https://skrl.readthedocs.io/)** library in Isaac Lab.
- Network (`SKRLPolicy`, Gaussian policy): `Linear(obs, 64) → ELU → Linear(64, 64) → ELU → Linear(64, act)`, with a learnable per-action `log_std`.
- Loaded from an skrl PPO `.pt` checkpoint (`policy` sub-state-dict); the environment `.yaml` supplies physics timestep, decimation, and per-joint properties (effort/velocity limits, stiffness, damping, default pose).
---
 
## Known limitations / TODO
 
- **Hard-coded absolute paths and IPs** — see the [Configuration](#configuration) table. Consider moving these to CLI args or a config file.
- **`utils/config_loader.py`** references `sys.maxsize`, but `import sys` is commented out; this only triggers when an effort/velocity limit is `None`/`inf`. Uncomment the import if you hit it.
- **Sim2Real** on physical hardware is still in progress.
---
 
## Acknowledgements
 
- Implementation adapted from **[louislelay/isaaclab_ur_reach_sim2real](https://github.com/louislelay/isaaclab_ur_reach_sim2real)**.
- `utils/config_loader.py` is adapted from **NVIDIA Isaac Lab** and retains NVIDIA's original copyright header; its use is subject to NVIDIA's license terms.
- Built on [NVIDIA Isaac Lab](https://isaac-sim.github.io/IsaacLab/), [skrl](https://skrl.readthedocs.io/), and the [Universal Robots ROS 2 Driver](https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver).
---
 
## Author
 
**Farkas Bálint Károly** — Doctoral researcher, Óbuda University
[GitHub](https://github.com/FarkasBalintKaroly) · [Homepage](https://farkasbalintkaroly.github.io/Balint-Farkas.github.io//) · [Google Scholar](https://scholar.google.com/citations?user=JKTiC4EAAAAJ) · [ORCID](https://orcid.org/0009-0008-4783-7226)
 
---
 
## License
 
Released under the **MIT License** — see [LICENSE](LICENSE). Note that `utils/config_loader.py` carries NVIDIA's own copyright and license terms, which apply to that file regardless of the project's MIT license.
 
