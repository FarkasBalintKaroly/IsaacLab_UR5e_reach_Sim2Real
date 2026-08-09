import numpy as np
from controllers.policy_controller import PolicyController, PolicyControllerSKRL


class URReachPolicy(PolicyControllerSKRL):
    """Policy controller for UR Reach using a pre-trained policy model."""

    def __init__(self) -> None:
        """Initialize the URReachPolicy instance."""
        super().__init__()
        self.dof_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"
        ]
        self.num_joints = len(self.dof_names)
        # Load the pre-trained policy model and environment configuration
        # YOU NEED TO CHANGE THE PATH
        self.load_policy(
            "/home/robotlab/isaaclab_ur_reach_sim2real/sample/ur_reach/ur5e_reach_policy.pt",
            "/home/robotlab/isaaclab_ur_reach_sim2real/sample/ur_reach/ur5e_reach_env.yaml",
        )

        self._action_scale = 0.5
        self._previous_action = np.zeros(12)
        self._policy_counter = 0
        self.target_command = np.array([0.5, 0.0, 0.2, 0.7071, 0.0, 0.7071, 0.0])

        self.has_joint_data = False
        self.current_joint_positions = np.zeros(6)
        self.current_joint_velocities = np.zeros(6)
        self.gripper_controller = GripperController(ip="192.168.1.101", port=63352)
        self._last_gripper_command = None

    def update_joint_state(self, position, velocity) -> None:
        """
        Update the current joint state.

        Args:
            position: A list or array of joint positions.
            velocity: A list or array of joint velocities.
        """
        self.current_joint_positions = np.array(position[:self.num_joints], dtype=np.float32)
        self.current_joint_velocities = np.array(velocity[:self.num_joints], dtype=np.float32)
        self.has_joint_data = True

    def _compute_observation(self, command: np.ndarray) -> np.ndarray:
        if not self.has_joint_data:
            return None

        obs = np.zeros(43, dtype=np.float32)
        obs[0:6] = self.current_joint_positions - self.default_pos
        obs[6:12] = self.current_joint_velocities
        obs[12:19] = command  # 3 pos + 4 quat
        obs[19:31] = self._previous_action  # teljes 12-dim action
        # 31-43 közé kerülhet padding vagy egyéb input (ha kell)
        return obs

    def forward(self, dt: float, command: np.ndarray) -> np.ndarray:
        """
        Compute the next joint positions based on the policy.

        Args:
            dt: Time step for the forward pass.
            command: The target command vector.

        Returns:
            The computed joint positions if joint data is available, otherwise None.
        """
        if not self.has_joint_data:
            return None

        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation(command)
            if obs is None:
                return None
            self.action = self._compute_action(obs)
            self._previous_action = self.action.copy()

            # === Gripper control ===
            gripper_action_value = self.action[-1]  # utolsó érték: gripper open/close
            gripper_threshold = 0.5

            if gripper_action_value > gripper_threshold:
                desired_gripper_state = "close"
            else:
                desired_gripper_state = "open"

            if desired_gripper_state != self._last_gripper_command:
                if desired_gripper_state == "close":
                    self.gripper_controller.close()
                else:
                    self.gripper_controller.open()
                self._last_gripper_command = desired_gripper_state
            # === Gripper control END ===

        # Védelem: ha még nincs action definiálva, addig ne küldjünk semmit
        if not hasattr(self, "action"):
            return None
            # Debug Logging (commented out)
            # print("\n=== Policy Step ===")
            # print(f"{'Command:':<20} {np.round(command, 4)}\n")
            # print("--- Observation ---")
            # print(f"{'Δ Joint Positions:':<20} {np.round(obs[:6], 4)}")
            # print(f"{'Joint Velocities:':<20} {np.round(obs[6:12], 4)}")
            # print(f"{'Command:':<20} {np.round(obs[12:19], 4)}")
            # print(f"{'Previous Action:':<20} {np.round(obs[19:25], 4)}\n")
            # print("--- Action ---")
            # print(f"{'Raw Action:':<20} {np.round(self.action, 4)}")
            # processed_action = self.default_pos + (self.action * self._action_scale)
            # print(f"{'Processed Action:':<20} {np.round(processed_action, 4)}")

        joint_positions = self.default_pos + (self.action[:self.num_joints] * self._action_scale)
        self._policy_counter += 1
        return joint_positions
