import io
from typing import Optional
import numpy as np
import yaml
from torch import nn
from skrl.models.torch import DeterministicMixin
from skrl.models.torch import GaussianMixin
import torch

from utils.config_loader import parse_env_config, get_physics_properties, get_robot_joint_properties

class PolicyController:
    """
    A controller that loads and executes a policy from a file.
    """

    def __init__(self) -> None:
        pass

    def load_policy(self, policy_file_path, policy_env_path) -> None:
        """
        Loads policy from a file.
        """
        print("\n=== Policy Loading ===")
        print(f"{'Model path:':<18} {policy_file_path}")
        print(f"{'Environment path:':<18} {policy_env_path}")

        with open(policy_file_path, "rb") as f:
            file = io.BytesIO(f.read())
        self.policy = torch.jit.load(file)
        self.policy_env_params = parse_env_config(policy_env_path)

        self._decimation, self._dt, self.render_interval = get_physics_properties(self.policy_env_params)

        print("\n--- Physics properties ---")
        print(f"{'Decimation:':<18} {self._decimation}")
        print(f"{'Timestep (dt):':<18} {self._dt}")
        print(f"{'Render interval:':<18} {self.render_interval}")

        self._max_effort, self._max_vel, self._stiffness, self._damping, self.default_pos, self.default_vel = get_robot_joint_properties(
            self.policy_env_params, self.dof_names
        )
        self.num_joints = len(self.dof_names)

        print("\n--- Robot joint properties ---")
        print(f"{'Number of joints:':<18} {self.num_joints}")
        print(f"{'Max effort:':<18} {self._max_effort}")
        print(f"{'Max velocity:':<18} {self._max_vel}")
        print(f"{'Stifness:':<18} {self._stiffness}")
        print(f"{'Damping:':<18} {self._damping}")
        print(f"{'Default position:':<18} {self.default_pos}")
        print(f"{'Default velocity:':<18} {self.default_vel}")

        print("\n=== Policy Loaded ===\n")

    def _compute_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Computes the action from the observation using the loaded policy.

        Args:
            obs (np.ndarray): The observation.

        Returns:
            np.ndarray: The action.
        """
        with torch.no_grad():
            obs = torch.from_numpy(obs).view(1, -1).float()
            action = self.policy(obs).detach().view(-1).numpy()
        return action

    def _compute_observation(self) -> NotImplementedError:
        """
        Computes the observation. Not implemented.
        """

        raise NotImplementedError(
            "Compute observation need to be implemented, expects np.ndarray in the structure specified by env yaml"
        )

    def forward(self) -> NotImplementedError:
        """
        Forwards the controller. Not implemented.
        """
        raise NotImplementedError(
            "Forward needs to be implemented to compute and apply robot control from observations"
        )
    


class SKRLPolicy(nn.Module, GaussianMixin):
    def __init__(self, observation_space: int, action_space: int):
        nn.Module.__init__(self)

        self.action_space = action_space
        GaussianMixin.__init__(self)

        self.net_container = nn.Sequential(
            nn.Linear(observation_space, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU()
        )

        self.policy_layer = nn.Linear(64, action_space)
        self.log_std_parameter = nn.Parameter(torch.zeros(action_space))

    def compute(self, inputs, role="policy"):
        x = self.net_container(inputs)
        return self.policy_layer(x), self.log_std_parameter



class PolicyControllerSKRL:
    """
    A controller that loads and executes a skrl-based Gaussian policy from a file.
    """

    def __init__(self) -> None:
        self.policy = None
        self.policy_env_params = None
        self._dt = None
        self._decimation = None
        self.render_interval = None

    def load_policy(self, policy_file_path: str, policy_env_path: str) -> None:
        """
        Loads policy from a skrl PPO-trained .pt checkpoint file and environment parameters from env.yaml.
        """
        print("\n=== Policy Loading ===")
        print(f"{'Model path:':<18} {policy_file_path}")
        print(f"{'Environment path:':<18} {policy_env_path}")

        # YAML betöltése – a környezet paramétereihez
        with open(policy_env_path, 'r') as f:
            self.policy_env_params = yaml.unsafe_load(f)

        obs_dim = 43
        act_dim = 12

        # Policy példányosítás
        self.policy = SKRLPolicy(observation_space=obs_dim, action_space=act_dim)

        # Checkpoint betöltése
        ckpt = torch.load(policy_file_path)
        self.policy.load_state_dict(ckpt["policy"], strict=False)  # csak a policy rész
        self.policy.eval()

        self._decimation, self._dt, self.render_interval = get_physics_properties(self.policy_env_params)
        self._max_effort, self._max_vel, self._stiffness, self._damping, self.default_pos, self.default_vel = get_robot_joint_properties(
            self.policy_env_params, self.dof_names
        )


        print("\n--- Policy configuration ---")
        print(f"{'Observation dim:':<18} {obs_dim}")
        print(f"{'Action dim:':<18} {act_dim}")
        print(f"{'Policy type:':<18} GaussianPolicy (skrl)")
        print(f"{'Decimation:':<18} {self._decimation}")
        print(f"{'Timestep (dt):':<18} {self._dt}")

        print("\n=== Policy Loaded ===\n")

    def _compute_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Computes the action from the observation using the loaded policy.

        Args:
            obs (np.ndarray): The observation of shape (41,)

        Returns:
            np.ndarray: The action of shape (7,)
        """
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).view(1, -1).float()
            action, _ = self.policy.compute(obs_tensor, role="policy")
            return action.detach().cpu().numpy().flatten()

    def _compute_observation(self) -> NotImplementedError:
        raise NotImplementedError(
            "Implement observation processing to produce a (43,) np.ndarray input vector"
        )

    def forward(self) -> NotImplementedError:
        raise NotImplementedError(
            "Implement forward pass that obtains observation, computes and applies action"
        )
