from typing import Tuple
from collections import deque
import torch
import torch.nn as nn
from src.dataset.normalizer import Normalizer

from ipdb import set_trace as bp  # noqa
from src.common.geometry import proprioceptive_quat_to_6d_rotation
from src.common.vision import FrontCameraTransform, WristCameraTransform

from pytorch3d.transforms import (
    rotation_6d_to_matrix,
    matrix_to_euler_angles,
    matrix_to_rotation_6d,
    euler_angles_to_matrix,
)

import time

# Update the PostInitCaller to be compatible
class PostInitCaller(type(torch.nn.Module)):
    def __call__(cls, *args, **kwargs):
        obj = super().__call__(*args, **kwargs)
        obj.__post_init__(*args, **kwargs)
        return obj


class Actor(torch.nn.Module, metaclass=PostInitCaller):
    obs_horizon: int
    action_horizon: int
    normalizer: Normalizer

    # Regularization
    feature_noise: bool = False
    feature_dropout: bool = False
    feature_layernorm: bool = False
    state_noise: bool = False

    encoding_dim: int
    augment_image: bool = False

    camera1_transform = WristCameraTransform(mode="eval")
    camera2_transform = FrontCameraTransform(mode="eval")

    encoder1: nn.Module
    encoder1_proj: nn.Module

    encoder2: nn.Module
    encoder2_proj: nn.Module

    # 在这里修改位姿估计的长度，由于用的是四元数表示，所以一个零件有7个数值
    # parts_poses_dim: int = 7
    parts_poses_dim: int = 35

    def __post_init__(self, *args, **kwargs):
        assert self.encoder1 is not None, "encoder1 is not defined"
        assert self.encoder2 is not None, "encoder2 is not defined"

        if self.feature_dropout:
            self.dropout = nn.Dropout(p=self.feature_dropout)

        if self.feature_layernorm:
            self.layernorm1 = nn.LayerNorm(self.encoding_dim).to(self.device)
            self.layernorm2 = nn.LayerNorm(self.encoding_dim).to(self.device)
            self.layernorm3 = nn.LayerNorm(self.parts_poses_dim).to(self.device)

        self.print_model_params()

    def print_model_params(self: torch.nn.Module):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total_params / 1_000_000:.2f}M")

        for name, submodule in self.named_children():
            params = sum(p.numel() for p in submodule.parameters())
            print(f"{name}: {params / 1_000_000:.2f}M parameters")

    def _normalized_obs(self, obs: deque, flatten: bool = True):
        """
        Normalize the observations

        Takes in a deque of observations and normalizes them
        And concatenates them into a single tensor of shape (n_envs, obs_horizon * obs_dim)
        """
        # Convert robot_state from obs_horizon x (n_envs, 14) -> (n_envs, obs_horizon, 14)
        # print(f'obs len: {len(obs)}')
        # print(f'o["robot_state"].shape: {obs[0]["robot_state"].shape}')
        robot_state = torch.cat([o["robot_state"].unsqueeze(1) for o in obs], dim=1)
        # print(f'robot_state.shape: {robot_state.shape}')

        # Convert the robot_state to use rot_6d instead of quaternion
        robot_state = proprioceptive_quat_to_6d_rotation(robot_state)
        # print(f'robot_state.shape_rot6d: {robot_state.shape}')

        # Normalize the robot_state
        nrobot_state = self.normalizer(robot_state, "robot_state", forward=True)

        B = nrobot_state.shape[0]

        # from furniture_bench.perception.image_utils import resize, resize_crop
        # Get size of the image
        img_size = obs[0]["color_image1"].shape[-3:]

        # Images come in as obs_horizon x (n_envs, 224, 224, 3) concatenate to (n_envs * obs_horizon, 224, 224, 3)
        image1 = torch.cat(
            [o["color_image1"].unsqueeze(1) for o in obs], dim=1
        ).reshape(B * self.obs_horizon, *img_size)
        image2 = torch.cat(
            [o["color_image2"].unsqueeze(1) for o in obs], dim=1
        ).reshape(B * self.obs_horizon, *img_size)

        # Move the channel to the front (B * obs_horizon, H, W, C) -> (B * obs_horizon, C, H, W)
        image1 = image1.permute(0, 3, 1, 2)
        image2 = image2.permute(0, 3, 1, 2)

        # Apply the transforms to resize the images to 224x224, (B * obs_horizon, C, 224, 224)
        image1: torch.Tensor = self.camera1_transform(image1)
        image2: torch.Tensor = self.camera2_transform(image2)

        # Place the channel back to the end (B * obs_horizon, C, 224, 224) -> (B * obs_horizon, 224, 224, C)
        image1 = image1.permute(0, 2, 3, 1)
        image2 = image2.permute(0, 2, 3, 1)

        # Encode the images and reshape back to (B, obs_horizon, -1)
        feature1: torch.Tensor = self.encoder1_proj(self.encoder1(image1)).reshape(
            B, self.obs_horizon, -1
        )
        feature2: torch.Tensor = self.encoder2_proj(self.encoder2(image2)).reshape(
            B, self.obs_horizon, -1
        )

        # Apply the regularization to the features
        feature1, feature2 = self.regularize_features(feature1, feature2)
        # 只取最后一个零件的位姿（即画面中最右边的桌腿）
        # parts_poses: torch.Tensor = torch.cat([o["parts_poses"].unsqueeze(1)[:, :, -7:] for o in obs], dim=1)
        # 取所有零件的位姿
        parts_poses: torch.Tensor = torch.cat([o["parts_poses"].unsqueeze(1) for o in obs], dim=1)
        # Reshape concatenate the features
        nobs = torch.cat([nrobot_state, feature1, feature2, parts_poses], dim=-1)
        # nobs = torch.cat([nrobot_state, feature1, feature2], dim=-1)

        if flatten:
            # (n_envs, obs_horizon, obs_dim) --> (n_envs, obs_horizon * obs_dim)
            nobs = nobs.flatten(start_dim=1)

        return nobs

    def regularize_features(
        self, feature1: torch.Tensor, feature2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.feature_layernorm:
            feature1 = self.layernorm1(feature1)
            feature2 = self.layernorm2(feature2)

        if self.training and self.feature_dropout:
            print("[WARNING] Make sure this is disabled during evaluation")
            feature1 = self.dropout(feature1)
            feature2 = self.dropout(feature2)

        if self.training and self.feature_noise:
            print("[WARNING] Make sure this is disabled during evaluation")
            # Add noise to the features
            feature1 = feature1 + torch.randn_like(feature1) * self.feature_noise
            feature2 = feature2 + torch.randn_like(feature2) * self.feature_noise

        return feature1, feature2

    def _training_obs(self, batch, flatten: bool = True):
        # The robot state is already normalized in the dataset
        nrobot_state = batch["robot_state"]
        B = nrobot_state.shape[0]

        # Check if we're in training mode and we want to add noise to the robot state

        if self.training and self.state_noise:
            # Add noise to the robot state akin to Ke et al., “Grasping with Chopsticks.”
            # Extract only the current position and orientation (x, y, x and 6D rotation)
            pos = nrobot_state[:, :, :3]
            rot_mat = rotation_6d_to_matrix(nrobot_state[:, :, 3:9])
            rot = matrix_to_euler_angles(rot_mat, "XYZ")

            # Add noise to the position with variance of of 1 cm
            pos = pos + torch.randn_like(pos) * 0.01

            # Sample random rotations in x, y, z Euler angles with variance of 0.1 rad
            d_rot = torch.randn_like(rot) * 0.1

            # Apply the noise rotation to the current rotation
            rot = matrix_to_rotation_6d(rot_mat @ euler_angles_to_matrix(d_rot, "XYZ"))

            # In 20% of observations, we now the noised position and rotation to the robot state
            mask = torch.rand(B) < 0.2
            nrobot_state[mask, :, :3] = pos[mask]
            nrobot_state[mask, :, 3:9] = rot[mask]

        if self.observation_type == "image":
            image1: torch.Tensor = batch["color_image1"]
            image2: torch.Tensor = batch["color_image2"]
            # parts_poses: (B, obs_horizon, parts*7)
            # 每个部件位姿信息由7个数值组成，前三个数值代表部件在空间中的Position，单位为m；后四个数值代表部件的姿态，以四元数形式表示
            # todo：比较部件姿态用四元数还是rpy好
            part_poses: torch.Tensor = batch["parts_poses"][:, :, -7:]
            # part_poses: torch.Tensor = batch["parts_poses"]
            # print(f"parts_poses_batch_size: {parts_poses.shape}")
            # print(f"image_batch_size: {image1.shape}")
            # time.sleep(50)
            # Reshape the images to (B * obs_horizon, H, W, C) for the encoder
            image1 = image1.reshape(B * self.obs_horizon, *image1.shape[-3:])
            image2 = image2.reshape(B * self.obs_horizon, *image2.shape[-3:])

            # Encode images and reshape back to (B, obs_horizon, encoding_dim)
            feature1 = self.encoder1_proj(self.encoder1(image1)).reshape(
                B, self.obs_horizon, -1
            )
            feature2 = self.encoder2_proj(self.encoder2(image2)).reshape(
                B, self.obs_horizon, -1
            )
            # print(nrobot_state.shape)
            # print(feature1.shape)
            # print(part_poses.shape)
            # time.sleep(10000)
            # Combine the robot_state and image features, (B, obs_horizon, obs_dim)
            nobs = torch.cat([nrobot_state, feature1, feature2], dim=-1)
            # Combine the parts poses, (B, obs_horizon, obs_dim)
            nobs = torch.cat([nobs, part_poses], dim=-1)
            nobs = nobs.flatten(start_dim=1) if flatten else nobs

        elif self.observation_type == "feature":
            # All observations already normalized in the dataset
            feature1 = self.encoder1_proj(batch["feature1"])
            feature2 = self.encoder2_proj(batch["feature2"])

            # Apply the regularization to the features
            feature1, feature2 = self.regularize_features(feature1, feature2)

            # Combine the robot_state and image features, (B, obs_horizon, obs_dim)
            nobs = torch.cat([nrobot_state, feature1, feature2], dim=-1)
            if flatten:
                # (B, obs_horizon, obs_dim) --> (B, obs_horizon * obs_dim)
                nobs = nobs.flatten(start_dim=1)

        else:
            raise ValueError(f"Invalid observation type: {self.observation_type}")

        return nobs

    def train_mode(self):
        """
        Set models to train mode
        """
        self.train()
        if self.augment_image:
            self.camera1_transform.train()
            self.camera2_transform.train()
        else:
            self.camera1_transform.eval()
            self.camera2_transform.eval()

    def eval_mode(self):
        """
        Set models to eval mode
        """
        self.eval()
        self.camera2_transform.eval()

    def action(self, obs: deque) -> torch.Tensor:
        """
        Given a deque of observations, predict the action

        The action is predicted for the next step for all the environments (n_envs, action_dim)
        """
        raise NotImplementedError

    def compute_loss(self, batch):
        raise NotImplementedError

    def set_task(self, task):
        """
        Set the task for the actor
        """
        pass
