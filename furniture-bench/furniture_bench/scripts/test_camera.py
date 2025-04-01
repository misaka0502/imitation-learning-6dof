import isaacgym
from isaacgym import gymapi, gymtorch
# import argparse
# import pickle

# import furniture_bench

# import gym
# import cv2
import torch
import numpy as np
from furniture_bench.sim_config import sim_config

isaac_gym = gymapi.acquire_gym()
sim = isaac_gym.create_sim(
    8,
    8,
    gymapi.SimType.SIM_PHYSX,
    sim_config["sim_params"],
)
num_per_row = int(np.sqrt(1))
spacing = 1.0
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)
env = isaac_gym.create_env(sim, env_lower, env_upper, num_per_row)
camera_cfg = gymapi.CameraProperties()
# camera_cfg.enable_tensors = True
camera_cfg.width = 1920
camera_cfg.height = 1080
camera_cfg.near_plane = 0.001
camera_cfg.far_plane = 2.0
camera_cfg.horizontal_fov = 69.4
camera_cfg = camera_cfg

camera_handle = isaac_gym.create_camera_sensor(env, camera_cfg)
cam_pos = gymapi.Vec3(0.90, -0.00, 0.65)
# cam_pos = gymapi.Vec3(0.90, -0.00, 0.80)
cam_target = gymapi.Vec3(-1, -0.00, 0.3)
isaac_gym.set_camera_location(camera_handle, env, cam_pos, cam_target)
front_cam_pos = np.array([cam_pos.x, cam_pos.y, cam_pos.z])
front_cam_target = np.array(
    [cam_target.x, cam_target.y, cam_target.z]
)

img = isaac_gym.get_camera_image(
    sim, env, camera_handle, gymapi.IMAGE_COLOR
)
img = torch.tensor(img)
print(type(img))