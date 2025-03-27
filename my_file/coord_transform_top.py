import torch
import numpy as np
import collections
from furniture_bench.config import config
from furniture_bench.utils.pose import get_mat
import furniture_bench.controllers.control_utils as C
import time

ROBOT_HEIGHT = 0.015
table_pos = np.array([0.8, 0.8, 0.4])
table_half_width = 0.015
table_surface_z = table_pos[2] + table_half_width
franka_pose = np.array(
    [0.5 * -table_pos[0] + 0.1, 0, table_surface_z + ROBOT_HEIGHT]
)
base_tag_from_robot_mat = config["robot"]["tag_base_from_robot_base"]
franka_from_origin_mat = get_mat(
    [franka_pose[0], franka_pose[1], franka_pose[2]],
    [0, 0, 0],
)

def sim_to_april_mat():
    return torch.tensor(
        np.linalg.inv(base_tag_from_robot_mat) @ np.linalg.inv(franka_from_origin_mat),
        device="cpu", dtype=torch.float64
    )

def sim_coord_to_april_coord(sim_coord_mat):
    return sim_to_april_mat() @ sim_coord_mat
# 你已有的相机位置和目标点
front_cam_pos = np.array([0.3, -0.65, 0.8])
front_cam_target = np.array([0.3, 0.8, 0.00])

# 假设你通过六维位姿估计获得了物体相对于相机的位姿
# 例如：object_pose_camera = [R|t]，其中R是旋转矩阵，t是平移向量

# 步骤1：构建相机到环境的变换矩阵
def transform_object_camera_to_world(object_pose_camera, cam_pos, cam_target):
    # Step 1: Compute camera coordinate axes
    z_camera = (cam_target - cam_pos) / np.linalg.norm(cam_target - cam_pos)  # Forward
    up_axis = np.array([0, 0, 1])  # Environment's up direction
    x_camera = np.cross(z_camera, up_axis)  # Right
    x_camera /= np.linalg.norm(x_camera)
    y_camera = np.cross(z_camera, x_camera)  # Down

    # Step 2: Construct camera rotation matrix
    R_camera_env = np.vstack([x_camera, y_camera, z_camera]).T

    # Step 3: Construct transformation matrices
    T_camera_env = np.eye(4)
    T_camera_env[:3, :3] = R_camera_env
    T_camera_env[:3, 3] = cam_pos

    T_object_camera = np.eye(4)
    T_object_camera[:3, :3] = object_pose_camera[:3, :3]
    T_object_camera[:3, 3] = object_pose_camera[:3, 3]

    # Step 4: Compute object's transformation in the environment
    T_object_env = np.dot(T_camera_env, T_object_camera)

    return T_object_env

leg_pose_foundationpose_path = "/home2/zxp/Projects/Juicer_ws/imitation-juicer/foundationpose/debug/2025-03-02_22-49-35/rollouts_ob/top"
leg_pose_april_path = "/home2/zxp/Projects/Juicer_ws/imitation-juicer/foundationpose/debug/2025-03-02_22-49-35/top_poses.txt"
leg_pose_april = np.loadtxt(leg_pose_april_path)
for i in range(1):
    leg_pose_foundationpose = np.loadtxt(f"{leg_pose_foundationpose_path}/{i:4d}.txt")
    leg_pose_foundationpose = transform_object_camera_to_world(leg_pose_foundationpose, front_cam_pos, front_cam_target)
    leg_pose_foundationpose = torch.tensor(leg_pose_foundationpose, device="cpu")
    pose_est_april_coord = np.concatenate(
        [
            *C.mat2pose(
                sim_coord_to_april_coord(
                    torch.tensor(leg_pose_foundationpose, device="cpu", dtype=torch.float64)
                )
            )
        ]
    )
    print(pose_est_april_coord)
    print(leg_pose_april[i])