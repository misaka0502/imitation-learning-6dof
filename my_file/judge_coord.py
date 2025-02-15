import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import time
from furniture_bench.config import config
from furniture_bench.utils.pose import get_mat
import furniture_bench.controllers.control_utils as C
import torch
# transform_dir = "/home2/zxp/Projects/Juicer_ws/imitation-juicer/foundationpose/debug/2025-01-13_15-21-47/rollouts_ob/0000000000000000000_begin.txt"
# quat_dir = "/home2/zxp/Projects/Juicer_ws/imitation-juicer/foundationpose/debug/2025-01-13_15-21-47/leg_poses.txt"

# T = np.loadtxt(transform_dir)
# rotation = T[:3, :3]

# det = np.linalg.det(rotation)
# if np.isclose(det, 1.0):
#     print("这是一个右手系旋转矩阵")
# elif np.isclose(det, -1.0):
#     print("这是一个左手系旋转矩阵")
# else:
#     print("矩阵可能有误，不是一个有效的旋转矩阵")

# quat = np.loadtxt(quat_dir)[0][-4:]
# rotation = R.from_quat(quat).as_matrix()
# det = np.linalg.det(rotation)
# if np.isclose(det, 1.0):
#     print("这是一个右手系旋转矩阵")
# elif np.isclose(det, -1.0):
#     print("这是一个左手系旋转矩阵")
# else:
#     print("矩阵可能有误，不是一个有效的旋转矩阵")

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

root_dir = "/home2/zxp/Projects/Juicer_ws/imitation-juicer/foundationpose/debug/run_sim_env/2025-01-16_01-56-32"
poses_est_dir = f"{root_dir}/ob_in_cam/"
pose_aprilatg_dir = f"{root_dir}/ob_in_cam_apriltag_sim/"

poses_est = []
pose_apriltag = []
for filename in sorted(os.listdir(poses_est_dir), key=lambda x: int(x.split('.')[0])):
    poses_est.append(np.loadtxt(poses_est_dir + filename))
for filename in sorted(os.listdir(pose_aprilatg_dir), key=lambda x: int(x.split('.')[0])):
    pose_apriltag.append(np.loadtxt(pose_aprilatg_dir + filename))

cam_pos = np.array([0.90, -0.00, 0.65])
cam_target = np.array([-1, -0.00, 0.3])
# Step 1: Compute camera rotation matrix
z_camera = (cam_target - cam_pos) / np.linalg.norm(cam_target - cam_pos)
up_axis = np.array([0, 0, 1])  # Assuming Z is the up axis
x_camera = np.cross(up_axis, z_camera)
x_camera /= np.linalg.norm(x_camera)
y_camera = np.cross(z_camera, x_camera)
R_camera_sim = np.vstack([x_camera, y_camera, z_camera]).T

# Step 2: Camera transformation matrix
T_camera_sim = np.eye(4)
T_camera_sim[:3, :3] = R_camera_sim
T_camera_sim[:3, 3] = cam_pos

# for i in range(len(poses_est)):
#     # pos_est = poses_est[i][:, 3].transpose()
#     # pos_est = pos_est * np.array([-1, -1, 1, 1])
#     # pos_est_sim = T_camera_sim @ pos_est
#     pose_est = poses_est[i]
#     # print(pose_est)
#     pose_est = pose_est * np.array([-1, -1, 1, 1]).reshape(4, -1)
#     # print(pose_est)
#     # time.sleep(100)
#     pos_est_sim = T_camera_sim @ pose_est

#     pose_est_aprilatg_coord = np.concatenate(
#         [
#             *C.mat2pose(
#                 sim_coord_to_april_coord(
#                     torch.tensor(pos_est_sim, device="cpu", dtype=torch.float64)
#                 )
#             )
#         ]
#     )
#     print(pose_est_aprilatg_coord, pose_apriltag[i])
#     time.sleep(100)

#     # if not np.isclose(pos_est_sim[:3], pose_apriltag[i][:3], atol=1e-2, rtol=1e-2).all():
#     #     print("寄", i)
#     #     print(pos_est_sim[:3], pose_apriltag[i][:3])
#         # time.sleep(1000)
i=50
pose_est = poses_est[i]
# print(pose_est)
pose_est = pose_est * np.array([-1, -1, 1, 1]).reshape(4, -1)
# print(pose_est)
# time.sleep(100)
pos_est_sim = T_camera_sim @ pose_est

pose_est_aprilatg_coord = np.concatenate(
    [
        *C.mat2pose(
            sim_coord_to_april_coord(
                torch.tensor(pos_est_sim, device="cpu", dtype=torch.float64)
            )
        )
    ]
)
print(pose_est_aprilatg_coord, pose_apriltag[i])
time.sleep(100)