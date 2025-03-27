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

def april_to_sim_mat():
        return franka_from_origin_mat @ base_tag_from_robot_mat

def sim_coord_to_april_coord(sim_coord_mat):
    return sim_to_april_mat() @ sim_coord_mat

def april_coord_to_sim_coord(april_coord_mat):
        """Converts AprilTag coordinate to simulator base_tag coordinate."""
        return april_to_sim_mat() @ april_coord_mat

def cam_coord_to_april_coord(pose_est_cam, cam_pos, cam_target, revise=False):
    cam_pos = np.array(cam_pos)
    cam_target = np.array(cam_target)
    z_camera = (cam_target - cam_pos) / np.linalg.norm(cam_target - cam_pos)
    up_axis = np.array([0, 0, 1])  # Assuming Z is the up axis
    x_camera = -np.cross(up_axis, z_camera)
    x_camera /= np.linalg.norm(x_camera)
    y_camera = np.cross(z_camera, x_camera)
    R_camera_sim = np.vstack([x_camera, y_camera, z_camera]).T
    T_camera_sim = np.eye(4)
    T_camera_sim[:3, :3] = R_camera_sim
    T_camera_sim[:3, 3] = cam_pos
    pos_est_sim = T_camera_sim @ pose_est_cam
    if revise:
        r_y = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
        pos_est_sim[:3, :3] = pos_est_sim[:3, :3] @ r_y
    pose_est_april_coord = np.concatenate(
        [
            *C.mat2pose(
                sim_coord_to_april_coord(
                    torch.tensor(pos_est_sim, device="cpu", dtype=torch.float64)
                )
            )
        ]
    )
    return pose_est_april_coord

def april_coord_to_cam_coord(pose_est_april, cam_pos, cam_target):
    cam_pos = np.array(cam_pos)
    cam_target = np.array(cam_target)
    z_camera = (cam_target - cam_pos) / np.linalg.norm(cam_target - cam_pos)
    up_axis = np.array([0, 0, 1])  # Assuming Z is the up axis
    x_camera = -np.cross(up_axis, z_camera)
    x_camera /= np.linalg.norm(x_camera)
    y_camera = np.cross(z_camera, x_camera)
    R_camera_sim = np.vstack([x_camera, y_camera, z_camera]).T
    T_camera_sim = np.eye(4)
    T_camera_sim[:3, :3] = R_camera_sim
    T_camera_sim[:3, 3] = cam_pos
    pose_est_april = torch.tensor(pose_est_april, device="cpu", dtype=torch.float64)
    pose_est_april_coord = april_coord_to_sim_coord(
                    C.pose2mat(pose_est_april[:3], pose_est_april[-4:],  device="cpu").numpy()
                )
    pose_est_april_coord = np.linalg.inv(T_camera_sim) @ pose_est_april_coord
    return pose_est_april_coord

leg_pose_foundationpose_path = "/home2/zxp/Projects/Juicer_ws/imitation-juicer/foundationpose/debug/2025-03-02_22-49-35/rollouts_ob/top"
leg_pose_april_path = "/home2/zxp/Projects/Juicer_ws/imitation-juicer/foundationpose/debug/2025-03-02_22-49-35/top_poses.txt"
leg_pose_april = np.loadtxt(leg_pose_april_path)
error_t = []
error_r = []
for i in range(len(leg_pose_april)):
    leg_pose_foundationpose = np.loadtxt(f"{leg_pose_foundationpose_path}/{i:4d}.txt")
    # leg_pose_foundationpose = cam_coord_to_april_coord(leg_pose_foundationpose, [0.90, -0.00, 0.65], [-1, -0.00, 0.3])
    leg_pose_foundationpose = cam_coord_to_april_coord(leg_pose_foundationpose, [0.3, -0.65, 0.8], [0.3, 0.8, 0.00], revise=True)
    print(leg_pose_foundationpose)
    print(leg_pose_april[i])
    time.sleep(10000)
    error_t.append(np.linalg.norm(leg_pose_foundationpose[:3] - leg_pose_april[i][:3]))

    rotation_april = C.pose2mat(torch.tensor(leg_pose_april[i][:3]), torch.tensor(leg_pose_april[i][-4:]), device="cpu")[:3, :3].numpy()
    rotation_foundationpose_april = C.pose2mat(torch.tensor(leg_pose_foundationpose[:3]), torch.tensor(leg_pose_foundationpose[-4:]), device="cpu")[:3, :3].numpy()
    cos_theta = (np.trace(np.dot(rotation_april.T, rotation_foundationpose_april)) - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 避免数值误差
    error_r.append(np.arccos(cos_theta) * 180 / np.pi)  # 转换为角度

te = np.mean(error_t)
re = np.mean(error_r)
print(te)
print(re)

# leg_pose_foundationpose_path = "/home2/zxp/Projects/Juicer_ws/imitation-juicer/foundationpose/debug/2025-03-02_22-49-35/rollouts_ob/top"
# leg_pose_april_path = "/home2/zxp/Projects/Juicer_ws/imitation-juicer/foundationpose/debug/2025-03-02_22-49-35/top_poses.txt"
# leg_pose_april = np.loadtxt(leg_pose_april_path)
# for i in range(1):
#     leg_pose_foundationpose = np.loadtxt(f"{leg_pose_foundationpose_path}/{i:4d}.txt")
#     leg_pose_cam = april_coord_to_cam_coord(leg_pose_april[i], [0.90, -0.00, 0.65], [-1, -0.00, 0.3])
#     print(leg_pose_foundationpose)
#     print(leg_pose_cam)
    