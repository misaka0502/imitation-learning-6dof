import isaacgym
from isaacgym import gymapi, gymtorch
import furniture_bench.controllers.control_utils as C
import torch
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from furniture_bench.config import config
from furniture_bench.utils.pose import get_mat

ROBOT_HEIGHT = 0.015
table_pos = gymapi.Vec3(0.8, 0.8, 0.4)
table_half_width = 0.015
table_surface_z = table_pos.z + table_half_width
franka_pose = gymapi.Transform()
franka_pose.p = gymapi.Vec3(
    0.5 * -table_pos.x + 0.1, 0, table_surface_z + ROBOT_HEIGHT
)
base_tag_from_robot_mat = config["robot"]["tag_base_from_robot_base"]
franka_from_origin_mat = get_mat(
    [franka_pose.p.x, franka_pose.p.y, franka_pose.p.z],
    [0, 0, 0],
)
def sim_to_april_mat():
    return torch.tensor(
        np.linalg.inv(base_tag_from_robot_mat)
        @ np.linalg.inv(franka_from_origin_mat),
        device="cpu", dtype=torch.float64
    )
    
def sim_coord_to_april_coord(sim_coord_mat):
    print(sim_to_april_mat().dtype, sim_coord_mat.dtype)
    return sim_to_april_mat() @ sim_coord_mat

def april_to_sim_mat():
    return franka_from_origin_mat @ base_tag_from_robot_mat

def april_coord_to_sim_coord(april_coord_mat):
    """Converts AprilTag coordinate to simulator base_tag coordinate."""
    return april_to_sim_mat() @ april_coord_mat

root_dir = "/home2/zxp/Projects/Juicer_ws/imitation-juicer/foundationpose/debug/run_sim_env/2025-01-14_17-30-58/rollouts_ob"
rotation_dir = f"{root_dir}/begin_pose.txt"
output_dir = f"{root_dir}/begin_pose_sim_coord.txt"

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

# Step 3: Object relative to camera transformation matrix
T_object_camera = np.loadtxt(rotation_dir)
T_object_camera = np.array([-1.992384046316146851e-01, -6.769002228975296021e-02, 8.426325917243957520e-01, 1])
# Step 4: Compute object relative to environment transformation matrix
# T_object_sim = np.dot(T_camera_sim, T_object_camera)
T_object_sim = T_camera_sim @ T_object_camera
# T_object_sim = torch.from_numpy(T_object_sim, )

# pose_aprilatg_coord = torch.concat(
#     [
#         *C.mat2pose(
#             sim_coord_to_april_coord(
#                 T_object_camera
#             )
#         )
#     ]
# )
# quat = pose_aprilatg_coord[-4:]
# rotation = R.from_quat(quat)
# euler = rotation.as_euler('xyz', degrees=True)

# rotation = R.from_matrix(T_object_sim[:3, :3])
# quat = rotation.as_quat()
# pose = np.concatenate([T_object_sim[:3, 3], quat])
# print(pose.shape)
# np.savetxt(output_dir, pose)
# print(T_camera_sim)
print(T_object_sim)
