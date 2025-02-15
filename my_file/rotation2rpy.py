from scipy.spatial.transform import Rotation as R
import numpy as np
import os

quat_dir = "/home2/zxp/Projects/Juicer_ws/imitation-juicer/my_file/parts_poses.txt"
root_dir = "/home2/zxp/Projects/Juicer_ws/imitation-juicer/foundationpose/debug/run_sim_env/2025-01-14_17-31-54"
rotation_dir = f"{root_dir}/rollouts_ob/begin_pose.txt"
output_dir = f"{root_dir}/rollouts_ob/begin_rpy.txt"

# euler_list = []
# id = 0
# for filename in os.listdir(rotation_dir):
#     T = np.loadtxt(rotation_dir + filename)
#     pos = T[:3, 3]
#     rotation_matrix = T[:3, :3]
#     rotation = R.from_matrix(rotation_matrix)
#     euler = rotation.as_euler('xyz', degrees=True)
#     euler_list.append(np.concatenate((pos, euler)))
#     id += 1
#     if id == 9:
#         break

T = np.loadtxt(rotation_dir)
pos = T[:3, 3]
rotation_matrix = T[:3, :3]
rotation = R.from_matrix(rotation_matrix)
euler = rotation.as_euler('xyz', degrees=True)

np.savetxt(output_dir, euler)