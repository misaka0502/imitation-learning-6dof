from scipy.spatial.transform import Rotation as R
import numpy as np
import os

root_dir = "/home2/zxp/Projects/Juicer_ws/imitation-juicer/foundationpose/debug/run_sim_env/2025-01-14_17-31-54"
quat_dir = f"{root_dir}/begin_parts_poses_leg.txt"
output_dir = f"{root_dir}/begin_parts_rpy_leg.txt"

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

quats = np.loadtxt(quat_dir)
quat = quats[-4:]
print(quat)
rotation = R.from_quat(quat)
euler = rotation.as_euler('xyz', degrees=True)

np.savetxt(output_dir, euler)