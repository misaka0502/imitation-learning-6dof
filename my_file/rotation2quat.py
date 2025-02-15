from scipy.spatial.transform import Rotation as R
import numpy as np
import os

quat_dir = "/home2/zxp/Projects/Juicer_ws/imitation-juicer/my_file/parts_poses.txt"
rotation_dir = "/home2/zxp/Projects/FoundationPose/debug/ob_in_cam/"
output_dir = "/home2/zxp/Projects/FoundationPose/debug/ob_in_cam_transformed/quat.txt"

T = np.loadtxt(rotation_dir + "0000000000000000000.txt")
rotation_matrix = T[:3, :3]
pos = T[:3, 3]
rotation = R.from_matrix(rotation_matrix)
quat = rotation.as_quat()
result = np.concatenate((pos, quat))
print(rotation_matrix)
print(pos)
print(quat)
print(result)

# quat_list = []
# id = 0
# for filename in os.listdir(rotation_dir):
#     T = np.loadtxt(rotation_dir + filename)
#     pos = T[:3, 3]
#     rotation_matrix = T[:3, :3]
#     rotation = R.from_matrix(rotation_matrix)
#     quat = rotation.as_quat()
#     quat_list.append(np.concatenate((pos, quat)))
#     id += 1
#     if id == 9:
#         break
# np.savetxt(output_dir, quat_list)