import numpy as np
import torch

pose_dir = "/home2/zxp/Projects/Juicer_ws/imitation-juicer/my_file/parts_poses.txt"
pose = np.loadtxt(pose_dir)
print(pose[0])