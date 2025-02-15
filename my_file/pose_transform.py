from scipy.spatial.transform import Rotation as R
import numpy as np
from math import cos, sin

transrorm_p = np.array([0.9, 0.0, 0.65])
# transrorm_p = np.array([0.65, 0.0, 0.9])
rotation = R.from_quat(np.array([-0.0, 0.995855, 0.0, 0.090958]))
# rotation = R.from_quat(np.array([0.453, -0.543, -0.543, 0.453]))
transform_r = R.as_matrix(rotation)
transform = np.eye(4)
transform[:3, :3] = transform_r
transform[:3, 3] = transrorm_p

# transform = np.array(
#     [
#         [0.0, -1.0, 0.0, 0.9],
#         [-0.1812, 0.0, 0.9834, 0.0],
#         [-0.983, 0.0, -0.181, 0.65],
#         [0.0, 0.0, 0.0, 1.0]
#     ]
# )

# coord_cam = np.array([1.992384046316146851e-01, 6.769002228975296021e-02, 8.426325917243957520e-01, 1])
# coord_cam = np.array([-6.769002228975296021e-02, 8.426325917243957520e-01, -1.992384046316146851e-01, 1])
coord_cam = np.array([-8.426325917243957520e-01, -1.992384046316146851e-01, 6.769002228975296021e-02, 1])
# coord_sim = np.linalg.inv(transform) @ coord_cam
coord_sim = np.linalg.inv(transform) @ coord_cam
print(coord_sim)
# print(transform)
print(np.linalg.inv(transform))
