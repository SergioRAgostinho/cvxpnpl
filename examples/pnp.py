import numpy as np
from cvxpnpl import pnp

# fix seed to allow for reproducible results
np.random.seed(0)
np.random.seed(42)

# instantiate a couple of points centered around the origin
pts = 0.6 * (np.random.random((6, 3)) - 0.5)

# Made up projective matrix
K = np.array([[160, 0, 320], [0, 120, 240], [0, 0, 1]])

# A pose
R_gt = np.array(
    [
        [-0.48048015, 0.1391384, -0.86589799],
        [-0.0333282, -0.98951829, -0.14050899],
        [-0.8763721, -0.03865296, 0.48008113],
    ]
)
t_gt = np.array([-0.10266772, 0.25450789, 1.70391109])

# Project points to 2D
pts_2d = (pts @ R_gt.T + t_gt) @ K.T
pts_2d = (pts_2d / pts_2d[:, -1, None])[:, :-1]

# Compute pose candidates. the problem is not minimal so only one
# will be provided
poses = pnp(pts_2d=pts_2d, pts_3d=pts, K=K)
R, t = poses[0]

print("Nr of possible poses:", len(poses))
print("R (ground truth):", R_gt, "R (estimate):", R, sep="\n")
print("t (ground truth):", t_gt)
print("t (estimate):", t)
