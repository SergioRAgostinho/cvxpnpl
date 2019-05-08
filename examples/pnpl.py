import numpy as np
from cvxpnpl import pnpl

# fix seed to allow for reproducible results
np.random.seed(0)
np.random.seed(42)

# instantiate a couple of points centered around the origin
pts = 0.6 * (np.random.random((4, 3)) - 0.5)

# 3D lines are parameterized as pts and direction stacked into a tuple
# instantiate a couple of points centered around the origin
pts_l = 0.6 * (np.random.random((4, 3)) - 0.5)
# generate normalized directions
directions = 2 * (np.random.random((4, 3)) - 0.5)
directions /= np.linalg.norm(directions, axis=1)[:, None]

line_3d = (pts_l, directions)

# Made up projective matrix
K = np.array([[160, 0, 320], [0, 120, 240], [0, 0, 1]])

# A pose
R_gt = np.array(
    [
        [0.89802142, -0.41500101, 0.14605372],
        [0.24509948, 0.7476071, 0.61725997],
        [-0.36535431, -0.51851499, 0.77308372],
    ]
)
t_gt = np.array([-0.0767557, 0.13917375, 1.9708239])

# sample 2 points from each line and stack all
pts_ls = np.hstack((pts_l, pts_l + directions)).reshape((-1, 3))
pts_all = np.vstack((pts, pts_ls))

# Project everything to 2D
pts_all_2d = (pts_all @ R_gt.T + t_gt) @ K.T
pts_all_2d = (pts_all_2d / pts_all_2d[:, -1, None])[:, :-1]

pts_2d = pts_all_2d[:4]
line_2d = pts_all_2d[4:].reshape((-1, 2, 2))

# Compute pose candidates. the problem is not minimal so only one
# will be provided
poses = pnpl(pts_2d=pts_2d, line_2d=line_2d, pts_3d=pts, line_3d=line_3d, K=K)
R, t = poses[0]

print("Nr of possible poses:", len(poses))
print("R (ground truth):", R_gt, "R (estimate):", R, sep="\n")
print("t (ground truth):", t_gt)
print("t (estimate):", t)
