import numpy as np
from cvxpnpl import pnl

# fix seed to allow for reproducible results
np.random.seed(0)
np.random.seed(42)

# 3D lines are parameterized as two points organized into a np.array
# with the layout (line, pts, dim)
# instantiate a couple of points centered around the origin
line_3d = 0.6 * (np.random.random((6, 2, 3)) - 0.5)

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

# Sample to points from the line and project them to 2D
pts_s = line_3d.reshape((-1, 3))
line_2d = (pts_s @ R_gt.T + t_gt) @ K.T

# this variable is organized as (line, point, dim)
line_2d = (line_2d / line_2d[:, -1, None])[:, :-1].reshape((-1, 2, 2))

# Compute pose candidates. the problem is not minimal so only one
# will be provided
poses = pnl(line_2d=line_2d, line_3d=line_3d, K=K)
R, t = poses[0]

print("Nr of possible poses:", len(poses))
print("R (ground truth):", R_gt, "R (estimate):", R, sep="\n")
print("t (ground truth):", t_gt)
print("t (estimate):", t)
