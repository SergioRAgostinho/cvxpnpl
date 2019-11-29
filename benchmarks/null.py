from cvxpnpl import _point_constraints
import numpy as np


def pnp_null(pts_2d, pts_3d, K):
    """Compute object poses from point 2D-3D correspondences.
    Variant A:
    - The redundant row orthonormality constained is removed.

    Arguments:
    pts_2d -- n x 2 np.array of 2D pixels
    pts_3d -- n x 3 np.array of 3D points
    K -- 3 x 3 np.array with the camera intrinsics
    eps -- numerical precision of the solver
    max_iters -- maximum number of iterations the solver is allowed to perform
    verbose -- print additional solver information to the console
    """
    # Extract point constraints
    (C1, C2, C3), (N1, N2, N3) = _point_constraints(pts_2d, pts_3d, K)

    # Compose block matrices
    C = np.vstack((C1, C2, C3))
    N = np.vstack((N1, N2, N3))

    B = np.linalg.solve(N.T @ N, N.T) @ C
    A = C - N @ B

    # Pick the smallest singular vector
    R = np.linalg.svd(A)[2][-1].reshape((3, 3)).T

    # Project to the orthogonal space
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    R *= np.sign(np.linalg.det(R))
    t = -B @ R.ravel("F")
    return [(R, t)]
