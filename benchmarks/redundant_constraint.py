import warnings

from cvxpnpl import (
    _constraint_ortho_det,
    _line_constraints,
    _point_constraints,
    _vech10,
    _vech10_inv,
)
import numpy as np
from scipy.sparse import csc_matrix
import scs


def _sdp_constraints_va():
    """Generates the static sdp constraints for the optimization problem
    Variant A:
    - The redundant row orthonormality constained is removed.
    """

    # Placeholder
    Ad = np.zeros((71, 55))
    E_ij = np.reshape(
        np.eye(3)[:, None, :, None] * np.eye(3)[None, :, None], (-1, 3, 3)
    )

    # Linear equalities
    # Z10,10 = 1
    Ad[0, -1] = 1

    # Rows and cols
    E_ij_rc = E_ij[[0, 1, 2, 4, 5, 8]]
    c = np.array((1, 0, 0, 1, 0, 1))
    for i in range(6):
        P = np.block(
            [[np.kron(E_ij_rc[i], np.eye(3)), np.zeros((9, 1))], [np.zeros(9), -c[i]],]
        )
        Ad[i + 1] = _vech10(0.5 * (P + P.T), 2)

    # Determinant
    E_ji_det = E_ij[[3, 3, 3, 7, 7, 7, 2, 2, 2]]
    e_l = np.kron(np.ones(3), np.eye(3)).T
    e_k = np.repeat(np.eye(3)[[2, 0, 1]], 3, axis=0)
    for i in range(9):
        S = np.array(
            [
                [0, -e_l[i, 2], e_l[i, 1]],
                [e_l[i, 2], 0, -e_l[i, 0]],
                [-e_l[i, 1], e_l[i, 0], 0],
            ]
        )
        P = np.block(
            [[np.kron(E_ji_det[i], S), np.zeros((9, 1))], [-np.kron(e_k[i], e_l[i]), 0]]
        )
        Ad[i + 7] = _vech10(0.5 * (P + P.T), 2)

    # Cones
    mask = np.concatenate((np.zeros((16, 55), dtype=bool), np.eye(55, dtype=bool)))
    Ad[mask] = -_vech10(np.ones((10, 10)), np.sqrt(2))

    # convert to csc matrix
    A = csc_matrix(Ad)

    # Constants
    b = np.zeros(71)
    b[0] = 1
    return A, b


_A_va, _b_va = _sdp_constraints_va()


def _solve_relaxation_va(A, B, eps=1e-9, max_iters=2500, verbose=False):
    """Given the linear system formed by the problem's geometric constraints,
    computes all possible poses.
    Variant A:
    - The redundant row orthonormality constained is removed.

    Arguments:
    A -- the matrix defining the homogeneous linear system formed from the problem's
    geometric constraints, such that A r = 0.
    B -- an auxiliary matrix which allows to retrieve the translation vector
    from an optimal rotation t = - B @ r.
    eps -- numerical precision of the solver
    max_iters -- maximum number of iterations the solver is allowed to perform
    verbose -- print additional information to the console
    """
    # Solve the QCQP using shor's relaxation
    # Construct Q
    Q = np.block([[A.T @ A, np.zeros((9, 1))], [np.zeros((1, 9)), 0]])

    # Invoke solver
    results = scs.solve(
        {"A": _A_va, "b": _b_va, "c": _vech10(Q, 2)},  # data
        {"f": 16, "l": 0, "q": [], "ep": 0, "s": [10]},  # cones
        verbose=verbose,
        eps=eps,
        max_iters=max_iters,
    )
    # Invoke solver
    Z = _vech10_inv(results["x"])
    if np.any(np.isnan(Z)):
        if verbose:
            warnings.warn(
                "The SDP solver did not return a valid solution. Increasing max_iters might solve the issue."
            )
        return [(np.full((3, 3), np.nan), np.full(3, np.nan))]
    vals, vecs = np.linalg.eigh(Z)

    # check for rank
    rank = np.sum(vals > 1e-3)
    r_c = None
    if rank == 1:
        r_c = (vecs[:-1, -1] / vecs[-1, -1])[None, :]
    else:
        r_c = _constraint_ortho_det(vecs, rank)

    poses = []
    for r in r_c:
        # Retrieve rotation and translation
        U, _, Vh = np.linalg.svd(r.reshape((3, 3)).T)
        R = U @ Vh
        t = -B @ R.ravel("F")
        poses.append((R, t))
    return poses


def pnp_va(pts_2d, pts_3d, K, eps=1e-9, max_iters=2500, verbose=False):
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

    # Solve the QCQP using shor's relaxation
    return _solve_relaxation_va(A, B, eps=eps, max_iters=max_iters, verbose=verbose)


def pnl_va(line_2d, line_3d, K, eps=1e-9, max_iters=2500, verbose=False):
    """Compute object poses from line 2D-3D correspondences.
    Variant A:
    - The redundant row orthonormality constained is removed.

    Arguments:
    line_2d -- n x 2 x 2 np.array organized as (line, pt, dim). Each line is defined
    by sampling 2 points from it. Each point is a pixel in 2D.
    line_3d -- A n x 2 x 3 np.array organized as (line, pt, dim). Each line is defined
    by 2 points. The points reside in 3D.
    K -- 3 x 3 np.array with the camera intrinsics.
    eps -- numerical precision of the solver
    max_iters -- maximum number of iterations the solver is allowed to perform
    verbose -- print additional solver information to the console
    """
    # Extract point constraints
    C, N = _line_constraints(line_2d, line_3d, K)

    # Compose block matrices
    B = np.linalg.solve(N.T @ N, N.T @ C)
    A = C - N @ B

    # Solve the QCQP using shor's relaxation
    return _solve_relaxation_va(A, B, eps=eps, max_iters=max_iters, verbose=verbose)


def pnpl_va(
    pts_2d, line_2d, pts_3d, line_3d, K, eps=1e-9, max_iters=2500, verbose=False
):
    """Compute object poses from point and line 2D-3D correspondences.
    Variant A:
    - The redundant row orthonormality constained is removed.

    Arguments:
    pts_2d -- n x 2 np.array of 2D pixels
    line_2d -- n x 2 x 2 np.array organized as (line, pt, dim). Each line is defined
    by sampling 2 points from it. Each point is a pixel in 2D.
    pts_3d -- n x 3 np.array of 3D points
    line_3d -- A n x 2 x 3 np.array organized as (line, pt, dim). Each line is defined
    by 2 points. The points reside in 3D.
    K -- 3 x 3 np.array with the camera intrinsics.
    eps -- numerical precision of the solver
    max_iters -- maximum number of iterations the solver is allowed to perform
    verbose -- print additional solver information to the console
    """
    # Extract point constraints
    (Cp1, Cp2, Cp3), (Np1, Np2, Np3) = _point_constraints(
        pts_2d=pts_2d.reshape((-1, 2)), pts_3d=pts_3d.reshape((-1, 3)), K=K
    )

    # Extract line constraints
    Cl, Nl = _line_constraints(line_2d.reshape((-1, 2, 2)), line_3d, K)

    # Compose block matrices
    C = np.vstack((Cp1, Cp2, Cp3, Cl))
    N = np.vstack((Np1, Np2, Np3, Nl))

    # Compose block matrices
    B = np.linalg.solve(N.T @ N, N.T @ C)
    A = C - N @ B

    # Solve the QCQP using shor's relaxation
    return _solve_relaxation_va(A, B, eps=eps, max_iters=max_iters, verbose=verbose)
