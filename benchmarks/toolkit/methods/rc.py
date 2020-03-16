import warnings

from cvxpnpl import _constraint_ortho_det, _vech10, _vech10_inv
import numpy as np
from scipy.sparse import csc_matrix
import scs


def _sdp_constraints_rc():
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


_A_rc, _b_rc = _sdp_constraints_rc()


def _solve_relaxation_rc(A, B, eps=1e-9, max_iters=2500, verbose=False):
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
        {"A": _A_rc, "b": _b_rc, "c": _vech10(Q, 2)},  # data
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
