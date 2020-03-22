from cvxpnpl import pnpl, _point_constraints, _line_constraints
import numpy as np

from .rc import _solve_relaxation_rc
from .utils import init_matlab, VakhitovHelper


# init matlab
_matlab = init_matlab()


def rc(pts_2d, line_2d, pts_3d, line_3d, K, eps=1e-9, max_iters=2500, verbose=False):
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
    return _solve_relaxation_rc(A, B, eps=eps, max_iters=max_iters, verbose=verbose)


class CvxPnPL:

    name = "CvxPnPL"

    @staticmethod
    def estimate_pose(K, pts_2d, line_2d, pts_3d, line_3d):
        # requires a minimum of 3 elements
        if (len(line_2d) + len(pts_2d)) < 3:
            return [(np.full((3, 3), np.nan), np.full(3, np.nan))]
        return pnpl(pts_2d, line_2d, pts_3d, line_3d, K)


class DLT:

    name = "DLT"
    loaded = _matlab is not None and _matlab.exist("DLT") > 0

    @staticmethod
    def estimate_pose(K, pts_2d, line_2d, pts_3d, line_3d):

        # compose all geometric constraints
        xxn, XXw = VakhitovHelper.points(pts_2d, pts_3d, K)
        xs, xe, Xs, Xe = VakhitovHelper.lines(line_2d, line_3d, K)

        # Invoke method on matlab
        R, t = _matlab.DLT(XXw, xxn, xs, xe, Xs, Xe, nargout=2)

        # Cast to numpy types
        return [(np.array(R), np.array(t).ravel())]


class EPnPL:

    name = "EPnPL"
    loaded = _matlab is not None and _matlab.exist("EPnPLS_GN") > 0

    @staticmethod
    def estimate_pose(K, pts_2d, line_2d, pts_3d, line_3d):

        # requires a minimum of 6 elements
        if (len(line_2d) + len(pts_2d)) < 6:
            return [(np.full((3, 3), np.nan), np.full(3, np.nan))]

        # compose all geometric constraints
        xxn, XXw = VakhitovHelper.points(pts_2d, pts_3d, K)
        xs, xe, Xs, Xe = VakhitovHelper.lines(line_2d, line_3d, K)

        # Invoke method on matlab
        try:
            # SVD occasionally is exploding
            R, t = _matlab.EPnPLS_GN(XXw, xxn, xs, xe, Xs, Xe, nargout=2)
        except:
            return [(np.full((3, 3), np.nan), np.full(3, np.nan))]

        # Cast to numpy types
        return [(np.array(R), np.array(t).ravel())]


class OPnPL:

    name = "OPnPL"
    loaded = _matlab is not None and _matlab.exist("OPnPL") > 0

    @staticmethod
    def estimate_pose(K, pts_2d, line_2d, pts_3d, line_3d):

        # compose all geometric constraints
        xxn, XXw = VakhitovHelper.points(pts_2d, pts_3d, K)
        xs, xe, Xs, Xe = VakhitovHelper.lines(line_2d, line_3d, K)

        # Invoke method on matlab
        Rs, ts = _matlab.OPnPL(XXw, xxn, xs, xe, Xs, Xe, nargout=2)
        Rs, ts = np.array(Rs), np.array(ts)

        # Detect if there's no multiple solutions
        if len(Rs.shape) == 2:
            return [(Rs, ts.ravel())]

        # This method returns multiple poses even in a non-minimal case
        # repackage results
        poses_out = []
        for i in range(Rs.shape[2]):
            R = Rs[:, :, i]
            t = ts[:, i]
            poses_out.append((R, t))
        return poses_out
