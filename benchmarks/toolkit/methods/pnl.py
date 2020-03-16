from cvxpnpl import pnl, _line_constraints
import numpy as np

from .rc import _solve_relaxation_rc
from .utils import init_matlab, VakhitovHelper

# init matlab
_matlab = init_matlab()


def rc(line_2d, line_3d, K, eps=1e-9, max_iters=2500, verbose=False):
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
    return _solve_relaxation_rc(A, B, eps=eps, max_iters=max_iters, verbose=verbose)


class CvxPnPL:

    name = "CvxPnPL"

    @staticmethod
    def estimate_pose(line_2d, line_3d, K):
        return pnl(line_2d, line_3d, K)


class EPnPL:

    name = "EPnPL"
    loaded = _matlab is not None and _matlab.exist("EPnPLS_GN") > 0

    @staticmethod
    def estimate_pose(line_2d, line_3d, K):

        # the method needs at least 6 lines to work
        if len(line_2d) < 6:
            return [(np.full((3, 3), np.nan), np.full(3, np.nan))]

        # compose all geometric constraints
        xxn = _matlab.double.empty(2, 0)
        XXw = _matlab.double.empty(3, 0)
        xs, xe, Xs, Xe = VakhitovHelper.lines(line_2d, line_3d, K)

        # Invoke method on matlab
        R, t = _matlab.EPnPLS_GN(XXw, xxn, xs, xe, Xs, Xe, nargout=2)

        # Cast to numpy types
        return [(np.array(R), np.array(t).ravel())]


class Mirzaei:
    """Ref [28] on the Vakhitov et al."""

    name = "Mirzaei"
    loaded = _matlab is not None and _matlab.exist("mirzWrapper") > 0

    @staticmethod
    def estimate_pose(line_2d, line_3d, K):

        # compose line constraints
        xs, xe, Xs, Xe = VakhitovHelper.lines(line_2d, line_3d, K)

        # Invoke method on matlab
        R, t = _matlab.mirzWrapper(xs, xe, Xs, Xe, nargout=2)

        # Cast to numpy types
        return [(np.array(R), np.array(t).ravel())]


class OPnPL:

    name = "OPnPL"
    loaded = _matlab is not None and _matlab.exist("OPnPL") > 0

    @staticmethod
    def estimate_pose(line_2d, line_3d, K):

        # compose all geometric constraints
        xxn = _matlab.double.empty(2, 0)
        XXw = _matlab.double.empty(3, 0)
        xs, xe, Xs, Xe = VakhitovHelper.lines(line_2d, line_3d, K)

        # Invoke method on matlab
        Rs, ts = _matlab.OPnPL(XXw, xxn, xs, xe, Xs, Xe, nargout=2)
        Rs, ts = np.array(Rs), np.array(ts)

        # Detect if there's no multiple solutions
        if len(Rs.shape) == 2:
            return [(Rs, ts.ravel())]

        # repackage results
        poses_out = []
        for i in range(Rs.shape[2]):
            R = Rs[:, :, i]
            t = ts[:, i]
            poses_out.append((R, t))
        return poses_out


class Pluecker:
    """Ref [28] on the Vakhitov et al."""

    name = "Plücker"
    loaded = _matlab is not None and _matlab.exist("plueckerWrapper") > 0

    @staticmethod
    def estimate_pose(line_2d, line_3d, K):

        # requires a minimum of 9 lines to work properly
        if len(line_2d) < 9:
            return [(np.full((3, 3), np.nan), np.full(3, np.nan))]

        # compose line constraints
        xs, xe, Xs, Xe = VakhitovHelper.lines(line_2d, line_3d, K)

        # Invoke method on matlab
        R, t = _matlab.plueckerWrapper(xs, xe, Xs, Xe, nargout=2)

        # Cast to numpy types
        return [(np.array(R), np.array(t).ravel())]


class RPnL:
    """Ref [44] on the Vakhitov et al."""

    name = "RPnL"
    loaded = _matlab is not None and _matlab.exist("PNLWrapper") > 0

    @staticmethod
    def estimate_pose(line_2d, line_3d, K):

        # compose line constraints
        xs, xe, Xs, Xe = VakhitovHelper.lines(line_2d, line_3d, K)

        # Invoke method on matlab
        R, t = _matlab.PNLWrapper(xs, xe, Xs, Xe, nargout=2)

        # Cast to numpy types
        return [(np.array(R), np.array(t).ravel())]
