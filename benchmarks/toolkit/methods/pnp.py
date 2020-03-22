from importlib import import_module

import cv2
from cvxpnpl import pnp, _point_constraints
import numpy as np

from .rc import _solve_relaxation_rc
from .utils import init_matlab, VakhitovHelper


# Dynamically import pyopengv
upnp = None
try:
    upnp = import_module("pyopengv").absolute_pose_upnp
except ModuleNotFoundError:
    pass


# init matlab
# _matlab = None
_matlab = init_matlab()


def null(pts_2d, pts_3d, K):
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


def rc(pts_2d, pts_3d, K, eps=1e-9, max_iters=2500, verbose=False):
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
    return _solve_relaxation_rc(A, B, eps=eps, max_iters=max_iters, verbose=verbose)


class CvxPnPL:

    name = "CvxPnPL"

    @staticmethod
    def estimate_pose(K, pts_2d, pts_3d):
        if len(pts_2d) < 3:
            return [(np.full((3, 3), np.nan), np.full(3, np.nan))]
        return pnp(pts_2d, pts_3d, K)


class EPnP:

    name = "EPnP"

    @staticmethod
    def estimate_pose(K, pts_2d, pts_3d):

        # doesn't work with less than 4 points
        if len(pts_2d) < 4:
            return [(np.full((3, 3), np.nan), np.full(3, np.nan))]

        _, rvec, tvec = cv2.solvePnP(
            objectPoints=pts_3d.astype(float),
            imagePoints=pts_2d.astype(float).reshape((-1, 1, 2)),
            cameraMatrix=K.astype(float),
            distCoeffs=None,
            flags=cv2.SOLVEPNP_EPNP,
        )
        R, _ = cv2.Rodrigues(rvec)
        return [(R, tvec.ravel())]


class OPnP:

    name = "OPnP"
    loaded = _matlab is not None and _matlab.exist("OPnP") > 0

    @staticmethod
    def estimate_pose(K, pts_2d, pts_3d):

        # trying to prevent blowups
        if len(pts_2d) < 3:
            return [(np.full((3, 3), np.nan), np.full(3, np.nan))]

        # compose point variables constraints
        xxn, XXw = VakhitovHelper.points(pts_2d, pts_3d, K)

        # Invoke method on matlab
        try:
            # SVD occasionally is exploding
            Rs, ts = _matlab.OPnP(XXw, xxn, nargout=2)
        except:
            return [(np.full((3, 3), np.nan), np.full(3, np.nan))]
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


class UPnP:

    name = "UPnP"
    loaded = upnp is not None

    @staticmethod
    def estimate_pose(K, pts_2d, pts_3d):

        # trying to prevent blowups
        if len(pts_2d) < 3:
            return [(np.full((3, 3), np.nan), np.full(3, np.nan))]

        # compute bearing vectors
        n = len(pts_3d)
        bearing = np.linalg.solve(K, np.vstack((pts_2d.T, np.ones(n)))).T
        bearing /= np.linalg.norm(bearing, axis=1)[:, None]

        # run pose estimation
        poses = upnp(bearing, pts_3d)

        # repackage results
        poses_out = []
        for T in poses:
            R = T[:, :3].T
            t = -R @ T[:, 3]
            poses_out.append((R, t))
        return poses_out
