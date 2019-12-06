from importlib import import_module
import time

import cv2
from cvxpnpl import pnp
import numpy as np

from toolkit.synth import PnPSynth
from toolkit.suite import init_matlab, Suite, VakhitovHelper, parse_arguments

# Dynamically import pyopengv
upnp = None
try:
    upnp = import_module("pyopengv").absolute_pose_upnp
except ModuleNotFoundError:
    pass


# init matlab
matlab = init_matlab()


class CvxPnPl:

    name = "cvxpnpl"


    @staticmethod
    def estimate_pose(pts_2d, pts_3d, K):
        return pnp(pts_2d, pts_3d, K)


class EPnP:

    name = "EPnP"

    @staticmethod
    def estimate_pose(pts_2d, pts_3d, K):

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
    loaded = matlab is not None and matlab.exist("OPnP") > 0

    @staticmethod
    def estimate_pose(pts_2d, pts_3d, K):

        # compose point variables constraints
        xxn, XXw = VakhitovHelper.points(pts_2d, pts_3d, K)

        # Invoke method on matlab
        Rs, ts = Suite.matlab_engine.OPnP(XXw, xxn, nargout=2)
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
    def estimate_pose(pts_2d, pts_3d, K):

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




if __name__ == "__main__":

    # reproducibility is a great thing
    np.random.seed(0)
    np.random.seed(42)

    # parse console arguments
    args = parse_arguments()

    # Just a loading data scenario
    if args.load:
        session = PnPSynth.load(args.load)
        session.print_timings()
        session.plot()
        quit()

    # run something
    session = PnPSynth(methods=[CvxPnPl, EPnP, OPnP, UPnP], n_runs=100)
    session.run(n_elements=[4, 6, 8, 10, 12], noise=[0.0, 1.0, 2.0])
    if args.save:
        session.save(args.save)
    session.print_timings()
    session.plot()
