import cv2
from cvxpnpl import pnp
import numpy as np
from pyopengv import absolute_pose_upnp as upnp

from suite import Suite, VakhitovHelper, parse_arguments


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


class PnPSynth(Suite):
    def __init__(self, methods=None, n_runs=10):

        super().__init__(
            methods=[CvxPnPl] if methods is None else methods, n_runs=n_runs
        )

    def estimate_pose(self, method, pts_2d, pts_3d, groundtruth):

        # run estimation method
        poses = method.estimate_pose(pts_2d, pts_3d, self.K)

        # it can happen that certain realizations with fewer elements
        # admit more than one valid pose. we use additional support points to
        # disambiguate
        if len(poses) == 1:
            return poses[0]

        # create support points
        R_gt, t_gt = groundtruth
        pts_s_3d, _, _ = self.instantiate(10 * len(pts_2d))
        pts_s_2d = self.project_points(pts_s_3d, R_gt, t_gt)

        # disambiguate poses
        min_idx = 0
        min_error = np.float("+inf")
        for i, (R, t) in enumerate(poses):
            pts_s_2d_e = self.project_points(pts_s_3d, R, t)
            err = np.sum(np.linalg.norm(pts_s_2d - pts_s_2d_e, axis=1))
            if err < min_error:
                min_error = err
                min_idx = i

        return poses[min_idx]

    def scenario(self, n_elements, noise):

        pts_3d, R, t = self.instantiate(n_elements)
        pts_2d = self.project_points(pts_3d, R, t)

        # Add gaussian noise to pixel projections
        pts_2d += np.random.normal(scale=noise, size=pts_2d.shape)
        return pts_2d, pts_3d, R, t

    def plot(self):
        super().plot("Points")

    def run(self, n_elements=None, noise=None):

        # Allocate storage and other stuff
        self.init_run(n_elements, noise)

        # Some printing aids
        print("Progress:   0%", end="", flush=True)
        n_prog = len(self.n_elements) * len(self.noise) * self.n_runs
        i_prog = 0

        for i, n_el in enumerate(self.n_elements):
            for j, noise_ in enumerate(self.noise):
                for l in range(self.n_runs):

                    # instantiate environment
                    pts_2d, pts_3d, R_gt, t_gt = self.scenario(
                        n_elements=n_el, noise=noise_
                    )

                    for k, method in enumerate(self.methods):

                        # estimate pose
                        R, t = self.estimate_pose(
                            method, pts_2d, pts_3d, groundtruth=(R_gt, t_gt)
                        )

                        # store error results in the object
                        ang, trans = PnPSynth.compute_pose_error(
                            groundtruth=(R_gt, t_gt), estimate=(R, t)
                        )
                        self.results["angular"][i, j, k, l] = ang
                        self.results["translation"][i, j, k, l] = trans

                    i_prog += 1
                    print(
                        "\rProgress: {:>3d}%".format(int(i_prog * 100 / n_prog)),
                        end="",
                        flush=True,
                    )

        print("\rProgress: 100%", flush=True)


if __name__ == "__main__":

    # reproducibility is a great thing
    np.random.seed(0)
    np.random.seed(42)

    # parse console arguments
    args = parse_arguments()

    # Just a loading data scenario
    if args.load:
        session = PnPSynth.load(args.load)
        session.plot()
        quit()

    # run something
    session = PnPSynth(methods=[CvxPnPl, EPnP, OPnP, UPnP], n_runs=100)
    session.run(n_elements=[4, 6, 8, 10, 12], noise=[0.0, 1.0, 2.0])
    if args.save:
        session.save(args.save)

    session.plot()
