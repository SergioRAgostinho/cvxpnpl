import time

import numpy as np
from cvxpnpl import pnl

from suite import Suite, VakhitovHelper, parse_arguments


class CvxPnPl:

    name = "cvxpnpl"

    @staticmethod
    def estimate_pose(line_2d, line_3d, K):
        return pnl(line_2d, line_3d, K)


class EPnPL:

    name = "EPnPL"

    @staticmethod
    def estimate_pose(line_2d, line_3d, K):

        # the method needs at least 6 lines to work
        if len(line_2d) < 6:
            return [(np.full((3, 3), np.nan), np.full(3, np.nan))]

        # compose all geometric constraints
        xxn = Suite.matlab_engine.double.empty(2, 0)
        XXw = Suite.matlab_engine.double.empty(3, 0)
        xs, xe, Xs, Xe = VakhitovHelper.lines(line_2d, line_3d, K)

        # Invoke method on matlab
        R, t = Suite.matlab_engine.EPnPLS_GN(XXw, xxn, xs, xe, Xs, Xe, nargout=2)

        # Cast to numpy types
        return [(np.array(R), np.array(t).ravel())]


class Mirzaei:
    """Ref [28] on the Vakhitov et al."""

    name = "Mirzaei"

    @staticmethod
    def estimate_pose(line_2d, line_3d, K):

        # compose line constraints
        xs, xe, Xs, Xe = VakhitovHelper.lines(line_2d, line_3d, K)

        # Invoke method on matlab
        R, t = Suite.matlab_engine.mirzWrapper(xs, xe, Xs, Xe, nargout=2)

        # Cast to numpy types
        return [(np.array(R), np.array(t).ravel())]


class OPnPL:

    name = "OPnPL"

    @staticmethod
    def estimate_pose(line_2d, line_3d, K):

        # compose all geometric constraints
        xxn = Suite.matlab_engine.double.empty(2, 0)
        XXw = Suite.matlab_engine.double.empty(3, 0)
        xs, xe, Xs, Xe = VakhitovHelper.lines(line_2d, line_3d, K)

        # Invoke method on matlab
        Rs, ts = Suite.matlab_engine.OPnPL(XXw, xxn, xs, xe, Xs, Xe, nargout=2)
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

    @staticmethod
    def estimate_pose(line_2d, line_3d, K):

        # requires a minimum of 9 lines to work properly
        if len(line_2d) < 9:
            return [(np.full((3, 3), np.nan), np.full(3, np.nan))]

        # compose line constraints
        xs, xe, Xs, Xe = VakhitovHelper.lines(line_2d, line_3d, K)

        # Invoke method on matlab
        R, t = Suite.matlab_engine.plueckerWrapper(xs, xe, Xs, Xe, nargout=2)

        # Cast to numpy types
        return [(np.array(R), np.array(t).ravel())]


class RPnL:
    """Ref [44] on the Vakhitov et al."""

    name = "RPnL"

    @staticmethod
    def estimate_pose(line_2d, line_3d, K):

        # compose line constraints
        xs, xe, Xs, Xe = VakhitovHelper.lines(line_2d, line_3d, K)

        # Invoke method on matlab
        R, t = Suite.matlab_engine.PNLWrapper(xs, xe, Xs, Xe, nargout=2)

        # Cast to numpy types
        return [(np.array(R), np.array(t).ravel())]


class PnLSynth(Suite):
    def __init__(self, methods=None, n_runs=10, timed=True):

        super().__init__(
            methods=[CvxPnPl] if methods is None else methods,
            n_runs=n_runs,
            timed=timed,
        )

    def estimate_pose(self, method, line_2d, line_3d, groundtruth):

        # time counting mechanism
        start = time.time()

        # run estimation method
        poses = method.estimate_pose(line_2d, line_3d, self.K)

        # elapsed time
        elapsed = time.time() - start

        # it can happen that certain realizations with fewer elements
        # admit more than one valid pose. we use additional support points to
        # disambiguate
        if len(poses) == 1:
            return poses[0], elapsed

        # create support points
        R_gt, t_gt = groundtruth
        pts_s_3d, _, _ = self.instantiate(10 * len(line_2d))
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

        return poses[min_idx], elapsed

    def scenario(self, n_elements, noise):

        pts_3d, R, t = self.instantiate(2 * n_elements)
        pts_2d = self.project_points(pts_3d, R, t)

        # Add gaussian noise to pixel projections
        pts_2d += np.random.normal(scale=noise, size=pts_2d.shape)

        # Rearrange the points into lines
        # [l00, l01, l10, l11, l20, l21]

        # 3D line is organized
        line_3d = pts_3d.reshape((n_elements, 2, 3))

        # Organized as 3x2x2 tensor. Lines x points x pixels
        line_2d = pts_2d.reshape((n_elements, 2, 2))
        return line_2d, line_3d, R, t

    def plot(self, tight=False):
        super().plot("Lines", tight)

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
                    line_2d, line_3d, R_gt, t_gt = self.scenario(
                        n_elements=n_el, noise=noise_
                    )

                    for k, method in enumerate(self.methods):

                        # estimate pose
                        (R, t), elapsed_time = self.estimate_pose(
                            method, line_2d, line_3d, groundtruth=(R_gt, t_gt)
                        )

                        # Sanitize results
                        if np.any(np.isnan(R)) or np.any(np.isnan(t)):
                            self.results["angular"][i, j, k, l] = np.nan
                            self.results["translation"][i, j, k, l] = np.nan
                            continue

                        # store error results in the object
                        ang, trans = PnLSynth.compute_pose_error(
                            groundtruth=(R_gt, t_gt), estimate=(R, t)
                        )
                        self.results["angular"][i, j, k, l] = ang
                        self.results["translation"][i, j, k, l] = trans
                        if self.timed:
                            self.results["time"][i, j, k, l] = elapsed_time

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
        session = PnLSynth.load(args.load)
        session.print_timings()
        session.plot()
        quit()

    # run something
    session = PnLSynth(
        methods=[CvxPnPl, EPnPL, Mirzaei, OPnPL, Pluecker, RPnL], n_runs=1000
    )
    session.run(n_elements=[4, 6, 8, 10, 12], noise=[0.0, 1.0, 2.0])
    if args.save:
        session.save(args.save)
    session.print_timings()
    session.plot()
