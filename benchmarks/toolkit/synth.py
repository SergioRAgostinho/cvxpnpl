import time
import numpy as np
from .suite import Suite

class PnPSynth(Suite):
    def __init__(self, methods=None, n_runs=10, timed=True):

        super().__init__(
            methods=[CvxPnPl] if methods is None else methods,
            n_runs=n_runs,
            timed=timed,
        )

    def estimate_pose(self, method, pts_2d, pts_3d, groundtruth):

        # time counting mechanism
        start = time.time()

        # run estimation method
        poses = method.estimate_pose(pts_2d, pts_3d, self.K)

        # elapsed time
        elapsed = time.time() - start

        # it can happen that certain realizations with fewer elements
        # admit more than one valid pose. we use additional support points to
        # disambiguate
        if len(poses) == 1:
            return poses[0], elapsed

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

        return poses[min_idx], elapsed

    def scenario(self, n_elements, noise):

        pts_3d, R, t = self.instantiate(n_elements)
        pts_2d = self.project_points(pts_3d, R, t)

        # Add gaussian noise to pixel projections
        pts_2d += np.random.normal(scale=noise, size=pts_2d.shape)
        return pts_2d, pts_3d, R, t

    def plot(self, tight=False):
        super().plot("Points", tight=tight)

    def plot_timings(self, tight=False):
        super().plot_timings("Points", tight=tight)

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
                        (R, t), elapsed_time = self.estimate_pose(
                            method, pts_2d, pts_3d, groundtruth=(R_gt, t_gt)
                        )

                        # Sanitize results
                        if np.any(np.isnan(R)) or np.any(np.isnan(t)):
                            self.results["angular"][i, j, k, l] = np.nan
                            self.results["translation"][i, j, k, l] = np.nan
                            continue

                        # store error results in the object
                        ang, trans = PnPSynth.compute_pose_error(
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


class PnPLSynth(Suite):
    def __init__(self, methods=None, n_runs=10, timed=True):

        super().__init__(
            methods=[CvxPnPl] if methods is None else methods,
            n_runs=n_runs,
            timed=timed,
        )

    def estimate_pose(self, method, pts_2d, line_2d, pts_3d, line_3d, groundtruth):

        # time counting mechanism
        start = time.time()

        # run estimation method
        poses = method.estimate_pose(pts_2d, line_2d, pts_3d, line_3d, self.K)

        # elapsed time
        elapsed = time.time() - start

        # it can happen that certain realizations with fewer elements
        # admit more than one valid pose. we use additional support points to
        # disambiguate
        if len(poses) == 1:
            return poses[0], elapsed

        # create support points
        R_gt, t_gt = groundtruth
        pts_s_3d, _, _ = self.instantiate(10 * (len(pts_2d) + 2 * len(line_2d)))
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

        # We need to ensure at least 1 point and 1 line
        n_p = np.random.randint(1, n_elements)
        n_l = n_elements - n_p

        pts_3d, R, t = self.instantiate(n_p + 2 * n_l)
        pts_2d = self.project_points(pts_3d, R, t)

        # Add gaussian noise to pixel projections
        pts_2d += np.random.normal(scale=noise, size=pts_2d.shape)

        # Rearrange the points into lines
        # [l00, l01, l10, l11, l20, l21]

        # 3D line is organized
        line_3d = pts_3d[n_p:].reshape((n_l, 2, 3))

        # Organized as 3x2x2 tensor. Lines x points x pixels
        line_2d = pts_2d[n_p:].reshape((n_l, 2, 2))
        return pts_2d[:n_p], line_2d, pts_3d[:n_p], line_3d, R, t

    def plot(self, tight=False):
        super().plot("Points and Lines", tight)

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
                    pts_2d, line_2d, pts_3d, line_3d, R_gt, t_gt = self.scenario(
                        n_elements=n_el, noise=noise_
                    )

                    for k, method in enumerate(self.methods):

                        # estimate pose
                        (R, t), elapsed_time = self.estimate_pose(
                            method,
                            pts_2d,
                            line_2d,
                            pts_3d,
                            line_3d,
                            groundtruth=(R_gt, t_gt),
                        )

                        # Sanitize results
                        if np.any(np.isnan(R)) or np.any(np.isnan(t)):
                            self.results["angular"][i, j, k, l] = np.nan
                            self.results["translation"][i, j, k, l] = np.nan
                            continue

                        # store error results in the object
                        ang, trans = PnPLSynth.compute_pose_error(
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
