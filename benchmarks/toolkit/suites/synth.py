from itertools import product
import time

from cycler import cycler
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from .suite import Suite, project_points, compute_pose_error

def aa2rm(aa):
    """Construct a rotation matrix from an axis angle representation"""
    angle = np.linalg.norm(aa)

    # np.finfo(float).eps -> 2.220446049250313e-16
    if angle < 2.220446049250313e-16:
        return np.eye(3)

    k = aa / angle
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
    return R


class SynthSuite(Suite):

    def __init__(self, methods=None, n_runs=10, timed=True):
        super().__init__(methods=methods, timed=timed)
        # Kinect V1 intrinsics
        self.K = np.array(
            [[572.41140, 0, 325.26110], [0, 573.57043, 242.04899], [0, 0, 1]]
        )
        self.K_inv = np.linalg.inv(self.K)

        # store simulation properties
        self.LENGTH = 0.6
        self.n_runs = n_runs
        self.noise = None
        self.n_elements = [4]


    def init_run(self, n_elements, noise):

        self.noise = [0.0] if noise is None else noise
        self.n_elements = [4] if n_elements is None else n_elements

        # Initialize storage
        n_noise = len(self.noise)
        n_methods = len(self.methods)
        n_el = len(n_elements)
        self.results = {
            "angular": np.empty((n_el, n_noise, n_methods, self.n_runs)),
            "translation": np.empty((n_el, n_noise, n_methods, self.n_runs)),
        }
        if self.timed:
            self.results["time"] = np.empty((n_el, n_noise, n_methods, self.n_runs))


    def instantiate(self, n_pts):
        """Instantiates a scenario very similar to the conditions of the LINEMOD
        dataset
        """

        # create all points
        pts = self.LENGTH * (np.random.random((n_pts, 3)) - 0.5)

        # Generate random rotation
        axis = np.random.random(3) - 0.5
        axis /= np.linalg.norm(axis)

        angle = 2 * np.pi * np.random.random(1)
        aa = angle * axis
        R = aa2rm(aa)

        # Generate random translation
        t = np.concatenate([np.random.random(2) - 0.5, 1.6 * np.random.random(1) + 0.6])
        return pts, R, t


    def plot(self, label, tight=False):
        """Generate plots"""

        # Creates two subplots and unpacks the output array immediately
        f, axes = plt.subplots(1, 2, figsize=(8, 4) if tight else (16, 9))
        data_t = ("angular", "translation")
        y_label = ("Angular Error (°)", r"Translation Error (%)")

        # method major, noise minor
        labels = [
            m.name + r", $\sigma$=" + str(int(n))
            for m, n in product(self.methods, self.noise)
        ]

        linestyle_all = [
            "-",
            "--",
            ":",
            "-.",
            (0, (3, 5, 1, 5)),
            (0, (3, 5, 1, 5, 1, 5)),
        ]
        markers_all = ["o", "v", "X", "s", "h", "D"]
        colors = plt.get_cmap("tab10")(range(len(self.methods)))
        style_cycler = cycler(
            color=colors, marker=markers_all[: len(self.methods)]
        ) * cycler(linestyle=linestyle_all[: len(self.noise)])

        for i, ax in enumerate(axes):
            # (elements, noise, methods, runs)
            results = self.results[data_t[i]]
            median = np.nanmedian(results, axis=3)
            if i == 1:
                median /= 0.01  # % representation for angles
                ax.yaxis.set_major_formatter(PercentFormatter(decimals=1))

            # (elements, noise, methods)
            # organize as (elements, methods X noise)
            median = np.transpose(median, (0, 2, 1))
            median = median.reshape((len(median), -1))
            ax.set_prop_cycle(style_cycler)
            n = len(self.noise)
            lineobjs = ax.plot(np.array(self.n_elements), median[:, :n], zorder=10)
            if median.shape[1] > n:
                lineobjs += ax.plot(np.array(self.n_elements), median[:, n:])
            ax.set_ylabel(y_label[i])
            ax.set_xlabel(label)

            ax.grid()
            ax.minorticks_on()

            y_lims_max = (4.0, 6.0)
            y_min, y_max = ax.get_ylim()
            ax.set_ylim(-0.20, min(y_lims_max[i], y_max))

        if tight:
            plt.tight_layout()

        for ax in axes:
            # Shrink current axis's height by 10% on the bottom
            box = ax.get_position()
            comp = 0.7 if tight else 0.85
            ax.set_position(
                [box.x0, box.y0 + box.height * (1 - comp), box.width, box.height * comp]
            )

        f.legend(
            lineobjs,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.05),
            ncol=len(self.methods),
        )
        plt.show()


    def plot_timings(self, label, tight=False):

        if not self.timed:
            warnings.warn("Timinings were not logged for this class. Doing nothing.")
            return

        markers_all = ["o", "v", "X", "s", "h", "D"]
        colors = plt.get_cmap("tab10")(range(len(self.methods)))
        style_cycler = cycler(color=colors, marker=markers_all[: len(self.methods)])
        plt.rc("axes", prop_cycle=style_cycler)

        median = 1000 * np.median(self.results["time"], axis=(1, 3))

        f = plt.figure(figsize=(4, 3) if tight else (8, 9))
        lineobjs = plt.plot(np.array(self.n_elements), median[:, 0], zorder=10)
        if median.shape[1] > 1:
            lineobjs += plt.loglog(np.array(self.n_elements), median[:, 1:])

        plt.xlabel(label)
        plt.ylabel("Runtime (ms)")
        plt.grid()
        plt.minorticks_on()

        if tight:
            plt.tight_layout()

        # Shrink current axis's height by 10% on the bottom
        ax = plt.gca()
        box = ax.get_position()
        comp = 0.80 if tight else 0.85
        ax.set_position(
            [box.x0, box.y0 + box.height * (1 - comp), box.width, box.height * comp]
        )

        f.legend(
            lineobjs,
            [m.name for m in self.methods],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.05),
            ncol=len(self.methods),
        )
        plt.show()


    def print_timings(self):
        if not self.timed:
            warnings.warn("Timinings were not logged for this class. Doing nothing.")
            return

        mean_times = 1000 * np.nanmean(self.results["time"], axis=(0, 1, 3))
        for i in range(len(mean_times)):
            print(self.methods[i].name + ":", str(mean_times[i]) + "ms")




    # def project_points(self, pts, R=np.eye(3), t=np.zeros(3)):
    #     pts_ = (pts @ R.T + t) @ self.K.T
    #     return (pts_ / pts_[:, -1, None])[:, :-1]




class PnPSynth(SynthSuite):
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
        poses = method.estimate_pose(self.K, pts_2d, pts_3d)

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
        pts_s_2d = project_points(pts_s_3d, self.K, R_gt, t_gt)

        # disambiguate poses
        min_idx = 0
        min_error = np.float("+inf")
        for i, (R, t) in enumerate(poses):
            pts_s_2d_e = project_points(pts_s_3d, self.K, R, t)
            err = np.sum(np.linalg.norm(pts_s_2d - pts_s_2d_e, axis=1))
            if err < min_error:
                min_error = err
                min_idx = i

        return poses[min_idx], elapsed

    def scenario(self, n_elements, noise):

        pts_3d, R, t = self.instantiate(n_elements)
        pts_2d = project_points(pts_3d, self.K, R, t)

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
                            if self.timed:
                                self.results["time"][i, j, k, l] = np.nan
                            continue

                        # store error results in the object
                        ang, trans = compute_pose_error((R_gt, t_gt), (R, t))

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


class PnLSynth(SynthSuite):
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
        pts_s_2d = project_points(pts_s_3d, self.K, R_gt, t_gt)

        # disambiguate poses
        min_idx = 0
        min_error = np.float("+inf")
        for i, (R, t) in enumerate(poses):
            pts_s_2d_e = project_points(pts_s_3d, self.K, R, t)
            err = np.sum(np.linalg.norm(pts_s_2d - pts_s_2d_e, axis=1))
            if err < min_error:
                min_error = err
                min_idx = i

        return poses[min_idx], elapsed

    def scenario(self, n_elements, noise):

        pts_3d, R, t = self.instantiate(2 * n_elements)
        pts_2d = project_points(pts_3d, self.K, R, t)

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
                            if self.timed:
                                self.results["time"][i, j, k, l] = np.nan
                            continue

                        # store error results in the object
                        ang, trans = compute_pose_error((R_gt, t_gt), (R, t))

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


class PnPLSynth(SynthSuite):
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
        pts_s_2d = project_points(pts_s_3d, self.K, R_gt, t_gt)

        # disambiguate poses
        min_idx = 0
        min_error = np.float("+inf")
        for i, (R, t) in enumerate(poses):
            pts_s_2d_e = project_points(pts_s_3d, self.K, R, t)
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
        pts_2d = project_points(pts_3d, self.K, R, t)

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
                            if self.timed:
                                self.results["time"][i, j, k, l] = np.nan
                            continue

                        # store error results in the object
                        ang, trans = compute_pose_error((R_gt, t_gt), (R, t))

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
