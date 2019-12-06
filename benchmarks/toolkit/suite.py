import argparse
from importlib import import_module
from itertools import product
import pickle
import warnings

from cycler import cycler
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

# Dynamically import matlab
matlab = None
_matlab_engine = None
try:
    matlab = import_module("matlab")
    matlab.engine = import_module("matlab.engine")
except ModuleNotFoundError:
    pass


def parse_arguments():

    parser = argparse.ArgumentParser()

    group_save_load = parser.add_mutually_exclusive_group()
    group_save_load.add_argument("--save", help="File path to store the session data.")
    group_save_load.add_argument("--load", help="File path to load and plot session data.")

    group_figures = parser.add_mutually_exclusive_group()
    group_figures.add_argument("--tight", help="Show tight figures.", action="store_true")
    group_figures.add_argument("--no-display", help="Don't display any figures.", action="store_true")

    parser.add_argument("--runs", type=int, default=1000, help="Number of runs each scenario is instantiated.")
    return parser.parse_args()


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


def angle(R):
    """The angle from a rotation matrix"""
    Ru = R.reshape((-1, 3, 3))
    U, _, Vh = np.linalg.svd(Ru)
    Ru = U @ Vh

    return np.arccos(np.clip(0.5*(Ru.trace(axis1=-2, axis2=-1) - 1), -1, 1)).squeeze()


def init_matlab():
    global _matlab_engine
    if matlab is None:
        return None

    if _matlab_engine is not None:
        return _matlab_engine

    # start the engine
    print("Launching MATLAB Engine: ", end="", flush=True)
    _matlab_engine = matlab.engine.start_matlab()
    print("DONE", flush=True)
    return _matlab_engine


class Suite:

    matlab_engine = None

    def __init__(self, methods=None, n_runs=10, timed=False):
        # Kinect V1 intrinsics
        self.K = np.array(
            [[572.41140, 0, 325.26110], [0, 573.57043, 242.04899], [0, 0, 1]]
        )
        self.K_inv = np.linalg.inv(self.K)

        # store simulation properties
        self.LENGTH = 0.6
        self.n_runs = n_runs
        self.methods = type(self).filter_methods(methods)
        self.noise = None
        self.n_elements = [4]

        # placeholder for result storage
        self.results = None

        # Are we benchamrking speed
        self.timed = timed

        # boot up Matlab Engine if needed
        if Suite.matlab_engine is None:
            Suite.matlab_engine = init_matlab()


    @staticmethod
    def filter_methods(methods):
        not_initialized = [method.name for method in methods ]
        not_initialized = []
        filtered = []
        for method in methods:
            if hasattr(method, "loaded") and not method.loaded:
                not_initialized.append(method.name)
            else:
                filtered.append(method)

        if len(not_initialized):
            warnings.warn(f"The dependencies for the following methods could not be loaded: {not_initialized}.\nDiscarding them from the benchmarks")

        return filtered

    @staticmethod
    def compute_pose_error(groundtruth, estimate):

        R_gt, t_gt = groundtruth
        R, t = estimate

        # Compute angular error
        R_err = np.linalg.solve(R_gt, R)
        ang = angle(R_err) * 180.0 / np.pi

        # Compute translation error
        trans = np.linalg.norm(t - t_gt) / np.linalg.norm(t_gt)
        return ang, trans

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

    @staticmethod
    def load(path):
        return pickle.load(open(path, "rb"))

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

        # import pdb; pdb.set_trace()

    def print_timings(self):
        if not self.timed:
            warnings.warn("Timinings were not logged for this class. Doing nothing.")
            return

        mean_times = 1000 * self.results["time"].mean(axis=(0, 1, 3))
        for i in range(len(mean_times)):
            print(self.methods[i].name + ":", str(mean_times[i]) + "ms")

    def project_points(self, pts, R=np.eye(3), t=np.zeros(3)):
        pts_ = (pts @ R.T + t) @ self.K.T
        return (pts_ / pts_[:, -1, None])[:, :-1]

    def save(self, path=None):
        pickle.dump(self, open(path, "wb"))
        print("Saved data to:", path)


class VakhitovHelper:
    """Utility functions to prepare inputs for what is requested
    by functions in Vakhitov's pnpl toolbox. We adopt the same naming
    convention the author used.
    """

    def lines(line_2d, line_3d, K):
        # set up bearing vectors
        bear = np.linalg.solve(
            K, np.vstack((line_2d.reshape((-1, 2)).T, np.ones((1, 2 * len(line_2d)))))
        ).T[:, :-1]
        bear = bear.reshape((-1, 2, 2))

        # Split points into start and end points
        xs = matlab.double(bear[:, 0, :].T.tolist())
        xe = matlab.double(bear[:, 1, :].T.tolist())
        Xs = matlab.double(line_3d[:, 0, :].T.tolist())
        Xe = matlab.double((line_3d[:, 1, :]).T.tolist())
        return xs, xe, Xs, Xe

    def points(pts_2d, pts_3d, K):
        # set up bearing vectors
        bear = np.linalg.solve(K, np.vstack((pts_2d.T, np.ones((1, len(pts_2d))))))

        # Rename vars to PnPL convention
        xxn = matlab.double(bear[:-1].tolist())
        XXw = matlab.double(pts_3d.T.tolist())
        return xxn, XXw
