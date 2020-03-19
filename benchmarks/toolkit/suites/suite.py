import pickle
from time import time
import warnings

import numpy as np


def angle(R):
    """The angle from a rotation matrix"""
    Ru = R.reshape((-1, 3, 3))
    U, _, Vh = np.linalg.svd(Ru)
    Ru = U @ Vh

    return np.arccos(np.clip(0.5 * (Ru.trace(axis1=-2, axis2=-1) - 1), -1, 1)).squeeze()


def project_points(pts, K=np.eye(3), R=np.eye(3), t=np.zeros(3)):
    pts_ = (pts @ R.T + t) @ K.T
    return (pts_ / pts_[:, -1, None])[:, :-1]


def compute_pose_error(groundtruth, estimate):

    R_gt, t_gt = groundtruth
    R, t = estimate

    # Compute angular error
    R_err = np.linalg.solve(R_gt, R)
    ang = angle(R_err) * 180.0 / np.pi

    # Compute translation error
    trans = np.linalg.norm(t - t_gt) / np.linalg.norm(t_gt)
    return ang, trans


class Suite:

    def __init__(self, methods, timed=True):
        # store simulation properties
        self.methods = type(self).filter_methods(methods)

        # placeholder for result storage
        self.results = None

        # Are we benchamrking speed
        self.timed = timed


    @staticmethod
    def filter_methods(methods):
        not_initialized = [method.name for method in methods]
        not_initialized = []
        filtered = []
        for method in methods:
            if hasattr(method, "loaded") and not method.loaded:
                not_initialized.append(method.name)
            else:
                filtered.append(method)

        if len(not_initialized):
            unable_to_load_dependencies = f"The dependencies for the following methods could not be loaded: {not_initialized}. Discarding them from the benchmarks"
            warnings.warn(unable_to_load_dependencies)

        return filtered


    @staticmethod
    def load(path):
        return pickle.load(open(path, "rb"))



    def save(self, path):
        pickle.dump(self, open(path, "wb"))
        print("Saved data to:", path)


    def estimate_pose(self, method, groundtruth, K, **kwargs):

        # time counting mechanism
        start = time()

        # run estimation method
        poses = method.estimate_pose(K, **kwargs)

        # elapsed time
        elapsed = time() - start

        # it can happen that certain realizations with fewer elements
        # admit more than one valid pose. we use additional support points to
        # disambiguate
        if len(poses) == 0:
            return (np.full((3, 3), np.nan), np.full(3, np.nan)), np.nan
        elif len(poses) == 1:
            return poses[0], elapsed

        # create support points to disambiguate multiple poses
        R_gt, t_gt = groundtruth
        pts_s_3d = np.random.random((20, 3)) - 0.5
        pts_s_2d = project_points(pts_s_3d, K, R_gt, t_gt)

        # disambiguate poses
        min_idx = 0
        min_error = np.float("+inf")
        for i, (R, t) in enumerate(poses):
            pts_s_2d_e = project_points(pts_s_3d, K, R, t)
            err = np.sum(np.linalg.norm(pts_s_2d - pts_s_2d_e, axis=1))
            if err < min_error:
                min_error = err
                min_idx = i

        return poses[min_idx], elapsed

