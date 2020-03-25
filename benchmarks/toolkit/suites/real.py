from abc import ABC, abstractmethod

import cv2
import numpy as np
import pandas as pd
from skimage.draw import line as raster_line

from .suite import Suite, project_points, compute_pose_error

# delete me
import matplotlib.pyplot as plt


def compute_3d_coordinates(oc, pts, model):
    if not len(pts):
        return np.empty((0, 3))

    colors = oc[pts[:, 1], pts[:, 0]]
    if np.any(colors[:, -1] != 255):
        raise NotImplementedError("The object coordinate masks have issues")

    return colors[:, :3] * model.size / 255 + model.min


def draw_lines(lines, img, color):

    paths = np.concatenate(
        [
            np.stack(
                raster_line(line[0, 1], line[0, 0], line[1, 1], line[1, 0]), axis=-1
            )
            for line in lines
        ]
    )
    out = img.copy()
    out[paths[:, 0], paths[:, 1]] = color
    return out


def extract_sift_keypoints(rgb):
    gray = cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    detections = sift.detect(gray, None)

    # store unique keypoints
    keypoints = np.unique(
        np.array([kp.pt for kp in detections]).astype(np.uint32), axis=0
    )
    return keypoints


def extract_line_segments(rgb):
    gray = cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2GRAY)
    ld = cv2.line_descriptor.LSDDetector_createLSDDetector()
    keylines = ld.detect(gray, 1, 1)

    paths = []
    idx = []
    for i, keyline in enumerate(keylines):
        start = np.round(keyline.getStartPoint()).astype(int)
        end = np.round(keyline.getEndPoint()).astype(int)

        path = np.stack(raster_line(start[1], start[0], end[1], end[0]), axis=-1)
        paths.append(path)
        idx.append(np.full(len(path), i))

    paths = np.concatenate(paths)
    idx = np.concatenate(idx)

    # ensure max bounds are not overstepped
    max_bound = np.array(rgb.shape[:2]) - 1
    paths = np.minimum(paths, max_bound)
    return paths, idx


def extract_point_correspondences(oid, frame, keypoints, model):
    # filter keypoints to object mask and object coordinate data
    pts_2d = keypoints[
        np.logical_and(
            frame["mask"][keypoints[:, 1], keypoints[:, 0]] == oid,
            frame["oc"][keypoints[:, 1], keypoints[:, 0], -1] == 255,
        )
    ]

    # objects get the corresponding object coordinates
    pts_3d = compute_3d_coordinates(frame["oc"], pts_2d, model)
    return pts_2d, pts_3d


def extract_line_correspondences(oid, frame, lines, model):
    paths, idx = lines

    # prune line segments to masks. assume masks are convex
    mask = np.logical_and(frame["mask"] == oid, frame["oc"][:, :, -1] == 255)
    line_2d = []
    for pid in range(idx[-1]):

        path = paths[idx == pid]
        if not np.any(mask[path[:, 0], path[:, 1]]):
            continue

        line = np.empty((2, 2), dtype=int)

        # clamp at start and at the end
        start, end = None, None
        for i, (r, c) in enumerate(path):
            if mask[r, c]:
                line[0] = (c, r)
                start = i
                break

        for i, (r, c) in enumerate(reversed(path)):
            if mask[r, c]:
                line[1] = (c, r)
                end = len(path) - i
                break

        # Reject very small segments
        if end - start < 5:
            continue

        line_2d.append(line)
    line_2d = np.array(line_2d)  # array can cope with empty lists

    # # debug
    # img = draw_lines(line_2d, frame["rgb"], np.array([255, 255, 255], dtype=np.uint8))
    # plt.imshow(img); plt.show()

    # # objects get the corresponding object coordinates
    line_3d = compute_3d_coordinates(
        frame["oc"], line_2d.reshape(-1, 2), model
    ).reshape(-1, 2, 3)
    return line_2d, line_3d


class RealSuite(Suite, ABC):
    def __init__(self, methods, timed=True):
        super().__init__(methods, timed)

        self.data = None  # dataset placeholder

        # Since each dataset has a different number of sequences, frames
        # objects per frames and even instance per objects, we need to
        # store everything in a flat array and store indexes for each
        # instance
        self.did = None  # datasets
        self.sid = None  # sequences
        self.fid = None  # frames
        self.oid = None  # objects

    def init_run(self, data):

        self.data = data

        self.results = {
            "angular": [],
            "translation": [],
        }
        if self.timed:
            self.results["time"] = []

        # Initialize accumulators
        self.did = []  # datasets
        self.sid = []  # sequences
        self.fid = []  # frames
        self.oid = []  # objects

    @abstractmethod
    def extract_features(self, rgb):
        pass

    @abstractmethod
    def extract_correspondences(self, oid, frame, features, model):
        pass

    def run(self, data):

        self.init_run(data)

        # Can we print some progress statistics
        n_prog, i_prog = 0, 0
        for ds in self.data:
            n_prog += len(ds)
        print("Progress:   0.00%", end="", flush=True)

        # Looping over datasets
        for did, ds in enumerate(self.data):
            # looping over sequences
            for sid, seq in enumerate(ds):

                # looping over frames
                for frame in seq:

                    # extract features in each frame
                    features = self.extract_features(frame["rgb"])

                    # Iterate through each object in frame
                    for oid, pose in frame["poses"].items():

                        # plt.imsave(f'/tmp/images/{seq.name:02d}_{frame["id"]:04d}.m.png', frame["mask"])
                        # plt.imsave(f'/tmp/images/{seq.name:02d}_{frame["id"]:04d}.o.png', frame["oc"])

                        mmask = frame["mask"].astype(bool)
                        moc = frame["oc"][:, :, -1] == 255
                        iou = np.sum(np.logical_and(mmask, moc)) / np.sum(
                            np.logical_or(mmask, moc)
                        )

                        # there are legit occlusion cases lower than 0.6 iou
                        if iou < 0.5:
                            error_msg = "IoU issues between mask and object coordinates"
                            raise RuntimeError(error_msg)

                        # extract correspondences
                        correspondences = self.extract_correspondences(
                            oid, frame, features, ds.models[str(oid)]
                        )

                        # Pre allocate placeholders storing results
                        nm = len(self.methods)
                        ang_all = np.full(nm, np.nan)
                        trans_all = np.full(nm, np.nan)
                        time_all = np.full(nm, np.nan)

                        groundtruth = (pose[:, :3], pose[:, -1])

                        for mid, method in enumerate(self.methods):

                            # get a pose estimate
                            (R, t), time_all[mid] = self.estimate_pose(
                                method, groundtruth, ds.camera.K, **correspondences
                            )

                            # Sanitize results
                            if np.any(np.isnan(R)) or np.any(np.isnan(t)):
                                continue

                            # store error results in the object
                            ang_all[mid], trans_all[mid] = compute_pose_error(
                                groundtruth, (R, t)
                            )

                        # let each method compute the pose compute pose
                        self.did.append(did)
                        self.sid.append(sid)
                        self.fid.append(frame["id"])
                        self.oid.append(oid)

                        self.results["angular"].append(ang_all)
                        self.results["translation"].append(trans_all)
                        if self.timed:
                            self.results["time"].append(time_all)

                    # progress only reported at frame level
                    i_prog += 1
                    percent = i_prog * 100 / n_prog
                    print(f"\rProgress: {percent:>6.2f}%", end="", flush=True)

        print("\rProgress: 100.00%", flush=True)

        # merge everything together
        self.did = np.array(self.did)
        self.sid = np.array(self.sid)
        self.fid = np.array(self.fid)
        self.oid = np.array(self.oid)

        self.results["angular"] = np.stack(self.results["angular"])
        self.results["translation"] = np.stack(self.results["translation"])
        if self.timed:
            self.results["time"] = np.stack(self.results["time"])

    def _aggregate_results(self):

        # build tables for angular error, translation errors, timings and nan counts
        angular = []
        translation = []
        timings = []
        nans = []

        dids = []
        sids = []

        # filter out all nans
        good_mask = np.logical_not(
            np.logical_or.reduce(np.isnan(self.results["angular"]).T)
        )

        # Looping over datasets
        for did, ds in enumerate(self.data):
            for sid, seq in enumerate(ds):

                dids.append(type(ds).__name__)
                # sids.append(str(seq.name))
                sids.append(type(ds).seq_names[sid])

                mask_with_nans = np.logical_and(self.did == did, self.sid == sid)
                mask = np.logical_and(mask_with_nans, good_mask)

                angular.append(np.nanmedian(self.results["angular"][mask], axis=0))
                translation.append(
                    np.nanmedian(self.results["translation"][mask], axis=0)
                )
                nans.append(
                    np.sum(np.isnan(self.results["angular"][mask_with_nans]), axis=0)
                )
                if self.timed:
                    timings.append(np.nanmean(self.results["time"][mask], axis=0))

        # last row is over the entire data set
        angular.append(np.nanmedian(self.results["angular"][good_mask], axis=0))
        translation.append(np.nanmedian(self.results["translation"][good_mask], axis=0))
        nans.append(np.sum(np.isnan(self.results["angular"]), axis=0))
        if self.timed:
            timings.append(np.nanmean(self.results["time"][good_mask], axis=0))
        # dids.append("all")
        # sids.append("all")

        # Aggregate
        angular = np.stack(angular)
        translation = np.stack(translation)
        timings = np.stack(timings)
        nans = np.stack(nans)

        return angular, translation, timings, nans, dids, sids

    def print(self, mode=None):

        angular, translation, timings, nans, dids, sids = self._aggregate_results()

        # build pandas table for pretty rendering
        for data, name, scale in zip(
            [angular, translation, timings, nans],
            [
                "Angular Error (°)",
                "Translation Error (%)",
                "Average Runtime (ms)",
                "NaN Counts",
            ],
            [1.0, 100.0, 1000.0, 1],
        ):
            print(name)
            df = pd.DataFrame(
                data * scale,
                index=[d + " " + s for d, s, in zip(dids, sids)] + ["All"],
                columns=[m.__name__ for m in self.methods],
            )
            if mode is None or mode == "console":
                print(df, "\n")
            elif mode == "latex":
                print(df.to_latex(), "\n")
            else:
                raise RuntimeError("Unknown mode '" + str(mode) + "'")

        # Combined angle and translation
        print("Angular Error (°) / Translation Error (‰)")
        str_format = np.frompyfunc(lambda x: f"{np.round(x, 3):.3f}", 1, 1)
        data = str_format(np.stack([angular, translation], axis=-1) * (1.0, 1000.0))
        if mode == "latex":
            # highlight the best
            bf_format = np.frompyfunc(lambda x: f"\\textbf{{{x}}}", 1, 1)
            best_angular = np.argmin(angular, axis=1)
            sidx = list(range(len(data)))
            data[sidx, best_angular, 0] = bf_format(data[sidx, best_angular, 0])

            best_translation = np.argmin(translation, axis=1)
            # import pdb; pdb.set_trace()
            data[sidx, best_translation, 1] = bf_format(data[sidx, best_translation, 1])
        data = data[:, :, 0] + " / " + data[:, :, 1]
        df = pd.DataFrame(
            data,
            index=[d[:4] + " " + s for d, s, in zip(dids, sids)] + ["All"],
            columns=[
                m.__name__ + " \\textdegree/\\textperthousand"
                if mode == "latex"
                else " (° / ‰)"
                for m in self.methods
            ],
        )
        if mode is None or mode == "console":
            print(df, "\n")
        elif mode == "latex":
            print(df.to_latex(escape=False), "\n")
        else:
            raise RuntimeError("Unknown mode '" + str(mode) + "'")


class PnPReal(RealSuite):
    def extract_features(self, rgb):
        return extract_sift_keypoints(rgb)

    def extract_correspondences(self, oid, frame, keypoints, model):
        pts_2d, pts_3d = extract_point_correspondences(oid, frame, keypoints, model)
        return {"pts_2d": pts_2d, "pts_3d": pts_3d}


class PnLReal(RealSuite):
    def extract_features(self, rgb):
        return extract_line_segments(rgb)

    def extract_correspondences(self, oid, frame, lines, model):
        line_2d, line_3d = extract_line_correspondences(oid, frame, lines, model)
        return {"line_2d": line_2d, "line_3d": line_3d}


class PnPLReal(RealSuite):
    def extract_features(self, rgb):
        keypoints = extract_sift_keypoints(rgb)
        keylines = extract_line_segments(rgb)
        return keypoints, keylines

    def extract_correspondences(self, oid, frame, features, model):
        keypoints, keylines = features
        pts_2d, pts_3d = extract_point_correspondences(oid, frame, keypoints, model)
        line_2d, line_3d = extract_line_correspondences(oid, frame, keylines, model)
        return {
            "pts_2d": pts_2d,
            "pts_3d": pts_3d,
            "line_2d": line_2d,
            "line_3d": line_3d,
        }
