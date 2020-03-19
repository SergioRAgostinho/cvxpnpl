from abc import ABC, abstractmethod

import cv2
import numpy as np
import pandas as pd

from .suite import Suite, project_points, compute_pose_error


def compute_3d_coordinates(oc, pts, model):
    colors = oc[pts[:, 1], pts[:, 0]]
    if np.any(colors[:, -1] != 255):
        raise NotImplementedError("The object coordinate masks have issues")

    return colors[:, :3] * model.size / 255 + model.min


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

    def print(self):

        # build tables for angular error, translation errors, timings and nan counts
        angular = []
        translation = []
        timings = []
        nans = []

        dids = []
        sids = []

        # Looping over datasets
        for did, ds in enumerate(self.data):
            for sid, seq in enumerate(ds):

                dids.append(type(ds).__name__)
                sids.append(str(seq.name))

                mask = np.logical_and(self.did == did, self.sid == sid)

                angular.append(np.nanmedian(self.results["angular"][mask], axis=0))
                translation.append(
                    np.nanmedian(self.results["translation"][mask], axis=0)
                )
                nans.append(np.sum(np.isnan(self.results["angular"][mask]), axis=0))
                if self.timed:
                    timings.append(np.nanmean(self.results["time"][mask], axis=0))

        # last row is over the entire data set
        angular.append(np.nanmedian(self.results["angular"], axis=0))
        translation.append(np.nanmedian(self.results["translation"], axis=0))
        nans.append(np.sum(np.isnan(self.results["angular"]), axis=0))
        if self.timed:
            timings.append(np.nanmean(self.results["time"], axis=0))
        dids.append("all")
        sids.append("all")

        # Aggregate
        angular = np.stack(angular)
        translation = np.stack(translation)
        timings = np.stack(timings)
        nans = np.stack(nans)

        # build pandas table for pretty rendering
        for data in [angular, translation, timings, nans]:
            df = pd.DataFrame(
                data,
                index=[d + " seq: " + s for d, s, in zip(dids, sids)],
                columns=[m.__name__ for m in self.methods],
            )
            print(df)


class PnPReal(RealSuite):

    def extract_features(self, rgb):
        gray = cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        detections = sift.detect(gray, None)

        # store unique keypoints
        keypoints = np.unique(
            np.array([kp.pt for kp in detections]).astype(np.uint32), axis=0
        )
        return keypoints

    def extract_correspondences(self, oid, frame, keypoints, model):
        # filter keypoints to object mask
        pts_2d = keypoints[frame["mask"][keypoints[:, 1], keypoints[:, 0]] == oid]

        # objects get the corresponding object coordinates
        pts_3d = compute_3d_coordinates(frame["oc"], pts_2d, model)
        return {"pts_2d": pts_2d, "pts_3d": pts_3d}
