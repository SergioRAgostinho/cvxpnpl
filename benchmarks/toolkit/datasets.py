from collections import namedtuple
import json
import os
from os.path import join as pjoin
from pathlib import Path

import numpy as np
from plymit import Ply
from PIL import Image

from .renderer import Renderer

Model = namedtuple(
    "Model",
    [
        "id",
        "points",
        "normals",
        "color",
        "faces",
        "diameter",
        "min",
        "size",
        "symmetries_discrete",
    ],
)

Camera = namedtuple("Camera", ["K", "size"])


class Dataset:
    def __init__(self, prefix):
        print("Initializing " + type(self).__name__)
        self.prefix = prefix
        self.camera = self._parse_camera()

        # Load models
        self.models = self._load_models()
        self.renderer = self._init_renderer()

        # Handle Partitions
        # we're only interested in the test partition here
        # self.train = type(self)._Partition(pjoin(self.prefix, "train"))
        # self.train = None
        self.test = type(self)._Partition(
            pjoin(self.prefix, "test"), self.models, self.renderer
        )

    def __iter__(self):
        return iter(self.test)

    def __len__(self):
        return self.test.n_frames

    def __getstate__(self):
        # save prefix only and reload database upon deserializing
        return {"prefix": self.prefix}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__init__(Path(self.prefix).parent)

    def _init_renderer(self):
        renderer = Renderer(False)
        renderer.load_models(list(self.models.values()))
        return renderer

    def _parse_camera(self):
        data = json.loads(open(pjoin(self.prefix, "camera.json")).read())
        camera = Camera(
            K=np.array(
                ((data["fx"], 0, data["cx"]), (0, data["fy"], data["cy"]), (0, 0, 1),)
            ),
            size=(data["width"], data["height"]),
        )
        return camera

    def _load_models(self):

        models = {}

        print("Reading ply files for models: ", end="", flush=True)

        # load model info. models_eval are lighter
        info = json.loads(
            open(pjoin(self.prefix, "models_eval", "models_info.json")).read()
        )
        for k, v in info.items():

            print(k, end=" ", flush=True)

            # load points, normals and color
            ply = Ply(pjoin(self.prefix, "models", "obj_{:06d}.ply".format(int(k))))

            # parse vertices
            points = []
            normals = []
            colors = []
            for vertex in ply.elementLists["vertex"]:
                points.extend([vertex.x, vertex.y, vertex.z])
                normals.extend([vertex.nx, vertex.ny, vertex.nz])
                colors.extend([vertex.red, vertex.green, vertex.blue])
            points = np.array(points, dtype=np.float32).reshape((-1, 3))
            normals = np.array(normals, dtype=np.float32).reshape((-1, 3))
            colors = np.array(colors, dtype=np.uint8).reshape((-1, 3))

            # faces
            faces = []
            for f in ply.elementLists["face"]:
                faces.extend(f.vertex_indices)
            faces = np.array(faces, dtype=np.uint32).reshape((-1, 3))

            # create model object
            models[k] = Model(
                int(k),
                points,
                normals,
                colors,
                faces,
                v["diameter"],
                np.array((v["min_x"], v["min_y"], v["min_z"])),
                np.array((v["size_x"], v["size_y"], v["size_z"])),
                [np.array(s).reshape((4, 4)) for s in v["symmetries_discrete"]]
                if "symmetries_discrete" in v
                else None,
            )
        print("DONE", flush=True)
        return models

    class _Partition:
        def __init__(self, prefix, models, renderer):

            self.prefix = prefix
            self.models = models
            self.renderer = renderer

            seq_names = sorted([d.name for d in os.scandir(prefix)])
            # seq_names = [seq_names[1]]
            self.sequences = [
                Dataset._Sequence(int(n), pjoin(prefix, n), models, renderer)
                for n in seq_names
            ]

            # store the total number of frames in the partition
            self.n_frames = 0
            for seq in self.sequences:
                self.n_frames += len(seq)

        def __iter__(self):
            return iter(self.sequences)

        def __len__(self):
            return len(self.sequences)

    class _Sequence:
        def __init__(self, name, prefix, models, renderer):

            self.name = name
            self.prefix = prefix
            self.models = models
            self.renderer = renderer

            # parse gt
            gt = json.loads(open(pjoin(prefix, "scene_gt.json")).read())
            self.poses = [None] * len(gt.keys())
            for k, v in gt.items():
                poses = {}
                for pose in v:
                    poses[pose["obj_id"]] = np.hstack(
                        (
                            np.array(pose["cam_R_m2c"]).reshape((3, 3)),
                            np.array(pose["cam_t_m2c"]).reshape((3, 1)),
                        )
                    )
                self.poses[int(k)] = poses

            # iterator stuff
            self.i = 0

        def __iter__(self):
            self.i = 0
            return self

        def __len__(self):
            return len(self.poses)
            # return 4

        def __next__(self):
            # reached the end. get out
            if self.i == len(self):
                raise StopIteration

            # generate object coordinates
            poses = self.poses[self.i]
            oc = self.renderer.object_coordinates(poses)

            # load visibility masks
            mask = self.fuse_masks(self.i, poses.keys())

            # return dictionary object with rgb, depth and poses
            data = {
                "id": self.i,
                "rgb": np.array(
                    Image.open(pjoin(self.prefix, "rgb", "{:06d}.png".format(self.i)))
                ),  # load rgb
                # "depth": np.array(
                #     Image.open(pjoin(self.prefix, "depth", "{:06d}.png".format(self.i)))
                # ),  # load depth
                "mask": mask,
                "oc": oc,
                "poses": poses,
            }
            self.i += 1
            return data

        def fuse_masks(self, frame, object_ids):
            masks = np.zeros(self.renderer.size[::-1], dtype=np.uint8)
            for i, oid in enumerate(object_ids):
                masks[
                    np.array(
                        Image.open(
                            pjoin(self.prefix, "mask_visib", f"{frame:06d}_{i:06d}.png")
                        )
                    )
                    > 127
                ] = oid
            return masks


class Linemod(Dataset):
    def __init__(self, prefix):
        super().__init__(pjoin(prefix, "lm"))


class Occlusion(Dataset):
    def __init__(self, prefix):
        super().__init__(pjoin(prefix, "lmo"))
