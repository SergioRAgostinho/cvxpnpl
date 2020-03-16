from collections import namedtuple
import json
import os
from os.path import join as pjoin

import moderngl as mgl
import numpy as np
from PIL import Image
from plymit import Ply

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


class Renderer:
    def __init__(
        self,
        use_gl_pose_convention=True,
        size=(640, 480),
        depth_limits=(20, 2000),
        K=None,
    ):

        self._ctx = mgl.create_standalone_context()
        self._ctx.enable(mgl.DEPTH_TEST)
        self._use_gl_pose_convention = use_gl_pose_convention
        self._fbo = self._ctx.simple_framebuffer(size, components=4)
        self._fbo.use()
        self._prog = self.init_programs()

        # All of these are set by set_opengl_modelview_matrix
        self.size = size
        self.depth_limits = depth_limits
        self.proj = Renderer._init_mvm(K, size, depth_limits)

        # model storage
        self._models = {}

    @staticmethod
    def _init_mvm(K=None, size=(640, 480), depth_limits=(20, 2000)):
        if K is None:
            K = np.array(
                [
                    [572.41140, 0.0, 325.26110],
                    [0.0, 573.57043, 242.04899],
                    [0.0, 0.0, 1.0],
                ]
            )

        # building the projection
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        w = size[0]
        l = -cx
        r = w + l

        h = size[1]
        t = cy
        b = t - h

        near, far = depth_limits
        proj = np.array(
            [
                [2 * fx / w, 0.0, (r + l) / w, 0.0],
                [0.0, -2 * fy / h, -(t + b) / h, 0.0],
                [
                    0.0,
                    0.0,
                    -(far + near) / (far - near),
                    -2 * far * near / (far - near),
                ],
                [0.0, 0.0, -1.0, 0.0],
            ]
        )
        return proj

    def init_programs(self):

        # object coordinates
        oc = self._ctx.program(
            vertex_shader="""
                #version 330

                in vec3 in_vert;

                out vec3 color;

                uniform mat4 mvp;
                uniform vec3 v_max;
                uniform vec3 v_min;

                void main() {
                    gl_Position = mvp * vec4(in_vert, 1.0);
                    color = (in_vert - v_min)/(v_max - v_min);
                }
            """,
            fragment_shader="""
                #version 330

                in vec3 color;

                out vec4 f_color;

                void main() {
                    // f_color = vec4(color.rg, 0.5 + 0.5*color.b, 1);
                    f_color = vec4(color.rgb, 1);
                }
            """,
        )
        return {"oc": oc}

    def object_coordinates(self, poses):

        # clear frame buffer
        self._fbo.clear(red=0.0, green=0.0, blue=0.0, alpha=0.0, depth=1.0)
        prog = self._prog["oc"]

        # Apply transformation between computer vision and opengl frame of reference
        s = 2 * int(self._use_gl_pose_convention) - 1
        T = np.diag((1, s, s))

        for i, pose in poses.items():
            mvp = self.proj @ np.vstack((T @ pose, (0, 0, 0, 1)))
            prog["mvp"].write(mvp.T.astype("f4").tobytes())
            model = self._models[str(i)]
            prog["v_max"].write(model["max"].astype("f4").tobytes())
            prog["v_min"].write(model["min"].astype("f4").tobytes())

            self._ctx.simple_vertex_array(
                prog, model["vbo"], "in_vert", index_buffer=model["ibo"]
            ).render()

        data = np.frombuffer(
            self._fbo.read(components=4), dtype=np.dtype("u1")
        ).reshape((480, 640, 4))
        return data

    def load_models(self, models):

        for model in models:

            # Store everything
            d = {
                "vbo": self._ctx.buffer(model.points.astype("f4").tobytes()),
                "ibo": self._ctx.buffer(model.faces.astype("i4").tobytes()),
                "min": model.min,
                "max": model.min + model.size,
            }
            self._models[model.id] = d


class LinemodOcclusion:
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

        def __next__(self):
            # reached the end. get out
            if self.i == len(self):
                raise StopIteration

            # generate object coordinates
            poses = self.poses[self.i]
            oc = self.renderer.object_coordinates(poses)

            # load visibility masks
            masks = {}
            for i, k in enumerate(poses.keys()):
                masks[k] = np.array(
                    Image.open(
                        pjoin(
                            self.prefix,
                            "mask_visib",
                            "{:06d}_{:06d}.png".format(self.i, i),
                        )
                    )
                )
                # import matplotlib.pyplot as plt
                # plt.imshow(img); plt.show()
                # import pdb; pdb.set_trace()

            # return dictionary object with rgb, depth and poses
            data = {
                "id": self.i,
                "rgb": np.array(
                    Image.open(pjoin(self.prefix, "rgb", "{:06d}.png".format(self.i)))
                ),  # load rgb
                "depth": np.array(
                    Image.open(pjoin(self.prefix, "depth", "{:06d}.png".format(self.i)))
                ),  # load depth
                "masks": masks,
                "oc": oc,
                "poses": poses,
            }
            self.i += 1
            return data

    class _Partition:
        def __init__(self, prefix, models, renderer):

            self.prefix = prefix
            self.models = models
            self.renderer = renderer

            seq_names = sorted([d.name for d in os.scandir(prefix)])
            self.sequences = [
                LinemodOcclusion._Sequence(int(n), pjoin(prefix, n), models, renderer)
                for n in seq_names
            ]

        def __iter__(self):
            return iter(self.sequences)

    def __init__(self, prefix):

        self.prefix = prefix
        self.camera = self._parse_camera()

        # Load models
        self.models = self._load_models()
        self.renderer = Renderer(False)
        self.renderer.load_models(list(self.models.values()))

        # Handle Partitions
        # self.train = type(self)._Partition(pjoin(self.prefix, "train"))
        self.train = None
        self.test = type(self)._Partition(
            pjoin(self.prefix, "test"), self.models, self.renderer
        )

    def _parse_camera(self):
        data = json.loads(open(pjoin(self.prefix, "camera.json")).read())
        camera = {
            "K": np.array(
                ((data["fx"], 0, data["cx"]), (0, data["fy"], data["cy"]), (0, 0, 1),)
            ),
            "size": (data["width"], data["height"]),
        }
        return camera

    def _load_models(self):

        models = {}

        # load model info
        info = json.loads(open(pjoin(self.prefix, "models", "models_info.json")).read())
        for k, v in info.items():

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
                k,
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
        return models


def parse_arguments():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", help="Dataset prefix folder")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()
    ds = LinemodOcclusion(args.prefix)

    for sequence in ds.test:
        for frame in sequence:
            import pdb

            pdb.set_trace()
            pass
