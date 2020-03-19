import moderngl as mgl
import numpy as np

# We only need a single stand alone context
_ctx = None
def _get_global_context():
    global _ctx
    if _ctx is None:
        _ctx = mgl.create_standalone_context()
        _ctx.enable(mgl.DEPTH_TEST)
    return _ctx


class Renderer:
    def __init__(
        self,
        use_gl_pose_convention=True,
        size=(640, 480),
        depth_limits=(20, 2000),
        K=None,
    ):

        self._ctx = _get_global_context()
        self._use_gl_pose_convention = use_gl_pose_convention
        self._fbo = self._ctx.simple_framebuffer(size, components=4)
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
        self._fbo.use()
        self._fbo.clear(red=0.0, green=0.0, blue=0.0, alpha=0.0, depth=1.0)
        prog = self._prog["oc"]

        # Apply transformation between computer vision and opengl frame of reference
        s = 2 * int(self._use_gl_pose_convention) - 1
        T = np.diag((1, s, s))

        for i, pose in poses.items():
            mvp = self.proj @ np.vstack((T @ pose, (0, 0, 0, 1)))
            prog["mvp"].write(mvp.T.astype("f4").tobytes())
            model = self._models[i]
            prog["v_max"].write(model["max"].astype("f4").tobytes())
            prog["v_min"].write(model["min"].astype("f4").tobytes())

            self._ctx.simple_vertex_array(
                prog, model["vbo"], "in_vert", index_buffer=model["ibo"]
            ).render()

        data = np.frombuffer(
            self._fbo.read(components=4), dtype=np.dtype("u1")
        ).reshape((self.size[1], self.size[0], 4))
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
