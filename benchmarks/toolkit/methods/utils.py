from importlib import import_module
import numpy as np


# Dynamically import matlab
_matlab = None
_matlab_engine = None
try:
    _matlab = import_module("matlab")
    _matlab.engine = import_module("matlab.engine")
except ModuleNotFoundError:
    pass


def init_matlab():
    global _matlab_engine
    if _matlab is None:
        return None

    if _matlab_engine is not None:
        return _matlab_engine

    # start the engine
    print("Launching MATLAB Engine: ", end="", flush=True)
    _matlab_engine = _matlab.engine.start_matlab()
    print("DONE", flush=True)
    return _matlab_engine



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
        xs = _matlab.double(bear[:, 0, :].T.tolist())
        xe = _matlab.double(bear[:, 1, :].T.tolist())
        Xs = _matlab.double(line_3d[:, 0, :].T.tolist())
        Xe = _matlab.double((line_3d[:, 1, :]).T.tolist())
        return xs, xe, Xs, Xe

    def points(pts_2d, pts_3d, K):
        # set up bearing vectors
        bear = np.linalg.solve(K, np.vstack((pts_2d.T, np.ones((1, len(pts_2d))))))

        # Rename vars to PnPL convention
        xxn = _matlab.double(bear[:-1].tolist())
        XXw = _matlab.double(pts_3d.T.tolist())
        return xxn, XXw
