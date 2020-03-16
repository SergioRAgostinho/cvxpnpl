# import os, sys, inspect, pathlib
# currentdir = pathlib.Path(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
# sys.path = [str(currentdir.parent)] + sys.path

import numpy as np

# from synth.pnp import *
from suite import parse_arguments


class PnPOcclusion:
    def __init__(self, prefix, timed=False):

        self.ds = LinemodOcclusion(prefix)
        self.timed = timed

        # placeholder for result storage
        self.results = None
        self.frame_id = None
        self.object_id = None


if __name__ == "__main__":

    # reproducibility is a great thing
    np.random.seed(0)
    np.random.seed(42)

    # parse console arguments
    args = parse_arguments()
