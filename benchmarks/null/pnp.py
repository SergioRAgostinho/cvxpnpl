from cvxpnpl import pnp
import numpy as np

from toolkit.synth import PnPSynth
from toolkit.null import pnp_null
from toolkit.suite import parse_arguments


class Baseline:

    name = "baseline"

    @staticmethod
    def estimate_pose(pts_2d, pts_3d, K):
        return pnp(pts_2d, pts_3d, K)


class Null:

    name = "null"

    @staticmethod
    def estimate_pose(pts_2d, pts_3d, K):
        return pnp_null(pts_2d, pts_3d, K)


if __name__ == "__main__":

    # reproducibility is a great thing
    np.random.seed(0)
    np.random.seed(42)

    # parse console arguments
    args = parse_arguments()

    # Just a loading data scenario
    if args.load:
        session = PnPSynth.load(args.load)
        session.print_timings()
        session.plot(tight=args.tight)
        quit()

    # run something
    session = PnPSynth(methods=[Baseline, Null], n_runs=args.runs)
    session.run(n_elements=[8, 10, 12, 14, 16], noise=[0.0, 1.0, 2.0])
    if args.save:
        session.save(args.save)
    session.print_timings()
    if not args.no_display:
        session.plot(tight=args.tight)
