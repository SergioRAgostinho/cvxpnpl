from cvxpnpl import pnl
import numpy as np

from pnl_synth import PnLSynth
from redundant_constraint import pnl_va
from suite import parse_arguments


class Baseline:

    name = "baseline"

    @staticmethod
    def estimate_pose(line_2d, line_3d, K):
        return pnl(line_2d, line_3d, K)


class Stripped:

    name = "stripped"

    @staticmethod
    def estimate_pose(line_2d, line_3d, K):
        return pnl_va(line_2d, line_3d, K)


if __name__ == "__main__":

    # reproducibility is a great thing
    np.random.seed(0)
    np.random.seed(42)

    # parse console arguments
    args = parse_arguments()

    # Just a loading data scenario
    if args.load:
        session = PnLSynth.load(args.load)
        session.print_timings()
        session.plot(tight=args.tight)
        quit()

    # run something
    session = PnLSynth(methods=[Baseline, Stripped], n_runs=1000)
    session.run(n_elements=[4, 6, 8, 10, 12], noise=[0.0, 1.0, 2.0])
    if args.save:
        session.save(args.save)
    session.print_timings()
    session.plot(tight=args.tight)
