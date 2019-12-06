from os.path import splitext
from cvxpnpl import pnp
import numpy as np

from toolkit.synth import PnPSynth
from toolkit.suite import parse_arguments


class Baseline:

    name = "baseline"

    @staticmethod
    def estimate_pose(pts_2d, pts_3d, K):
        return pnp(pts_2d, pts_3d, K)


if __name__ == "__main__":

    # reproducibility is a great thing
    np.random.seed(0)
    np.random.seed(42)

    # parse console arguments
    args = parse_arguments()

    # Just a loading data scenario
    if args.load:
        session = PnPSynth.load(args.load)
        session.plot_timings(tight=args.tight)
        quit()

    # run something
    session = PnPSynth(methods=[Baseline], n_runs=args.runs)
    session.run(
        n_elements=range(4, 11),
        noise=[0.0, 1.0, 2.0],
    )
    if args.save:
        filename, ext = splitext(args.save)
        session.save(filename + ".low" + ext)
    if not args.no_display:
        session.plot_timings(tight=args.tight)

    session.run(
        # n_elements=np.logspace(np.log10(4), np.log10(1e7), num=20, dtype=int),
        n_elements=np.linspace(200, 10000, num=20, dtype=int),
        noise=[0.0, 1.0, 2.0],
    )
    if args.save:
        filename, ext = splitext(args.save)
        session.save(filename + ".high" + ext)
    if not args.no_display:
        session.plot_timings(tight=args.tight)
