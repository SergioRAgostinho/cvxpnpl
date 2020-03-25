from os.path import splitext
import numpy as np

from toolkit.methods.pnp import CvxPnPL
from toolkit.suites import parse_arguments, PnPSynth


class Baseline(CvxPnPL):
    name = "baseline"


# reproducibility is a great thing
np.random.seed(0)
np.random.seed(42)

# parse console arguments
args = parse_arguments()

# Just a loading data scenario
if args.load:
    session = PnPSynth.load(args.load)
    session.plot_timings(legend=False, tight=args.tight)
    quit()

# run something
session = PnPSynth(methods=[Baseline], n_runs=args.runs)
session.run(
    n_elements=range(4, 11), noise=[0.0, 1.0, 2.0],
)
if args.save:
    filename, ext = splitext(args.save)
    session.save(filename + ".low" + ext)
if not args.no_display:
    session.plot_timings(tight=args.tight)

session.run(
    n_elements=np.linspace(200, 10000, num=20, dtype=int), noise=[0.0, 1.0, 2.0],
)
if args.save:
    filename, ext = splitext(args.save)
    session.save(filename + ".high" + ext)
if not args.no_display:
    session.plot_timings(tight=args.tight)
