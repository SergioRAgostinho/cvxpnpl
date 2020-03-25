import numpy as np

from toolkit.methods.pnl import CvxPnPL, EPnPL, Mirzaei, OPnPL, Pluecker, RPnL
from toolkit.suites import parse_arguments, PnLSynth


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
session = PnLSynth(
    methods=[CvxPnPL, EPnPL, OPnPL, Mirzaei, Pluecker, RPnL], n_runs=args.runs
)
session.run(n_elements=[4, 6, 8, 10, 12], noise=[0.0, 1.0, 2.0])
if args.save:
    session.save(args.save)
session.print_timings()
if not args.no_display:
    session.plot(tight=args.tight)
