import numpy as np

from toolkit.methods.pnpl import CvxPnPL, DLT, EPnPL, OPnPL
from toolkit.suites import parse_arguments, PnPLReal
from toolkit.datasets import Linemod, Occlusion


# reproducibility is a great thing
np.random.seed(0)
np.random.seed(42)


# parse console arguments
args = parse_arguments()

# Just a loading data scenario
if args.load:
    session = PnPLReal.load(args.load)
    session.print()
    quit()

# run something
session = PnPLReal(methods=[CvxPnPL, DLT, EPnPL, OPnPL])
session.run(data=[Linemod(args.datasets_prefix), Occlusion(args.datasets_prefix)])
# session.run(data=[Linemod(args.datasets_prefix)])
if args.save:
    session.save(args.save)
session.print()
