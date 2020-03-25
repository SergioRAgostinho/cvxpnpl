import numpy as np

from toolkit.methods.pnp import CvxPnPL, EPnP, OPnP, UPnP
from toolkit.suites import parse_arguments, PnPReal
from toolkit.datasets import Linemod, Occlusion


# reproducibility is a great thing
np.random.seed(0)
np.random.seed(42)


# parse console arguments
args = parse_arguments()

# Just a loading data scenario
if args.load:
    session = PnPReal.load(args.load)
    session.print(args.print_mode)
    quit()

# run something
session = PnPReal(methods=[CvxPnPL, EPnP, OPnP, UPnP])
session.run(data=[Linemod(args.datasets_prefix), Occlusion(args.datasets_prefix)])
# session.run(data=[Occlusion(args.datasets_prefix)])
if args.save:
    session.save(args.save)
session.print()
