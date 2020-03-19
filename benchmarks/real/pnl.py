import numpy as np

from toolkit.methods.pnp import CvxPnPL
# from toolkit.suites import parse_arguments, PnLReal
from toolkit.datasets import Linemod, Occlusion

# to move later
from toolkit.suites import parse_arguments, RealSuite

# class PnPReal(RealSuite):
#     pass

# reproducibility is a great thing
np.random.seed(0)
np.random.seed(42)


# parse console arguments
args = parse_arguments()

# Just a loading data scenario
if args.load:
    session = PnPReal.load(args.load)
    # import pdb; pdb.set_trace()
    session.print()
    quit()

# run something
session = PnLReal(methods=[CvxPnPL])
session.run(data=[Linemod(args.datasets_prefix), Occlusion(args.datasets_prefix)])
# session.run(data=[Occlusion(args.datasets_prefix)])
if args.save:
    session.save(args.save)
session.print()
# if not args.no_display:
#     session.plot(tight=args.tight)
