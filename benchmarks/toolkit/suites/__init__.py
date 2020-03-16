import argparse

from .synth import *

def parse_arguments():

    parser = argparse.ArgumentParser()

    group_save_load = parser.add_mutually_exclusive_group()
    group_save_load.add_argument("--save", help="File path to store the session data.")
    group_save_load.add_argument(
        "--load", help="File path to load and plot session data."
    )

    group_figures = parser.add_mutually_exclusive_group()
    group_figures.add_argument(
        "--tight", help="Show tight figures.", action="store_true"
    )
    group_figures.add_argument(
        "--no-display", help="Don't display any figures.", action="store_true"
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=1000,
        help="Number of runs each scenario is instantiated.",
    )
    return parser.parse_args()
