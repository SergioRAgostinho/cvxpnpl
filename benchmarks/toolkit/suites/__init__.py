import argparse

from .suite import *
from .synth import *
from .real import *


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

    parser.add_argument(
        "--datasets-prefix",
        default="data",
        help="Specifies the prefix folder holding all datasets. If no single folder exists, consider creating one with the aid of symbolic links.",
    )

    parser.add_argument(
        "--print-mode",
        default=None,
        choices=["console", "latex"],
        help="Specializes the printing to console to generate LaTeX friendly tables.",
    )
    return parser.parse_args()
