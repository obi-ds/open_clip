"""Arguments related to prompts"""
import argparse


def get_instruct_arguments():
    """
    Return arguments for prompt processing
    Returns:
        (parser): Argument parser for prompt arguments
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--number-of-instructions",
        type=int,
        nargs='+',
        default=None,
        help="The number of instruction examples to use",
    )
    parser.add_argument(
        "--training-type",
        choices=["future_instruct_trajectory", "any_instruct_trajectory", "any_instruct", "trajectory"],
        help="The type of training to perform"
    )
    parser.add_argument(
        "--label-type",
        choices=["binary", "binary_strict", "multiclass"],
        help="The type of labels to use for instruct tasks"
    )
    return parser