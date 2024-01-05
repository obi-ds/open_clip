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
        default=2,
        help="The number of instruction examples to use",
    )
    return parser