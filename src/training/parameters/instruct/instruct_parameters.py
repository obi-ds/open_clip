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
        default=1,
        help="The number of instruction examples to use",
    )
    parser.add_argument(
        "--eval-mode",
        default=False,
        action="store_true",
        help="Whether to run the main script in eval mode only"
    )
    return parser