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
        "--k-shot",
        type=int,
        nargs='+',
        default=0,
        help="The number of k-shot examples to use",
    )
    return parser