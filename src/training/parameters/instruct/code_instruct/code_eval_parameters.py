"""Parameters for eval prompts"""
import argparse


def get_code_eval_arguments():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--eval-code",
        type=str,
        default=None,
        help="The code to evaluate"
    )
    parser.add_argument(
        "--eval-start-time",
        type=int,
        default=None,
        help="The start time period of the code evaluation prompt",
    )
    parser.add_argument(
        "--eval-end-time",
        type=int,
        default=None,
        help="The end time period of the code evaluation prompt",
    )
    parser.add_argument(
        "--eval-label",
        type=str,
        default=None,
        help="Evaluate a specific label",
    )
    parser.add_argument(
        "--eval-strict",
        default=False,
        action="store_true",
        help="Whether to run the eval with strict control over time period"
    )
    return parser