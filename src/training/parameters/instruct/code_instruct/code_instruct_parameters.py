"""Parameters for train prompts"""
import argparse


def get_code_instruct_arguments():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--billable-probability",
        type=float,
        default=0.0,
        help="Probability to get the billable icd code as is"
    )
    parser.add_argument(
        "--top-non-probability",
        type=float,
        default=1.0,
        help="Probability to get the top most icd code in the hierarchy"
    )
    parser.add_argument(
        "--mixed-non-probability",
        type=float,
        default=0.0,
        help="Probability to get any code in the hierarchy"
    )
    parser.add_argument(
        "--lowercase-code-text",
        default=False,
        action="store_true",
        help="Whether to lowercase code text"
    )
    parser.add_argument(
        "--distance_threshold",
        type=int,
        default=60,
        help="Threshold to use for clustering encounters into bins",
    )
    parser.add_argument(
        "--shuffle-bins",
        default=False,
        action="store_true",
        help="Whether to shuffle bins within a code"
    )
    return parser