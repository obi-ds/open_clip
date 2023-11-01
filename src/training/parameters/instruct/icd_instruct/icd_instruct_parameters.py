"""Parameters for train prompts"""
import argparse


def get_icd_instruct_arguments():
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
        "--random-negative-probability",
        type=float,
        default=1.0,
        help="Probability of sampling negative codes randomly"
    )
    parser.add_argument(
        "--lock-range",
        default=False,
        action="store_true",
        help="Whether to used fixed time ranges in the prompt"
    )
    parser.add_argument(
        "--lowercase_icd_text",
        default=False,
        action="store_true",
        help="Whether to lowercase icd text"
    )
    return parser