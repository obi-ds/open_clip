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
        "--distance-threshold",
        type=int,
        nargs='+',
        help="Threshold to use for clustering encounters into bins - can specify multiple thresholds",
    )
    parser.add_argument(
        "--shuffle-bins",
        default=False,
        action="store_true",
        help="Whether to shuffle bins within a code"
    )
    parser.add_argument(
        '--negatives-type',
        default=None,
        type=str,
        choices=['random', 'cached', 'random_cached'],
        help="How to sample negatives"
    )
    parser.add_argument(
        "--time-period-range",
        nargs='+',
        default=None,
        help="If we want to train or evaluate with fixed time periods",
    )
    parser.add_argument(
        "--future-only",
        default=False,
        action="store_true",
        help="Whether to only make predictions for future time periods"
    )
    parser.add_argument(
        "--past-only-labs",
        default=False,
        action="store_true",
        help="Whether to only make predictions for future time periods"
    )
    parser.add_argument(
        "--fixed-position-range",
        default=False,
        action="store_true",
        help="Use position ranges - 1, 3, 6, 12"
    )
    parser.add_argument(
        "--include-reference-range",
        default=False,
        action="store_true",
        help="Include reference ranges for lab values"
    )
    parser.add_argument(
        "--time-negatives-buffer",
        default=None,
        type=int,
        help="Make negatives stricter - a code is a negative if it doesn't occur in the prediction range +- this buffer"
    )
    parser.add_argument(
        "--fine-tune-code",
        type=str,
        default=None,
        help="The code to fine tune when training type is single"
    )
    parser.add_argument(
        "--update-code-counts",
        default=False,
        action="store_true",
        help="Update code counts and IDFs"
    )
    parser.add_argument(
        "--update-lab-counts",
        default=False,
        action="store_true",
        help="Update lab counts and IDFs"
    )
    parser.add_argument(
        "--training-eval-codes",
        type=str,
        nargs='+',
        help="The codes we want to evaluate during training",
    )


    return parser