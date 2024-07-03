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
        "--k-shot",
        type=int,
        nargs='+',
        default=None,
        help="The number of k_shots to use",
    )
    parser.add_argument(
        "--k-shot-demographics",
        type=int,
        nargs='+',
        default=None,
        help="The number of k_shots to use for demographic data",
    )
    parser.add_argument(
        "--k-shot-labs",
        type=int,
        nargs='+',
        default=None,
        help="The number of k_shots to use for demographic data",
    )
    parser.add_argument(
        "--k-shot-ecg-attributes",
        type=int,
        nargs='+',
        default=None,
        help="The number of k_shots to use for demographic data",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs='+',
        default=None,
        help="The training tasks",
    )
    parser.add_argument(
        "--task-shuffle",
        default=False,
        action="store_true",
        help="Whether to shuffle the tasks"
    )
    parser.add_argument(
        "--add-img-token",
        default=False,
        action="store_true",
        help="Whether to add a special token to separate image and text - applicable for decoder only model"
    )
    parser.add_argument(
        "--token-loss-weighting",
        default=False,
        action="store_true",
        help="Whether to do a instance weighted token loss"
    )
    parser.add_argument(
        "--eval-mode",
        default=False,
        action="store_true",
        help="Whether to run the main script in eval mode only"
    )
    parser.add_argument(
        "--loss-function",
        default=None,
        choices=['clip', 'coca', 'focal', 'lm', 'lm_z'],
        type=str,
        help="What loss function to use for training"
    )
    return parser