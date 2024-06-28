"""Parameters for eval prompts"""
import argparse


def get_prompt_eval_arguments():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--demographic-prompt-attributes",
        nargs='+',
        default=None,
        help="The demographic attributes to use in the prompt"
    )
    parser.add_argument(
        "--ecg-prompt-attributes",
        nargs='+',
        default=None,
        help="The ECG attributes to use in the prompt"
    )
    parser.add_argument(
        "--lab-prompt-attributes",
        type=str,
        default=None,
        help="The lab attributes to use in the prompt"
    )
    parser.add_argument(
        "--diagnosis-prompt-attributes",
        type=str,
        default=None,
        help="The diagnosis attributes to use in the prompt"
    )
    return parser