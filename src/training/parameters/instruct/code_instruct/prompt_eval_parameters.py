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
    return parser