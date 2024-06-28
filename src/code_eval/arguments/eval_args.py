import argparse

from training.parameters.instruct.code_instruct import (
    get_code_instruct_arguments,
    get_encounter_file_arguments,
    get_code_eval_arguments,
    get_prompt_eval_arguments
)
from training.parameters.instruct import get_instruct_arguments
from training.parameters.open_clip_default import get_open_clip_arguments

def parse_args(args):

    code_prompt_arguments = get_code_instruct_arguments()
    encounter_file_arguments = get_encounter_file_arguments()
    open_clip_arguments = get_open_clip_arguments()
    prompt_arguments = get_instruct_arguments()
    code_eval_arguments = get_code_eval_arguments()
    prompt_eval_arguments = get_prompt_eval_arguments()
    eval_arguments = get_eval_arguments()

    parser = argparse.ArgumentParser(
        parents=[
            code_prompt_arguments,
            encounter_file_arguments,
            open_clip_arguments,
            prompt_arguments,
            code_eval_arguments,
            prompt_eval_arguments,
            eval_arguments
        ],
    )

    args = parser.parse_args(args)

    return args


def get_eval_arguments():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--gpu",
        type=str,
        required=True,
        help="The GPU to run eval on",
    )
    parser.add_argument(
        "--phecode-file",
        type=str,
        required=True,
        help="The location to the phecode file",
    )
    parser.add_argument(
        "--start",
        type=int,
        required=True,
        help="The start position of the phecode file",
    )
    parser.add_argument(
        "--end",
        type=int,
        required=True,
        help="The end position of the phecode file",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        required=True,
        help="Where to write the binned data",
    )
    parser.add_argument(
        "--file-suffix",
        type=str,
        required=True,
        help="A suffix to distinguish between different dataset",
    )
    parser.add_argument(
        "--overwrite",
        type=bool,
        default=False,
        help="Overwrite existing results",
    )
    return parser
