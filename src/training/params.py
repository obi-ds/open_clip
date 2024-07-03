import argparse

from .parameters.instruct.code_instruct import (
    get_code_instruct_arguments,
    get_encounter_file_arguments,
    get_code_eval_arguments,
    get_prompt_eval_arguments
)
from .parameters.instruct import get_instruct_arguments
from .parameters.open_clip_default import get_open_clip_arguments

def parse_args(args):

    code_prompt_arguments = get_code_instruct_arguments()
    encounter_file_arguments = get_encounter_file_arguments()
    open_clip_arguments = get_open_clip_arguments()
    prompt_arguments = get_instruct_arguments()
    code_eval_arguments = get_code_eval_arguments()
    prompt_eval_arguments = get_prompt_eval_arguments()

    parser = argparse.ArgumentParser(
        parents=[
            code_prompt_arguments,
            encounter_file_arguments,
            open_clip_arguments,
            prompt_arguments,
            code_eval_arguments,
            prompt_eval_arguments
        ],
    )

    args = parser.parse_args(args)

    return args
