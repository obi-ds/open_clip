import argparse
import ast

import argparse

from .parameters.instruct.code_instruct import (
    get_code_instruct_arguments,
    get_encounter_file_arguments,
)
from .parameters.instruct import get_instruct_arguments
from .parameters.open_clip_default import get_open_clip_arguments

def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    model_name = model_name.lower()
    if "vit" in model_name:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        kw = {}
        for value in values:
            key, value = value.split('=')
            try:
                kw[key] = ast.literal_eval(value)
            except ValueError:
                kw[key] = str(value)  # fallback to string (avoid need to escape on command line)
        setattr(namespace, self.dest, kw)

def parse_args(args):

    code_prompt_arguments = get_code_instruct_arguments()
    encounter_file_arguments = get_encounter_file_arguments()
    open_clip_arguments = get_open_clip_arguments()
    prompt_arguments = get_instruct_arguments()

    parser = argparse.ArgumentParser(
        parents=[
            code_prompt_arguments,
            encounter_file_arguments,
            open_clip_arguments,
            prompt_arguments
        ],
    )

    args = parser.parse_args(args)

    # If some params are not passed, we use the default values based on model name.
    default_params = get_default_params(args.model)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)

    return args
