import numpy as np
import torch
import random
from typing import Union

from .demographics import DemographicPredictionPrompt
from .labs import LabPredictionPrompt
from .instruct_tokenizer import GPT2InstructTokenizer, InstructTokenizer


class InstructPrompt(object):

    def __init__(
            self,
            task_list,
            instruct_tokenizer: Union[GPT2InstructTokenizer, InstructTokenizer],
    ):
        self._task_list = task_list
        self._instruct_tokenizer = instruct_tokenizer

    def process_sample_from_args(self, sample, args, demographic=None, lab=None):
        task_instructions = list()

        for task in self._task_list:
            if isinstance(task, DemographicPredictionPrompt) and demographic is not None:
                instructions = task.process_sample(
                    sample=sample, args=args, ignore_instruction=True, category=demographic
                )
            elif isinstance(task, LabPredictionPrompt) and lab is not None:
                instructions = task.process_sample(sample=sample, args=args, lab=lab)
            else:
                raise NotImplementedError()
            task_instructions.extend(instructions)

        if self._instruct_tokenizer is None:
            return task_instructions

        input_ids, _ = self._instruct_tokenizer.get_tokens(
            task_instructions=task_instructions, return_tensor=True
        )
        return input_ids

    def get_task_separator_instruction(self):
        if self._instruct_tokenizer is None:
            return '\n'
        else:
            return self._instruct_tokenizer.get_eos_token() + '\n'
