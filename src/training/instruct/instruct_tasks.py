import numpy as np
import torch
import random
from typing import Union

from .demographics import DemographicPredictionTask, DemographicPredictionPrompt
from .instruct_tokenizer import HFInstructTokenizer, InstructTokenizer


class InstructTasks(object):

    def __init__(
            self,
            task_list,
            instruct_tokenizer: Union[HFInstructTokenizer, InstructTokenizer],
    ):
        self._task_list = task_list
        self._instruct_tokenizer = instruct_tokenizer

    def process_sample_from_args(self, sample, args):
        task_instructions = list()
        if args.task_shuffle:
            random.shuffle(self._task_list)

        if args.add_img_token:
            task_instructions.append([self.get_img_sep_token_instruction(), '', True, False])

        for task in self._task_list:
            if isinstance(task, DemographicPredictionPrompt):
                instructions = task.process_sample(
                    sample=sample, args=args, attributes=args.demographic_prompt_attributes
                )
            elif isinstance(task, DemographicPredictionTask):
                instructions = task.process_sample(
                    sample=sample, args=args, ignore_instruction=self.get_ignore_instruction_demographics(
                        focal_loss=args.focal_loss
                    )
                )
            else:
                instructions = task.process_sample(sample=sample, args=args)
            task_instructions.extend(instructions)
            if len(instructions):
                task_instructions.append([self.get_task_separator_instruction(), '', True, False])

        if self._instruct_tokenizer is None:
            return task_instructions

        input_ids, labels = self._instruct_tokenizer.get_tokens(
            task_instructions=task_instructions, return_tensor=True
        )
        input_ids = input_ids.unsqueeze(0)
        labels = labels.unsqueeze(0)

        return torch.cat([input_ids, labels])

    def get_task_separator_instruction(self):
        if self._instruct_tokenizer is None:
            return '\n'
        else:
            return self._instruct_tokenizer.get_eos_token() + '\n'

    def get_img_sep_token_instruction(self):
        if self._instruct_tokenizer is None:
            return '\n'
        else:
            return self._instruct_tokenizer.get_bos_token() + '\n'

    @staticmethod
    def get_ignore_instruction_demographics(focal_loss):
        """
        Demographics overfit - so we don't always train on it

        Returns:

        """
        if focal_loss:
            return np.random.rand() > 1
        else:
            return np.random.rand() > 0.5
