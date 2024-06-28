import numpy as np
import torch
import random
from typing import Union

from .demographics import DemographicPredictionTask, DemographicPredictionPrompt
from .labs import LabPredictionPrompt
from .diagnosis import DiagnosisLabelPredictionPrompt
from .instruct_tokenizer import InstructTokenizer


class InstructTasks(object):

    def __init__(
            self,
            task_list,
            instruct_tokenizer: InstructTokenizer,
            token_loss_weighting: bool
    ):
        self._task_list = task_list
        self._instruct_tokenizer = instruct_tokenizer
        self._token_loss_weighting = token_loss_weighting

    def process_sample_from_args(self, sample, args):
        task_instructions = list()
        if args.task_shuffle:
            random.shuffle(self._task_list)

        if args.add_img_token:
            task_instructions.append([self.get_img_sep_token_instruction(), '', True, False, -100])

        for task in self._task_list:
            if isinstance(task, DemographicPredictionPrompt):
                instructions = task.process_sample(
                    sample=sample, args=args, attributes=args.demographic_prompt_attributes
                )
            elif isinstance(task, LabPredictionPrompt):
                instructions = task.process_sample(
                    sample=sample, args=args, labs=args.lab_prompt_attributes
                )
            elif isinstance(task, DiagnosisLabelPredictionPrompt):
                instructions = task.process_sample(
                    sample=sample, args=args, diagnoses=args.diagnosis_prompt_attributes
                )
            elif isinstance(task, DemographicPredictionTask):
                instructions = task.process_sample(
                    sample=sample, args=args, ignore_instruction=self.get_ignore_instruction_demographics(
                        focal_loss=args.loss_function == 'focal'
                    )
                )
            else:
                instructions = task.process_sample(sample=sample, args=args)

            task_instructions.extend(instructions)

            # if len(instructions):
            #     task_instructions.append([self.get_task_separator_instruction(), '', True, False, -100])

        if self._instruct_tokenizer is None:
            return task_instructions

        input_ids, labels, weights = self._instruct_tokenizer.get_tokens(
            task_instructions=task_instructions, return_tensor=True
        )
        input_ids = input_ids.unsqueeze(0)
        labels = labels.unsqueeze(0)

        if self._token_loss_weighting:
            weights = weights.unsqueeze(0)
            return torch.cat([input_ids, labels, weights])
        else:
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
