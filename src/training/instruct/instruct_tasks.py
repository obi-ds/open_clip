import torch
from typing import Union

from .instruct_tokenizer import GPT2InstructTokenizer, InstructTokenizer
from .codes import CodeLabelPredictionTask


class InstructTasks(object):

    def __init__(
            self,
            code_instruct_task: CodeLabelPredictionTask,
            instruct_tokenizer: Union[GPT2InstructTokenizer, InstructTokenizer]
    ):
        self._code_trajectory_instruct_task = code_instruct_task
        self._instruct_tokenizer = instruct_tokenizer

    def process_sample_from_args(self, sample, args):
        task_instructions = self._code_trajectory_instruct_task.process_sample(sample=sample, args=args)
        if self._instruct_tokenizer is None:
            return task_instructions

        input_ids, labels = self._instruct_tokenizer.get_tokens(
            task_instructions=task_instructions, return_tensor=True
        )
        input_ids = input_ids.unsqueeze(0)
        labels = labels.unsqueeze(0)

        return torch.cat([input_ids, labels])
