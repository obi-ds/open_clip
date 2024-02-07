import torch
from typing import Union

from .multi_tokenizer import MultiTokenizer
from .multi_instruct_tokenizer import GPT2MultiInstructTokenizer
from .codes import CodeTrajectoryPredictionTask


class MultiInstructTrajectory(object):

    def __init__(
            self,
            code_trajectory_instruct_task: CodeTrajectoryPredictionTask,
            multi_instruct_tokenizer: Union[GPT2MultiInstructTokenizer, MultiTokenizer]
    ):
        self._code_trajectory_instruct_task = code_trajectory_instruct_task
        self._multi_instruct_tokenizer = multi_instruct_tokenizer

    def process_sample_from_args(self, sample, args):
        task_instructions = self._code_trajectory_instruct_task.process_sample(sample=sample, args=args)
        if self._multi_instruct_tokenizer is None:
            return task_instructions

        input_ids, labels = self._multi_instruct_tokenizer.get_tokens(
            multi_task_instructions=task_instructions, return_tensor=True
        )
        input_ids = input_ids.unsqueeze(0)
        labels = labels.unsqueeze(0)

        return torch.cat([input_ids, labels])
