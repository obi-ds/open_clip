"""Make training data"""
import numpy as np
from typing import Sequence, Any

from .instruct_interface import InstructInterface


class ICDInstruct(InstructInterface):
    """
    Make training data from icd codes
    """

    def __init__(
            self,
            instruction_tasks: Sequence[Any],
            tokenizer,
            max_seq_length: int = 77,
            pad_id: int = 0,
    ):
        """
        Initialize variables
        Args:
            pre-defined list
            instruction_tasks (Sequence[Any]): A list of icd instruction tasks. At any point one of these
            tasks is randomly chosen and trained on
            tokenizer (): The tokenizer used for training
            max_seq_length (int, defaults to `77`): The maximum sequence length allowed by model/tokenizer
            pad_id (int, defaults to `0`): The id used for padding tokens
        """
        super().__init__(tokenizer=tokenizer, max_seq_length=max_seq_length, pad_id=pad_id)
        self._instruction_tasks = instruction_tasks

    def process_sample_from_args(self, sample, args):
        instruction_task = np.random.choice(self._instruction_tasks)
        instructions = instruction_task.process_sample_from_args(
            sample=sample,
            args=args,
        )
        input_text, output_text = self.process_instructions(instructions=instructions)
        return self.tokenize_instruction(input_text=input_text, output_text=output_text)