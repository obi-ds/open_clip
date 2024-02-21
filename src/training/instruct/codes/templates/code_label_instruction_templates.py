"""
Define the diagnostic code status classification task - The task uses templates, inputs and targets.
This class will return the instruction string. It will make use of the  instruction data
and the instruction template to create the instruction string.
"""
import numpy as np
from typing import Tuple, Union


class CodeLabelPredictionInstructionTemplate(object):
    """
    Define the code trajectory prediction task. Encapsulate the templates
    inputs, outputs to build the final instruction. The instruction is built based
    on the trajectory of encounters.
    """

    def __init__(
            self,
            past_input,
            future_input,
            targets: str = '',
            inputs_prefix: str = "",
            targets_prefix: str = "",
            x_y_delimiter: str = " ",
            example_separator: str = '\n',
            task_definition: str = '',
            task_name: str = 'code_label_prediction'
    ):
        """
        Initialize the variables

        Args:
            past_input (str): The string that represents the past-time instruction input
            future_input (str): The string that represents the future-time instruction input
            targets (str): The string that represents the instruction target
            inputs_prefix (str): Append this prefix to the instruction input
            targets_prefix (str): Append this prefix to the instruction target
            x_y_delimiter (str): Delimiter between instruction input and target
            time code attributes to string
            task_definition (str, defaults to): Add the task definition to the instruction
            task_name (str, defaults to `status_classification`): A name to define this task.
            Used to distinguish from other tasks
        """
        self._task_name = task_name
        self._task_definition = task_definition
        self._past_input = past_input
        self._future_input = future_input
        self._targets = targets
        self._inputs_prefix = inputs_prefix
        self._targets_prefix = targets_prefix
        self._x_y_delimiter = x_y_delimiter
        self._example_separator = example_separator

    def get_task_definition(self) -> str:
        """
        Return the task definition

        Returns:
            (str): String containing the definition of the task
        """
        return self._task_definition

    def get_instruction(self, diagnosis: str, time: int, value: Union[str, int, float]) -> Tuple[str, str]:
        """
        Given a positive diagnosis and the relative time of diagnosis
        return the instruction input and instruction target containing
        these attributes

        Args:
            diagnosis (str): A medical/billing diagnosis
            time (int): The time of diagnosis
            value (Union[str, int, float]): A value to represent the status of the diagnosis

        Returns:
            (Tuple[str, str]): Tuple that contains the positive instruction input and instruction target
        """
        if time < 0:
            instruction_input = self.get_past_instruction_input(diagnosis=diagnosis, time=str(np.abs(time)))
        else:
            instruction_input = self.get_future_instruction_input(diagnosis=diagnosis, time=str(np.abs(time)))
        instruction_input = self.fix_time_string(instruction_input, np.abs(time))
        instruction_target = self.get_instruction_target(value=value)

        return instruction_input, instruction_target

    def get_past_instruction_input(self, diagnosis: str, time: str) -> str:
        """
        Add the diagnosis to the instruction input

        Args:
            diagnosis (str): A medical/billing diagnosis
            time (str): The time of diagnosis

        Returns:
            (str): Instruction input that contains the diagnosis
        """
        return self._inputs_prefix + self._past_input.format(diagnosis=diagnosis, time=time) + self._x_y_delimiter

    def get_future_instruction_input(self, diagnosis: str, time: str) -> str:
        """
        Add the diagnosis to the instruction input

        Args:
            diagnosis (str): A medical/billing diagnosis
            time (str): The time of diagnosis

        Returns:
            (str): Instruction input that contains the diagnosis
        """
        return self._inputs_prefix + self._future_input.format(diagnosis=diagnosis, time=time) + self._x_y_delimiter


    def get_instruction_target(self, value: Union[str, int, float]) -> str:
        """
        Return the code value - which represents some function of the code
        in the time bin

        Args:
            value (Union[str, int, float]): A value associated with the diagnosis
            - e.g. count of occurrences of the code in the bin

        Returns:
            (str): The instruction target with the label
        """
        return str(value)

    def fix_time_string(self, instruction_input, time):
        if time == 1:
            return instruction_input.replace('months', 'month')
        else:
            return instruction_input

    def get_example_separator_instruction(self) -> Tuple[str, str, bool]:
        """
        This instruction is added as a separator between instructions

        Returns:
            (List[str, str, bool]): Example separator instruction
        """
        # \n is the separator string, '' is the label (empty - don't train)
        # True indicates ignore this instruction for training loss
        return self._example_separator, '', True


