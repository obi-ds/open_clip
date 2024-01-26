"""
Define the diagnostic code status classification task - The task uses templates, inputs and targets.
This class will return the instruction string. It will make use of the  instruction data
and the instruction template to create the instruction string.
"""
from typing import Tuple, Union

from .code_trajectories_template import (
    TimeBinPeriodTemplate,
    TimeBinCodeTemplate,
    DescriptionInstructionTemplate
)


class CodeTrajectoryPredictionInstructions(object):
    """
    Define the code trajectory prediction task. Encapsulate the templates
    inputs, outputs to build the final instruction. The instruction is built based
    on the trajectory of encounters.
    """

    def __init__(
            self,
            time_bin_period_instruction_template: TimeBinPeriodTemplate,
            time_bin_code_instruction_template: TimeBinCodeTemplate,
            code_description_instruction_template: DescriptionInstructionTemplate,
            task_definition: str = '',
            task_name: str = 'code_trajectory_prediction'
    ):
        """
        Initialize the variables

        Args:
            time_bin_period_instruction_template (TimeBinPeriodTemplate): The template for converting the
            time bin to string
            time_bin_code_instruction_template (TimeBinCodeTemplate): The template for converting the
            time code attributes to string
            task_definition (str, defaults to): Add the task definition to the instruction
            task_name (str, defaults to `status_classification`): A name to define this task.
            Used to distinguish from other tasks
        """
        self.task_name = task_name
        self._task_definition = task_definition
        self._time_bin_period_instruction_template = time_bin_period_instruction_template
        self._time_bin_code_instruction_template = time_bin_code_instruction_template
        self._code_description_instruction_template = code_description_instruction_template

    def get_task_definition(self) -> str:
        """
        Return the task definition

        Returns:
            (str): String containing the definition of the task
        """
        return self._task_definition

    def get_code_instruction(self, diagnosis: str, value: Union[str, int, float]) -> Tuple[str, str]:
        """
        Given a positive diagnosis and the relative time of diagnosis
        return the instruction input and instruction target containing
        these attributes

        Args:
            diagnosis (str): A medical/billing diagnosis
            value (Union[str, int, float]): A value to represent the diagnosis in the bin

        Returns:
            (Tuple[str, str]): Tuple that contains the positive instruction input and instruction target
        """
        instruction_input = self._time_bin_code_instruction_template.get_instruction_input(diagnosis=diagnosis)
        instruction_target = self._time_bin_code_instruction_template.get_instruction_target(value=value)

        return instruction_input, instruction_target

    def get_time_instruction(self, start_time: Union[str, int], end_time: Union[str, int]) -> Tuple[str, str]:
        """
        Add the start time and end time to the instruction

        Args:
            start_time (Union[str, int]): The start time of the bin
            end_time (Union[str, int]): The end time of the bin

        Returns:
            (str): Instruction input that contains the bin start and end times
        """
        instruction_input = self._time_bin_period_instruction_template.get_instruction_input(
            start_time=start_time,
            end_time=end_time
        )
        instruction_target = self._time_bin_period_instruction_template.get_instruction_target()

        return instruction_input, instruction_target

    def get_code_description_instruction(self, diagnosis: str, description: str) -> Tuple[str, str]:
        """
        Given a positive diagnosis and the relative time of diagnosis
        return the instruction input and instruction target containing
        these attributes

        Args:
            diagnosis (str): A medical/billing diagnosis
            description (str): A textual description of the string

        Returns:
            (Tuple[str, str]): Tuple that contains the positive instruction input and instruction target
        """
        instruction_input = self._code_description_instruction_template.get_instruction_input(diagnosis=diagnosis)
        instruction_target = self._code_description_instruction_template.get_instruction_target(description=description)

        return instruction_input, instruction_target

