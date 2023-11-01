"""Define instruction templates for icd classification"""
from typing import Union

from .icd_instruction_template_interface import ICDInstructionTemplateInterface


class ICDStatusInstructionTemplate(ICDInstructionTemplateInterface):
    """
    Define the instruction template for tasks involving ICD status classification.
    """

    def __init__(
            self,
            inputs,
            targets,
            inputs_prefix: str = "",
            targets_prefix: str = "",
            x_y_delimiter: str = "  ",
            example_separator: str = "\n\n\n"
    ):
        """
        Initialize the variables
        Args:
            inputs (str): The string that represents the instruction input
            targets (str): The string that represents the instruction target
            inputs_prefix (str): Append this prefix to the instruction input
            targets_prefix (str): Append this prefix to the instruction target
            x_y_delimiter (str): Delimiter between instruction input and target
            example_separator (str): Delimiter between multiple instruction examples - few shot
        """
        super().__init__(
            inputs,
            targets,
            inputs_prefix,
            targets_prefix,
            x_y_delimiter,
            example_separator
        )

    def get_instruction_input(self, diagnosis: str, time: Union[str, int]) -> str:
        """
        Add the diagnosis and time of diagnosis to the instruction input
        Args:
            diagnosis (str): A medical/billing diagnosis
            time (Union[str, int]): The time of diagnosis in months from a given point in time

        Returns:
            (str): Instruction input that contains diagnosis and relative time
        """
        # If the time is 1, replace months with month.
        # We're assuming the "inputs" argument contains months - we need to make this
        # more flexible
        # TODO: Make flexible - remove hard coded month
        if time == 1:
            return (
                    self._inputs_prefix +
                    self._inputs.format(diagnosis=diagnosis, time=time).replace('months', 'month') +
                    self._x_y_delimiter
            )
        else:
            return self._inputs_prefix + self._inputs.format(diagnosis=diagnosis, time=time) + self._x_y_delimiter

    def get_instruction_target(self, answer: str) -> str:
        """
        Return the instruction target with the specified target
        Args:
            answer (str): The label that is present in the instruction target

        Returns:
            (str): The instruction target with the label
        """
        return self._targets_prefix + self._targets.format(answer=answer) + self._example_separator

class ICDStatusRangeInstructionTemplate(ICDInstructionTemplateInterface):
    """
    Define the instruction template for tasks involving ICD status classification.
    """

    def __init__(
            self,
            inputs,
            targets,
            inputs_prefix: str = "",
            targets_prefix: str = "",
            x_y_delimiter: str = "  ",
            example_separator: str = "\n\n\n"
    ):
        """
        Initialize the variables
        Args:
            inputs (str): The string that represents the instruction input
            targets (str): The string that represents the instruction target
            inputs_prefix (str): Append this prefix to the instruction input
            targets_prefix (str): Append this prefix to the instruction target
            x_y_delimiter (str): Delimiter between instruction input and target
            example_separator (str): Delimiter between multiple instruction examples - few shot
        """
        super().__init__(
            inputs,
            targets,
            inputs_prefix,
            targets_prefix,
            x_y_delimiter,
            example_separator
        )

    def get_instruction_input(self, diagnosis: str, start_time: Union[str, int], end_time: Union[str, int]) -> str:
        """
        Add the diagnosis and time of diagnosis to the instruction input
        Args:
            diagnosis (str): A medical/billing diagnosis
            start_time (Union[str, int]): The start time of diagnosis in months from a given point in time
            end_time (Union[str, int]): The end time of diagnosis in months from a given point in time

        Returns:
            (str): Instruction input that contains diagnosis and relative time
        """
        # If the time is 1, replace months with month.
        # We're assuming the "inputs" argument contains months - we need to make this
        # more flexible
        # TODO: Make flexible - remove hard coded month
        return self._inputs_prefix + self._inputs.format(
            diagnosis=diagnosis,
            start_time=start_time,
            end_time=end_time
        ) + self._x_y_delimiter

    def get_instruction_target(self, answer: str) -> str:
        """
        Return the instruction target with the specified target
        Args:
            answer (str): The label that is present in the instruction target

        Returns:
            (str): The instruction target with the label
        """
        return self._targets_prefix + self._targets.format(answer=answer) + self._example_separator


class ICDT2EInstructionTemplate(ICDInstructionTemplateInterface):
    """
    Define the instruction template for tasks involving ICD time to event prediction.
    """

    def __init__(
            self,
            inputs,
            targets,
            inputs_prefix: str = "",
            targets_prefix: str = "",
            x_y_delimiter: str = "  ",
            example_separator: str = "\n\n\n"
    ):
        """
        Initialize the variables
        Args:
            inputs (str): The string that represents the instruction input
            targets (str): The string that represents the instruction target
            inputs_prefix (str): Append this prefix to the instruction input
            targets_prefix (str): Append this prefix to the instruction target
            x_y_delimiter (str): Delimiter between instruction input and target
            example_separator (str): Delimiter between multiple instruction examples - few shot
        """
        super().__init__(
            inputs,
            targets,
            inputs_prefix,
            targets_prefix,
            x_y_delimiter,
            example_separator
        )

    def get_instruction_input(self, diagnosis: str, start_time: Union[str, int], end_time: Union[str, int]) -> str:
        """
        Add the diagnosis and time of diagnosis to the instruction input
        Args:
            diagnosis (str): A medical/billing diagnosis
            start_time (Union[str, int]): The start time of diagnosis in months from a given point in time
            end_time (Union[str, int]): The end time of diagnosis in months from a given point in time

        Returns:
            (str): Instruction input that contains diagnosis and relative time
        """
        # If the time is 1, replace months with month.
        # We're assuming the "inputs" argument contains months - we need to make this
        # more flexible
        # TODO: Make flexible - remove hard coded month
        return self._inputs_prefix + self._inputs.format(
            diagnosis=diagnosis,
            start_time=start_time,
            end_time=end_time
        ) + self._x_y_delimiter

    def get_instruction_target(self, answer: str) -> str:
        """
        Return the instruction target with the specified target
        Args:
            answer (str): The label that is present in the instruction target

        Returns:
            (str): The instruction target with the label
        """
        return self._targets_prefix + self._targets.format(answer=answer) + self._example_separator


