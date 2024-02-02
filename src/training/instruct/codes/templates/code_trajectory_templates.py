"""
Define instruction templates for code trajectory with
time bins
"""
from typing import Union

from .instruction_template import InstructionTemplateInterface


class TimeBinPeriodTemplate(InstructionTemplateInterface):
    """
    The class defines the template that will be used when creating the
    string that will contain the positions indicating the start and end of
    the time bins
    """
    def __init__(
            self,
            inputs,
            targets: str = '',
            inputs_prefix: str = "",
            targets_prefix: str = "",
            x_y_delimiter: str = " :\n",
            example_separator: str = " :\n\n\n"
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

    def get_instruction_input(self, start_time: Union[str, int], end_time: Union[str, int]) -> str:
        """
        Add the start time and end time of the bin to the instruction

        Args:
            start_time (Union[str, int]): The start time of the bin
            end_time (Union[str, int]): The end time of the bin

        Returns:
            (str): Instruction input that contains the bin start and end times
        """
        if start_time == end_time:
            input_string = self._inputs[:self._inputs.find('}') + 1]
        else:
            input_string = self._inputs
        return self._inputs_prefix + input_string.format(start_time=start_time, end_time=end_time) + self._x_y_delimiter

    def get_instruction_target(self) -> str:
        """
        Return empty string - because we don't train on the string
        containing the time bins - only instruction input is used

        Returns:
            (str): The instruction target with the label
        """
        return ''


class TimeBinCodeTemplate(InstructionTemplateInterface):
    """
    The class defines the template that will be used when creating the
    string that will contain the codes and some code value within the time bin
    """
    def __init__(
            self,
            inputs,
            targets: str = '',
            inputs_prefix: str = "",
            targets_prefix: str = "",
            x_y_delimiter: str = " : ",
            example_separator: str = "\n"
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

    def get_instruction_input(self, diagnosis: str) -> str:
        """
        Add the diagnosis to the instruction input

        Args:
            diagnosis (str): A medical/billing diagnosis

        Returns:
            (str): Instruction input that contains the diagnosis
        """
        return self._inputs_prefix + self._inputs.format(diagnosis=diagnosis) + self._x_y_delimiter

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
        return str(value) + self._example_separator


class DescriptionInstructionTemplate(InstructionTemplateInterface):
    """
    Define the instruction template for tasks involving code status classification.
    """

    def __init__(
            self,
            inputs,
            targets,
            inputs_prefix: str = "",
            targets_prefix: str = "",
            x_y_delimiter: str = " : ",
            example_separator: str = "\n"
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

    def get_instruction_input(self, diagnosis: str) -> str:
        """
        Add the diagnosis code to the instruction input

        Args:
            diagnosis (str): A medical/billing diagnosis

        Returns:
            (str): Instruction input that contains diagnosis
        """
        return self._inputs_prefix + self._inputs.format(diagnosis=diagnosis) + self._x_y_delimiter

    def get_instruction_target(self, description: str) -> str:
        """
        Return the instruction target with the specified target - possibly a description
        of the diagnosis

        Args:
            description (str): The label that is present in the instruction target

        Returns:
            (str): The instruction target with the label
        """
        return self._targets_prefix + self._targets.format(description=description) + self._example_separator