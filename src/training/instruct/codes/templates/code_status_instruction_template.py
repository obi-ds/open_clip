"""Define instruction templates for code classification"""
from typing import Union

from .code_instruction_template_interface import CodeInstructionTemplateInterface


class CodeStatusInstructionTemplate(CodeInstructionTemplateInterface):
    """
    Define the instruction template for tasks involving code status classification.
    """

    def __init__(
            self,
            inputs,
            targets,
            inputs_prefix: str = "",
            targets_prefix: str = "",
            x_y_delimiter: str = ": ",
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


class CodeT2EInstructionTemplate(CodeInstructionTemplateInterface):
    """
    Define the instruction template for tasks involving diagnostic time to event prediction.
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

    def get_instruction_input(self, diagnosis: str) -> str:
        """
        Add the diagnosis and time of diagnosis to the instruction input
        Args:
            diagnosis (str): A medical/billing diagnosis

        Returns:
            (str): Instruction input that contains diagnosis and relative time
        """
        return self._inputs_prefix + self._inputs.format(
            diagnosis=diagnosis,
        ) + self._x_y_delimiter

    def get_instruction_target(self, sign: str, answer: Union[str, int]) -> str:
        """
        Return the instruction target with the specified target
        Args:
            sign (str): Indicate whether it is a positive or negative number (+ or -)
            answer (Union[str, int]): The label that is present in the instruction target

        Returns:
            (str): The instruction target with the label
        """
        return self._targets_prefix + self._targets.format(sign=sign, answer=answer) + self._example_separator


