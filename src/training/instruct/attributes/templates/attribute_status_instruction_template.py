"""Define instruction templates for code classification"""
from typing import Union

from .attribute_instruction_template_interface import AttributeInstructionTemplateInterface


class AttributeStatusInstructionTemplate(AttributeInstructionTemplateInterface):
    """
    Define the instruction template for tasks involving patient attributes (e.g Age).
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

    def get_instruction_input(self, attribute: str) -> str:
        """
        Add the patient attribute to the instruction input
        Args:
            attribute (str): The patient attribute

        Returns:
            (str): Instruction input that contains patient attribute
        """
        return self._inputs_prefix + self._inputs.format(attribute=attribute) + self._x_y_delimiter

    def get_instruction_target(self, answer: str) -> str:
        """
        Return the instruction target with the specified target
        Args:
            answer (str): The label that is present in the instruction target

        Returns:
            (str): The instruction target with the label
        """
        return self._targets_prefix + self._targets.format(answer=answer) + self._example_separator
