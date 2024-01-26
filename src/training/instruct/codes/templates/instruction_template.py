"""
Define instruction templates for code classification
The classes will define the format of the instruction
"""


class InstructionTemplateInterface(object):
    """
    Define the instruction template interface for tasks involving diagnostic codes.
    """

    def __init__(
            self,
            inputs: str,
            targets: str,
            inputs_prefix: str = "",
            targets_prefix: str = "",
            x_y_delimiter: str = "\n\n",
            example_separator: str = "\n\n\n",
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

        self._inputs = inputs
        self._targets = targets
        self._inputs_prefix = inputs_prefix
        self._targets_prefix = targets_prefix
        self._x_y_delimiter = x_y_delimiter
        self._example_separator = example_separator

    def get_instruction_input(self, *args) -> str:
        """
        Returns the instruction input - behaviour is defined by subclass

        Args:
            *args: Arguments accepted by subclass

        Returns:
            (str): The instruction input
        """
        raise NotImplementedError('Implement in subclass')

    def get_instruction_target(self, *args):
        """
        Returns the instruction target - behaviour is defined by subclass

        Args:
            *args: Arguments accepted by subclass

        Returns:
            (str): The instruction target
        """
        raise NotImplementedError('Implement in subclass')
