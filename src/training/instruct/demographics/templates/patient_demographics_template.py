"""
Define the patient demographic prediction task - The task uses templates, inputs and targets.
This class will return the instruction string. It will make use of the  instruction data
and the instruction template to create the instruction string.
"""
from typing import Tuple, Union


class PatientDemographicsTemplate(object):
    """
    Define the code trajectory prediction task. Encapsulate the templates
    inputs, outputs to build the final instruction. The instruction is built based
    on the trajectory of encounters.
    """

    def __init__(
            self,
            inputs,
            targets: str = '',
            inputs_prefix: str = "",
            targets_prefix: str = "",
            x_y_delimiter: str = " ",
            example_separator: str = '\n',
            task_definition: str = '',
            task_name: str = 'patient_demographics'
    ):
        """
        Initialize the variables

        Args:
            inputs (str): The string that represents the future-time instruction input
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
        self._inputs = inputs
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
        return self._task_definition + '\n'

    def get_instruction(self, category: str, value: Union[str, int, float]) -> Tuple[str, str]:
        """


        Args:
            category (str): A demographic category
            value (Union[str, int, float]): A value to represent the category of the demographic

        Returns:
            (Tuple[str, str]): Tuple that contains the positive instruction input and instruction target
        """
        instruction_input = self.get_instruction_input(category=category)
        instruction_target = self.get_instruction_target(value=value)

        return instruction_input, instruction_target

    def get_instruction_input(self, category: str) -> str:
        """
        Add the diagnosis to the instruction input

        Args:
            category (str): A demographic category

        Returns:
            (str): Instruction input that contains the diagnosis
        """
        return self._inputs_prefix + self._inputs.format(category=category) + self._x_y_delimiter

    def get_instruction_target(self, value: Union[str, int, float]) -> str:
        """
        Return the code value - which represents some function of the code
        in the time bin - present or absent

        Args:
            value (Union[str, int, float]): A value associated with the diagnosis
            - e.g. count of occurrences of the code in the bin

        Returns:
            (str): The instruction target with the label
        """
        return str(value) + self._example_separator

    def get_example_separator(self) -> str:
        """
        This instruction is added as a separator between instructions

        Returns:
            (str): Example separator string
        """
        # \n is the separator string, '' is the label (empty - don't train)
        # True indicates ignore this instruction for training loss
        return self._example_separator

    def get_task_separator_instruction(self):
        """
        Insert this instruction after this task

        Returns:

        """
        return [self._example_separator, '', True, False, -100]
