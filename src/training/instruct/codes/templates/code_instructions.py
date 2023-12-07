"""Define the diagnostic code status classification task - The task uses templates, inputs and targets"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Union

from .code_status_instruction_template import (
    CodeStatusInstructionTemplate,
    CodeT2EInstructionTemplate
)

class CodeStatusInstructionsInterface(object):
    """
    Define the diagnostic code status classification task interface. Encapsulate the templates
    inputs, outputs to build the final instruction. The instruction is built based
    on whether events occurred in the past/future and are positive/negative
    """

    def __init__(
            self,
            instruction_template_future: CodeStatusInstructionTemplate,
            instruction_template_past: CodeStatusInstructionTemplate,
            positive_answer_target: str,
            negative_answer_target: str,
            task_definition: str = '',
            task_name: str = 'status_classification_interface'
    ):
        """
        Initialize the variables
        Args:
            instruction_template_future (CodeStatusInstructionTemplate): The template for code status for future events.
            instruction_template_past (CodeStatusInstructionTemplate): The template for code status for past events.
            positive_answer_target (str):
            negative_answer_target (str):
            task_definition (str, defaults to): Add the task definition to the instruction
            task_name (str, defaults to `status_classification`): A name to define this task.
            Used to distinguish from other tasks
        """
        self.task_name = task_name
        self._task_definition = task_definition
        self._instruction_template_future = instruction_template_future
        self._instruction_template_past = instruction_template_past
        self._positive_answer_target = positive_answer_target
        self._negative_answer_target = negative_answer_target

    def get_task_definition(self) -> str:
        """
        Return the task definition
        Returns:
            (str): String containing the definition of the task
        """
        return self._task_definition

    def get_instruction(self, diagnosis: str, position: int, target: str) -> Tuple[str, str]:
        """
        Given a positive diagnosis and the relative time of diagnosis
        return the instruction input and instruction target containing
        these attributes
        Args:
            diagnosis (str): A medical/billing diagnosis
            position (int): The relative time of diagnosis
            target (str): The answer for the prompt

        Returns:
            (Tuple[str, str]): Tuple that contains the positive instruction input and instruction target
        """
        raise NotImplementedError('Implement in subclass')

    def get_positive_instruction(self, diagnosis: str, position: int) -> Tuple[str, str]:
        """
        Given a positive diagnosis and the relative time of diagnosis
        return the instruction input and instruction target containing
        these attributes
        Args:
            diagnosis (str): A medical/billing diagnosis
            position (int): The relative time of diagnosis

        Returns:
            (Tuple[str, str]): Tuple that contains the positive instruction input and instruction target
        """
        raise NotImplementedError('Implement in subclass')

    def get_negative_instruction(
            self,
            diagnosis: str,
            position: int,
    ) -> Tuple[str, str]:
        """
        Given a diagnosis and the relative time of diagnosis
        return the instruction input and instruction target such that
        the position/time is replaced such that the patient should
        not have been diagnosed with the diagnosis at that time
        Args:
            diagnosis (str): A medical/billing diagnosis
            position (int): The relative time of diagnosis

        Returns:
            (Tuple[str, str]): Tuple that contains the negative instruction input and instruction target
        """
        raise NotImplementedError('Implement in subclass')

    def get_positive_instructions(
            self,
            diagnosis_list: pd.Series,
            positions_list: pd.Series
    ) -> List[Tuple[str, str]]:
        """
        Given a list of positive diagnoses and the relative time of diagnosis
        return a list that contains the instruction input and instruction target
        containing these attributes for each element in the list
        Args:
            diagnosis_list (pd.Series): Series containing medical/billing diagnosis
            positions_list (pd.Series): Series containing the relative time of diagnosis

        Returns:
            (List[Tuple[str, str]]): List that contains the instruction input and instruction target
            for each input element
        """
        raise NotImplementedError('Implement in subclass')

    def get_negative_instructions(
            self,
            diagnosis_list: pd.Series,
            positions_list: pd.Series
    ) -> List[Tuple[str, str]]:
        """
        Given a list of diagnoses and the relative time of them
        return a list that contains the instruction input and instruction target
        such that the position/time is replaced such that the patient should
        not have been diagnosed with the diagnosis at that time
        Args:
            diagnosis_list (pd.Series): Series containing medical/billing diagnosis
            positions_list (pd.Series): Series containing the relative time of diagnosis

        Returns:
            (List[Tuple[str, str]]): List that contains the instruction input and instruction target
            for each input element
        """
        raise NotImplementedError('Implement in subclass')

    def get_future_input(self, diagnosis: str, time: int) -> str:
        """
        Return the instruction input, given a diagnosis and position occurring in the future
        Args:
            diagnosis (str): A medical/billing diagnosis
            time (Union[str, int]): The relative time of diagnosis

        Returns:
            (str): Instruction input for a future time period

        """
        return self._instruction_template_future.get_instruction_input(
            diagnosis=diagnosis,
            time=time
        )

    def get_past_input(self, diagnosis: str, time: int) -> str:
        """
        Return the instruction input, given a diagnosis and position occurring in the future
        Args:
            diagnosis (str): A medical/billing diagnosis
            time (Union[str, int]): The relative time of diagnosis

        Returns:
            (str): Instruction input for a pastime period

        """
        return self._instruction_template_past.get_instruction_input(
            diagnosis=diagnosis,
            time=time
        )

    def get_future_output(self, target: str) -> str:
        """
        Return the instruction input, given an instruction target label
        Args:
            target (str): Instruction target label

        Returns:
            (str): Instruction target for a future time period

        """
        return self._instruction_template_future.get_instruction_target(
            answer=target
        )

    def get_past_output(self, target: str) -> str:
        """
        Return the instruction input, given an instruction target label
        Args:
            target (str): Instruction target label

        Returns:
            (str): Instruction target for a pastime period

        """
        return self._instruction_template_past.get_instruction_target(
            answer=target
        )

class CodeStatusClassificationInstructions(CodeStatusInstructionsInterface):
    """
    Define the diagnostic status classification task. Encapsulate the templates
    inputs, outputs to build the final instruction. The instruction is built based
    on whether events occurred in the past/future and are positive/negative
    """

    def __init__(
            self,
            instruction_template_future: CodeStatusInstructionTemplate,
            instruction_template_past: CodeStatusInstructionTemplate,
            positive_answer_target: str,
            negative_answer_target: str,
            task_definition: str = '',
            task_name: str = 'status_range_classification'
    ):
        """
        Initialize the variables
        Args:
            instruction_template_future (CodeStatusInstructionTemplate): The template for code status for future events.
            instruction_template_past (CodeStatusInstructionTemplate): The template for code status for past events.
            positive_answer_target (str): The string used to represent a positive diagnosis
            negative_answer_target (str): The string used to represent a negative diagnosis
            select a random range from a given start and end time
            task_definition (str, defaults to): Add the task definition to the instruction
            task_name (str, defaults to `status_range_classification`): A name to define this task.
            Used to distinguish from other tasks
        """
        super().__init__(
            instruction_template_future,
            instruction_template_past,
            positive_answer_target,
            negative_answer_target,
            task_definition,
            task_name
        )

    def get_instruction(
            self,
            diagnosis: str,
            position: float,
            target: str,
    ) -> Tuple[str, str]:
        """
        Given a positive diagnosis and the relative time of diagnosis
        return the instruction input and instruction target containing
        these attributes, and a time range for this diagnosis
        Args:
            diagnosis (str): A medical/billing diagnosis
            position (float): The relative time of diagnosis
            target (str): The answer for the prompt

        Returns:
            (Tuple[str, str]): Tuple that contains the positive instruction input and instruction target
        """

        if position >= 0:
            instruction_input = self.get_future_input(diagnosis=diagnosis, time=int(position))
            instruction_target = self.get_future_output(target=target)
        else:
            instruction_input = self.get_past_input(diagnosis=diagnosis, time=np.abs(int(position)))
            instruction_target = self.get_past_output(target=target)
        return instruction_input, instruction_target

    def get_positive_instruction(
            self,
            diagnosis: str,
            position: float,
    ) -> Tuple[str, str]:
        """
        Given a positive diagnosis and the relative time of diagnosis
        return the instruction input and instruction target containing
        these attributes, and a time range for this diagnosis
        Args:
            diagnosis (str): A medical/billing diagnosis
            position (float): The relative time of diagnosis

        Returns:
            (Tuple[str, str]): Tuple that contains the positive instruction input and instruction target
        """
        return self.get_instruction(
            diagnosis=diagnosis,
            position=position,
            target=self._positive_answer_target,
        )

    def get_negative_instruction(
            self,
            diagnosis: str,
            position: float,
    ) -> Tuple[str, str]:
        """
        Given a diagnosis and the relative time of diagnosis
        return the instruction input and instruction target such that
        the position/time is replaced such that the patient should
        not have been diagnosed with the diagnosis at that time.
        This is done w.r.t a time range for the diagnosis
        Args:
            diagnosis (str): A medical/billing diagnosis
            position (float): The relative time of diagnosis

        Returns:
            (Tuple[str, str]): Tuple that contains the negative instruction input and instruction target
        """

        # Get a time period to represent the negative diagnosis
        # Essentially the diagnosis was not diagnosed during this period

        # If the value is positive use future instruction template
        # If a diagnosis occurs between 0 and 1, we do negatives in the pas
        # because the diagnosis occurred in the lowest future range - we can't do
        # did the diagnosis occur between months 0 & 1 and set the answer to negative
        # This is assuming negative codes are sampled from patients history - where we
        # are using the time range to sample negatives. If diagnosis of I48 was in months 3-4
        # we could set diagnosis of I48 between months 0-1 as negative
        # However is lock range is True, we assume all negatives are sampled randomly
        # and don't have this constraint
        return self.get_instruction(
            diagnosis=diagnosis,
            position=position,
            target=self._negative_answer_target,
        )

    def get_positive_instructions(
            self,
            diagnosis_list: pd.Series,
            positions_list: pd.Series,

    ) -> List[Tuple[str, str]]:
        """
        Given a list of positive diagnoses and the relative time of diagnosis
        return a list that contains the instruction input and instruction target
        containing these attributes for each element in the list
        Args:
            diagnosis_list (pd.Series): Series containing medical/billing diagnosis
            positions_list (pd.Series): Series containing the relative time of diagnosis

        Returns:
            (List[Tuple[str, str]]): List that contains the instruction input and instruction target
            for each input element
        """
        return [
            self.get_positive_instruction(
                diagnosis=diagnosis,
                position=position,
            )
            if position < 0 else
            self.get_positive_instruction(
                diagnosis=diagnosis,
                position=position,
            )
            for diagnosis, position in zip(diagnosis_list, positions_list)
        ]

    def get_negative_instructions(
            self,
            diagnosis_list: pd.Series,
            positions_list: pd.Series,
    ) -> List[Tuple[str, str]]:
        """
        Given a list of positive diagnoses and the relative time of diagnosis
        return a list that contains the instruction input and instruction target
        containing these attributes for each element in the list
        Args:
            diagnosis_list (pd.Series): Series containing medical/billing diagnosis
            positions_list (pd.Series): Series containing the relative time of diagnosis

        Returns:
            (List[Tuple[str, str]]): List that contains the instruction input and instruction target
            for each input element
        """
        return [
            self.get_negative_instruction(
                diagnosis=diagnosis,
                position=position,
            )
            if position < 0
            else self.get_negative_instruction(
                diagnosis=diagnosis,
                position=position,
            )
            for diagnosis, position in zip(diagnosis_list, positions_list)
        ]

class CodeT2EPredictionInstructions(object):
    """
    Define the code time to event prediction task. Encapsulate the templates
    inputs, outputs to build the final instruction. The instruction is built based
    on whether events occurred in the past/future and are positive/negative
    """

    def __init__(
            self,
            instruction_template: CodeT2EInstructionTemplate,
            task_definition: str = '',
            task_name: str = 't2e_prediction',
            negative_answer_target=np.inf,
    ):
        """
        Initialize the variables
        Args:
            instruction_template (CodeT2EInstructionTemplate): The template for diagnostic code
             time to event predictions.
            task_definition (str, defaults to): Add the task definition to the instruction
            task_name (str, defaults to `t2e_prediction`): A name to define this task.
            Used to distinguish from other tasks
            negative_answer_target (float, defaults to `inf`): The time to event value used for negatives
        """
        self.task_name = task_name
        self._task_definition = task_definition
        self._instruction_template = instruction_template
        self._negative_answer_target = negative_answer_target

    def get_task_definition(self) -> str:
        """
        Return the task definition
        Returns:
            (str): String containing the definition of the task
        """
        return self._task_definition

    def get_positive_instruction(self, diagnosis: str, position: float) -> Tuple[str, str]:
        """
        Given a positive diagnosis and the relative time of diagnosis
        return the instruction input and instruction target containing
        these attributes
        Args:
            diagnosis (str): A medical/billing diagnosis
            position (float): The relative time of diagnosis

        Returns:
            (Tuple[str, str]): Tuple that contains the positive instruction input and instruction target
        """
        instruction_input = self.get_input(diagnosis=diagnosis)
        instruction_target = self.get_output(position=position)

        return instruction_input, instruction_target

    def get_negative_instruction(
            self,
            diagnosis: str,
            position: int,
    ) -> Tuple[str, str]:
        """
        Given a diagnosis and the relative time of diagnosis
        return the instruction input and instruction target such that
        the position/time is replaced such that the patient should
        not have been diagnosed with the diagnosis at that time
        Args:
            diagnosis (str): A medical/billing diagnosis
            position (float): The relative time of diagnosis

        Returns:
            (Tuple[str, str]): Tuple that contains the negative instruction input and instruction target
        """
        instruction_input = self.get_input(diagnosis=diagnosis)
        instruction_target = self.get_output(position=self._negative_answer_target)

        return instruction_input, instruction_target

    def get_positive_instructions(
            self,
            diagnosis_list: pd.Series,
            positions_list: pd.Series
    ) -> List[Tuple[str, str]]:
        """
        Given a list of positive diagnoses and the relative time of diagnosis
        return a list that contains the instruction input and instruction target
        containing these attributes for each element in the list
        Args:
            diagnosis_list (pd.Series): Series containing medical/billing diagnosis
            positions_list (pd.Series): Series containing the relative time of diagnosis

        Returns:
            (List[Tuple[str, str]]): List that contains the instruction input and instruction target
            for each input element
        """
        return [
            self.get_positive_instruction(diagnosis=diagnosis, position=position)
            for diagnosis, position in zip(diagnosis_list, positions_list)
        ]

    def get_negative_instructions(
            self,
            diagnosis_list: pd.Series,
            positions_list: pd.Series
    ) -> List[Tuple[str, str]]:
        """
        Given a list of diagnoses and the relative time of them
        return a list that contains the instruction input and instruction target
        such that the position/time is replaced such that the patient should
        not have been diagnosed with the diagnosis at that time
        Args:
            diagnosis_list (pd.Series): Series containing medical/billing diagnosis
            positions_list (pd.Series): Series containing the relative time of diagnosis

        Returns:
            (List[Tuple[str, str]]): List that contains the instruction input and instruction target
            for each input element
        """
        return [
            self.get_negative_instruction(diagnosis=diagnosis, position=position)
            for diagnosis, position in zip(diagnosis_list, positions_list)
        ]

    def get_input(self, diagnosis: str) -> str:
        """
        Return the instruction input, given a diagnosis and position occurring in the future
        Args:
            diagnosis (str): A medical/billing diagnosis

        Returns:
            (str): Instruction input for a future time period

        """
        return self._instruction_template.get_instruction_input(
            diagnosis=diagnosis,
        )

    def get_output(self, position: float) -> str:
        """
        Return the instruction output, given an instruction target label
        Args:
            position (float): Instruction target label - position (time) of event

        Returns:
            (str): Instruction target for t2e prediction

        """
        sign = self.get_sign(position=position)
        return self._instruction_template.get_instruction_target(
            sign=sign,
            answer=self.get_position_string(position=position)
        )

    @staticmethod
    def get_sign(position: float):
        """
        Return the instruction output, given an instruction target label
        Args:
            position (float): The sign of position (time) of an event - to indicate past, future or negative

        Returns:
            (str): Sign of the position that indicates whether event occurred in the past, future or never

        """
        if position == np.inf:
            return '~'
        elif position < 0:
            return '-'
        else:
            return '+'


    @staticmethod
    def get_position_string(position: float):
        """
        Given a position value - process and return the string representation
        Args:
            position (float): The position (time) of the event

        Returns:
            (str): String representation of the position
        """
        return str(position) if position == np.inf else str(int(np.abs(position)))

