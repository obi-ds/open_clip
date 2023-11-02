"""Define the ICD status classification task - The task uses templates, inputs and targets"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Union

from .icd_status_instruction_template import (
    ICDStatusInstructionTemplate,
    ICDStatusRangeInstructionTemplate,
    ICDT2EInstructionTemplate
)

class ICDStatusClassificationInstructionsInterface(object):
    """
    Define the ICD status classification task. Encapsulate the templates
    inputs, outputs to build the final instruction. The instruction is built based
    on whether events occurred in the past/future and are positive/negative
    """

    def __init__(
            self,
            instruction_template_future: Union[ICDStatusInstructionTemplate, ICDStatusRangeInstructionTemplate],
            instruction_template_past: Union[ICDStatusInstructionTemplate, ICDStatusRangeInstructionTemplate],
            positive_answer_target: str,
            negative_answer_target: str,
            task_definition: str = '',
            task_name: str = 'status_classification_interface'
    ):
        """
        Initialize the variables
        Args:
            instruction_template_future (ICDStatusInstructionTemplate): The template for icd status for future events.
            instruction_template_past (ICDStatusInstructionTemplate): The template for icd status for past events.
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
            encounter_positions: pd.Series
    ) -> Tuple[str, str]:
        """
        Given a diagnosis and the relative time of diagnosis
        return the instruction input and instruction target such that
        the position/time is replaced such that the patient should
        not have been diagnosed with the diagnosis at that time
        Args:
            diagnosis (str): A medical/billing diagnosis
            position (int): The relative time of diagnosis
            encounter_positions (pd.Series): The positions of all other encounters/diagnoses

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

class ICDStatusClassificationInstructions(ICDStatusClassificationInstructionsInterface):
    """
    Define the ICD status classification task. Encapsulate the templates
    inputs, outputs to build the final instruction. The instruction is built based
    on whether events occurred in the past/future and are positive/negative
    """

    def __init__(
            self,
            instruction_template_future: ICDStatusInstructionTemplate,
            instruction_template_past: ICDStatusInstructionTemplate,
            positive_answer_target: str,
            negative_answer_target: str,
            task_definition: str = '',
            task_name: str = 'status_classification'
    ):
        """
        Initialize the variables
        Args:
            instruction_template_future (ICDStatusInstructionTemplate): The template for icd status for future events.
            instruction_template_past (ICDStatusInstructionTemplate): The template for icd status for past events.
            positive_answer_target (str):
            negative_answer_target (str):
            task_definition (str, defaults to): Add the task definition to the instruction
            task_name (str, defaults to `status_classification`): A name to define this task.
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
        # Positive diagnosis that occur in the future
        if position > 0:
            instruction_input = self.get_future_input(diagnosis=diagnosis, time=position)
            instruction_target = self.get_future_output(target=self._positive_answer_target)
        # If position is 0, we assume the diagnosis happened within the next 1 month
        elif position == 0:
            instruction_input = self.get_future_input(diagnosis=diagnosis, time=1)
            instruction_target = self.get_future_output(target=self._positive_answer_target)
        # Positive diagnosis that occurred in the past
        else:
            instruction_input = self.get_past_input(diagnosis=diagnosis, time=np.abs(position))
            instruction_target = self.get_past_output(target=self._positive_answer_target)
        return instruction_input, instruction_target

    def get_negative_instruction(
            self,
            diagnosis: str,
            position: int,
            encounter_positions: pd.Series
    ) -> Tuple[str, str]:
        """
        Given a diagnosis and the relative time of diagnosis
        return the instruction input and instruction target such that
        the position/time is replaced such that the patient should
        not have been diagnosed with the diagnosis at that time
        Args:
            diagnosis (str): A medical/billing diagnosis
            position (int): The relative time of diagnosis
            encounter_positions (pd.Series): The positions of all other encounters/diagnoses

        Returns:
            (Tuple[str, str]): Tuple that contains the negative instruction input and instruction target
        """
        # Get a time period to represent the negative diagnosis
        # Essentially the diagnosis was not diagnosed during this period
        negative_position = self.get_negative_position(position=position, encounter_positions=encounter_positions)
        # If the value is positive use future instruction template
        if negative_position > 0:
            instruction_input = self.get_future_input(diagnosis=diagnosis, time=negative_position)
            instruction_target = self.get_future_output(target=self._negative_answer_target)
        elif negative_position == 0:
            instruction_input = self.get_past_input(diagnosis=diagnosis, time=1)
            instruction_target = self.get_past_output(target=self._negative_answer_target)
        else:
            instruction_input = self.get_past_input(diagnosis=diagnosis, time=np.abs(negative_position))
            instruction_target = self.get_past_output(target=self._negative_answer_target)
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
            self.get_negative_instruction(diagnosis=diagnosis, position=position, encounter_positions=positions_list)
            for diagnosis, position in zip(diagnosis_list, positions_list)
        ]


    @staticmethod
    def get_negative_position(
            position: int,
            encounter_positions: pd.Series
    ) -> int:
        """

        Args:
            position (int): The current position of the encounter/diagnosis
            encounter_positions (pd.Series): The positions of all other encounters/diagnoses

        Returns:
            (int): The position value that will be used to represent negative diagnosis
        """
        # Get all the positions prior the current "position"
        # Then randomly pick one to use as the negative position
        # This means that the diagnosis was not made at time
        # "negative_position", but it was made at a later time "position"
        # hence why we get all positions lesser than current "position"
        negative_positions = encounter_positions[encounter_positions < position]
        # If there is no position lower, subtract 6 from position and return
        if len(negative_positions) == 0:
            # Pick random number
            return position - 1
        else:
            return np.random.choice(negative_positions)



class ICDStatusRangeClassificationInstructions(ICDStatusClassificationInstructionsInterface):
    """
    Define the ICD status classification task. Encapsulate the templates
    inputs, outputs to build the final instruction. The instruction is built based
    on whether events occurred in the past/future and are positive/negative
    """

    def __init__(
            self,
            instruction_template_future: ICDStatusRangeInstructionTemplate,
            instruction_template_past: ICDStatusRangeInstructionTemplate,
            positive_answer_target: str,
            negative_answer_target: str,
            task_definition: str = '',
            task_name: str = 'status_range_classification'
    ):
        """
        Initialize the variables
        Args:
            instruction_template_future (ICDStatusInstructionTemplate): The template for icd status for future events.
            instruction_template_past (ICDStatusInstructionTemplate): The template for icd status for past events.
            positive_answer_target (str):
            negative_answer_target (str):
            lock_range (bool, `defaults to False`): Whether to use pre-defined ranges or sample ranges
            select randomly from given range
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

    def get_positive_instruction(
            self,
            diagnosis: str,
            position: float,
            start_time: int = None,
            end_time: int = None
    ) -> Tuple[str, str]:
        """
        Given a positive diagnosis and the relative time of diagnosis
        return the instruction input and instruction target containing
        these attributes
        Args:
            diagnosis (str): A medical/billing diagnosis
            position (float): The relative time of diagnosis
            start_time: The start limit for the position range
            end_time: The end limit for the position range

        Returns:
            (Tuple[str, str]): Tuple that contains the positive instruction input and instruction target
        """
        # Positive diagnosis that occur in the future

        start_time, end_time = self.get_position_range(
            position=np.abs(int(position)),
            start_range_limit=start_range_limit,
            end_range_limit=end_range_limit
        ) if not self._lock_range else (np.abs(int(start_range_limit)), np.abs(int(end_range_limit)))

        if start_time >= end_time:
            print('Position', position)
            print('Start time', start_time)
            print('End time', end_time)
            raise ValueError('Start time needs to be less than end time')

        if position >= 0:
            instruction_input = self.get_future_input(diagnosis=diagnosis, start_time=start_time, end_time=end_time)
            instruction_target = self.get_future_output(target=self._positive_answer_target)
        # If position is 0, we assume the diagnosis happened within the next 1-month
        # Positive diagnosis that occurred in the past
        else:
            instruction_input = self.get_past_input(diagnosis=diagnosis, start_time=start_time, end_time=end_time)
            instruction_target = self.get_past_output(target=self._positive_answer_target)
        return instruction_input, instruction_target

    def get_negative_instruction(
            self,
            diagnosis: str,
            position: float,
            start_range_limit: int,
            end_range_limit: int
    ) -> Tuple[str, str]:
        """
        Given a diagnosis and the relative time of diagnosis
        return the instruction input and instruction target such that
        the position/time is replaced such that the patient should
        not have been diagnosed with the diagnosis at that time
        Args:
            diagnosis (str): A medical/billing diagnosis
            position (float): The relative time of diagnosis
            start_range_limit: The start limit for the position range
            end_range_limit: The end limit for the position range

        Returns:
            (Tuple[str, str]): Tuple that contains the negative instruction input and instruction target
        """
        # Get a time period to represent the negative diagnosis
        # Essentially the diagnosis was not diagnosed during this period

        # If the value is positive use future instruction template
        if position >= 1 or (position == 0 and self._lock_range):
            start_time, end_time = self.get_future_negative_range(
                position=position,
                start_range_limit=start_range_limit,
                end_range_limit=end_range_limit
            )
            instruction_input = self.get_future_input(diagnosis=diagnosis, start_time=start_time, end_time=end_time)
            instruction_target = self.get_future_output(target=self._negative_answer_target)
        else:
            start_time, end_time = self.get_past_negative_range(
                position=position,
                start_range_limit=start_range_limit,
                end_range_limit=end_range_limit
            )
            instruction_input = self.get_past_input(diagnosis=diagnosis, start_time=start_time, end_time=end_time)
            instruction_target = self.get_past_output(target=self._negative_answer_target)
        if start_time >= end_time:
            print('Position', position)
            print('Start time', start_time)
            print('End time', end_time)
            raise ValueError('Start time needs to be less than end time')
        return instruction_input, instruction_target

    def get_positive_instructions(
            self,
            diagnosis_list: pd.Series,
            positions_list: pd.Series,
            past_start_range_limit: int,
            past_end_range_limit: int,
            future_start_range_limit: int,
            future_end_range_limit: int

    ) -> List[Tuple[str, str]]:
        """
        Given a list of positive diagnoses and the relative time of diagnosis
        return a list that contains the instruction input and instruction target
        containing these attributes for each element in the list
        Args:
            diagnosis_list (pd.Series): Series containing medical/billing diagnosis
            positions_list (pd.Series): Series containing the relative time of diagnosis
            past_start_range_limit: The start limit for the past position range
            past_end_range_limit: The end limit for the past position range
            future_start_range_limit: The start limit for the future position range
            future_end_range_limit: The end limit for the future position range

        Returns:
            (List[Tuple[str, str]]): List that contains the instruction input and instruction target
            for each input element
        """
        return [
            self.get_positive_instruction(
                diagnosis=diagnosis,
                position=position,
                start_range_limit=past_start_range_limit,
                end_range_limit=past_end_range_limit
            )
            if position < 0 else
            self.get_positive_instruction(
                diagnosis=diagnosis,
                position=position,
                start_range_limit=future_start_range_limit,
                end_range_limit=future_end_range_limit
            )
            for diagnosis, position in zip(diagnosis_list, positions_list)
        ]

    def get_negative_instructions(
            self,
            diagnosis_list: pd.Series,
            positions_list: pd.Series,
            past_start_range_limit: int = 0,
            past_end_range_limit: int = 12,
            future_start_range_limit: int = 0,
            future_end_range_limit: int = 12
    ) -> List[Tuple[str, str]]:
        """
        Given a list of positive diagnoses and the relative time of diagnosis
        return a list that contains the instruction input and instruction target
        containing these attributes for each element in the list
        Args:
            diagnosis_list (pd.Series): Series containing medical/billing diagnosis
            positions_list (pd.Series): Series containing the relative time of diagnosis
            past_start_range_limit: The start limit for the past position range
            past_end_range_limit: The end limit for the past position range
            future_start_range_limit: The start limit for the future position range
            future_end_range_limit: The end limit for the future position range

        Returns:
            (List[Tuple[str, str]]): List that contains the instruction input and instruction target
            for each input element
        """
        return [
            self.get_negative_instruction(
                diagnosis=diagnosis,
                position=position,
                start_range_limit=past_start_range_limit,
                end_range_limit=past_end_range_limit
            )
            if position <= 0
            else self.get_negative_instruction(
                diagnosis=diagnosis,
                position=position,
                start_range_limit=future_start_range_limit,
                end_range_limit=future_end_range_limit
            )
            for diagnosis, position in zip(diagnosis_list, positions_list)
        ]

    def get_future_input(self, diagnosis: str, start_time: int, end_time: int) -> str:
        """
        Return the instruction input, given a diagnosis and position occurring in the future
        Args:
            diagnosis (str): A medical/billing diagnosis
            start_time (int): The start time for a time range
            end_time (int): The end time for a time range

        Returns:
            (str): Instruction input for a future time period

        """
        return self._instruction_template_future.get_instruction_input(
            diagnosis=diagnosis,
            start_time=start_time,
            end_time=end_time
        )

    def get_past_input(self, diagnosis: str, start_time: int, end_time: int) -> str:
        """
        Return the instruction input, given a diagnosis and position occurring in the future
        Args:
            diagnosis (str): A medical/billing diagnosis
            start_time (int): The start time for a time range
            end_time (int): The end time for a time range

        Returns:
            (str): Instruction input for a pastime period

        """
        return self._instruction_template_past.get_instruction_input(
            diagnosis=diagnosis,
            start_time=start_time,
            end_time=end_time
        )

    def get_future_negative_range(self, position, start_range_limit, end_range_limit):
        if self._lock_range:
            return np.abs(int(start_range_limit)), np.abs(int(end_range_limit))
        else:
            negative_position = np.random.choice(range(start_range_limit, int(position)))
            return self.get_position_range(
                position=negative_position,
                start_range_limit=start_range_limit,
                end_range_limit=int(position)
            )

    def get_past_negative_range(self, position, start_range_limit, end_range_limit):
        if self._lock_range:
            return np.abs(int(start_range_limit)), np.abs(int(end_range_limit))
        else:
            position = np.abs(int(position))
            start_position = position if position == 0 else position + 1
            if start_position == end_range_limit:
                end_range_limit += 1

            negative_position = np.random.choice(range(start_position, end_range_limit))
            return self.get_position_range(
                position=negative_position,
                start_range_limit=start_position,
                end_range_limit=end_range_limit
            )

    @staticmethod
    def get_position_range(position: int, start_range_limit: int, end_range_limit):
        """
        Returns a range object such that "position" is contained within the range
        Args:
            position (int): A given position
            start_range_limit: The start limit for the range
            end_range_limit: The end limit for the range

        Returns:
            start_time (int): The start time for a time range
            end_time (int): The end time for a time range
        """
        start_time = np.random.choice(range(start_range_limit, position + 1))
        end_time = np.random.choice(range(position + 1, end_range_limit + 1))
        return start_time, end_time


class ICDT2EPredictionInstructions(object):
    """
    Define the ICD status classification task. Encapsulate the templates
    inputs, outputs to build the final instruction. The instruction is built based
    on whether events occurred in the past/future and are positive/negative
    """

    def __init__(
            self,
            instruction_template: ICDT2EInstructionTemplate,
            task_definition: str = '',
            task_name: str = 't2e_prediction',
            negative_answer_target=np.inf,
    ):
        """
        Initialize the variables
        Args:
            instruction_template (ICDT2EInstructionTemplate): The template for icd time to event predictions.
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

