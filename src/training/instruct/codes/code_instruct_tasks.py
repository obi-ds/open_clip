"""Make training data"""
import random
import numpy as np
import pandas as pd
from scipy.special import softmax
from typing import Union, Tuple, List, Any, Optional, Sequence, NoReturn

from .processing.code_convert import ICDConvert
from .processing.encounter_dataframe_process import EncounterDataframeProcess
from .processing.dataframe_sampling import DataFrameSampling
from .processing.negative_code_sampling import NegativeICDSampling

np.random.seed(42)


class CodeStatusClassificationTask(object):
    """
    Make instruction tuning training data from diagnostic codes
    """

    def __init__(
            self,
            encounter_dataframe_process: EncounterDataframeProcess,
            dataframe_sampling: DataFrameSampling,
            code_instructions: Any,
            code_convert: ICDConvert,
            negative_code_sampling: Optional[NegativeICDSampling] = None,
            patient_id_column: str = 'PatientID',
            code_column: str = 'ICD10CD',
            position_column: str = 'position',
            instruction_label: str = 'instruction_label'
    ):
        """
        Initialize variables
        Args:
            encounter_dataframe_process (EncounterDataframeProcess): Dataframe processing object
            dataframe_sampling (DataFrameSampling): Dataframe sampling object
            negative_code_sampling (NegativeICDSampling, defaults to `None`): Object to sample negatives from a
            pre-defined list
            code_instructions (Any): A list of code instruction tasks. At any point one of these
            tasks is randomly chosen and trained on
            code_convert (ICDConvert): The object to map codes to other codes in hierarchy and their textual
            descriptions.
            patient_id_column (str, defaults to `PatientID`): The column name for the patient id
            code_column (str, defaults to `ICD10CD`): The column that contains the codes
            position_column (str, defaults to `positions`): The column where position values
            (e.g. time difference in days or log days) will be stored
            instruction_label (str, defaults to `instruction_label`): Label assigned to the instructions - either
            positive or negative
        """
        self._encounter_dataframe_process = encounter_dataframe_process
        self._dataframe_sampling = dataframe_sampling
        self._negative_code_sampling = negative_code_sampling
        self._code_instructions = code_instructions
        self._code_convert = code_convert
        self._patient_id_column = patient_id_column
        self._code_column = code_column
        self._position_column = position_column
        self._instruction_label = instruction_label

    def get_full_encounter_history(
            self,
            patient_id: Union[str, int],
            current_time: str,
            use_log_position: bool,
            time_difference_normalize: int,
    ) -> pd.DataFrame:
        """
        Return the encounter history of a patient - if the patient does not have an encounter history
        return an empty encounter history
        Args:
            patient_id (Union[str, int]): The unique id of the patient we need to process
            current_time (str): The timestamp of the sample we are processing - used to calculate time deltas
            with respect to other encounters
            use_log_position (bool): Whether to keep time deltas in days or as log value of days
            time_difference_normalize (int, defaults to `30`): Normalize time difference by this value
            (e.g. 30 normalizes it to months)

        Returns:
            (pd.DataFrame): The encounter history
        """
        try:
            return self._encounter_dataframe_process.get_patient_sequence(
                patient_id=patient_id,
                current_time=current_time,
                use_log_position=use_log_position,
                time_difference_normalize=time_difference_normalize
            )
        except KeyError:
            # TODO: A fix for now - because we have missing data - ideally we should not
            #  use this code - or come up with a better fix
            position_range = np.arange(6, 7, step=0.5)
            # Sample codes and create a random encounter dataframe
            encounter_history = self.get_random_encounters(
                size=50,
                position_range=position_range,
                exclude_codes=pd.Series([])
            )
            return encounter_history

    def load_encounter_history_for_task(
            self,
            encounter_history: pd.DataFrame,
            past_time_delta: str,
            future_time_delta: str,
    ) -> NoReturn:
        """
        Given the encounter history, map codes, filter out duplicate codes and filter
        the history based on time range. Keep only those entries
        that occur within the time range. If encounter history - create a history
        by randomly generating codes and positions. This will only be used to sample
        negatives
        Args:
            encounter_history (pd.DataFrame): The dataframe containing all encounters
            past_time_delta (str): Filter out encounters beyond this point in time
            future_time_delta (str): Filter out encounters beyond this point in time

        """

        encounter_history, all_positives = self._encounter_dataframe_process.filter_encounter_history_for_task(
            encounter_history=encounter_history,
            past_time_delta=past_time_delta,
            future_time_delta=future_time_delta,
        )

        positives = self.process_encounter_positives(
            encounter_history=encounter_history,
            positives=all_positives
        )

        return encounter_history, all_positives, positives

    def get_sampling_weights(self, encounter_history:pd.DataFrame) -> Sequence[float]:
        """
        Return the weights for sampling - based on position
        Returns:
            encounter_history (pd.DataFrame): The encounter history of the patient
            (Sequence[float]): Sequence of weights
        """
        # Increase sample probability of past over future
        positions = encounter_history[self._position_column] * -1
        return softmax(positions)

    def convert_samples_to_instructions(
            self,
            instruction_samples: pd.DataFrame,
            shuffle,
    ) -> List[Tuple[str, str]]:
        """
        Return the instruction input and targets
        for the sampled instruction code status classification task with time range
        Args:
            instruction_samples (pd.DataFrame): The samples that will be used as instructions
            shuffle (bool, defaults to `True`): Whether to shuffle the list of instructions.

        Returns:
            (List[Tuple[str, str]]): A list that contains tuples which have the instruction input and target.
            The list will contain only 1 element for zero shot training
        """
        instructions = list()
        for row in instruction_samples[
            [self._code_column, self._position_column, self._instruction_label]
        ].itertuples(index=False):
            code, position, instruction_label = row[0], row[1], row[2]
            if instruction_label:
                instruct_string = self._code_instructions.get_positive_instruction(
                    diagnosis=self._code_convert.transform_code(code=code),
                    position=position,
                )
            else:
                instruct_string = self._code_instructions.get_negative_instruction(
                    diagnosis=self._code_convert.transform_code(code=code),
                    position=position,
                )
            instructions.append(instruct_string)

        if shuffle:
            random.shuffle(instructions)

        # TODO: Add the task definition to the list?
        return instructions

    def get_instruction_samples(
            self,
            encounter_history: pd.DataFrame,
            exclude_codes_from_negatives: pd.Series,
            k_shot: int,
            sort_position: bool,
            encounter_negatives: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Given the encounter history, map codes, filter out duplicate codes and filter
        the history based on time range. Keep only those entries
        that occur within the time range. If encounter history - create a history
        by randomly generating codes and positions. This will only be used to sample
        negatives. Then sample the dataframe for encounters to use as positive and
        negative samples. Use these samples to generate instructions and
        return the instruction input and targets for the sampled instruction code
        status classification task with time range
        Args:
            encounter_history (pd.DataFrame): The encounter history of the patient
            exclude_codes_from_negatives (pd.Series): List of codes that should not be used for negative sampling
            k_shot (int): The number of k-shot examples to use
            sort_position (bool): Whether to sort instructions by position
            encounter_negatives (Optional[pd.DataFrame], defaults to `None`): If passed, use this to sample negatives

        Returns:
            (pd.DataFrame): The instructions samples
        """
        number_of_positives, number_of_negatives = self.get_positive_negative_sizes(
            max_size=k_shot + 1,
            number_of_encounters=encounter_history.shape[0]
        )

        positive_weights = self.get_sampling_weights(encounter_history=encounter_history)
        negative_weights = self.get_sampling_weights(
            encounter_history=encounter_history if encounter_negatives is None else encounter_negatives
        )

        # Sample a subset of positives from dataframe
        sampled_positives = self.get_positive_samples(
            encounter_history=encounter_history,
            size=number_of_positives,
            weights=positive_weights
        )
        sampled_positives[self._instruction_label] = True

        sampled_negatives = self.get_negative_samples(
            encounter_history=encounter_history,
            encounter_negatives=encounter_negatives,
            exclude_codes=exclude_codes_from_negatives,
            size=number_of_negatives,
            weights=negative_weights
        )
        sampled_negatives[self._instruction_label] = False

        instruction_samples = pd.concat([sampled_positives, sampled_negatives])

        if sort_position:
            instruction_samples.sort_values(by=self._position_column)

        return instruction_samples

    def get_task_instructions(
            self,
            encounter_history: pd.DataFrame,
            all_positives: pd.Series,
            positives: pd.Series,
            k_shot: int,
            sort_position: bool,
            encounter_negatives: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
        """
        Given the encounter history, map codes, filter out duplicate codes and filter
        the history based on time range. Keep only those entries
        that occur within the time range. If encounter history - create a history
        by randomly generating codes and positions. This will only be used to sample
        negatives. Then sample the dataframe for encounters to use as positive and
        negative samples. Use these samples to generate instructions
        Args:
            encounter_history (pd.DataFrame): The encounter history of the patient
            all_positives (pd.Series): The codes present in the entire encounter history of the patient
            positives (pd.Series): The codes present in the current encounter history
            k_shot (int): The number of k-shot examples to use
            sort_position (bool): Whether to sort instructions by position
            encounter_negatives (Optional[pd.DataFrame], defaults to `None`): If passed, use this to sample negatives

        Returns:
            instruction_samples (pd.DataFrame): The instructions samples
            instructions (List[Tuple[str, str]]): A list that contains tuples which have the instruction
            input and target. The list will contain only 1 element for zero shot training
        """
        # TODO: Keep track of sampled positives - and make sure we don't re-sample them
        #  as positives - we cna re-sample as negatives - instead of filtering the dataframe

        exclude_codes_from_negatives = self.get_exclude_codes_from_negatives(
            all_positives=all_positives,
            positives=positives
        )

        instruction_samples = self.get_instruction_samples(
            encounter_history=encounter_history,
            exclude_codes_from_negatives=exclude_codes_from_negatives,
            k_shot=k_shot,
            sort_position=sort_position,
            encounter_negatives=encounter_negatives
        )

        instructions = self.convert_samples_to_instructions(
            instruction_samples=instruction_samples,
            shuffle=not sort_position
        )
        return instruction_samples, instructions

    def get_positive_samples(
            self,
            encounter_history: pd.DataFrame,
            size: int,
            weights: Optional[Sequence[float]] = None
    ) -> pd.DataFrame:
        """
        Returns the positive samples - sampled from the encounter history
        Args:
            encounter_history (pd.DataFrame): The encounter history of the patient
            size (int): How many samples to return
            weights (Optional[Sequence[float]], defaults to `None`): The sampling weights

        Returns:
            (pd.DataFrame): A subset of the encounters that will be used as positive
            samples
        """
        # Sample a subset of positives from dataframe
        return self._dataframe_sampling.sample_dataframe(
            dataframe=encounter_history,
            sample_size=size,
            weights=weights
        )

    def get_negative_samples(
            self,
            encounter_history: pd.DataFrame,
            encounter_negatives: pd.DataFrame,
            exclude_codes: pd.Series,
            size: int,
            weights: Optional[Sequence[float]] = None
    ) -> pd.DataFrame:
        """
        Returns the negatives samples - sampled based on the given encounter history
        Args:
            encounter_history (pd.DataFrame): The encounter history of the patient
            encounter_negatives (pd.DataFrame): The encounter history of the patient
            exclude_codes (pd.Series): List of codes that should not be part of negative samples
            size (int): How many samples to return
            weights (Optional[Sequence[float]], defaults to `None`): The sampling weights

        Returns:
            (pd.DataFrame): A subset of the encounters that will be used as positive
            samples
        """
        if encounter_negatives is not None:
            # Sample a subset of positives from dataframe
            sampled_negatives = self.get_negative_samples_from_encounter_negatives(
                encounter_negatives=encounter_negatives,
                exclude_codes=exclude_codes,
                size=size,
                weights=weights
            )
            position_range = np.arange(
                encounter_history[self._position_column].min(),
                encounter_history[self._position_column].max(),
                step=0.5
            )
        else:
            # Sample a subset of positives from dataframe
            sampled_negatives = self.get_negative_samples_from_encounter_history(
                encounter_history=encounter_history,
                exclude_codes=exclude_codes,
                size=size,
                weights=weights
            )
            position_range = np.arange(
                encounter_history[self._position_column].min(),
                encounter_history[self._position_column].max(),
                step=0.5
            )
        if len(sampled_negatives) != size:
            random_encounters = self.get_random_encounters(
                size=size - len(sampled_negatives),
                position_range=position_range,
                exclude_codes=exclude_codes
            )
            sampled_negatives = pd.concat(
                [sampled_negatives[self._code_column, self._position_column], random_encounters]
            )
        return sampled_negatives

    def get_negative_samples_from_encounter_negatives(
            self,
            encounter_negatives: pd.DataFrame,
            exclude_codes: pd.Series,
            size: int,
            weights: Optional[Sequence[float]] = None
    ) -> pd.DataFrame:
        """
        Returns the negatives samples - sampled based on the given encounter history
        Args:
            encounter_negatives (pd.DataFrame): The encounter history of the patient
            exclude_codes (pd.Series): List of codes that should not be part of negative samples
            size (int): How many samples to return
            weights (Optional[Sequence[float]], defaults to `None`): The sampling weights

        Returns:
            (pd.DataFrame): A subset of the encounters that will be used as positive
            samples
        """

        return self._dataframe_sampling.sample_dataframe(
            dataframe=encounter_negatives[~encounter_negatives[self._code_column].isin(exclude_codes)],
            sample_size=size,
            weights=weights
        )

    def get_negative_samples_from_encounter_history(
            self,
            encounter_history: pd.DataFrame,
            exclude_codes: pd.Series,
            size: int,
            weights: Optional[Sequence[float]] = None
    ) -> pd.DataFrame:
        """
        Returns the negatives samples - sampled based on the encounter history
        Args:
            encounter_history (pd.DataFrame): The encounter history of the patient
            exclude_codes (pd.Series): List of codes that should not be part of negative samples
            size (int): How many samples to return
            weights (Optional[Sequence[float]], defaults to `None`): The sampling weights

        Returns:
            (pd.DataFrame): A subset of the encounters that will be used as positive
            samples
        """
        # Sample a subset of positives from dataframe
        # TODO: Need to fix this code
        return self._dataframe_sampling.sample_dataframe(
            dataframe=encounter_history[~encounter_history[self._code_column].isin(exclude_codes)],
            sample_size=size,
            weights=weights
        )

    def filter_encounter_history_by_selected_sampled(
            self,
            encounter_history: pd.DataFrame,
            position: Union[int, float],
    ) -> NoReturn:
        """
        Once we sample an encounter, and we have a condition that we want
        to sample encounters in a sorted order - we need to ignore all
        previous encounters.
        Args:
            encounter_history (pd.DataFrame): The encounter history of the patient
            position (Union[int, float]): Ignore all encounters prior to this

        """
        # TODO: This will get rid of any codes that are in "position" but haven't been
        #  sampled - easier to implement - but ideally we would check if there are any
        #  codes at "position" that haven't been sampled and keep those codes.
        position_filter = encounter_history[self._position_column] > position
        # if code is None:
        #     code_filter = False
        # else:
        #     code_filter = (
        #             (self._encounter_history[self._code_column] == code)
        #             & (self._encounter_history[self._position_column] == position)
        #     )
        return encounter_history[position_filter]

    def process_encounter_positives(
            self,
            encounter_history: pd.DataFrame,
            positives: pd.Series
    ) -> Union[List[str], pd.Series]:
        """
        Scan the list of positives and check if there are any that are
        chronic or transient - and modify the list accordingly
        Args:
            encounter_history (pd.DataFrame): The encounter history of patient
            positives (pd.DataFrame): The list of all positives

        Returns:
            (): A modified list of positives
        """
        return encounter_history[self._code_column]

    @staticmethod
    def get_exclude_codes_from_negatives(
            all_positives: pd.Series,
            positives: pd.Series
    ) -> pd.Series:
        """
        Return a list of codes that should not be sampled when sampling negatives
        Args:
            all_positives (pd.Series): The codes present in the entire encounter history of the patient
            positives (pd.Series): The codes present in the current encounter history

        Returns:
            (pd.Series): List of codes that should not be used for negative sampling
        """
        return all_positives

    def get_random_encounters(
            self,
            size: int,
            position_range: np.array,
            exclude_codes: pd.Series
    ) -> pd.DataFrame:
        """
        Use this function to create a random dataframe populated
        with codes and positions
        Args:
            size (int): The size of the dataframe
            position_range (np.array): Position values are assigned based on this range
            exclude_codes (pd.Series): Exclude these codes when randomly sampling negatives

        Returns:
            (pd.DataFrame): A randomly generated encounter dataframe
        """

        # Sample negative codes
        codes = self._negative_code_sampling.get_codes(
            positives=exclude_codes,
            k=size
        )
        # Sample position values
        positions = np.random.choice(position_range, size)
        # Use these to create and return a dataframe
        return pd.DataFrame(
            {self._code_column: codes, self._position_column: positions}
        )

    @staticmethod
    def get_positive_negative_sizes(max_size: int, number_of_encounters: int) -> Tuple[int, int]:
        """
        For a given k-shot problem, return how many positive samples
        and negative samples should be present in the instruction

        Args:
            max_size (int): The maximum number of positives and negatives
            number_of_encounters (int): The total number of encounters
        Returns:
            (Tuple[int, int]): The number of positive and negative examples
        """
        # For zero shot, we have either one positive or one negative
        positive = 0
        negative = 0
        for k in range(max_size):
            if np.random.rand() <= 0.5 and positive < number_of_encounters:
                positive += 1
            else:
                negative += 1
        return positive, negative


class CodeT2EPredictionTask(CodeStatusClassificationTask):
    """
    Make training data from icd codes
    """

    def __init__(
            self,
            encounter_dataframe_process: EncounterDataframeProcess,
            dataframe_sampling: DataFrameSampling,
            code_instructions: Any,
            code_convert: ICDConvert,
            negative_code_sampling: Optional[NegativeICDSampling] = None,
            patient_id_column: str = 'PatientID',
            code_column: str = 'ICD10CD',
            position_column: str = 'position',
            instruction_label: str = 'instruction_label'
    ):
        """
        Initialize variables
        Args:
            encounter_dataframe_process (EncounterDataframeProcess): Dataframe processing object
            dataframe_sampling (DataFrameSampling): Dataframe sampling object
            negative_code_sampling (NegativeICDSampling, defaults to `None`): Object to sample negatives from a
            pre-defined list
            code_instructions (Any): A list of code instruction tasks. At any point one of these
            tasks is randomly chosen and trained on
            code_convert (ICDConvert): The object to map codes to their textual descriptions'
            patient_id_column (str, defaults to `PatientID`): The column name for the patient id
            code_column (str, defaults to `ICD10CD`): The column that contains the icd codes
            position_column (str, defaults to `positions`): The column where position values
            (e.g. time difference in days or log days) will be stored
            instruction_label (str, defaults to `instruction_label`): Label assigned to the instructions - either
            positive or negative
        """
        super().__init__(
            encounter_dataframe_process=encounter_dataframe_process,
            dataframe_sampling=dataframe_sampling,
            code_instructions=code_instructions,
            code_convert=code_convert,
            negative_code_sampling=negative_code_sampling,
            patient_id_column=patient_id_column,
            code_column=code_column,
            position_column=position_column,
            instruction_label=instruction_label
        )
