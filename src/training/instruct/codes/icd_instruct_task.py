"""Make training data"""
import math
import random
import numpy as np
import pandas as pd
from typing import Union, Tuple, List, Any, Optional

from .processing.icd_convert import ICDConvert
from .processing.encounter_dataframe_process import EncounterDataframeProcess
from .processing.dataframe_sampling import DataFrameSampling
from .processing.negative_icd_sampling import NegativeICDSampling

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
            position_column: str = 'position'
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
        """
        self._encounter_dataframe_process = encounter_dataframe_process
        self._dataframe_sampling = dataframe_sampling
        self._negative_code_sampling = negative_code_sampling
        self._code_instructions = code_instructions
        self._code_convert = code_convert
        self._patient_id_column = patient_id_column
        self._code_column = code_column
        self._position_column = position_column

    def process_sample_from_args(self, sample, args):
        """

        Args:
            sample:
            args:

        Returns:

        """
        return self.process_sample(
            patient_id=sample[args.patient_id_column],
            current_time=sample[args.sample_result_date_column],
            past_time_delta=args.past_time_delta,
            future_time_delta=args.future_time_delta,
            use_log_position=args.use_log_position,
            k_shot=args.k_shot,
            random_negative_probability=args.random_negative_probability,
            time_difference_normalize=args.time_difference_normalize,
            shuffle=args.shuffle
        )

    def map_encounter_codes(self, encounter_history: pd.DataFrame) -> pd.DataFrame:
        """
        Map codes from one format to another and return the dataframe
        with the mapped codes
        Args:
            encounter_history (pd.DataFrame): The input dataframe

        Returns:
            encounter_history (pd.DataFrame): The dataframe with codes mapped
        """
        # Codes can be mapped from leaf to top level - e.g. I48.1 to I48
        encounter_history[self._code_column] = self._code_convert.get_converted_codes(
            codes=encounter_history[self._code_column]
        )
        return encounter_history

    def filter_encounter_codes(self, encounter_history: pd.DataFrame) -> pd.DataFrame:
        """
        Drop duplicate entries - entries that repeat codes
        Args:
            encounter_history (pd.DataFrame): The input dataframe

        Returns:
            encounter_history (pd.DataFrame): The dataframe with duplicate codes dropped
        """
        # Remove any duplicate entries. When removing duplicates - keep the earliest
        # entry (hence sorted by position)
        encounter_history.sort_values(by=self._position_column, inplace=True)
        encounter_history.drop_duplicates(subset=self._code_column, inplace=True)
        return encounter_history

    def filter_encounter_history_time_delta(
            self,
            encounter_history: pd.DataFrame,
            past_time_delta: str,
            future_time_delta:str
    ) -> pd.DataFrame:
        """
        Filter encounter history based on the given time range. Keep only those entries
        that occur within the time range
        Args:
            encounter_history (pd.DataFrame): The input dataframe
            past_time_delta (str): Filter out encounters beyond this point in time
            future_time_delta (str): Filter out encounters beyond this point in time

        Returns:
            encounter_history (pd.DataFrame): Dataframe that contains only those entries within the time frame
        """
        if past_time_delta is not None or future_time_delta is not None:
            # Filter based on time range
            # Keep only the rows within this range
            time_filter = self._encounter_dataframe_process.get_time_filter(
                time_difference=encounter_history[self._encounter_dataframe_process.time_difference_column],
                past_time_delta=past_time_delta,
                future_time_delta=future_time_delta
            )
            encounter_history = self._encounter_dataframe_process.filter_dataframe(
                dataframe=encounter_history,
                filter_mask=time_filter
            )
        return encounter_history

    def fix_empty_encounter_history(
            self,
            past_time_delta: str,
            future_time_delta: str,
            k_shot: int,
            time_difference_normalize: int,
            exclude_codes: pd.Series
    ) -> pd.DataFrame:
        """
        Check if encounter history is empty - if it is, create a random encounter dataframe.
        This is done by randomly sampling negative codes
        Args:
            exclude_codes (pd.Series): Exclude these codes when randomly sampling negatives
            with respect to other encounters
            past_time_delta (str): Filter out encounters beyond this point in time
            future_time_delta (str): Filter out encounters beyond this point in time
            k_shot (int): The number of k-shot examples to use
            time_difference_normalize (int, defaults to `30`): Normalize time difference by this value
            (e.g. 30 normalizes it to months)

        Returns:
            encounter_history (pd.DataFrame): The original encounter dataframe or a random dataframe if the
            original contained 0 entries
        """

        # Assign random position values (within the given range) to the randomly sampled codes
        position_range = np.arange(
            -(math.floor(pd.to_timedelta(past_time_delta).days / time_difference_normalize)),
            (math.ceil(pd.to_timedelta(future_time_delta).days / time_difference_normalize))
        )

        # Sample codes and create a random encounter dataframe
        encounter_history = self.get_random_encounters(
            size=k_shot + 1,
            position_range=position_range,
            exclude_codes=exclude_codes
        )

        return encounter_history

    def filter_encounter_history_for_task(
            self,
            encounter_history: pd.DataFrame,
            past_time_delta: str,
            future_time_delta: str,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Given the encounter history, map codes, filter out duplicate codes and filter
        the history based on time range. Keep only those entries
        that occur within the time range
        Args:
            encounter_history (pd.DataFrame): The dataframe containing all encounters
            past_time_delta (str): Filter out encounters beyond this point in time
            future_time_delta (str): Filter out encounters beyond this point in time

        Returns:
            encounter_history (pd.DataFrame): The encounter history after mapping codes, dropping duplicates and
            filtering by time range
            all_positives (pd.Series): The codes present in the entire encounter history of the patient
        """

        encounter_history = self.filter_encounter_codes(
            encounter_history=self.map_encounter_codes(
                encounter_history=encounter_history
            )
        )

        all_positives = encounter_history[self._code_column]

        encounter_history = self.filter_encounter_history_time_delta(
            encounter_history=encounter_history,
            past_time_delta=past_time_delta,
            future_time_delta=future_time_delta
        )

        return encounter_history, all_positives

    def process_encounter_history_for_task(
            self,
            encounter_history: pd.DataFrame,
            past_time_delta: str,
            future_time_delta: str,
            k_shot: int,
            time_difference_normalize: int,
    ) -> Tuple[pd.DataFrame, pd.Series, int]:
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
            k_shot (int): The number of k-shot examples to use
            time_difference_normalize (int, defaults to `30`): Normalize time difference by this value
            (e.g. 30 normalizes it to months)

        Returns:
            encounter_history (pd.DataFrame): The encounter history after mapping codes, dropping duplicates and
            filtering by time range
            all_positives (pd.Series): The codes present in the entire encounter history of the patient
            number_of_encounters (int): The number of encounters in the input dataframe - before random sampling,
            if done.
        """

        encounter_history, all_positives = self.filter_encounter_history_for_task(
            encounter_history=encounter_history,
            past_time_delta=past_time_delta,
            future_time_delta=future_time_delta,
        )

        number_of_encounters = encounter_history.shape[0]

        if number_of_encounters == 0:
            encounter_history = self.fix_empty_encounter_history(
                past_time_delta=past_time_delta,
                future_time_delta=future_time_delta,
                k_shot=k_shot,
                time_difference_normalize=time_difference_normalize,
                exclude_codes=all_positives
            )

        return encounter_history, all_positives, number_of_encounters

    def get_positive_negative_samples(
            self,
            encounter_history: pd.DataFrame,
            exclude_codes: pd.Series,
            k_shot: int,
            random_negative_probability: float,
            number_of_encounters: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Given the encounter history dataframe, sample the dataframe for encounters to use as positive and
        negative samples
        Args:
            encounter_history (pd.Dataframe): The encounter history of the patient
            exclude_codes (pd.Series): Exclude these codes when randomly sampling negatives
            with respect to other encounters
            k_shot (int): The number of k-shot examples to use
            random_negative_probability (float, defaults to 0.5): Probability of sampling negatives randomly
            number_of_encounters (int): The number of encounters in original encounter dataframe

        Returns:
            sampled_positives (pd.DataFrame): The samples that will be used to generate positive instruction examples
            sampled_negatives (pd.DataFrame): The samples that will be used to generate negative instruction examples
        """

        # Compute how many samples to extract from code history dataframe
        # For a given k_shot we can extract only positives, only negatives
        # or a mix of both
        number_of_positives, number_of_negatives = self.get_positive_negative_sizes(
            max_size=min(k_shot + 1, encounter_history.shape[0]),
            number_of_encounters=number_of_encounters
        )

        # Sample a subset of negatives from dataframe
        sampled_positives = self.get_positive_samples(
            encounter_history=encounter_history,
            size=number_of_positives
        )

        sampled_negatives = self.get_negative_samples(
            encounter_history=encounter_history,
            size=number_of_negatives,
            random_negative_probability=random_negative_probability,
            exclude_codes=exclude_codes
        )

        # Get the instructions - this should contain the instruction inputs and instruction targets
        return sampled_positives, sampled_negatives

    def process_samples_for_task(
            self,
            encounter_history: pd.DataFrame,
            past_time_delta: str,
            future_time_delta: str,
            k_shot: int,
            random_negative_probability: float,
            time_difference_normalize: int,
    ):
        """
        Given the encounter history, map codes, filter out duplicate codes and filter
        the history based on time range. Keep only those entries
        that occur within the time range. If encounter history - create a history
        by randomly generating codes and positions. This will only be used to sample
        negatives. Then sample the dataframe for encounters to use as positive and
        negative samples
        Args:
            encounter_history (pd.Dataframe): The encounter history of the patient
            with respect to other encounters
            past_time_delta (str): Filter out encounters beyond this point in time
            future_time_delta (str): Filter out encounters beyond this point in time
            k_shot (int): The number of k-shot examples to use
            random_negative_probability (float, defaults to 0.5): Probability of sampling negatives randomly
            time_difference_normalize (int, defaults to `30`): Normalize time difference by this value
            (e.g. 30 normalizes it to months)

        Returns:
            sampled_positives (pd.DataFrame): The samples that will be used to generate positive instruction examples
            sampled_negatives (pd.DataFrame): The samples that will be used to generate negative instruction examples
        """

        encounter_history, all_positives, number_of_encounters = self.process_encounter_history_for_task(
            encounter_history=encounter_history,
            past_time_delta=past_time_delta,
            future_time_delta=future_time_delta,
            k_shot=k_shot,
            time_difference_normalize=time_difference_normalize,
        )

        sampled_positives, sampled_negatives = self.get_positive_negative_samples(
            encounter_history=encounter_history,
            k_shot=k_shot,
            random_negative_probability=random_negative_probability,
            number_of_encounters=number_of_encounters,
            exclude_codes=all_positives
        )

        return sampled_positives, sampled_negatives

    def process_sample(
            self,
            patient_id: Union[str, int],
            current_time: str,
            past_time_delta: str,
            future_time_delta: str,
            use_log_position: bool,
            k_shot: int,
            random_negative_probability: float,
            time_difference_normalize: int,
            shuffle: bool,
    ) -> List[Tuple[str, str]]:
        """
        Given the encounter history, map codes, filter out duplicate codes and filter
        the history based on time range. Keep only those entries
        that occur within the time range. If encounter history - create a history
        by randomly generating codes and positions. This will only be used to sample
        negatives. Then sample the dataframe for encounters to use as positive and
        negative samples. Use these samples to generate instructions
        Args:
            patient_id (Union[str, int]): The unique id of the patient we need to process
            current_time (str): The timestamp of the sample we are processing - used to calculate time deltas
            with respect to other encounters
            past_time_delta (str): Filter out encounters beyond this point in time
            future_time_delta (str): Filter out encounters beyond this point in time
            use_log_position (bool): Whether to keep time deltas in days or as log value of days
            k_shot (int): The number of k-shot examples to use
            random_negative_probability (float, defaults to 0.5): Probability of sampling negatives randomly
            time_difference_normalize (int, defaults to `30`): Normalize time difference by this value
            (e.g. 30 normalizes it to months)
            shuffle (bool): Whether to shuffle the list of instructions.

        Returns:
            (List[Tuple[str, str]]): The instructions that will be used for training
        """

        k_shot = np.random.choice(k_shot)

        full_encounter_history = self._encounter_dataframe_process.get_patient_sequence(
            patient_id=patient_id,
            current_time=current_time,
            use_log_position=use_log_position,
            time_difference_normalize=time_difference_normalize
        )

        sampled_positives, sampled_negatives = self.process_samples_for_task(
            encounter_history=full_encounter_history,
            past_time_delta=past_time_delta,
            future_time_delta=future_time_delta,
            k_shot=k_shot,
            random_negative_probability=random_negative_probability,
            time_difference_normalize=time_difference_normalize
        )

        # Get the instructions - this should contain the instruction inputs and instruction targets
        return self.process_task_instructions(
            sampled_positives=sampled_positives,
            sampled_negatives=sampled_negatives,
            shuffle=shuffle,
        )

    def get_positive_samples(self, encounter_history: pd.DataFrame, size: int) -> pd.DataFrame:
        """
        Returns the positive samples - sampled from the encounter history
        Args:
            encounter_history (pd.DataFrame): The dataframe containing patient encounters
            size (int): How many samples to return

        Returns:
            (pd.DataFrame): A subset of the encounters that will be used as positive
            samples
        """
        # Sample a subset of positives from dataframe
        return self._dataframe_sampling.sample_dataframe(
            dataframe=encounter_history,
            sample_size=size,
            weights=None
        )

    def get_negative_samples(
            self,
            encounter_history: pd.DataFrame,
            size: int,
            random_negative_probability: float,
            exclude_codes: pd.Series
    ) -> pd.DataFrame:
        """
        Returns the negative samples - sampled from the encounter history or randomly sampled
        Exclude any codes to be sampled as negatives when sampling by passing exclude codes
        argument
        Args:
            encounter_history (pd.DataFrame): The dataframe containing patient encounters
            size (int): How many samples to return
            random_negative_probability (float, defaults to 0.5): Probability of sampling negatives randomly
            exclude_codes (pd.Series): Exclude these codes when randomly sampling negatives

        Returns:
            (pd.DataFrame): A subset of the encounters that will be used as negative
            samples. It could also be a dataset with negatives sampled randomly from an
            external source
        """

        # Sample a subset of negatives from dataframe
        sampled_negatives = self._dataframe_sampling.sample_dataframe(
            dataframe=encounter_history,
            sample_size=size,
            weights=None
        )

        # Sample negatives uniformly from a given set of codes
        # Exclude codes if desired
        if np.random.rand() < random_negative_probability:
            codes = self._negative_code_sampling.get_negatives(
                positives=exclude_codes,
                k=size
            )
            sampled_negatives[self._code_column] = codes

        return sampled_negatives

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
        codes = self._negative_code_sampling.get_negatives(
            positives=exclude_codes,
            k=size
        )
        # Sample position values
        positions = np.random.choice(position_range, size)
        # Use these to create and return a dataframe
        return pd.DataFrame(
            {self._code_column: codes, self._position_column: positions}
        )

    def process_task_instructions(
            self,
            sampled_positives: pd.DataFrame,
            sampled_negatives: pd.DataFrame,
            shuffle,
    ) -> List[Tuple[str, str]]:
        """
        Return the instruction input and targets
        for the sampled instruction icd status classification task
        Args:
            sampled_positives (pd.DataFrame): The dataframe that is used for positive diagnosis
            sampled_negatives (pd.DataFrame): The dataframe that will be used for negative diagnosis
            shuffle (bool, defaults to `True`): Whether to shuffle the list of instructions.

        Returns:
            (List[Tuple[str, str]]): A list that contains tuples which have the instruction input and target.
            The list will contain only 1 element for zero shot training
        """
        # Get the task definition string
        task_definition = self._code_instructions.get_task_definition()

        # Get positive instruction inputs and target
        positive_instructions = self._code_instructions.get_positive_instructions(
            diagnosis_list=sampled_positives[self._code_column].apply(self._code_convert.transform_code),
            positions_list=sampled_positives[self._position_column]
        )

        negative_instructions = self._code_instructions.get_negative_instructions(
            diagnosis_list=sampled_negatives[self._code_column].apply(self._code_convert.transform_code),
            positions_list=sampled_negatives[self._position_column]
        )

        # Concatenate list of positive and negative instructions and shuffle the list
        instructions = positive_instructions + negative_instructions
        if shuffle:
            random.shuffle(instructions)

        # Add the task definition to the list
        return [task_definition] + instructions

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
        if number_of_encounters == 0:
            return 0, max_size
        positive = 0
        negative = 0
        for k in range(max_size):
            if np.random.rand() <= 0.5:
                positive += 1
            else:
                negative += 1
        return positive, negative


class CodeStatusRangeClassificationTask(CodeStatusClassificationTask):
    """
    Make instruction tuning training data from diagnostic codes. The instruction
    asks the model to predict whether a patient is diagnosed with a given condition
    within a given time range.
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
            position_column: str = 'position'
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
            code_column (str, defaults to `ICD10CD`): The column that contains the codes
            position_column (str, defaults to `positions`): The column where position values
            (e.g. time difference in days or log days) will be stored
        """
        super().__init__(
            encounter_dataframe_process=encounter_dataframe_process,
            dataframe_sampling=dataframe_sampling,
            code_instructions=code_instructions,
            code_convert=code_convert,
            negative_code_sampling=negative_code_sampling,
            patient_id_column=patient_id_column,
            code_column=code_column,
            position_column=position_column
        )

    def process_sample(
            self,
            patient_id: Union[str, int],
            current_time: str,
            past_time_delta: str,
            future_time_delta: str,
            use_log_position: bool,
            k_shot: int,
            random_negative_probability: float,
            time_difference_normalize: int,
            shuffle: bool
    ):
        """
        Given the encounter history, map codes, filter out duplicate codes and filter
        the history based on time range. Keep only those entries
        that occur within the time range. If encounter history - create a history
        by randomly generating codes and positions. This will only be used to sample
        negatives. Then sample the dataframe for encounters to use as positive and
        negative samples. Use these samples to generate instructions
        Args:
            patient_id (Union[str, int]): The unique id of the patient we need to process
            current_time (str): The timestamp of the sample we are processing - used to calculate time deltas
            with respect to other encounters
            past_time_delta (str): Filter out encounters beyond this point in time
            future_time_delta (str): Filter out encounters beyond this point in time
            use_log_position (bool): Whether to keep time deltas in days or as log value of days
            k_shot (int): The number of k-shot examples to use
            random_negative_probability (float, defaults to 0.5): Probability of sampling negatives randomly
            time_difference_normalize (int, defaults to `30`): Normalize time difference by this value
            (e.g. 30 normalizes it to months)
            shuffle (bool): Whether to shuffle the list of instructions.

        Returns:
            (List[Tuple[str, str]]): The instructions that will be used for training
        """

        k_shot = np.random.choice(k_shot)

        full_encounter_history = self._encounter_dataframe_process.get_patient_sequence(
            patient_id=patient_id,
            current_time=current_time,
            use_log_position=use_log_position,
            time_difference_normalize=time_difference_normalize
        )

        sampled_positives, sampled_negatives = self.process_samples_for_task(
            encounter_history=full_encounter_history,
            past_time_delta=past_time_delta,
            future_time_delta=future_time_delta,
            k_shot=k_shot,
            random_negative_probability=random_negative_probability,
            time_difference_normalize=time_difference_normalize
        )

        # Create the time range for both and future.
        # These represent the lowest and highest possible value that can be used
        # for a time range
        past_start_range_limit = 0
        future_start_range_limit = 0
        past_end_range_limit = math.ceil(pd.to_timedelta(past_time_delta).days /  time_difference_normalize)
        future_end_range_limit = math.ceil(pd.to_timedelta(future_time_delta).days /  time_difference_normalize)

        # Get the instructions - this should contain the instruction inputs and instruction targets
        return self.process_task_instructions(
            sampled_positives=sampled_positives,
            sampled_negatives=sampled_negatives,
            shuffle=shuffle,
            past_start_range_limit=past_start_range_limit,
            past_end_range_limit=past_end_range_limit,
            future_start_range_limit=future_start_range_limit,
            future_end_range_limit=future_end_range_limit
        )

    def process_task_instructions(
            self,
            sampled_positives: pd.DataFrame,
            sampled_negatives: pd.DataFrame,
            shuffle,
            past_start_range_limit: int = None,
            past_end_range_limit: int = None,
            future_start_range_limit: int = None,
            future_end_range_limit: int = None
    ) -> List[Tuple[str, str]]:
        """
        Return the instruction input and targets
        for the sampled instruction code status classification task with time range
        Args:
            sampled_positives (pd.DataFrame): The dataframe that is used for positive diagnosis
            sampled_negatives (pd.DataFrame): The dataframe that will be used for negative diagnosis
            shuffle (bool, defaults to `True`): Whether to shuffle the list of instructions.
            past_start_range_limit (int): The start limit for the past position range
            past_end_range_limit (int): The end limit for the past position range
            future_start_range_limit (int): The start limit for the future position range
            future_end_range_limit (int): The end limit for the future position range

        Returns:
            (List[Tuple[str, str]]): A list that contains tuples which have the instruction input and target.
            The list will contain only 1 element for zero shot training
        """

        # Get the task definition string
        task_definition = self._code_instructions.get_task_definition()

        # Get positive instruction inputs and target
        positive_instructions = self._code_instructions.get_positive_instructions(
            diagnosis_list=sampled_positives[self._code_column].apply(self._code_convert.transform_code),
            positions_list=sampled_positives[self._position_column],
            past_start_range_limit=past_start_range_limit,
            past_end_range_limit=past_end_range_limit,
            future_start_range_limit=future_start_range_limit,
            future_end_range_limit=future_end_range_limit
        )

        # Get negative instruction inputs and target
        negative_instructions = self._code_instructions.get_negative_instructions(
            diagnosis_list=sampled_negatives[self._code_column].apply(self._code_convert.transform_code),
            positions_list=sampled_negatives[self._position_column],
            past_start_range_limit=past_start_range_limit,
            past_end_range_limit=past_end_range_limit,
            future_start_range_limit=future_start_range_limit,
            future_end_range_limit=future_end_range_limit
        )

        # Concatenate list of positive and negative instructions and shuffle the list
        instructions = positive_instructions + negative_instructions
        if shuffle:
            random.shuffle(instructions)

        # Add the task definition to the list
        return [task_definition] + instructions


class CodeStatusRangeClassificationTaskEval(CodeStatusRangeClassificationTask):
    """
    Make instruction tuning evaluation data from diagnostic codes. This constructs data
    for a given code. The instruction asks the model to predict whether a patient is
    diagnosed with that condition within a given time range.
    """

    def __init__(
            self,
            evaluate_attribute: Tuple[str, Tuple[int, int], Optional[bool]],
            encounter_dataframe_process: EncounterDataframeProcess,
            dataframe_sampling: DataFrameSampling,
            code_instructions: Any,
            code_convert: ICDConvert,
            negative_code_sampling: Optional[NegativeICDSampling] = None,
            patient_id_column: str = 'PatientID',
            code_column: str = 'ICD10CD',
            position_column: str = 'position',
            example_attributes: Optional[List[Tuple[str, Tuple[int, int], Optional[bool]]]] = None,
    ):
        """
        Initialize variables
        Args:
            evaluate_attribute(Tuple[str, Tuple[int, int], Optional[bool]]): The code that we want to evaluate -
            it will contain the code and the time range we want to check the presence or absence of that code
            encounter_dataframe_process (EncounterDataframeProcess): Dataframe processing object
            dataframe_sampling (DataFrameSampling): Dataframe sampling object
            negative_code_sampling (NegativeICDSampling, defaults to `None`): Object to sample negatives from a
            pre-defined list
            code_instructions (Any): A list of code instruction tasks. At any point one of these
            tasks is randomly chosen and trained on
            code_convert (ICDConvert): The object to map codes to their textual descriptions'
            patient_id_column (str, defaults to `PatientID`): The column name for the patient id
            code_column (str, defaults to `ICD10CD`): The column that contains the codes
            position_column (str, defaults to `positions`): The column where position values
            (e.g. time difference in days or log days) will be stored
            example_attributes(): Any few shot prompts to add while evaluating. If we want to add any
            additional descriptions to the prompt we add it through this
        """
        super().__init__(
            encounter_dataframe_process=encounter_dataframe_process,
            code_instructions=code_instructions,
            dataframe_sampling=dataframe_sampling,
            negative_code_sampling=negative_code_sampling,
            code_convert=code_convert,
            patient_id_column=patient_id_column,
            code_column=code_column,
            position_column=position_column
        )
        self._evaluate_attribute = evaluate_attribute
        self._example_attributes = example_attributes

    def get_encounter_range(
            self,
            encounter_history: pd.DataFrame,
            start_time: int,
            end_time: int,
            time_period: int
    ) -> pd.DataFrame:
        """
        Filter the encounter history to return only those entries that occur within
        the given time range
        Args:
            encounter_history (pd.DataFrame): The dataframe that contains encounter history
            start_time (int): The start time of the range
            end_time (int): The end time of the range
            time_period (int): Flag to indicate whether the time range is in the past or future

        Returns:
            (pd.DataFrame): The filtered dataframe
        """
        if time_period == 1:
            return encounter_history[
                (encounter_history[self._position_column] >= start_time)
                & (encounter_history[self._position_column] < end_time)
                ]
        elif time_period == -1:
            return encounter_history[
                (encounter_history[self._position_column] <= start_time)
                & (encounter_history[self._position_column] > end_time)
                ]
        else:
            raise ValueError('Invalid time period')

    def __get_answer(
            self,
            encounter_history: pd.DataFrame,
            code: str,
            regex: bool
    ) -> int:
        """
        Check if a given code exists in a given encounter history
        Args:
            encounter_history (pd.DataFrame): The encounter history
            code (str): A given diagnostic code
            regex (bool): Whether to perform regex match or not

        Returns:
            (int): Count of the number of matches
        """
        return sum(encounter_history[self._code_column].str.contains(code, regex=regex))

    def get_answer(
            self,
            encounter_history: pd.DataFrame,
            code: str,
            start_time: int,
            end_time: int,
            answer: Optional[bool],
            regex: bool
    ):
        """
        Check if a given code exists in a given encounter history
        Args:
            encounter_history (pd.DataFrame): The encounter history dataframe
            code (str): The code to check - present or not in encounter history
            start_time (int): The start time of the period to check in
            end_time (int): The end time of the period to check in
            answer (bool): True if code is present, False otherwise
            regex (bool): Whether to use regex matching when checking presence of code

        Returns:
            (bool): A boolean indicating whether a given code in present in the encounter history
        """

        if answer is None:
            # Select the dataframe values within the time range
            encounter_subset = self.get_encounter_range(
                encounter_history=encounter_history,
                start_time=start_time,
                end_time=end_time,
                time_period=-1 if end_time < 0 else 1
            )

            # Find the answer
            answer = self.__get_answer(encounter_history=encounter_subset, code=code, regex=regex)

        return answer

    def process_example_attributes(
            self,
            example_attributes: List[Tuple[str, Tuple[int, int], Optional[bool]]],
            encounter_history: pd.DataFrame,
            regex: bool
    ):
        """
        Given the encounter history and a set of prompt codes and evaluation codes.
        First check if we need to find the presence or absence of the prompt codes
        in the user specified time ranges, if the user has not specified it.
        Then check if the evaluation code is present or absent in the time range.
        Once we have positives and negatives - build the prompt instruction
        Args:
            example_attributes (List[Tuple[str, Tuple[int, int], Optional[bool]]]): List containing the prompt codes
            along with their time ranges and optionally a flag indicating their presence or absence in the encounter
            history.
            encounter_history (pd.DataFrame): The encounter dataframe
            regex (bool): Whether to use regex matching when checking presence of code

        Returns:
            (List[Tuple[str, str]]): The instruction prompt that will be used for evaluation
        """
        # Initialize instructions with task definition
        instructions = list()
        instructions.append(self._code_instructions.get_task_definition())

        for example_attribute in example_attributes:
            icd_code = example_attribute[0]
            start_time, end_time = example_attribute[1]
            # Check if the code is present in the time range defined by start and end time
            answer = self.get_answer(
                encounter_history=encounter_history,
                code=icd_code,
                start_time=start_time,
                end_time=end_time,
                answer=example_attribute[2],
                regex=regex
            )

            # If the code is present - generate a positive instruction prompt
            if answer:
                instruction = self._code_instructions.get_positive_instruction(
                    diagnosis=self._code_convert.transform_code(icd_code),
                    position=-1 if end_time < 0 else 1,
                    start_range_limit=start_time,
                    end_range_limit=end_time
                )
            # If the code is not present - generate a negative instruction prompt
            else:
                instruction = self._code_instructions.get_negative_instruction(
                    diagnosis=self._code_convert.transform_code(icd_code),
                    position=-1 if end_time < 0 else 1,
                    start_range_limit=start_time,
                    end_range_limit=end_time
                )
            instructions.append(instruction)
        return instructions

    def process_sample(
            self,
            patient_id: Union[str, int],
            current_time: str,
            past_time_delta: str,
            future_time_delta: str,
            use_log_position: bool,
            k_shot: int,
            random_negative_probability: float,
            time_difference_normalize: int,
            shuffle: bool
    ):
        """
        Given the encounter history, map codes, filter out duplicate codes and filter
        the history based on time range. Keep only those entries
        that occur within the time range. Then check if the code specified in evaluate
        attribute and optionally the example attributes is present in the encounter history.
        Build the positive and negative instructions/prompt accordingly
        Args:
            patient_id (Union[str, int]): The unique id of the patient we need to process
            current_time (str): The timestamp of the sample we are processing - used to calculate time deltas
            with respect to other encounters
            past_time_delta (str): Filter out encounters beyond this point in time
            future_time_delta (str): Filter out encounters beyond this point in time
            use_log_position (bool): Whether to keep time deltas in days or as log value of days
            k_shot (int): The number of k-shot examples to use
            random_negative_probability (float, defaults to 0.5): Probability of sampling negatives randomly
            time_difference_normalize (int, defaults to `30`): Normalize time difference by this value
            (e.g. 30 normalizes it to months)
            shuffle (bool): Whether to shuffle the list of instructions.

        Returns:
            (List[Tuple[str, str]]): The instructions that will be used for evaluation
        """

        k_shot = np.random.choice(k_shot)

        # Get encounter history of patient
        encounter_history = self._encounter_dataframe_process.get_patient_sequence(
            patient_id=patient_id,
            current_time=current_time,
            use_log_position=use_log_position,
            time_difference_normalize=time_difference_normalize
        )

        # Map and filter encounter dataframe
        encounter_history, _ = self.filter_encounter_history_for_task(
            encounter_history=encounter_history,
            past_time_delta=past_time_delta,
            future_time_delta=future_time_delta,
        )

        if self._example_attributes is None and k_shot == 0:
            example_attributes = [self._evaluate_attribute]
        elif self._example_attributes is None:
            # sampled_positives, sampled_negatives = self.process_samples_for_task(
            #     encounter_history=encounter_history,
            #     past_time_delta=past_time_delta,
            #     future_time_delta=future_time_delta,
            #     k_shot=k_shot,
            #     random_negative_probability=random_negative_probability,
            #     time_difference_normalize=time_difference_normalize
            # )
            example_attributes = None
        else:
            # The evaluate attribute will be at the end of the prompt
            example_attributes = self._example_attributes + [self._evaluate_attribute]

        # Get the instructions - this should contain the instruction inputs and instruction targets
        return self.process_example_attributes(
            example_attributes=example_attributes,
            encounter_history=encounter_history,
            regex=False
        )


class ICDT2EPredictionTask(CodeStatusClassificationTask):
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
            position_column: str = 'position'
    ):
        """
        Initialize variables
        Args:
            encounter_dataframe_process (EncounterDataframeProcess): Dataframe processing object
            dataframe_sampling (DataFrameSampling): Dataframe sampling object
            negative_code_sampling (NegativeICDSampling, defaults to `None`): Object to sample negatives from a
            pre-defined list
            code_instructions (Any): A list of icd instruction tasks. At any point one of these
            tasks is randomly chosen and trained on
            code_convert (ICDConvert): The object to map codes to their textual descriptions'
            patient_id_column (str, defaults to `PatientID`): The column name for the patient id
            code_column (str, defaults to `ICD10CD`): The column that contains the icd codes
            position_column (str, defaults to `positions`): The column where position values
            (e.g. time difference in days or log days) will be stored
        """
        super().__init__(
            encounter_dataframe_process=encounter_dataframe_process,
            dataframe_sampling=dataframe_sampling,
            code_instructions=code_instructions,
            code_convert=code_convert,
            negative_code_sampling=negative_code_sampling,
            patient_id_column=patient_id_column,
            code_column=code_column,
            position_column=position_column
        )

    def process_sample(
            self,
            patient_id: Union[str, int],
            current_time: str,
            past_time_delta: str,
            future_time_delta: str,
            use_log_position: bool,
            k_shot: int,
            random_negative_probability: float,
            time_difference_normalize: int,
            shuffle: bool,
    ):
        """
        Given the encounter history, map codes, filter out duplicate codes and filter
        the history based on time range. Keep only those entries
        that occur within the time range. If encounter history - create a history
        by randomly generating codes and positions. This will only be used to sample
        negatives. Then sample the dataframe for encounters to use as positive and
        negative samples. Use these samples to generate instructions
        Args:
            patient_id (Union[str, int]): The unique id of the patient we need to process
            current_time (str): The timestamp of the sample we are processing - used to calculate time deltas
            with respect to other encounters
            past_time_delta (str): Filter out encounters beyond this point in time
            future_time_delta (str): Filter out encounters beyond this point in time
            use_log_position (bool): Whether to keep time deltas in days or as log value of days
            k_shot (int): The number of k-shot examples to use
            random_negative_probability (float, defaults to 0.5): Probability of sampling negatives randomly
            time_difference_normalize (int, defaults to `30`): Normalize time difference by this value
            (e.g. 30 normalizes it to months)
            shuffle (bool): Whether to shuffle the list of instructions.

        Returns:
            (List[Tuple[str, str]]): The instructions that will be used for training
        """

        assert random_negative_probability == 1.0, "random negative probability needs to be 1"

        k_shot = np.random.choice(k_shot)

        full_encounter_history = self._encounter_dataframe_process.get_patient_sequence(
            patient_id=patient_id,
            current_time=current_time,
            use_log_position=use_log_position,
            time_difference_normalize=time_difference_normalize
        )

        sampled_positives, sampled_negatives = self.process_samples_for_task(
            encounter_history=full_encounter_history,
            past_time_delta=past_time_delta,
            future_time_delta=future_time_delta,
            k_shot=k_shot,
            random_negative_probability=random_negative_probability,
            time_difference_normalize=time_difference_normalize
        )

        # Get the instructions - this should contain the instruction inputs and instruction targets
        return self.process_task_instructions(
            sampled_positives=sampled_positives,
            sampled_negatives=sampled_negatives,
            shuffle=shuffle,
        )
