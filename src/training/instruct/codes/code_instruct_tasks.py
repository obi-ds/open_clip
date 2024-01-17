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
        self.patient_id_column = patient_id_column
        self.code_column = code_column
        self.position_column = position_column
        self.instruction_label = instruction_label

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
            # TODO: A fix for now - because we have missing data - ideally we should not have missing data
            position_range = np.arange(-6, 7, step=0.5)
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
        all_positives = self._encounter_dataframe_process.get_all_positives(
            encounter_history=encounter_history
        )
        encounter_history = self._encounter_dataframe_process.filter_encounter_history_for_task(
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
        # TODO: Sampling strategy
        positions = encounter_history[self.position_column] * -1
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
            [self.code_column, self.position_column, self.instruction_label]
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

        return instructions

    def get_instruction_samples(
            self,
            encounter_history: pd.DataFrame,
            encounter_negatives: pd.DataFrame,
            k_shot: int,
            sort_position: bool,
            exclude_codes_from_negatives: Optional[pd.Series] = None,
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
            encounter_negatives (pd.DataFrame): The dataframe to sample negatives from
            k_shot (int): The number of k-shot examples to use
            sort_position (bool): Whether to sort instructions by position
            exclude_codes_from_negatives (Optional[pd.Series], defaults to `None`   ): List of codes that should not
            be used for negative sampling

        Returns:
            (pd.DataFrame): The instructions samples
        """
        # Get the number of positives and negative samples to use in the instructions
        number_of_positives, number_of_negatives = self.get_positive_negative_sizes(
            k_shot=k_shot, number_of_encounters=encounter_history.shape[0]
        )

        # Sample a subset of positives from dataframe
        sampled_positives = self.get_positive_samples(
            encounter_history=encounter_history,
            size=number_of_positives,
        )
        sampled_positives[self.instruction_label] = True

        sampled_negatives = self.get_negative_samples(
            encounter_negatives=encounter_negatives,
            size=number_of_negatives,
            exclude_codes=exclude_codes_from_negatives
        )
        sampled_negatives[self.instruction_label] = False


        instruction_samples = self.concatenate_instruction_samples(
            sampled_positives=sampled_positives,
            sampled_negatives=sampled_negatives
        )

        if sort_position:
            instruction_samples.sort_values(by=self.position_column)

        return instruction_samples

    @staticmethod
    def concatenate_instruction_samples(
            sampled_positives: pd.DataFrame,
            sampled_negatives: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Concatenate the positive and negative samples
        Args:
            sampled_positives (pd.DataFrame): The sampled positives
            sampled_negatives (pd.DataFrame): The sampled negatives

        Returns:
            (pd.DataFrame): The concatenated dataframe
        """
        if not sampled_positives.empty and not sampled_negatives.empty:
            return pd.concat([sampled_positives, sampled_negatives])
        elif not sampled_negatives.empty:
            return sampled_negatives
        elif not sampled_positives.empty:
            return sampled_positives
        else:
            raise ValueError('No instruction samples')

    def get_task_instructions(
            self,
            encounter_history: pd.DataFrame,
            all_positives: pd.Series,
            positives: pd.Series,
            encounter_negatives: pd.DataFrame,
            k_shot: int,
            sort_position: bool,
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
            encounter_negatives (pd.DataFrame): Use this to sample negatives

        Returns:
            instruction_samples (pd.DataFrame): The instructions samples
            instructions (List[Tuple[str, str]]): A list that contains tuples which have the instruction
            input and target. The list will contain only 1 element for zero shot training
        """

        exclude_codes_from_negatives = self.get_exclude_codes_from_negatives(
            all_positives=all_positives,
            positives=positives
        )

        encounter_negatives = self.get_exclude_codes_dataframe(
            encounter_dataframe=encounter_negatives,
            exclude_codes=exclude_codes_from_negatives
        )

        instruction_samples = self.get_instruction_samples(
            encounter_history=encounter_history,
            encounter_negatives=encounter_negatives,
            k_shot=k_shot,
            sort_position=sort_position,
            exclude_codes_from_negatives=exclude_codes_from_negatives
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
    ) -> pd.DataFrame:
        """
        Returns the positive samples - sampled from the encounter history
        Args:
            encounter_history (pd.DataFrame): The encounter history of the patient
            size (int): How many samples to return

        Returns:
            (pd.DataFrame): A subset of the encounters that will be used as positive
            samples
        """
        # TODO: Positive sampling weights
        # weights = self.get_sampling_weights(encounter_history=encounter_history)
        weights = None
        # Sample a subset of positives from dataframe
        return self._dataframe_sampling.sample_dataframe(
            dataframe=encounter_history,
            sample_size=size,
            weights=weights
        )

    def get_negative_samples(
            self,
            encounter_negatives: pd.DataFrame,
            size: int,
            exclude_codes: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Returns the negatives samples - sampled based on the given encounter history
        Args:
            encounter_negatives (pd.DataFrame): The encounter history of the patient
            size (int): How many samples to return
            exclude_codes (Optional[pd.Series], defaults to `None`): Codes excluded from negative sampling

        Returns:
            (pd.DataFrame): A subset of the encounters that will be used as positive
            samples
        """
        # TODO: Negative sampling weights
        sampled_negatives = self.get_negative_samples_from_encounter_negatives(
            encounter_negatives=encounter_negatives,
            size=min(size, len(encounter_negatives)),
        )

        if len(sampled_negatives) != size:
            position_range = self.get_position_range_from_encounter_dataframe(
                encounter_dataframe=encounter_negatives
            )
            random_encounters = self.get_random_encounters(
                size=size - len(sampled_negatives),
                position_range=position_range,
                exclude_codes=exclude_codes
            )
            if len(sampled_negatives) > 0:
                sampled_negatives = pd.concat(
                    [sampled_negatives[[self.code_column, self.position_column]], random_encounters]
                )
            else:
                sampled_negatives = random_encounters
        return sampled_negatives

    def get_negative_samples_from_encounter_negatives(
            self,
            encounter_negatives: pd.DataFrame,
            size: int,
    ) -> pd.DataFrame:
        """
        Returns the negatives samples - sampled based on the given encounter history
        Args:
            encounter_negatives (pd.DataFrame): The encounter history of the patient
            size (int): How many samples to return

        Returns:
            (pd.DataFrame): A subset of the encounters that will be used as positive
            samples
        """
        weights = None
        return self._dataframe_sampling.sample_dataframe(
            dataframe=encounter_negatives,
            sample_size=size,
            weights=weights
        )

    def get_exclude_codes_dataframe(self, encounter_dataframe: pd.DataFrame, exclude_codes: pd.Series) -> pd.DataFrame:
        """
        Given an input dataframe - return the original dataframe without the
        codes given in exclude codes - basically exclude these codes from the
        dataframe and return
        Args:
            encounter_dataframe (pd.DataFrame): The input dataframe
            exclude_codes (pd.Series): The codes to exclude

        Returns:
            (pd.DataFrame): Dataframe with code excluded
        """
        return encounter_dataframe[~encounter_dataframe[self.code_column].isin(exclude_codes)]

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
        return encounter_history[self.code_column]

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

    def get_position_range_from_encounter_dataframe(self, encounter_dataframe: pd.DataFrame) -> np.array:
        """
        Get a position range to create new random encounters. The positions
        will be based on existing position values, whereas the codes will
        be sampled
        Args:
            encounter_dataframe:

        Returns:
            (np.array): A range of position values
        """
        if encounter_dataframe.empty:
            return np.arange(0, 1, step=0.5)
        else:
            return np.arange(
                encounter_dataframe[self.position_column].min(),
                encounter_dataframe[self.position_column].max() + 1,
                step=0.5
            )

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
            {self.code_column: codes, self.position_column: positions}
        )

    @staticmethod
    def get_positive_negative_sizes(k_shot: int, number_of_encounters: int) -> Tuple[int, int]:
        """
        For a given k-shot problem, return how many positive samples
        and negative samples should be present in the instruction

        Args:
            k_shot (int): The number of k_shot examples
            number_of_encounters (int): The total number of encounters
        Returns:
            (Tuple[int, int]): The number of positive and negative examples
        """
        number_of_positives = np.random.choice(np.arange(0, min(k_shot + 1, number_of_encounters) + 1))
        number_of_negatives = k_shot + 1 - number_of_positives
        return number_of_positives, number_of_negatives


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
