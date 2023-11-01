"""Make training data"""
import random
import numpy as np
import pandas as pd
from typing import Union, Tuple, List, Any, Optional

from .processing.icd_convert import ICDConvert
from .processing.encounter_dataframe_process import EncounterDataframeProcess
from .processing.dataframe_sampling import DataFrameSampling
from .processing.negative_icd_sampling import NegativeICDSampling

np.random.seed(42)


class ICDStatusClassificationTask(object):
    """
    Make training data from icd codes
    """

    def __init__(
            self,
            encounter_dataframe_process: EncounterDataframeProcess,
            dataframe_sampling: DataFrameSampling,
            icd_instructions: Any,
            icd_convert: ICDConvert,
            negative_icd_sampling: Optional[NegativeICDSampling] = None,
            patient_id_column: str = 'PatientID',
            icd_10_column: str = 'ICD10CD',
            position_column: str = 'position'
    ):
        """
        Initialize variables
        Args:
            encounter_dataframe_process (EncounterDataframeProcess): Dataframe processing object
            dataframe_sampling (DataFrameSampling): Dataframe sampling object
            negative_icd_sampling (NegativeICDSampling, defaults to `None`): Object to sample negatives from a
            pre-defined list
            icd_instructions (Any): A list of icd instruction tasks. At any point one of these
            tasks is randomly chosen and trained on
            icd_convert (ICDConvert): The object to map codes to their textual descriptions'
            patient_id_column (str, defaults to `PatientID`): The column name for the patient id
            icd_10_column (str, defaults to `ICD10CD`): The column that contains the icd codes
            position_column (str, defaults to `positions`): The column where position values
            (e.g. time difference in days or log days) will be stored
        """
        self._encounter_dataframe_process = encounter_dataframe_process
        self._dataframe_sampling = dataframe_sampling
        self._negative_icd_sampling = negative_icd_sampling
        self._icd_instructions = icd_instructions
        self._icd_convert = icd_convert
        self._patient_id_column = patient_id_column
        self._icd_10_column = icd_10_column
        self._position_column = position_column

    def process_sample_from_args(self, sample, args):
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

    def get_instruction_samples(
            self,
            patient_id: Union[str, int],
            current_time: str,
            past_time_delta: str,
            future_time_delta: str,
            use_log_position: bool,
            k_shot: int,
            random_negative_probability: float,
            time_difference_normalize: int
    ):
        """
        Build the instruction training data based on encounter history, relative times and negative sampling
        Extract the data and pass the relevant to data to the sampled instruction task. Return a tuple
        that contains the input and label ids for the given instruction task
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

        Returns:

            training_data (str): The train data after applying various functions, transformations
            and sampling
        """
        k_shot = np.random.choice(k_shot)

        # Get encounter history of patient
        encounter_history = self._encounter_dataframe_process.get_patient_sequence(
            patient_id=patient_id,
            current_time=current_time,
            past_time_delta=past_time_delta,
            future_time_delta=future_time_delta,
            use_log_position=use_log_position,
            time_difference_normalize=time_difference_normalize
        )

        # Check if dataframe contains any data
        # If not, create a random dataframe - will only be used
        # as negatives
        number_of_encounters = encounter_history.shape[0]

        position_range = np.arange(
            -int(np.ceil(pd.to_timedelta(past_time_delta).days / time_difference_normalize)),
            int(np.ceil(pd.to_timedelta(future_time_delta).days / time_difference_normalize))
        )

        if number_of_encounters == 0:
            encounter_history = self.get_random_encounters(
                size=k_shot + 1,
                position_range=position_range
            )

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
            random_negative_probability=0 if number_of_encounters == 0 else random_negative_probability
        )

        # Get the instructions - this should contain the instruction inputs and instruction targets
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
    ):
        """
        Build the instruction training data based on encounter history, relative times and negative sampling
        Extract the data and pass the relevant to data to the sampled instruction task. Return a tuple
        that contains the input and label ids for the given instruction task
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

            training_data (str): The train data after applying various functions, transformations
            and sampling
        """

        sampled_positives, sampled_negatives = self.get_instruction_samples(
            patient_id=patient_id,
            current_time=current_time,
            past_time_delta=past_time_delta,
            future_time_delta=future_time_delta,
            use_log_position=use_log_position,
            k_shot=k_shot,
            random_negative_probability=random_negative_probability,
            time_difference_normalize=time_difference_normalize
        )

        # Get the instructions - this should contain the instruction inputs and instruction targets
        return self.process_status_classification_instructions(
            sampled_positives=sampled_positives,
            sampled_negatives=sampled_negatives,
            shuffle=shuffle,
        )

    def get_positive_samples(self, encounter_history: pd.DataFrame, size: int) -> pd.DataFrame:
        """
        Returns the positive samples
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
            random_negative_probability
    ) -> pd.DataFrame:
        """
        Returns the positive samples
        Args:
            encounter_history (pd.DataFrame): The dataframe containing patient encounters
            size (int): How many samples to return
            random_negative_probability (float, defaults to 0.5): Probability of sampling negatives randomly

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

        # Get negative instruction inputs and target
        if np.random.rand() < random_negative_probability:
            codes = self._negative_icd_sampling.get_negatives(
                positives=encounter_history[self._icd_10_column],
                k=size
            )
            sampled_negatives[self._icd_10_column] = codes

        return sampled_negatives


    def get_random_encounters(self, size: int, position_range: int) -> pd.DataFrame:
        """
        Use this function to create a random dataframe populated
        with codes and positions
        Args:
            size (int): The size of the dataframe
            position_range (int): Position values are assigned based on this range

        Returns:
            (pd.DataFrame): A random dataframe
        """
        codes = self._negative_icd_sampling.get_negatives(
            positives=pd.Series([]),
            k=size
        )
        positions = np.random.choice(position_range, size)
        return pd.DataFrame(
            {self._icd_10_column: codes, self._position_column: positions}
        )

    def process_status_classification_instructions(
            self,
            sampled_positives: pd.DataFrame,
            sampled_negatives: pd.DataFrame,
            shuffle,
    ):
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
        task_definition = self._icd_instructions.get_task_definition()

        # Get positive instruction inputs and target
        positive_instructions = self._icd_instructions.get_positive_instructions(
            diagnosis_list=sampled_positives[self._icd_10_column].apply(self._icd_convert.get_converted_code),
            positions_list=sampled_positives[self._position_column]
        )

        negative_instructions = self._icd_instructions.get_negative_instructions(
            diagnosis_list=sampled_negatives[self._icd_10_column].apply(self._icd_convert.get_converted_code),
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


class ICDStatusRangeClassificationTask(ICDStatusClassificationTask):
    """
    Make training data from icd codes
    """

    def __init__(
            self,
            encounter_dataframe_process: EncounterDataframeProcess,
            dataframe_sampling: DataFrameSampling,
            icd_instructions: Any,
            icd_convert: ICDConvert,
            negative_icd_sampling: Optional[NegativeICDSampling] = None,
            patient_id_column: str = 'PatientID',
            icd_10_column: str = 'ICD10CD',
            position_column: str = 'position'
    ):
        """
        Initialize variables
        Args:
            encounter_dataframe_process (EncounterDataframeProcess): Dataframe processing object
            dataframe_sampling (DataFrameSampling): Dataframe sampling object
            negative_icd_sampling (NegativeICDSampling, defaults to `None`): Object to sample negatives from a
            pre-defined list
            icd_instructions (Any): A list of icd instruction tasks. At any point one of these
            tasks is randomly chosen and trained on
            icd_convert (ICDConvert): The object to map codes to their textual descriptions'
            patient_id_column (str, defaults to `PatientID`): The column name for the patient id
            icd_10_column (str, defaults to `ICD10CD`): The column that contains the icd codes
            position_column (str, defaults to `positions`): The column where position values
            (e.g. time difference in days or log days) will be stored
        """
        super().__init__(
            encounter_dataframe_process=encounter_dataframe_process,
            dataframe_sampling=dataframe_sampling,
            icd_instructions=icd_instructions,
            icd_convert=icd_convert,
            negative_icd_sampling=negative_icd_sampling,
            patient_id_column=patient_id_column,
            icd_10_column=icd_10_column,
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
        Build the instruction training data based on encounter history, relative times and negative sampling
        Extract the data and pass the relevant to data to the sampled instruction task. Return a tuple
        that contains the input and label ids for the given instruction task
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

            training_data (str): The train data after applying various functions, transformations
            and sampling
        """

        sampled_positives, sampled_negatives = self.get_instruction_samples(
            patient_id=patient_id,
            current_time=current_time,
            past_time_delta=past_time_delta,
            future_time_delta=future_time_delta,
            use_log_position=use_log_position,
            k_shot=k_shot,
            random_negative_probability=random_negative_probability,
            time_difference_normalize=time_difference_normalize,
        )

        past_start_range_limit = 0
        future_start_range_limit = 0
        past_end_range_limit = (pd.to_timedelta(past_time_delta).days //  time_difference_normalize) + 1
        future_end_range_limit = (pd.to_timedelta(future_time_delta).days //  time_difference_normalize) + 1

        # Get the instructions - this should contain the instruction inputs and instruction targets
        return self.process_status_classification_instructions(
            sampled_positives=sampled_positives,
            sampled_negatives=sampled_negatives,
            shuffle=shuffle,
            past_start_range_limit=past_start_range_limit,
            past_end_range_limit=past_end_range_limit,
            future_start_range_limit=future_start_range_limit,
            future_end_range_limit=future_end_range_limit
        )

    def process_status_classification_instructions(
            self,
            sampled_positives: pd.DataFrame,
            sampled_negatives: pd.DataFrame,
            shuffle,
            past_start_range_limit: int = 0,
            past_end_range_limit: int = 12,
            future_start_range_limit: int = 0,
            future_end_range_limit: int = 12
    ):
        """
        Return the instruction input and targets
        for the sampled instruction icd status classification task
        Args:
            sampled_positives (pd.DataFrame): The dataframe that is used for positive diagnosis
            sampled_negatives (pd.DataFrame): The dataframe that will be used for negative diagnosis
            shuffle (bool, defaults to `True`): Whether to shuffle the list of instructions.
            past_start_range_limit: The start limit for the past position range
            past_end_range_limit: The end limit for the past position range
            future_start_range_limit: The start limit for the future position range
            future_end_range_limit: The end limit for the future position range

        Returns:
            (List[Tuple[str, str]]): A list that contains tuples which have the instruction input and target.
            The list will contain only 1 element for zero shot training
        """
        # Get the task definition string
        task_definition = self._icd_instructions.get_task_definition()

        # Get positive instruction inputs and target
        positive_instructions = self._icd_instructions.get_positive_instructions(
            diagnosis_list=sampled_positives[self._icd_10_column].apply(self._icd_convert.get_converted_code),
            positions_list=sampled_positives[self._position_column],
            past_start_range_limit=past_start_range_limit,
            past_end_range_limit=past_end_range_limit,
            future_start_range_limit=future_start_range_limit,
            future_end_range_limit=future_end_range_limit
        )

        negative_instructions = self._icd_instructions.get_negative_instructions(
            diagnosis_list=sampled_negatives[self._icd_10_column].apply(self._icd_convert.get_converted_code),
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


class ICDStatusRangeClassificationTaskEval(ICDStatusClassificationTask):
    """
    Given data, build a label prompt to evaluate the model
    """

    def __init__(
            self,
            evaluate_attribute: List[Tuple[str, Tuple[int, int], Optional[bool]]],
            encounter_dataframe_process: EncounterDataframeProcess,
            dataframe_sampling: DataFrameSampling,
            icd_instructions: Any,
            icd_convert: ICDConvert,
            negative_icd_sampling: Optional[NegativeICDSampling] = None,
            patient_id_column: str = 'PatientID',
            icd_10_column: str = 'ICD10CD',
            position_column: str = 'position',
            example_attributes: Optional[List[Tuple[str, Tuple[int, int], Optional[bool]]]] = None,
            start_range_limit_column: str = 'start_range_limit',
            end_range_limit_column: str = 'end_range_limit'
    ):
        super().__init__(
            encounter_dataframe_process=encounter_dataframe_process,
            icd_instructions=icd_instructions,
            dataframe_sampling=dataframe_sampling,
            negative_icd_sampling=negative_icd_sampling,
            icd_convert=icd_convert,
            patient_id_column=patient_id_column,
            icd_10_column=icd_10_column,
            position_column=position_column
        )
        self._evaluate_attribute = evaluate_attribute
        self._example_attributes = example_attributes
        self._start_range_limit_column = start_range_limit_column
        self._end_range_limit_column = end_range_limit_column

    def get_encounter_range(self, encounter_history, start_time, end_time, time_period):
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

    def __get_answer(self, encounter_history, icd_code):
        return sum(encounter_history[self._icd_10_column].str.contains(icd_code))

    def get_sample(self, icd_code, start_time, end_time):
        return {
            self._icd_10_column: icd_code,
            self._position_column: -1 if start_time < 0 else 1,
            self._start_range_limit_column: start_time,
            self._end_range_limit_column: end_time
        }

    def get_answer(self, encounter_history, icd_code, start_time, end_time, answer):

        if answer is None:
            # Select the dataframe values within the time range
            encounter_subset = self.get_encounter_range(
                encounter_history=encounter_history,
                start_time=start_time,
                end_time=end_time,
                time_period=-1 if end_time < 0 else 1
            )

            # Find the answer
            answer = self.__get_answer(encounter_history=encounter_subset, icd_code=icd_code)

        return answer

    def process_example_attributes(self, example_attributes, encounter_history):
        # Initialize instructions with task definition
        instructions = list()
        instructions.append(self._icd_instructions.get_task_definition())

        for example_attribute in example_attributes:
            icd_code = example_attribute[0]
            start_time, end_time = example_attribute[1]
            answer = self.get_answer(
                encounter_history=encounter_history,
                icd_code=icd_code,
                start_time=start_time,
                end_time=end_time,
                answer=example_attribute[2]
            )

            if answer:
                instruction = self._icd_instructions.get_positive_instruction(
                    diagnosis=self._icd_convert.get_converted_code(icd_code),
                    position=-1 if end_time < 0 else 1,
                    start_range_limit=start_time,
                    end_range_limit=end_time
                )
            else:
                instruction = self._icd_instructions.get_negative_instruction(
                    diagnosis=self._icd_convert.get_converted_code(icd_code),
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

        # Get encounter history of patient
        encounter_history = self._encounter_dataframe_process.get_patient_sequence(
            patient_id=patient_id,
            current_time=current_time,
            past_time_delta=past_time_delta,
            future_time_delta=future_time_delta,
            use_log_position=use_log_position,
            time_difference_normalize=time_difference_normalize
        )

        # evaluate_attribute = np.random.choice(self._evaluate_attribute)
        evaluate_attribute = self._evaluate_attribute[0]

        if self._example_attributes is None:
            example_attributes = [evaluate_attribute]
        else:
            # The evaluate attribute will be at the end of the prompt
            example_attributes = self._example_attributes + evaluate_attribute

        # Get the instructions - this should contain the instruction inputs and instruction targets
        return self.process_example_attributes(
            example_attributes=example_attributes,
            encounter_history=encounter_history
        )
