"""Make training data"""
import random
import itertools
import numpy as np
import pandas as pd
from typing import Union, Tuple, List, Optional, Sequence

from .processing.code_convert import ICDConvert, PHEConvert
from .processing.encounter_dataframe_process import EncounterDataframeProcess
from .processing.dataframe_sampling import GroupBySampling
from .processing.data_bins import AgglomerativeDataBins
from .processing.negative_code_sampling import NegativeCodeCacheSampling
from .templates import CodeLabelPredictionInstructionTemplate

np.random.seed(42)


class CodeLabelPredictionTask(object):
    """
    Make instruction tuning training data from diagnostic codes
    """

    def __init__(
            self,
            encounter_dataframe_process: EncounterDataframeProcess,
            dataframe_sampling: GroupBySampling,
            code_instructions: CodeLabelPredictionInstructionTemplate,
            code_convert: Union[ICDConvert, PHEConvert],
            time_bins: Union[AgglomerativeDataBins, Sequence[AgglomerativeDataBins]],
            negative_code_sampling: NegativeCodeCacheSampling,
            fixed_position_range: bool,
            patient_id_column: str = 'PatientID',
            code_column: str = 'phecode',
            position_column: str = 'position',
            bins_column: str = 'bins',
            bin_start_column: str = 'min',
            bin_end_column: str = 'max',
            label_column: str = 'label',
            idf_column: str = 'idf',
            tf_column: str = 'tf',
            weights_column: str = 'weights',
            ignore_instruction_column: str = 'ignore_instruction',
            position_range_column: str = 'position_range',
            current_bin_value: int = -100,
            prediction_range_limit: int = None,
    ):
        """
        Initialize variables

        Args:
            encounter_dataframe_process (EncounterDataframeProcess): Dataframe processing object
            dataframe_sampling (GroupBySampling): Dataframe sampling object
            code_instructions (Any): A list of code instruction tasks. At any point one of these
            tasks is randomly chosen and trained on
            code_convert (Union[ICDConvert, PHEConvert]): The object to map codes to other codes in hierarchy and
            their textual descriptions.
            patient_id_column (str, defaults to `PatientID`): The column name for the patient id
            code_column (str, defaults to `phecode`): The column that contains the codes
            position_column (str, defaults to `positions`): The column where position values
            (e.g. time difference in days or log days) will be stored
            bins_column (str, defaults to `bins`): Column that stores the assigned bin
            bin_start_column (str, defaults to `min`): Column that stores the start position of each bin
            bin_end_column (str, defaults to `max`): Column that stores the end position of each bin
            label_column (str, defaults to `count`): The column that stores some value that represents
            the code in a time bin
            fixed_position_range (bool): Use position ranges - 1, 3, 6, 12
            ignore_instruction_column (str, defaults to `ignore_instruction`): Column that stores whether
            to ignore the instruction in that row
        """
        self._encounter_dataframe_process = encounter_dataframe_process
        self._dataframe_sampling = dataframe_sampling
        self._code_instructions = code_instructions
        self._code_convert = code_convert
        self._patient_id_column = patient_id_column
        self._code_column = code_column
        self._position_column = position_column
        self._time_bins = [time_bins] if not isinstance(time_bins, list) else  time_bins
        self._bins_column = bins_column
        self._bin_start_column = bin_start_column
        self._bin_end_column = bin_end_column
        self._label_column = label_column
        self._ignore_instruction_column = ignore_instruction_column
        self._negative_code_sampling = negative_code_sampling
        self._position_range_column = position_range_column
        self._current_bin_value = current_bin_value
        self._idf_column = idf_column
        self._tf_column = tf_column
        self._weights_column = weights_column

        if fixed_position_range:
            # self._position_ranges = [
            #     (-1, 30), (30, 90), (90, 180),
            #     (180, 360), (360, np.inf)
            # ]
            self._position_ranges = [
                (-7, 180), (180, np.inf)
            ]
        else:
            self._position_ranges = [
                (-1, 30), (30, 60), (60, 90), (90, 120), (120, 150), (150, 180),
                (180, 210), (210, 240), (240, 270), (270, 300), (300, 330),
                (330, 360), (360, np.inf)
            ]
        # We mainly use this to exclude any codes or encounters occurring in the (360, inf) time period
        if prediction_range_limit is None:
            self._prediction_range_limit = len(self._position_ranges)
        else:
            self._prediction_range_limit = prediction_range_limit

    def process_sample(self, sample, args):
        """
        Process sample

        Args:
            sample:
            args:

        Returns:

        """

        # Get the full encounter history
        encounter_history = self.get_full_encounter_history(
            patient_id=sample[args.patient_id_column],
            current_time=sample[args.sample_result_date_column],
            use_log_position=args.use_log_position,
            time_difference_normalize=args.time_difference_normalize
        )

        # If for some reason we don't have encounter data for this person - we ignore training
        # on this person
        if self._encounter_dataframe_process.check_patient_id(patient_id=sample[args.patient_id_column]):
            ignore_instruction = False
        else:
            ignore_instruction = True

        # Filter encounters by time if the time delta values are not None
        # Compute which time bins the encounters belong too - based on some clustering
        # Also compute the position range buckets they fall into
        encounter_history = self.transform_encounter_history_for_task(
            encounter_history=encounter_history,
            past_time_delta=args.past_time_delta,
            future_time_delta=args.future_time_delta,
        )

        # Get the bin period we will be making predictions for.
        # The list can contain multiple elements - we can make prediction for
        # multiple time bins
        prediction_ranges = self.get_prediction_ranges(
            future_only=args.future_only,
            number_of_ranges=self.sample_from_list(args.number_of_instructions),
            fixed_range=args.time_period_range
        )

        # Store all the prompt elements (input, output)
        all_instructions = list()

        for prediction_range in prediction_ranges:

            k_shot = self.sample_from_list(args.k_shot)

            # Get the task instruction - which should indicate we are making prediction for a given
            # time range
            all_instructions.append(self.get_task_instruction(prediction_range=prediction_range))

            # Get dataframe that contain encounter in the current time period
            current_time_period = self.get_current_time_period(
                encounter_history=encounter_history, prediction_range=prediction_range
            )

            # Sample counts for negatives and positives
            # K shot will give how many samples we want for a given prediction range
            sample_counts = self.get_sample_sizes(
                max_size=k_shot,
                sample_limits=[k_shot, current_time_period[self._code_column].nunique()],
            )

            exclude_codes_for_negatives = self.get_exclude_codes_for_negatives(
                encounter_history=encounter_history,
                prediction_range=prediction_range,
                time_negatives_buffer=args.time_negatives_buffer
            )

            # Get a dataframe that can be used to sample negatives
            # We exclude only the codes that are part of the current time period. A code
            # that occurs outside this period can be used as a negative - we are essentially
            # saying does this code occur in the period or not
            encounter_negatives = self._negative_code_sampling.get_encounter_negatives_from_cache_process_and_update(
                patient_id=sample[args.patient_id_column],
                current_time=sample[args.sample_result_date_column],
                prediction_range=prediction_range,
                use_log_position=args.use_log_position,
                time_difference_normalize=args.time_difference_normalize,
                exclude_codes=exclude_codes_for_negatives,
                minimum_negatives_size=sample_counts[0],
                code_column=self._code_column,
                position_column=self._position_column
            )

            # We just keep it uniform with how we processed
            # the encounter history for the patient. We add position ranges
            # and time bins to the negatives as well.
            encounter_negatives = self.transform_encounter_history_for_task(
                encounter_history=encounter_negatives,
                past_time_delta=args.past_time_delta,
                future_time_delta=args.future_time_delta,
            )

            # Imagine adding this sample (ecg or blood) to another patients encounter history
            # We then are basically comparing - what are the codes that occurred in the actual history
            # in the given time range - and what are the codes that occurred in this other person history
            # and saying the codes that occurred for the other person that didn't occur for the current person
            # are negatives
            negatives = self.get_current_time_period(
                encounter_history=encounter_negatives, prediction_range=prediction_range
            )
            if len(negatives) < sample_counts[0]:
                negatives = encounter_negatives

            histories = [negatives, current_time_period]

            # Now that we have sizes - we move on to sampling the codes
            # We try and sample such that we get codes spread across the time range
            # and not just concentrated in one portion of the time range - for both
            # positives and negatives
            sampled_codes = [
                self.sample_codes(encounter_history=encounter_history, max_limit=counts)
                for encounter_history, counts in zip(histories, sample_counts)
            ]

            label_samples = [
                self.prepare_sampled_codes(
                    codes=codes,
                    label_string=self.get_positive_labels_string() if label == 1 else self.get_negative_labels_string(),
                    prediction_range=prediction_range,
                    ignore_instruction=ignore_instruction,
                )
                for label, codes in enumerate(sampled_codes)
            ]

            # Combine the positive and negative samples
            instruction_samples = pd.concat(label_samples).sample(frac=1)

            # Convert samples to text instructions (prompt)
            all_instructions.extend(
                self.convert_samples_to_instructions(
                    instruction_samples=instruction_samples,
                )
            )

        return all_instructions

    def get_full_encounter_history(
            self,
            patient_id: Union[str, int],
            current_time: str,
            use_log_position: bool,
            time_difference_normalize: int,
    ) -> pd.DataFrame:
        """
        Return the encounter history of a patient - if the patient does not have an encounter history
        return a random history (temporary fix)

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
        encounter_history = self._encounter_dataframe_process.get_patient_encounter_history_with_position(
            patient_id=patient_id,
            current_time=current_time,
            use_log_position=use_log_position,
            time_difference_normalize=time_difference_normalize
        )
        encounter_history.dropna(subset=self._code_column, inplace=True)
        return encounter_history

    def transform_encounter_history_for_task(
            self,
            encounter_history: pd.DataFrame,
            past_time_delta: Optional[str] = None,
            future_time_delta: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Filter encounters by time if the time delta values are not None
        Compute which time bins the encounters belong too - based on some clustering
        Also compute the position range buckets they fall into.

        Args:
            encounter_history (pd.DataFrame): The full encounter history of the patient
            past_time_delta (str, defaults to `None`): Filter out encounters beyond this point in time
            future_time_delta (str, defaults to `None`): Filter out encounters beyond this point in time

        Returns:
            encounter_history (pd.DataFrame): The processed dataframe for the task
        """

        # Time filter
        if past_time_delta is not None or future_time_delta is not None:
            encounter_history = self.filter_time_encounter_dataframe(
                encounter_history=encounter_history,
                past_time_delta=past_time_delta,
                future_time_delta=future_time_delta,
            )

        if len(encounter_history) == 0:
            # FIXME: Raise an error if the number of encounters is 0 - ideally we should not have any
            #  empty encounters - or use this only for negatives
            encounter_history = self.get_random_encounters()

        encounter_history[self._bins_column] = self.get_bins(
            bin_value_column=encounter_history[self._position_column]
        )

        encounter_history[self._position_range_column] = self.get_position_range(
            position_values=encounter_history[self._position_column]
        )

        encounter_history[self._code_column] = encounter_history[self._code_column].astype(str)

        return encounter_history

    def get_prediction_ranges(
            self,
            future_only: bool,
            number_of_ranges: int,
            fixed_range: Tuple[int, int] = None
    ) -> List[Tuple[int, int]]:
        """
        Returns a list of tuples that contain the start and end times of the bin
        we will be using as the current bin period - this is the period we
        will make predictions for.

        Args:
            fixed_range (Tuple[int, int], defaults to `None`): Return the fixed range
            future_only (bool): Use ranges occurring only in the future
            number_of_ranges (int): The number of prediction ranges to sample

        Returns:
            (List[Tuple[int, int]]): The start and end position of the current bin period
        """

        # If we are using a fixed range - just return the range as is
        if fixed_range is not None:
            return [fixed_range]

        prediction_ranges = list()

        past = np.arange(-self._prediction_range_limit + 1, 0)
        future = np.arange(1, self._prediction_range_limit)
        if future_only:
            position_ranges = future
        else:
            position_ranges = np.append(past, future)

        sampled_position_ranges = np.random.choice(
            position_ranges,
            size=min(len(position_ranges), number_of_ranges),
            replace=False
        )

        # Convert the position range to an actual range tuple that consists of start and end time
        for sampled_position_range in sampled_position_ranges:
            range_period = self.get_range_period(
                position_range=sampled_position_range
            )
            prediction_ranges.append(range_period)

        return prediction_ranges

    def get_current_time_period(
            self, encounter_history: pd.DataFrame, prediction_range: Tuple[int, int]
    ) -> pd.DataFrame:
        """
        Return encounters within the current time range

        Args:
            encounter_history:
            prediction_range:

        Returns:

        """
        if prediction_range[1] <= 0:
            position_range_filter = (encounter_history[self._position_range_column] < 0)
        else:
            position_range_filter = (encounter_history[self._position_range_column] > 0)

        current_time_period = encounter_history[
            (encounter_history[self._position_column] >= prediction_range[0]) &
            (encounter_history[self._position_column] <= prediction_range[1]) &
            position_range_filter
            ]
        bins = current_time_period[self._bins_column].unique()
        return encounter_history[encounter_history[self._bins_column].isin(bins)]

    def get_exclude_codes_for_negatives(
            self,
            encounter_history: pd.DataFrame,
            prediction_range: Tuple[int, int],
            time_negatives_buffer: bool
    ) -> pd.Series:
        """
        Return a list of codes that should not be sampled when sampling negatives

        Args:
            encounter_history (pd.Dataframe): The encounter history of the patient
            prediction_range (Tuple[int, int]): The current prediction range
            time_negatives_buffer (bool): Make negatives stricter - a code is a negative if it does not occur in the
            prediction range

        Returns:
            (pd.Series): List of codes that should not be used for negative sampling
        """
        if time_negatives_buffer is not None:
            buffered_range = prediction_range[0] - time_negatives_buffer, prediction_range[1] + time_negatives_buffer
            buffered_time_period = self.get_current_time_period(
                encounter_history=encounter_history, prediction_range=buffered_range
            )
            bins = buffered_time_period[self._bins_column].unique()
            # In case there is any code that is i not in the current time period but is part of the bin that is in the
            # current time period - basically if the bin spans over two months - don't sample codes from this
            # bin as negatives
            return encounter_history[encounter_history[self._bins_column].isin(bins)][self._code_column]
        else:
            return encounter_history[self._code_column]

    def get_sample_sizes(
            self,
            max_size: int,
            sample_limits: List[int],
    ) -> List[int]:
        """
        For a given k-shot problem, return how many positive samples
        and negative samples should be present in the instruction

        Args:
            max_size (int): The maximum number of positives and negatives
            sample_limits (int): The total number of encounters. We can't sample more than this for
            say the positive or negatives

        Returns:
            (List[int]): The number of positive and negative examples
        """
        # Positive and negative have equal probability
        probabilities = [0.5, 0.5]
        indexes = [0, 1]
        counts = [0, 0]

        while max_size and sum(sample_limits) > 0:
            # If we exceed the sample limit - re-normalize the probability - so that we only sample
            # from the portion that has data
            probabilities = self.__normalize_probability(probabilities=probabilities, limits=sample_limits)
            index = np.random.choice(indexes, p=probabilities)
            counts[index] += 1
            sample_limits[index] -= 1
            max_size -= 1

        return counts

    def sample_codes(self, encounter_history, max_limit):
        """
        Sample codes from the dataframe - sample such that we have good
        representation from different time ranges - for example

        Args:
            encounter_history:
            max_limit:

        Returns:

        """
        tf_idf = self.get_tf_idf(encounter_history=encounter_history)
        return (
            tf_idf
            .sample(n=max_limit, weights=tf_idf[self._weights_column] if not tf_idf.empty else None)
        )

    def get_tf_idf(self, encounter_history):
        """
        Get tf idf scores

        Args:
            encounter_history:

        Returns:

        """
        tf_idf = (
            encounter_history
            .groupby(self._code_column)
            .agg({self._code_column: 'count', self._idf_column: 'first'})
            .rename(columns={self._code_column: 'count'})
            .reset_index()
        )
        tf_idf[self._weights_column] = tf_idf['count'] * tf_idf[self._idf_column]
        return tf_idf

    def prepare_sampled_codes(
            self, codes, label_string, prediction_range, ignore_instruction
    ):
        """
        Define the start and end times and the label string for the sampled codes

        Args:
            codes:
            label_string:
            prediction_range:
            ignore_instruction:

        Returns:

        """
        codes[self._label_column] = label_string
        codes[self._bin_start_column] = prediction_range[0]
        codes[self._bin_end_column] = prediction_range[1]
        codes[self._ignore_instruction_column] = ignore_instruction
        return codes

    def convert_samples_to_instructions(
            self,
            instruction_samples: pd.DataFrame,
    ) -> List[Tuple[str, str, bool]]:
        """
        Return the instruction input and targets
        for the sampled instruction code status classification task with time range

        Args:
            instruction_samples (pd.DataFrame): The samples that will be used as instructions

        Returns:
            (List[Tuple[str, str, bool]]): A list that contains tuples which have the instruction input and target.
            The list will contain only 1 element for zero shot training
        """

        instructions = list()
        for row in instruction_samples[
            [self._code_column, self._label_column, self._ignore_instruction_column]
        ].itertuples(index=False):
            code, label, ignore_instruction = row[0], row[1], row[2]
            code_instruct_string = self._code_instructions.get_instruction(
                diagnosis=self._code_convert.transform_code(code=code),
                value=label
            )
            instructions.append(code_instruct_string + (ignore_instruction,))
        return instructions

    def get_random_encounters(self) -> pd.DataFrame:
        """
        Use this function to create a random dataframe populated
        with codes and positions

        Args:

        Returns:
            (pd.DataFrame): A randomly generated encounter dataframe
        """
        # Use these to create and return a dataframe
        return pd.DataFrame(
            {self._code_column: [], self._position_column: [], self._idf_column: []}
        )

    def filter_time_encounter_dataframe(
            self,
            encounter_history: pd.DataFrame,
            past_time_delta: str,
            future_time_delta: str
    ):
        """
        Given the encounter history apply a time
        filter if passed and return the resulting dataframe

        Args:
            encounter_history (pd.DataFrame): The full encounter history of the patient
            past_time_delta (str): Filter out encounters beyond this point in time
            future_time_delta (str): Filter out encounters beyond this point in time

        Returns:
            encounter_history (pd.DataFrame): The processed dataframe for the task
        """
        return self._encounter_dataframe_process.filter_encounter_history_time_delta(
            encounter_history=encounter_history,
            past_time_delta=past_time_delta,
            future_time_delta=future_time_delta,
        )

    def get_bins(self, bin_value_column: pd.Series) -> Union[pd.Series, int]:
        """
        Given the input dataframe - return the dataframe with each row
        put into specific bins

        Args:
            bin_value_column (pd.DataFrame): The column to use to calculate the bins

        Returns:
            (Union[int, pd.Series]): Computed bins
        """
        if len(bin_value_column) <= 1:
            return 0
        time_bins_object = np.random.choice(self._time_bins)
        return time_bins_object.get_bins(bin_value_column)

    def get_position_range(self, position_values: pd.Series) -> pd.Series:
        """
        Given a position - assign  it a value - depending on which position range it
        belongs to

        Args:
            position_values (pd.Series): The position values

        Returns:
            index (int): A value that assign the position to a defined position range
        """
        return position_values.apply(
            self.__assign_position_range
        )

    def __assign_position_range(self, position: Union[int, float]) -> int:
        """
        Given a position - assign  it a value - depending on which position range it
        belongs to

        Args:
            position (Union[int, float]): A position value

        Returns:
            index (int): A value that assign the position to a defined position range
        """
        # Add 1 to the index - to avoid returning a zero
        for index, position_range in enumerate(self._position_ranges):
            if position_range[0] < abs(position) <= position_range[1]:
                if position < 0:
                    return -index - 1
                else:
                    return index + 1

    def get_ranges_for_prediction(self, position_ranges: pd.Series, future_only: bool) -> np.array:
        """
        Given the position range values - return the position ranges
        which can be for sampling and training

        Args:
            position_ranges (pd.Series): A series containing the position ranges
            future_only (bool): Return ranges occurring only in the future

        Returns:
            (np.array): An array of unique position ranges
        """
        # Get all the position ranges that encounters fall into for this person
        ranges = position_ranges.unique()
        # Return the ranges within a 1-year period - ranges outside this are considered too far out
        if future_only:
            # Return ranges only in the future
            return ranges[(np.abs(ranges) != self._prediction_range_limit) & (ranges >= 1)]
        else:
            # Return past and future
            return ranges[np.abs(ranges) != self._prediction_range_limit]

    @staticmethod
    def create_range_period(sampled_range_period: Tuple[int, int], future_only: bool) -> Tuple[int, int]:
        """
        Give a range - create a range period - that starts at 0 and ends at
        the given range or starts at the given range and ends at 0

        Args:
            sampled_range_period (Tuple[int, int]): The start and end period of a month
            future_only (bool): Whether the returned prediction range should only be in the future

        Returns:
            (Tuple[int, int]): The prediction range
        """
        if np.random.rand() <= 0.5 or future_only:
            return 0, sampled_range_period[1]
        else:
            return -sampled_range_period[1], 0

    def get_range_period(self, position_range: int) -> Tuple[int, int]:
        """
        Give a position range - create a range period - that starts at 0 and ends at
        the given range or starts at the given range and ends at 0

        Args:
            position_range (Tuple[int, int]): The start and end period of a month

        Returns:
            (Tuple[int, int]): The prediction range
        """
        range_period = self._position_ranges[np.abs(position_range) - 1]
        # Past range
        if position_range < 0:
            return -range_period[1], 0
        # Future range
        else:
            return 0, range_period[1]

    def __normalize_probability(self, probabilities, limits):
        """
        Give probabilities - re normalize them if any of the limits reaches zero

        Args:
            probabilities:
            limits:

        Returns:

        """
        weights = [int(sample_limit >= 1) for sample_limit in limits]
        probabilities = self.__get_probability(
            np.array(
                [probability * weight for probability, weight in
                 zip(probabilities, weights)])
        )
        return probabilities

    @staticmethod
    def __get_probability(weights):
        """
        Given a set of weights return the probability

        Args:
            weights:

        Returns:

        """
        return weights / weights.sum()

    @staticmethod
    def sample_from_list(shots):
        """
        Sample an element from a list

        Args:
            shots:

        Returns:

        """
        return np.random.choice(shots)

    @staticmethod
    def process_time_period(start_time, end_time):
        """
        Given a start time and end time in days - return the time range
        in a single month format - to be used in the string prompts

        Args:
            start_time:
            end_time:

        Returns:

        """
        if start_time < 0:
            return int(np.floor(start_time / 30))
        elif end_time > 0:
            return int(np.ceil(end_time / 30))
        else:
            print(start_time, end_time)
            raise ValueError('Invalid start and end time')

    @staticmethod
    def get_positive_labels_string():
        """
        Return the string used to denote the positive label

        Returns:

        """
        return 'yes'

    @staticmethod
    def get_negative_labels_string():
        """
        Return the string used to denote the negative label

        Returns:

        """
        return 'no'

    def get_task_instruction(self, prediction_range):
        """
        Return a task instruction based on the task definition and prediction range

        Args:
            prediction_range:

        Returns:

        """
        time = self.process_time_period(start_time=prediction_range[0], end_time=prediction_range[1])
        task_definition = self._code_instructions.get_task_definition(time=time)
        return task_definition, '', True






class HierarchicalCodeLabelPredictionTask(CodeLabelPredictionTask):
    """
    Make instruction tuning training data from diagnostic codes
    """

    def __init__(
            self,
            encounter_dataframe_process: EncounterDataframeProcess,
            dataframe_sampling: GroupBySampling,
            code_instructions: CodeLabelPredictionInstructionTemplate,
            code_convert: Union[ICDConvert, PHEConvert],
            time_bins: Union[AgglomerativeDataBins, Sequence[AgglomerativeDataBins]],
            negative_code_sampling: NegativeCodeCacheSampling,
            fixed_position_range: bool,
            patient_id_column: str = 'PatientID',
            code_column: str = 'phecode',
            position_column: str = 'position',
            bins_column: str = 'bins',
            bin_start_column: str = 'min',
            bin_end_column: str = 'max',
            label_column: str = 'label',
            ignore_instruction_column: str = 'ignore_instruction',
            position_range_column: str = 'position_range',
            current_bin_value: int = -100,
            prediction_range_limit: int = None,
            sample: bool = True
    ):
        """
        Initialize variables

        Args:
            encounter_dataframe_process (EncounterDataframeProcess): Dataframe processing object
            dataframe_sampling (GroupBySampling): Dataframe sampling object
            code_instructions (Any): A list of code instruction tasks. At any point one of these
            tasks is randomly chosen and trained on
            code_convert (Union[ICDConvert, PHEConvert]): The object to map codes to other codes in hierarchy and
            their textual descriptions.
            patient_id_column (str, defaults to `PatientID`): The column name for the patient id
            code_column (str, defaults to `phecode`): The column that contains the codes
            position_column (str, defaults to `positions`): The column where position values
            (e.g. time difference in days or log days) will be stored
            bins_column (str, defaults to `bins`): Column that stores the assigned bin
            bin_start_column (str, defaults to `min`): Column that stores the start position of each bin
            bin_end_column (str, defaults to `max`): Column that stores the end position of each bin
            label_column (str, defaults to `count`): The column that stores some value that represents
            the code in a time bin
            ignore_instruction_column (str, defaults to `ignore_instruction`): Column that stores whether
            to ignore the instruction in that row
        """
        super().__init__(
            encounter_dataframe_process=encounter_dataframe_process,
            dataframe_sampling=dataframe_sampling,
            code_instructions=code_instructions,
            code_convert=code_convert,
            time_bins=time_bins,
            negative_code_sampling=negative_code_sampling,
            patient_id_column=patient_id_column,
            code_column=code_column,
            position_column=position_column,
            bins_column=bins_column,
            bin_start_column=bin_start_column,
            bin_end_column=bin_end_column,
            label_column=label_column,
            ignore_instruction_column=ignore_instruction_column,
            position_range_column=position_range_column,
            current_bin_value=current_bin_value,
            prediction_range_limit=prediction_range_limit,
            fixed_position_range=fixed_position_range
        )
        self._trie = negative_code_sampling.get_trie()
        self._tree = negative_code_sampling.get_tree()
        self._sample = sample

    def process_sample(self, sample, args):
        """
        Process sample

        Args:
            sample:
            args:

        Returns:

        """

        # Get the full encounter history
        encounter_history = self.get_full_encounter_history(
            patient_id=sample[args.patient_id_column],
            current_time=sample[args.sample_result_date_column],
            use_log_position=args.use_log_position,
            time_difference_normalize=args.time_difference_normalize
        )

        # If for some reason we don't have encounter data for this person - we ignore training
        # on this person
        if self._encounter_dataframe_process.check_patient_id(patient_id=sample[args.patient_id_column]):
            ignore_instruction = False
        else:
            ignore_instruction = True

        encounter_history = self.transform_encounter_history_for_task(
            encounter_history=encounter_history,
            past_time_delta=args.past_time_delta,
            future_time_delta=args.future_time_delta,
        )

        #  Get the bin period we will be making predictions for
        prediction_ranges = self.get_prediction_ranges(
            future_only=args.future_only,
            number_of_ranges=self.sample_from_list(args.number_of_instructions),
            fixed_range=args.time_period_range
        )

        all_instructions = list()

        for prediction_range in prediction_ranges:

            # Get the task instruction - which should indicate we are making prediction for a given
            # time range
            all_instructions.append(self.get_task_instruction(prediction_range=prediction_range))

            # Get dataframe that contain encounter in the current time period
            current_time_period = self.get_current_time_period(
                encounter_history=encounter_history, prediction_range=prediction_range
            )
            # Get only the unique codes - keep only the earliest occurrences of each code basically
            # FIXME
            # current_time_period_unique = self.get_unique_codes(current_time_period)
            current_time_period_unique = current_time_period

            # Positive hash should contain a set of codes. It should contain all the ancestors for a given
            # code - but not their siblings. The ignore_hash is used when the note is labeled with a non leaf
            # code - we want to tell the prompt to not use the leaf codes - because we don't know which leaf code
            # as positive. If there is a code in ignore hash but not in positive hash - don't add that code to the
            # prompt
            positive_hash = self.get_positives_hash(codes=current_time_period_unique[self._code_column])
            ignore_hash = self.get_ignore_hash(codes=current_time_period_unique[self._code_column])

            # Define how to sample blocks, shuffle them, drop them and also how to do these
            # within blocks
            sample_in_level_strategy = self.get_sample_in_level_strategy()
            shuffle_probability = self.get_shuffle_probability(sample_in_level_strategy)
            drop_probability = self.get_drop_probability()
            consistent_sample = self.get_consistent_sample(sample_in_level_strategy=sample_in_level_strategy)
            random_sample = self.get_random_sample(sample_in_level_strategy=sample_in_level_strategy)
            # This will only be used when sample strategy is either random or consistent. This won't have
            # any effect if we are not sampling
            sample_size_limit = self.get_sample_size_limit()

            # Get the instruction level blocks
            # Each block is a list of codes, where each code has a positive or negative label
            instructions_at_levels = list(
                self.get_hierarchy(
                    positive_hash=positive_hash,
                    ignore_hash=ignore_hash,
                    consistent_sample=consistent_sample,
                    sample_size_limit=sample_size_limit
                )
            )
            # Shuffle the blocks - based on probability
            instructions_at_levels = self.shuffle_levels(instructions_at_levels, shuffle_probability=shuffle_probability)
            # Drop some blocks - based on probability
            instructions_at_levels = self.drop_levels(instructions_at_levels, drop_probability=drop_probability)

            # Go through each block and the codes that it contains - maybe shuffle them randomly
            # random shuffle only happens if strategy is 'random'
            for level, positive_codes_at_level, negative_codes_at_level in instructions_at_levels:

                all_instructions.append(self.get_level_instruction(level=level))

                positive_codes_at_level, negative_codes_at_level = self.random_sample_in_level(
                    positive_codes_at_level=positive_codes_at_level,
                    negative_codes_at_level=negative_codes_at_level,
                    random_sample=random_sample,
                    sample_size_limit=sample_size_limit
                )

                instruction_samples = self.prepare_hierarchy_codes(
                    positive_codes_at_level=positive_codes_at_level,
                    negative_codes_at_level=negative_codes_at_level,
                    prediction_range=prediction_range,
                    ignore_instruction=ignore_instruction
                )

                all_instructions.extend(
                    self.convert_samples_to_instructions(
                        instruction_samples=instruction_samples,
                    )
                )

        return all_instructions

    def get_positives_hash(self, codes):
        """
        Return a set that contains the codes that contains the codes passed in
        the arguments and its prefix (i.e. all parents until root). This basically
        indicates that the codes and their ancestors are all positives - but the
        siblings aren't automatically considered a positive

        Args:
            codes:

        Returns:

        """
        positive_hash = set()
        for code in codes:
            code_key = '/'.join(self._negative_code_sampling.get_code_representation(code))
            prefixes = list(self._trie.prefixes(code_key))
            for key, value in prefixes:
                positive_hash.add(value)
        return positive_hash

    def get_ignore_hash(self, codes):
        """
        This will contain all the leaf nodes for a given code. If for some reason we have
        a label for a parent code, but not for the leaf code - we don't want to include
        the leaf node in prediction. Because we don't know its label for sure (when they
        have siblings). If a code is present in positives hash and ignore hash - we have a
        label. If a code is present in ignore hash but not it in positive hash - we are
        unsure of its label. If a code is not present in either - then it is a negative

        Args:
            codes:

        Returns:

        """
        ignore_hash = dict()
        for code in codes:
            code_key = '/'.join(self._negative_code_sampling.get_code_representation(code))
            suffixes = list(self._trie[code_key:])
            for value in suffixes[1:]:
                ignore_hash[value] = 1
        return ignore_hash

    def get_hierarchy(self, positive_hash, ignore_hash, consistent_sample, sample_size_limit):
        """
        Go through the list of positive codes and build a tree structure that represents the
        ancestors and siblings of these codes and also assign labels to them. For example
        if we have CV404.1, we would add CV, CV_404 to the tree and assign positive labels to them
        while we assign negative labels to siblings to DE ..., CV_405, CV_406 ..., CV_404.2
        Args:
            positive_hash:
            ignore_hash:
            consistent_sample:
            sample_size_limit:

        Returns:

        """
        level = 0
        parent_codes = None
        parents_without_negatives_probability = np.random.rand()
        while True:
            # Starting from the root to the leaf level - retrieve the positive and negative
            # codes at each level. The hashes already have information about the codes at all levels
            positive_codes_at_level, negative_codes_at_level = self.get_codes_at_level(
                parent_codes=parent_codes, positive_hash=positive_hash, ignore_hash=ignore_hash
            )

            # If there are no positive codes - we've mostly likely reached the leaf, and we can now
            # break out of the loop
            if not positive_codes_at_level and not negative_codes_at_level:
                break

            # Sample codes within the level
            positive_codes_at_level, negative_codes_at_level = self.consistent_sample_in_level(
                positive_codes_at_level=positive_codes_at_level,
                negative_codes_at_level=negative_codes_at_level,
                consistent_sample=consistent_sample,
                sample_size_limit=sample_size_limit
            )


            # When we retrieve codes for the next level - we only retrieve the children of the
            # codes that were positive in this level - the loop in the get_codes_at_level function
            # will start from this list of codes
            if parents_without_negatives_probability < 0.2:
                parent_codes = positive_codes_at_level
            else:
                parent_codes = self.get_parent_codes_for_next_level_with_negatives(
                    positive_codes_at_level=positive_codes_at_level,
                    negative_codes_at_level=negative_codes_at_level,
                    level=level
                )

            yield (
                level,
                positive_codes_at_level,
                negative_codes_at_level,
            )
            level += 1

    @staticmethod
    def shuffle_levels(instructions_at_levels, shuffle_probability):
        """
        Shuffle the instruction level blocks

        Args:
            instructions_at_levels:
            shuffle_probability:

        Returns:

        """
        if np.random.rand() < shuffle_probability:
            random.shuffle(instructions_at_levels)
        return instructions_at_levels

    def drop_levels(self, instructions_at_levels, drop_probability):
        """
        Drop some instruction level blocks.

        Args:
            instructions_at_levels:
            drop_probability:

        Returns:

        """
        number_of_levels = len(instructions_at_levels)
        if np.random.rand() < drop_probability and number_of_levels > 1:
            keep_choice = self.get_sample_size(total_number_of_samples=number_of_levels - 2) + 1
            indexes = random.sample(range(number_of_levels), k=keep_choice)
            # Keep the order of the input list the same
            indexes.sort()
            instructions_at_levels = [instructions_at_levels[i] for i in indexes]
        return instructions_at_levels

    def get_codes_at_level(self, parent_codes, positive_hash, ignore_hash):
        """
        Create a list of positive and negative codes and use the tree structure of
        the codes to add siblings and ancestors to the respective lists.

        Args:
            parent_codes:
            positive_hash:
            ignore_hash:

        Returns:

        """
        positives = list()
        negatives = list()
        # This is the case when we get root codes like CV, DE, GU ...
        # They all originate from this pseudo root node
        if parent_codes is None:
            parent_codes = ['root']
        # For every parent code on the same level in the tree
        for parent_code in parent_codes:
            # Get its children
            for code_at_level in self._tree.children(parent_code):
                code_tag = code_at_level.tag
                # Check if this code has occurred in the patient history - the hash should have this information
                label = self.get_label(code=code_tag, positive_hash=positive_hash, ignore_hash=ignore_hash)
                # Assign to a list accordingly
                if label == 1:
                    positives.append(code_tag)
                elif label == 0:
                    negatives.append(code_tag)

        return positives, negatives

    def consistent_sample_in_level(
            self,
            positive_codes_at_level,
            negative_codes_at_level,
            consistent_sample,
            sample_size_limit
    ):
        """
        Given a set of positive and negative codes at a level in the tree
        Return a subset of these codes (via sampling) or don't sample

        Args:
            positive_codes_at_level:
            negative_codes_at_level:
            consistent_sample:
            sample_size_limit:

        Returns:

        """
        if consistent_sample:
            # There will be cases where there are a lot more say CV codes than EM codes
            # and any sampling shown below will reflect this. This might induce a bias where
            # our training data sees more of codes that tend to be split into more categories
            # to avoid this - we use this function.
            # The function will try and return equal number of codes from the CV and EM splits
            # and try and avoid the distribution mismatch. This is done by sampling equal number
            # of children from each parent. We get the parents of these codes and sample
            # equal number from each parent
            positive_codes_at_level, negative_codes_at_level = self.prepare_codes_at_level_for_sampling(
                positive_codes_at_level=positive_codes_at_level,
                negative_codes_at_level=negative_codes_at_level
            )
            # Get how many samples will be there - compute sample size
            positive_sample_size, negative_sample_size = self.get_positive_negative_sample_size(
                number_of_positive_samples=len(positive_codes_at_level),
                number_of_negative_samples=len(negative_codes_at_level),
                sample_size_limit=sample_size_limit
            )
            # Ensure that we sample at least one positive - if we don't decrease negative by 1 and increase
            # positive by 1
            if positive_sample_size == 0 and len(positive_codes_at_level):
                positive_sample_size += 1
                if sample_size_limit:
                    negative_sample_size -= 1
            return (
                random.sample(positive_codes_at_level, k=positive_sample_size),
                random.sample(negative_codes_at_level, k=negative_sample_size)
            )
        else:
            return positive_codes_at_level, negative_codes_at_level

    def random_sample_in_level(
            self,
            positive_codes_at_level,
            negative_codes_at_level,
            random_sample,
            sample_size_limit
    ):
        """
        Randomly sample codes within a level. We sample from the positive codes
        and the negative codes independently

        Args:
            positive_codes_at_level:
            negative_codes_at_level:
            random_sample:
            sample_size_limit:

        Returns:

        """
        if random_sample:
            # There will be cases where there are a lot more say CV codes than EM codes
            # and any sampling shown below will reflect this. This might induce a bias where
            # our training data sees more of codes that tend to be split into more categories
            # to avoid this - we use this function.
            # The function will try and return equal number of codes from the CV and EM splits
            # and try and avoid the distribution mismatch. This is done by sampling equal number
            # of children from each parent. We get the parents of these codes and sample
            # equal number from each parent
            positive_codes_at_level, negative_codes_at_level = self.prepare_codes_at_level_for_sampling(
                positive_codes_at_level=positive_codes_at_level,
                negative_codes_at_level=negative_codes_at_level
            )
            positive_sample_size, negative_sample_size = self.get_positive_negative_sample_size(
                number_of_positive_samples=len(positive_codes_at_level),
                number_of_negative_samples=len(negative_codes_at_level),
                sample_size_limit=sample_size_limit
            )
            return (
                random.sample(positive_codes_at_level, k=positive_sample_size),
                random.sample(negative_codes_at_level, k=negative_sample_size)
            )
        else:
            return positive_codes_at_level, negative_codes_at_level

    def get_parent_codes_for_next_level_with_negatives(
            self, positive_codes_at_level, negative_codes_at_level, level
    ):
        """
        Get the codes we are going to use to get codes for the next level.
        While it makes sense to use the positives from this level and get the
        codes for the next level. There is a scenario where we would want to
        propagate some negatives as well. The reason being - say we have a codee
        EM_202.22 at the leaf - if we return only positives we know that EM, EM202
        and EM_202.2 are all positive (which is how we end up here via the tree).
        Since we use block level shuffling etc., and would also like to make yes/no
        predictions on any leaf node - we want the model to learn predictions at the
        leaf level for codes that were negative for their parent.

        Case 1:
            - Training:
            CV: Yes, DE: No, GU: No
            CV_403: Yes, CV_404: no
            - Eval time:
            EM_209 - The model would assume that EM was seen in an earlier level

        But we don't want that - hence we propagate negatives across levels
        So, that when we test a leaf code directly the model is not making the assumption
        that its parent was seen earlier in the hierarchy - and we also get training
        examples of leaf codes without their ancestors being observed

        Args:
            positive_codes_at_level:
            negative_codes_at_level:
            level:

        Returns:

        """
        # We do the same as  we did for code sampling -  check prepare_codes_at_level_for_sampling
        # function for more information
        if not negative_codes_at_level:
            return positive_codes_at_level

        negative_codes_at_level = self.get_codes_at_level_for_sampling(
            codes_at_level=negative_codes_at_level, probability=1
        )

        # The first list will contain codes that are negative but their parent is a
        # positive -  which could mean that their sibling is a positive
        # The second list will contain codes that are negative and their root code
        # has never been seen.
        # Pos - CV, DE, GU
        # Neg 1: Can have CV_403, DE_509, DE_708, GU_33
        # Neg 2: Will have ID_001, NS_238
        negative_with_root, negative_without_root = self.split_negative_codes_at_level(
            positive_codes_at_level=positive_codes_at_level,
            negative_codes_at_level=negative_codes_at_level
        )

        minimum_samples = max(
            1, np.min([len(positive_codes_at_level), len(negative_with_root), len(negative_without_root)])
        )
        bias = level * 2 + 1

        # We can sample at most to this limit.
        sample_size = self.get_sample_size(total_number_of_samples=max(1, minimum_samples - bias)) + bias

        # Get the sampling sizes
        negative_with_root_size, negative_without_root_size = self.get_positive_negative_sample_size_fixed(
            number_of_positive_samples=len(negative_with_root),
            number_of_negative_samples=len(negative_without_root),
            sample_size=sample_size,
            distribution_threshold=1.0
        )
        negative_parents_for_next_level = (
                random.sample(negative_with_root, k=negative_with_root_size) +
                random.sample(negative_without_root, k=negative_without_root_size)
        )
        # print(negative_with_root_size, negative_without_root_size)
        return positive_codes_at_level + negative_parents_for_next_level

    def prepare_hierarchy_codes(
            self,
            positive_codes_at_level,
            negative_codes_at_level,
            prediction_range,
            ignore_instruction
    ):
        """

        Args:
            positive_codes_at_level:
            negative_codes_at_level:
            prediction_range:
            ignore_instruction:

        Returns:

        """

        positives_at_level = self.prepare_sampled_codes(
            codes=positive_codes_at_level,
            label_string=self.get_positive_labels_string(),
            prediction_range=prediction_range,
            ignore_instruction=ignore_instruction
        )
        negatives_at_level = self.prepare_sampled_codes(
            codes=negative_codes_at_level,
            label_string=self.get_negative_labels_string(),
            prediction_range=prediction_range,
            ignore_instruction=ignore_instruction
        )

        return pd.DataFrame([*positives_at_level, *negatives_at_level]).sample(frac=1)

    def prepare_codes_at_level_for_sampling(self, positive_codes_at_level, negative_codes_at_level):
        """
        There will be cases where there are a lot more say CV codes than EM codes
        and any sampling shown below will reflect this. This might induce a bias where
        our training data sees more of codes that tend to be split into more categories
        to avoid this - we use this function.
        The function will try and return equal number of codes from the CV and EM splits
        and try and avoid the distribution mismatch. This is done by sampling equal number
        of children from each parent. We get the parents of these codes and sample
        equal number from each parent

        Args:
            positive_codes_at_level:
            negative_codes_at_level:

        Returns:

        """
        # Return this 50% of the time - will  reflect the original distribution
        if np.random.rand() < 0.5:
            # Return based on distribution
            return positive_codes_at_level, negative_codes_at_level
        else:
            # Give equal to probability of occurrence to a code for a given parent
            # 50% of the time we consider the parent of a code to be the root
            # (e.g. CV, EM, DE). In this case we basically say that we sample equal
            # number of CV codes, DE codes and so on. The rest of the time we take
            # the immediate parent od the code (e.g., CV_403, CV_404, EM_202, DE_680)
            # If we sampled 2 codes - each of the CV_403 ... will have 2 codes
            # (CV_403.1, CV_403.2, EM_202.4, EM_202.7 ...). In the first case we keep
            # the distribution of disease categories uniform, and in the second we keep
            # the parent distribution uniform
            probability = 0 if np.random.rand() < 0.5 else 1
            if positive_codes_at_level:
                positive_codes_at_level = self.get_codes_at_level_for_sampling(
                    codes_at_level=positive_codes_at_level, probability=probability
                )
            if negative_codes_at_level:
                negative_codes_at_level = self.get_codes_at_level_for_sampling(
                    codes_at_level=negative_codes_at_level, probability=probability
                )
            return positive_codes_at_level, negative_codes_at_level

    def get_codes_at_level_for_sampling(self, codes_at_level, probability):
        """
        Given a set of codes - return a subset such that the counts of their
        parents are uniform

        Args:
            codes_at_level:
            probability:

        Returns:

        """
        # Get the parent codes
        parents = {self.__get_parent(code, probability): list() for code in codes_at_level}
        # For each parent get the list of children it has
        for code in codes_at_level:
            parents[self.__get_parent(code, probability)].append(code)
        # We know how many children each parent has - we choose the sample size
        # to be the size of the parent with the least amount of children. This
        # way when we sample - the list will contain codes - such that
        # all their parents will have equal counts
        sample_size = min([len(codes) for _, codes in parents.items()])
        # Sample codes from each parent
        sampled_codes = [random.sample(codes, k=sample_size) for _, codes in parents.items()]
        return list(itertools.chain.from_iterable(sampled_codes))

    def __get_parent(self, code, probability):
        """
        Given a code - return either its root parent
        or its immediate parent

        Args:
            code:
            probability:

        Returns:

        """
        if np.random.rand() < probability:
            return code.split('_')[0]
        else:
            return self._tree.parent(code).tag

    @staticmethod
    def split_negative_codes_at_level(positive_codes_at_level, negative_codes_at_level):
        """
        Given positive and negative codes - split the negative codes such that  one split
        contains codes that have ancestors in the positive set and the other split
        contains codes that do not have ancestors in the positive set

        Args:
            positive_codes_at_level:
            negative_codes_at_level:

        Returns:

        """
        positive_roots = {code.split('_')[0] for code in positive_codes_at_level}
        negative_with_root = [code for code in negative_codes_at_level if code.split('_')[0] in positive_roots]
        negative_without_root = [code for code in negative_codes_at_level if code.split('_')[0] not in positive_roots]
        return negative_with_root, negative_without_root

    def get_positive_negative_sample_size(
            self, number_of_positive_samples, number_of_negative_samples, sample_size_limit
    ):
        """
        Given the sizes of positive and negative lists - return how many elements
        we want to sample from these lists

        Args:
            number_of_positive_samples:
            number_of_negative_samples:
            sample_size_limit:

        Returns:

        """
        # If there is no limit - we can sample anywhere from [1, size]
        if sample_size_limit is None:
            # Get the sample sizes and then sample
            positive_sample_size, negative_sample_size = self.get_positive_negative_sample_size_random(
                number_of_positive_samples=number_of_positive_samples,
                number_of_negative_samples=number_of_negative_samples
            )
        else:
            # We can sample at most to this limit.
            sample_size = self.get_sample_size(total_number_of_samples=sample_size_limit)
            positive_sample_size, negative_sample_size = self.get_positive_negative_sample_size_fixed(
                number_of_positive_samples=number_of_positive_samples,
                number_of_negative_samples=number_of_negative_samples,
                sample_size=max(1, sample_size)
            )
        return positive_sample_size, negative_sample_size

    def get_positive_negative_sample_size_random(
            self,
            number_of_positive_samples,
            number_of_negative_samples
    ):
        """
        Given the number of positive codes and negative codes - we can sample two ways
        We sample uniformly - in which case if there are a lot more negative codes than
        positive codes - we end up with more negative sampled codes.
        Or we can also try and keep the amount of codes sampled from positive and negative
        to be about the same - we do this by getting which of the wto has the least codes
        and limit our sample size to this minimum value

        Args:
            number_of_positive_samples:
            number_of_negative_samples:

        Returns:

        """
        positive_sample_size = 0
        negative_sample_size = 0
        # We want at least one code to be sampled which means the sum of the size should be greater than 0
        while positive_sample_size + negative_sample_size == 0:
            if np.random.rand() < 0.5:
                # Sample according to distribution - should reflect the probability
                positive_sample_size = self.get_sample_size(total_number_of_samples=number_of_positive_samples)
                negative_sample_size = self.get_sample_size(total_number_of_samples=number_of_negative_samples)
            else:
                # Sample similar quantity
                minimum_samples = min(number_of_positive_samples, number_of_negative_samples)
                positive_sample_size = self.get_sample_size(total_number_of_samples=minimum_samples)
                negative_sample_size = self.get_sample_size(total_number_of_samples=minimum_samples)

        return positive_sample_size, negative_sample_size

    def get_positive_negative_sample_size_fixed(
            self,
            number_of_positive_samples,
            number_of_negative_samples,
            sample_size,
            distribution_threshold: float = 0.5
    ):
        """
        This function samples such that the total number of samples will be equal to
        sample_size. Similar to random sampling - we can sample according to distribution
        or sample similar quantities

        Args:
            number_of_positive_samples:
            number_of_negative_samples:
            sample_size:
            distribution_threshold:

        Returns:

        """
        # Decrease sample size if we don't have enough samples
        sample_size = min(sample_size, number_of_negative_samples + number_of_positive_samples)
        if np.random.rand() < distribution_threshold:
            # Sample similar quantity
            positive_sample_size, negative_sample_size = self.get_positive_negative_sizes(
                max_size=sample_size,
                number_of_positive_samples=number_of_positive_samples,
                number_of_negative_samples=number_of_negative_samples,
                probability_threshold=0.5
            )
        else:
            # Sample according to distribution
            probability_threshold = number_of_positive_samples / (number_of_positive_samples + number_of_negative_samples)
            positive_sample_size, negative_sample_size = self.get_positive_negative_sizes(
                max_size=sample_size,
                number_of_positive_samples=number_of_positive_samples,
                number_of_negative_samples=number_of_negative_samples,
                probability_threshold=probability_threshold
            )
        return positive_sample_size, negative_sample_size

    def prepare_sampled_codes(
            self, codes, label_string, prediction_range, ignore_instruction
    ):
        """
        Define the start and end times and the label string for the sampled codes

        Args:
            codes:
            label_string:
            prediction_range:
            ignore_instruction:

        Returns:

        """
        return [
            {
                self._code_column: code,
                self._label_column: label_string,
                self._bin_start_column: prediction_range[0],
                self._bin_end_column: prediction_range[1],
                self._ignore_instruction_column: ignore_instruction
            }
            for code in codes
        ]

    @staticmethod
    def get_positive_negative_sizes(
            max_size: int,
            number_of_positive_samples: int,
            number_of_negative_samples: int,
            probability_threshold: float
    ) -> Tuple[int, int]:
        """
        For a given k-shot problem, return how many positive samples
        and negative samples should be present in the instruction

        Args:
            max_size (int): The maximum number of positives and negatives
            number_of_positive_samples (int): The total number of positive samples
            number_of_negative_samples (int): The total number of negative samples
            probability_threshold (float): Probability of sampling positive

        Returns:
            (Tuple[int, int]): The number of positive and negative examples
        """
        # For zero shot, we have either one positive or one negative
        positive = 0
        negative = 0
        for k in range(max_size):
            probability = np.random.rand()
            # Add as long as there is space, and it meets the probability threshold
            if probability <= probability_threshold:
                if positive < number_of_positive_samples:
                    positive += 1
                elif negative < number_of_negative_samples:
                    negative += 1
            else:
                if negative < number_of_negative_samples:
                    negative += 1
                elif positive < number_of_positive_samples:
                    positive += 1
        return positive, negative

    @staticmethod
    def get_sample_size(total_number_of_samples):
        """
        Return a number from the given choices

        Args:
            total_number_of_samples:

        Returns:

        """
        sample_size_choices = np.arange(total_number_of_samples + 1)
        return np.random.choice(sample_size_choices)

    @staticmethod
    def get_label(code, positive_hash, ignore_hash):
        """
        Return true if the code has been seen in an encounter (will be part)
        of positive hash. Return false if not part of positive hash and ignore hash.

        Args:
            code:
            positive_hash:
            ignore_hash:

        Returns:

        """
        if code in positive_hash:
            return 1
        elif code in ignore_hash:
            return None
        else:
            return 0

    @staticmethod
    def get_sample_in_level_strategy():
        """
        Sampling within the levels
        With consistent sampling - we sample within level but also ensure that
        a code shown in the next level has its parent shown in the previous level.
        This constraint is not applicable for random. The rest of the time we don't
        sample codes within a level
        Consistent:
            - Level 0:
                - I20: Yes
                - I90: No
                - I39: yes
            - Level 1:
                - I20.1: No
                - I20.3: Yes
        Random:
            - Level 0:
                - I90: No
                - I39: yes
            - Level 1:
                - I20.1: No
                - I20.3: Yes

        Returns:

        """
        return np.random.choice(['no_sampling', 'consistent', 'random'], p=[0.5, 0.25, 0.25])

    @staticmethod
    def get_shuffle_probability(sample_in_level_strategy):
        """
        The probability of shuffling level blocks - if we are maintaining
        a consistent parent child relationship - we should not shuffle
        and hence we return 0

        Args:
            sample_in_level_strategy:

        Returns:

        """
        if sample_in_level_strategy == 'consistent':
            return 0.0
        else:
            return 0.5

    @staticmethod
    def get_drop_probability():
        """
        The probability of dropping one or more level blocks

        Returns:

        """
        return 0.5

    @staticmethod
    def get_consistent_sample(sample_in_level_strategy):
        """
        If we are sampling consistent - this should return one.
        We use this to tell the hierarchy portion of the code
        to sample such that parent child relationship is maintained
        as shown in get_sample_in_level_strategy

        Args:
            sample_in_level_strategy:

        Returns:

        """
        return sample_in_level_strategy == 'consistent'

    @staticmethod
    def get_random_sample(sample_in_level_strategy):
        """
        If we don't maintain consistent parent child relationship
        we randomly sample 50% of the time

        Args:
            sample_in_level_strategy:

        Returns:

        """
        return sample_in_level_strategy == 'random'

    def get_sample_size_limit(self):
        """
        In some scenarios - we might want to cap how many codes we can at most sample within a level
        If this is none, there is no such limit, and we can sample from [1, total], if there is a limit
        we can only sample [1, limit]

        Returns:

        """
        if np.random.rand() < 0.5:
            # TODO - is 5 a good number?
            return self.get_sample_size(total_number_of_samples=4) + 1
        else:
            return None

    @staticmethod
    def get_level_instruction(level) -> Tuple[str, str, bool]:
        """
        This instruction is added as a separator between instructions

        Returns:
            (List[str, str, bool]): Example separator instruction
        """
        # \n is the separator string, '' is the label (empty - don't train)
        # True indicates ignore this instruction for training loss
        return f'Level {level}\n', '', True


class SingleCodeLabelPredictionTask(CodeLabelPredictionTask):
    """
    Make instruction tuning training data from diagnostic codes
    """

    def __init__(
            self,
            encounter_dataframe_process: EncounterDataframeProcess,
            dataframe_sampling: GroupBySampling,
            code_instructions: CodeLabelPredictionInstructionTemplate,
            code_convert: Union[ICDConvert, PHEConvert],
            time_bins: Union[AgglomerativeDataBins, Sequence[AgglomerativeDataBins]],
            negative_code_sampling: NegativeCodeCacheSampling,
            fixed_position_range: bool,
            patient_id_column: str = 'PatientID',
            code_column: str = 'phecode',
            position_column: str = 'position',
            bins_column: str = 'bins',
            bin_start_column: str = 'min',
            bin_end_column: str = 'max',
            label_column: str = 'label',
            ignore_instruction_column: str = 'ignore_instruction',
            position_range_column: str = 'position_range',
            current_bin_value: int = -100,
            prediction_range_limit: int = None,
    ):
        """
        Initialize variables

        Args:
            encounter_dataframe_process (EncounterDataframeProcess): Dataframe processing object
            dataframe_sampling (GroupBySampling): Dataframe sampling object
            code_instructions (Any): A list of code instruction tasks. At any point one of these
            tasks is randomly chosen and trained on
            code_convert (Union[ICDConvert, PHEConvert]): The object to map codes to other codes in hierarchy and
            their textual descriptions.
            patient_id_column (str, defaults to `PatientID`): The column name for the patient id
            code_column (str, defaults to `phecode`): The column that contains the codes
            position_column (str, defaults to `positions`): The column where position values
            (e.g. time difference in days or log days) will be stored
            bins_column (str, defaults to `bins`): Column that stores the assigned bin
            bin_start_column (str, defaults to `min`): Column that stores the start position of each bin
            bin_end_column (str, defaults to `max`): Column that stores the end position of each bin
            label_column (str, defaults to `count`): The column that stores some value that represents
            the code in a time bin
            ignore_instruction_column (str, defaults to `ignore_instruction`): Column that stores whether
            to ignore the instruction in that row
        """
        super().__init__(
            encounter_dataframe_process=encounter_dataframe_process,
            dataframe_sampling=dataframe_sampling,
            code_instructions=code_instructions,
            code_convert=code_convert,
            time_bins=time_bins,
            negative_code_sampling=negative_code_sampling,
            patient_id_column=patient_id_column,
            code_column=code_column,
            position_column=position_column,
            bins_column=bins_column,
            bin_start_column=bin_start_column,
            bin_end_column=bin_end_column,
            label_column=label_column,
            ignore_instruction_column=ignore_instruction_column,
            position_range_column=position_range_column,
            current_bin_value=current_bin_value,
            prediction_range_limit=prediction_range_limit,
            fixed_position_range=fixed_position_range
        )

    def process_sample(self, sample, args):
        """
        Process sample

        Args:
            sample:
            args:

        Returns:

        """

        regex = False

        fine_tune_code = args.fine_tune_code

        encounter_history = self.get_full_encounter_history(
            patient_id=sample[args.patient_id_column],
            current_time=sample[args.sample_result_date_column],
            use_log_position=args.use_log_position,
            time_difference_normalize=args.time_difference_normalize
        )

        # If for some reason we don't have encounter data for this person - we ignore training
        # on this person
        if self._encounter_dataframe_process.check_patient_id(patient_id=sample[args.patient_id_column]):
            ignore_instruction = False
        else:
            ignore_instruction = True

        encounter_history = self.transform_encounter_history_for_task(
            encounter_history=encounter_history,
            past_time_delta=args.past_time_delta,
            future_time_delta=args.future_time_delta,
        )

        code_occurrences = self.get_code_occurrences(
            encounter_history=encounter_history, code=fine_tune_code, regex=regex
        )

        # Get the bin period we will be making predictions for.
        # The list can contain multiple elements - we can make prediction for
        # multiple time bins
        prediction_ranges = self.get_prediction_ranges(
            future_only=args.future_only,
            number_of_ranges=self.sample_from_list(args.number_of_instructions),
            fixed_range=args.time_period_range,
            code_occurrences=code_occurrences
        )

        # Store all the prompt elements (input, output)
        all_instructions = list()

        for prediction_range in prediction_ranges:

            # Get the task instruction - which should indicate we are making prediction for a given
            # time range
            all_instructions.append(self.get_task_instruction(prediction_range=prediction_range))

            # Get dataframe that contain encounter in the past, current and future time periods
            current_time_period = self.get_current_time_period(
                encounter_history=encounter_history, prediction_range=prediction_range
            )
            current_code_occurrences = self.get_code_occurrences(
                encounter_history=current_time_period, code=fine_tune_code, regex=regex
            )

            label = self.get_label(code_occurrences=current_code_occurrences)

            fine_tune_code_sample = self.get_code_sample(
                code=fine_tune_code,
                start_time=prediction_range[0],
                end_time=prediction_range[1],
                label=self.get_true_label_string(label=label),
                ignore_instruction=ignore_instruction
            )

            # Convert samples to text instructions (prompt)
            all_instructions.extend(
                self.convert_samples_to_instructions(
                    instruction_samples=fine_tune_code_sample,
                )
            )

        return all_instructions

    def get_code_occurrences(self, encounter_history, code, regex):
        """
        Check where all the code has occurred

        Args:
            encounter_history (pd.DataFrame): The encounter history of the patient
            code (str): The code we fine tune on
            regex (bool): Whether to use regex

        Returns:

        """
        return encounter_history[
            self.check_code_exists(encounter_history=encounter_history, code=code, regex=regex)
        ]

    def get_prediction_ranges(
            self,
            future_only: bool,
            number_of_ranges: int,
            fixed_range: Tuple[int, int] = None,
            code_occurrences: pd.DataFrame = None,
    ) -> List[Tuple[int, int]]:
        """
        Returns a list of tuples that contain the start and end times of the bin
        we will be using as the current bin period - this is the period we
        will make predictions for.

        Args:
            fixed_range (Tuple[int, int], defaults to `None`): Return the fixed range
            future_only (bool): Use ranges occurring only in the future
            number_of_ranges (int): The number of prediction ranges to sample
            code_occurrences (pd.DataFrame): The dataframe containing all the occurrences of the code

        Returns:
            (List[Tuple[int, int]]): The start and end position of the current bin period
        """

        # TODO: Remove position ranges where any code from hierarchy is present
        if code_occurrences.empty:
            return super().get_prediction_ranges(
                future_only=future_only,
                fixed_range=fixed_range,
                number_of_ranges=number_of_ranges
            )
        else:
            return super().get_prediction_ranges(
                future_only=future_only,
                fixed_range=fixed_range,
                number_of_ranges=number_of_ranges
            )

    @staticmethod
    def get_label(code_occurrences: pd.DataFrame) -> bool:
        """
        Check if a given code exists in a given encounter history
        Args:
            code_occurrences (pd.DataFrame): The dataframe containing all the occurrences of the code

        Returns:
            (str): The input code or sub code
            (bool): Whether the code or it's sub code is in the encounter history
        """
        return not code_occurrences.empty

    def get_code_sample(
            self,
            code: str,
            start_time: int,
            end_time: int,
            label: str,
            ignore_instruction: bool
    ) -> pd.DataFrame:
        """
        Create a row/dataframe that contains details about the code

        Args:
            code (str): The code to evaluate
            start_time (int): The start time period of the code evaluation prompt
            end_time (int): The end time period of the code evaluation prompt
            label (str): The label to evaluate
            ignore_instruction (bool): Whether to ignore instruction while computing loss etc

        Returns:
            (pd.DataFrame): The dataframe that contains the code, start and end times
        """
        code_sample = {
            self._code_column: [code],
            self._label_column: [label],
            self._bin_start_column: [start_time],
            self._bin_end_column: [end_time],
            self._ignore_instruction_column: [ignore_instruction]
        }
        return pd.DataFrame(code_sample)

    def get_true_label_string(self, label):
        """
        Get the label string
        Args:
            label:

        Returns:

        """
        if label:
            return self.get_positive_labels_string()
        else:
            return self.get_negative_labels_string()

    def check_code_exists(self, encounter_history, code, regex):
        """
        Check if the fine-tuning code exists in the dataframe - or if any
        of its children exist
        Args:
            encounter_history:
            code:
            regex:

        Returns:

        """
        return encounter_history[self._code_column].str.contains(code, regex=regex)
