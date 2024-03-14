"""Make training data"""
import random
import numpy as np
import pandas as pd
from collections import Counter
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
        self._position_ranges = [
            (-1, 30), (30, 60), (60, 90), (90, 120), (120, 150), (150, 180),
            (180, 210), (210, 240), (240, 270), (270, 300), (300, 330),
            (330, 360), (360, np.inf)
        ]
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

        encounter_history = self.transform_encounter_history_for_task(
            encounter_history=encounter_history,
            past_time_delta=args.past_time_delta,
            future_time_delta=args.future_time_delta,
        )

        #  Get the bin period we will be making predictions for
        prediction_ranges = self.get_prediction_range(
            encounter_history,
            future_only=args.future_only,
            number_of_ranges=self.sample_from_list(args.number_of_instructions),
            fixed_range=args.time_period_range
        )

        all_instructions = list()

        for prediction_range in prediction_ranges:

            # Get dataframe that contain encounter in the past, current and future time periods
            current_time_period = self.get_current_time_period(
                encounter_history=encounter_history, prediction_range=prediction_range
            )

            # Get a dataframe that can be used to sample negatives
            encounter_negatives = self._negative_code_sampling.get_encounter_negatives_from_cache_process_and_update(
                patient_id=sample[args.patient_id_column],
                current_time=sample[args.sample_result_date_column],
                use_log_position=args.use_log_position,
                time_difference_normalize=args.time_difference_normalize,
                exclude_codes=self.get_exclude_codes_for_negatives(
                    encounter_history=encounter_history, current_time_period=current_time_period
                ),
                code_column=self._code_column,
                position_column=self._position_column
            )

            encounter_negatives = self.transform_encounter_history_for_task(
                encounter_history=encounter_negatives,
                past_time_delta=args.past_time_delta,
                future_time_delta=args.future_time_delta,
            )

            negatives = self.get_current_time_period(
                encounter_history=encounter_negatives, prediction_range=prediction_range
            )

            # TODO - compute counts and first and last visit for each code
            histories = [negatives, current_time_period]
            histories_unique = [self.get_unique_codes(history) for history in histories]
            # print(f'Prediction Range: {prediction_range}, Negatives count: {len(histories_unique[0])}, Positives count: {len(histories_unique[1])} \n')

            # Sample counts for negatives, current new, current old, future new and future old
            sample_counts = self.get_sample_sizes(
                max_size=self.sample_from_list(args.k_shot),
                sample_limits=[len(dataframe) for dataframe in histories_unique],
            )

            sampled_codes = [
                self.sample_codes(encounter_history=encounter_history, max_limit=counts)
                for encounter_history, counts in zip(histories_unique, sample_counts)
            ]

            label_samples = [
                self.prepare_sampled_codes(
                    codes=codes,
                    label_string=self.get_positive_labels_string() if label == 1 else self.get_negative_labels_string(),
                    prediction_range=prediction_range
                ) for label, codes in enumerate(sampled_codes)
            ]

            # TODO: Shuffle or sort?
            instruction_samples = pd.concat(label_samples).sample(frac=1)

            all_instructions.extend(
                self.convert_samples_to_instructions(
                    instruction_samples=instruction_samples,
                    prediction_range=prediction_range
                )
            )
            all_instructions.append(self._code_instructions.get_example_separator_instruction())

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
        try:
            encounter_history = self._encounter_dataframe_process.get_patient_encounter_history_with_position(
                patient_id=patient_id,
                current_time=current_time,
                use_log_position=use_log_position,
                time_difference_normalize=time_difference_normalize
            )
            encounter_history.dropna(subset=self._code_column, inplace=True)
            return encounter_history
        except KeyError:
            # FIXME: A fix for now - because we have missing data - ideally we should not have missing data
            # Sample codes and create a random encounter dataframe
            return self.get_random_encounters()

    def transform_encounter_history_for_task(
            self,
            encounter_history: pd.DataFrame,
            past_time_delta: Optional[str] = None,
            future_time_delta: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Given the encounter history, filter out any NA codes and apply a time
        filter if passed and return the resulting dataframe

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

    def get_prediction_range(
            self,
            encounter_history: pd.DataFrame,
            future_only: bool,
            number_of_ranges: int,
            fixed_range: Tuple[int, int] = None
    ) -> List[Tuple[int, int]]:
        """
        Returns a tuple that contains the start and end times of the bin
        we will be using as the current bin period - this is the period we
        will make predictions for.

        Args:
            encounter_history (pd.DataFrame): The encounter history of the patient
            fixed_range (Tuple[int, int], defaults to `None`): Return the fixed range
            future_only (bool): Use ranges occurring only in the future
            number_of_ranges (int): The number of prediction ranges to sample

        Returns:
            (List[Tuple[int, int]]): The start and end position of the current bin period
        """

        if fixed_range is not None:
            return [fixed_range]

        # Choose only the bins in the future
        # Bins in the future have a positive value
        position_ranges = self.get_ranges_for_prediction(
            position_ranges=encounter_history[self._position_range_column],
            future_only=future_only
        )

        if len(position_ranges) == 0:
            sampled_range_periods = random.choices(self._position_ranges[0:1], k=1)
            prediction_ranges = [
                self.create_range_period(sampled_range_period=sampled_range_period, future_only=future_only)
                for sampled_range_period in sampled_range_periods
            ]
            return prediction_ranges

        prediction_ranges = list()
        sampled_position_ranges = np.random.choice(
            position_ranges, size=min(len(position_ranges), number_of_ranges), replace=False
        )

        for sampled_position_range in sampled_position_ranges:
            range_period = self.get_range_period(
                position_range=sampled_position_range
            )
            prediction_ranges.append(range_period)

        return prediction_ranges

    def get_time_periods(
            self, encounter_history: pd.DataFrame, prediction_range: Tuple[int, int]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Return three dataframes - one which contains encounters before the current bin period,
        one which contains encounters belonging to the current bin period and one which contains
        encounters after the current bin period

        Args:
            encounter_history (pd.DataFrame): The encounter dataframe
            prediction_range (Tuple[int, int]): The start and end positions of the current bin period

        Returns:
            (Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]): Dataframes that contains encounter before,
            during and after the current bin
        """
        past_time_period = self.get_past_time_period(encounter_history, prediction_range)
        current_time_period = self.get_current_time_period(encounter_history, prediction_range)
        future_time_period = self.get_future_time_period(encounter_history, prediction_range)
        return past_time_period, current_time_period, future_time_period

    def get_unique_codes(self, encounter_history):
        """
        Return the unique codes from the dataframe
        Args:
            encounter_history:

        Returns:

        """
        # Keep the first to keep the first occurrence of the code
        return encounter_history.drop_duplicates(self._code_column, keep='first')

    def sample_codes(self, encounter_history, max_limit):
        """
        Sample codes from the dataframe - sample such that we have good
        representation from different time ranges - for example

        Args:
            encounter_history:
            max_limit:

        Returns:

        """
        # Get the maximum number of codes you can sample from each sub range
        range_count_limits = (
            encounter_history
            .groupby(self._position_range_column)[self._position_range_column]
            .count()
            .to_dict()
        )
        max_limit = min(sum(range_count_limits.values()), max_limit)
        position_ranges = encounter_history[self._position_range_column].unique()
        # Get how many codes we are going to sample from each range
        range_counts = self.get_position_range_counts(
            position_ranges=position_ranges,
            range_count_limits=range_count_limits,
            max_limit=max_limit
        )
        return (
            encounter_history
            .groupby(self._position_range_column)
            .apply(lambda x: x.sample(n=range_counts[x.name]))
            .reset_index(drop=True)
        )

    def prepare_sampled_codes(self, codes, label_string, prediction_range):
        """
        Define the start and end times and the label string for the sampled codes

        Args:
            codes:
            label_string:
            prediction_range:

        Returns:

        """
        codes[self._label_column] = label_string
        codes[self._bin_start_column] = prediction_range[0]
        codes[self._bin_end_column] = prediction_range[1]
        codes[self._ignore_instruction_column] = False
        return codes

    def convert_samples_to_instructions(
            self,
            instruction_samples: pd.DataFrame,
            prediction_range: Tuple[int, int]
    ) -> List[Tuple[str, str, bool]]:
        """
        Return the instruction input and targets
        for the sampled instruction code status classification task with time range

        Args:
            instruction_samples (pd.DataFrame): The samples that will be used as instructions
            prediction_range: (Tuple[int, int]): The time range we are making predictions for

        Returns:
            (List[Tuple[str, str, bool]]): A list that contains tuples which have the instruction input and target.
            The list will contain only 1 element for zero shot training
        """

        instructions = list()
        time = self.process_time_period(start_time=prediction_range[0], end_time=prediction_range[1])
        task_definition = self._code_instructions.get_task_definition(time=time)
        instructions.append((task_definition, '') + (True,))
        instructions.append(self._code_instructions.get_example_separator_instruction())
        for row in instruction_samples[
            [self._code_column, self._label_column, self._ignore_instruction_column]
        ].itertuples(index=False):
            code, label, ignore_instruction = row[0], row[1], row[2]
            code_instruct_string = self._code_instructions.get_instruction(
                diagnosis=self._code_convert.transform_code(code=code),
                value=label
            )
            instructions.append(code_instruct_string + (ignore_instruction,))
            instructions.append(self._code_instructions.get_example_separator_instruction())
        return instructions

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

    def get_ranges_for_prediction(self, position_ranges: pd.Series, future_only: bool) -> np.array:
        """
        Given the position range values - return the unique position ranges
        Args:
            position_ranges (pd.Series): A series containing the position ranges
            future_only (bool): Return ranges occurring only in the future

        Returns:
            (np.array): An array of unique position ranges
        """
        ranges = position_ranges.unique()
        if future_only:
            return ranges[(np.abs(ranges) != self._prediction_range_limit) & (ranges >= 1)]
        else:
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

    def get_past_time_period(
            self, encounter_history: pd.DataFrame, prediction_range: Tuple[int, int]
    ) -> pd.DataFrame:
        """
        Return encounters in the past - before the start of the prediction range

        Args:
            encounter_history (pd.DataFrame):
            prediction_range:

        Returns:

        """
        return (
            encounter_history[encounter_history[self._position_column] < prediction_range[0]]
        )

    def get_current_time_period(
            self, encounter_history: pd.DataFrame, prediction_range: Tuple[int, int]
    ) -> pd.DataFrame:
        """
        Return encounters within the current range

        Args:
            encounter_history:
            prediction_range:

        Returns:

        """
        return (
            encounter_history[
                (encounter_history[self._position_column] >= prediction_range[0])
                & (encounter_history[self._position_column] <= prediction_range[1])
                ]
        )

    def get_future_time_period(
            self, encounter_history: pd.DataFrame, prediction_range
    ) -> pd.DataFrame:
        """
        Return encounters in the future - after the end of the prediction range

        Args:
            encounter_history:
            prediction_range:

        Returns:

        """
        return (
            encounter_history[encounter_history[self._position_column] > prediction_range[1]]
        )

    @staticmethod
    def get_position_range_counts(position_ranges, range_count_limits, max_limit):
        """
        Return the number of codes to sample from each sub range - each range should
        get about a similar number of counts - but should also not exceed
        the range limits

        Args:
            position_ranges:
            range_count_limits:
            max_limit:

        Returns:

        """
        range_counts = Counter()
        total_count = 0
        while total_count < max_limit:
            np.random.shuffle(position_ranges)
            for position_range in position_ranges:
                if range_counts[position_range] < range_count_limits[position_range]:
                    range_counts[position_range] += 1
                    total_count += 1
                    if total_count == max_limit:
                        return range_counts
        return range_counts

    def get_positive_labels_string(self):
        """
        Return the string used to denote the positive label

        Returns:

        """
        return 'yes'

    def get_negative_labels_string(self):
        """
        Return the string used to denote the negative label

        Returns:

        """
        return 'no'

    def process_time_period(self, start_time, end_time):
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
        elif start_time == 0 and end_time == 0:
            return 1
        else:
            print(start_time, end_time)
            raise ValueError('Invalid start and end time')

    def get_ignore_instructions(self, instruction_samples: pd.DataFrame) -> Union[List[bool], pd.Series]:
        """
        Return a dataframe of bools that represents whether to ignore instructions in the
        output/labels

        Args:
           instruction_samples (pd.DataFrame): The samples that will be used as instructions

        Returns:
            ignore_instruction (Optional[Sequence[bool]]): Keep these instructions in the
            input but don't use them as labels - ignore their output while training.
        """
        # We don't ignore any instruction
        return [False] * len(instruction_samples)

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
            {self._code_column: [], self._position_column: []}
        )

    def get_exclude_codes_for_negatives(
            self,
            encounter_history: pd.DataFrame,
            current_time_period: pd.DataFrame
    ) -> pd.Series:
        """
        Return a list of codes that should not be sampled when sampling negatives
        Args:
            encounter_history (pd.Dataframe): The encounter history of the patient
            current_time_period (pd.DataFrame): The encounters in a given time range

        Returns:
            (pd.Series): List of codes that should not be used for negative sampling
        """
        bins = current_time_period[self._bins_column].unique()
        # In case there is any code that is i not in the current time period but is part of the bin that is in the
        # current time period - basically if the bin spans over two months - don't sample codes from this
        # bin as negatives
        return encounter_history[encounter_history[self._bins_column].isin(bins)][self._code_column]

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
            sample_limits (int): The total number of encounters in a given time range
        Returns:
            (List[int]): The number of positive and negative examples
        """

        probabilities = [0.5, 0.5]
        indexes = [0, 1]
        counts = [0, 0]

        while max_size and sum(sample_limits) > 0:
            probabilities = self.__normalize_probability(probabilities=probabilities, limits=sample_limits)
            index = np.random.choice(indexes, p=probabilities)
            counts[index] += 1
            sample_limits[index] -= 1
            max_size -= 1

        return counts

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
            prediction_range_limit=prediction_range_limit
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

        encounter_history = self.transform_encounter_history_for_task(
            encounter_history=encounter_history,
            past_time_delta=args.past_time_delta,
            future_time_delta=args.future_time_delta,
        )

        #  Get the bin period we will be making predictions for
        prediction_range = self.get_prediction_range(
            encounter_history,
            future_only=args.future_only,
            fixed_range=args.time_period_range,
            code=fine_tune_code,
            regex=regex
        )[0]

        # Get dataframe that contain encounter in the past, current and future time periods
        _, current_time_period, _ = self.get_time_periods(
            encounter_history=encounter_history, prediction_range=prediction_range
        )

        prompt_code, true_label = self.get_prompt_code_and_label(
            encounter_history=current_time_period,
            code=fine_tune_code,
            regex=regex
        )

        eval_sample = self.get_code_sample(
            code=prompt_code,
            start_time=prediction_range[0],
            end_time=prediction_range[1],
            label=self.get_true_label_string(true_label=true_label),
            ignore_instruction=False
        )

        return self.convert_samples_to_instructions(eval_sample, prediction_range=prediction_range)

    def get_prediction_range(
            self,
            encounter_history: pd.DataFrame,
            future_only: bool,
            fixed_range: Tuple[int, int] = None,
            code: str = None,
            regex: bool = False
    ) -> List[Tuple[int, int]]:
        """
        Returns a tuple that contains the start and end times of the bin
        we will be using as the current bin period - this is the period we
        will make predictions for.

        Args:
            encounter_history (pd.DataFrame): The encounter history of the patient
            future_only (bool): Use ranges occurring only in the future
            fixed_range (Tuple[int, int], defaults to `None`): Return the fixed range
            code (str): The code we are evaluating
            regex (bool, defaults to `False`): Regex for matching the code

        Returns:
            (Tuple[int, int]): The start and end position of the current bin period
        """

        matched_codes = encounter_history[
            self.check_code_exists(encounter_history=encounter_history, code=code, regex=regex)
        ]

        # TODO: Remove position ranges where any code from hierarchy is present
        if matched_codes.empty:
            return super().get_prediction_range(
                encounter_history=encounter_history,
                future_only=future_only,
                fixed_range=fixed_range,
                number_of_ranges=1
            )
        else:
            return super().get_prediction_range(
                encounter_history=matched_codes,
                future_only=future_only,
                fixed_range=fixed_range,
                number_of_ranges=1
            )

    def get_prompt_code_and_label(
            self,
            encounter_history: pd.DataFrame,
            code: str,
            regex: bool
    ) -> Tuple[str, bool]:
        """
        Check if a given code exists in a given encounter history
        Args:
            encounter_history (pd.DataFrame): The encounter history
            code (str): A given diagnostic code
            regex (bool): Whether to perform regex match or not

        Returns:
            (str): The input code or sub code
            (bool): Whether the code or it's sub code is in the encounter history
        """
        matched_codes = encounter_history[
            self.check_code_exists(encounter_history=encounter_history, code=code, regex=regex)
        ]
        if len(matched_codes) == 0:
            return np.random.choice(self.get_sub_codes(code=code)), False
        else:
            return code, True

    def get_code_sample(
            self,
            code: str,
            start_time: int,
            end_time: int,
            label: str,
            ignore_instruction: bool
    ) -> pd.DataFrame:
        """
        Create a row/dataframe that contains details about the evaluation code

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

    def get_true_label_string(self, true_label):
        """
        Get the label string
        Args:
            true_label:

        Returns:

        """
        if true_label:
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

    def get_sub_codes(self, code: str) -> List[str]:
        """
        Given a code - return all sub codes of this. This would
        be all codes that contain this string.
        Args:
            code (str): A given diagnostic code

        Returns:
            (List[str]): All sub codes of the given code
        """
        return [code]

    def get_ignore_instructions(self, instruction_samples: pd.DataFrame) -> Union[List[bool], pd.Series]:
        """
        Return a dataframe of bools that represents whether to ignore instructions in the
        output/labels

        Args:
           instruction_samples (pd.DataFrame): The samples that will be used as instructions

        Returns:
            ignore_instruction (Optional[Sequence[bool]]): Keep these instructions in the
            input but don't use them as labels - ignore their output while training.
        """
        # Ignore all instructions that serve as context - won't be used for metrics/training
        return [True] * len(instruction_samples)


    @staticmethod
    def get_eval_instruction_samples(
            instruction_samples: pd.DataFrame,
            eval_sample: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add the eval sample to the instruction samples and return. The returned dataframe can
        then be used to create the prompt - which can then be passed to the model for evaluation.

        Args:
            instruction_samples (pd.DataFrame): Contains the samples/encounters from the trajectory
            eval_sample (pd.DataFrame): The eval sample

        Returns:
            (pd.DataFrame): Dataframe that contains both the trajectory samples and the eval sample
        """
        return pd.concat([instruction_samples, eval_sample])
