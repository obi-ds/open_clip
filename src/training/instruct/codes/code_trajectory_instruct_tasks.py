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

        # Get a dataframe that can be used to sample negatives
        encounter_negatives = self._negative_code_sampling.get_encounter_negatives_from_cache_process_and_update(
            encounter_history=encounter_history,
            patient_id=sample[args.patient_id_column],
            current_time=sample[args.sample_result_date_column],
            use_log_position=args.use_log_position,
            time_difference_normalize=args.time_difference_normalize,
            exclude_codes=self.get_exclude_codes_for_negatives(encounter_history=encounter_history),
            code_column=self._code_column,
            position_column=self._position_column
        )

        encounter_history = self.transform_encounter_history_for_task(
            encounter_history=encounter_history,
            past_time_delta=args.past_time_delta,
            future_time_delta=args.future_time_delta,
        )
        encounter_negatives = self.transform_encounter_history_for_task(
            encounter_history=encounter_negatives,
            past_time_delta=args.past_time_delta,
            future_time_delta=args.future_time_delta,
        )

        #  Get the bin period we will be making predictions for
        prediction_range = self.get_prediction_range(encounter_history)

        # Get dataframe that contain encounter in the past, current and future time periods
        past_time_period, current_time_period, future_time_period = self.get_time_periods(
            encounter_history=encounter_history, prediction_range=prediction_range
        )

        current_time_period_unique = self.get_unique_codes(current_time_period)

        max_number_positives, max_number_negatives = self.get_positive_negative_sizes(
            max_size=args.number_of_instructions,
            number_of_encounters=len(current_time_period_unique)
        )

        positive_codes = self.sample_codes(
            encounter_history=current_time_period_unique,
            max_limit=max_number_positives
        )
        # TODO: Add a constraint based on a prediction range - like how we do for positives?
        negative_codes = self.sample_codes(
            encounter_history=self.get_unique_codes(encounter_negatives),
            max_limit=max_number_negatives,
        )

        positive_label_samples = self.prepare_sampled_codes(
            codes=positive_codes, label_string=self.get_positive_labels_string(), prediction_range=prediction_range
        )
        negative_label_samples = self.prepare_sampled_codes(
            codes=negative_codes, label_string=self.get_negative_labels_string(), prediction_range=prediction_range
        )

        instruction_samples = self.combine_positive_and_negative_labels(
            positive_labels=positive_label_samples,
            negative_labels=negative_label_samples,
            shuffle=args.shuffle_bins
        )

        return self.convert_samples_to_instructions(instruction_samples)

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

    def get_prediction_range(self, encounter_history: pd.DataFrame) -> Tuple[int, int]:
        """
        Returns a tuple that contains the start and end times of the bin
        we will be using as the current bin period - this is the period we
        will make predictions for.

        Args:
            encounter_history (pd.DataFrame): The encounter history of the patient

        Returns:
            (Tuple[int, int]): The start and end position of the current bin period
        """

        # Choose only the bins in the future
        # Bins in the future have a positive value
        ranges = self.get_ranges_for_prediction(position_ranges=encounter_history[self._position_range_column])

        if len(ranges) == 0:
            sampled_range = random.choice(self._position_ranges[:-1])
            if np.random.rand() <= 0.5:
                return 0, sampled_range[1]
            else:
                return -sampled_range[1], 0

        sampled_range = np.random.choice(ranges)

        if sampled_range < 0:
            end_position = (
                encounter_history[encounter_history[self._position_range_column] == sampled_range][
                    self._position_column
                ]
                .min()
            )
            return end_position, 0
        else:
            end_position = (
                encounter_history[encounter_history[self._position_range_column] == sampled_range][
                    self._position_column
                ]
                .max()
            )
            return 0, end_position

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
        return encounter_history.drop_duplicates(self._code_column, keep='first')

    def sample_codes(self, encounter_history, max_limit):

        range_count_limits = (
            encounter_history
            .groupby(self._position_range_column)[self._position_range_column]
            .count()
            .to_dict()
        )
        max_limit = min(sum(range_count_limits.values()), max_limit)
        position_ranges = encounter_history[self._position_range_column].unique()
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

        codes[self._label_column] = label_string
        codes[self._bin_start_column] = prediction_range[0]
        codes[self._bin_end_column] = prediction_range[1]
        codes[self._ignore_instruction_column] = False
        return codes

    def combine_positive_and_negative_labels(self, positive_labels, negative_labels, shuffle):
        instruction_samples = self.concatenate_instruction_samples(
            sampled_positives=positive_labels, sampled_negatives=negative_labels
        )
        if shuffle:
            return instruction_samples.sample(frac=1)
        else:
            return instruction_samples.sort_values(by=self._position_column)

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
            [self._code_column, self._label_column, self._bin_start_column,
             self._bin_end_column, self._ignore_instruction_column]
        ].itertuples(index=False):
            code, label, start_time, end_time, ignore_instruction = row[0], row[1], row[2], row[3], row[4]
            time = self.process_time_period(start_time=start_time, end_time=end_time)
            code_instruct_string = self._code_instructions.get_instruction(
                diagnosis=self._code_convert.transform_code(code=code),
                time=time,
                value=label
            )
            # code_instruct_string = self._code_instructions.get_instruction(
            #     diagnosis=code,
            #     time=time,
            #     value=label
            # )
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

    def get_ranges_for_prediction(self, position_ranges: pd.Series) -> np.array:
        """
        Given the position range values - return the unique position ranges
        Args:
            position_ranges (pd.Series): A series containing the position ranges

        Returns:
            (np.array): An array of unique position ranges
        """
        ranges = position_ranges.unique()
        return ranges[np.abs(ranges) != self._prediction_range_limit]

    def get_past_time_period(
            self, encounter_history: pd.DataFrame, prediction_range: Tuple[int, int]
    ) -> pd.DataFrame:
        return (
            encounter_history[encounter_history[self._position_column] < prediction_range[0]]
        )

    def get_current_time_period(
            self, encounter_history: pd.DataFrame, prediction_range: Tuple[int, int]
    ) -> pd.DataFrame:
        return (
            encounter_history[
                (encounter_history[self._position_column] >= prediction_range[0])
                & (encounter_history[self._position_column] <= prediction_range[1])
                ]
        )

    def get_future_time_period(
            self, encounter_history: pd.DataFrame, prediction_range
    ) -> pd.DataFrame:
        return (
            encounter_history[encounter_history[self._position_column] > prediction_range[1]]
        )

    @staticmethod
    def get_position_range_counts(position_ranges, range_count_limits, max_limit):
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
        return 'yes'

    def get_negative_labels_string(self):
        return 'no'

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
            return pd.concat([sampled_positives, sampled_negatives])

    def process_time_period(self, start_time, end_time):
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
            encounter_history: pd.DataFrame
    ) -> pd.Series:
        """
        Return a list of codes that should not be sampled when sampling negatives
        Args:
            encounter_history (pd.Dataframe): The encounter history of the patient

        Returns:
            (pd.Series): List of codes that should not be used for negative sampling
        """
        return encounter_history[self._code_column]

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
