"""Make training data"""
import numpy as np
import pandas as pd
from typing import Union, Tuple, List, Optional, Sequence

from .processing.code_convert import ICDConvert, PHEConvert
from .processing.encounter_dataframe_process import EncounterDataframeProcess
from .processing.dataframe_sampling import GroupBySampling
from .processing.data_bins import AgglomerativeDataBins
from .processing.negative_code_sampling import NegativeCodeCacheSampling
from .templates import CodeLabelPredictionInstructionTemplate, HierarchicalCodeLabelPredictionInstructionTemplate

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
            seq2seq_column: str = 'seq2seq',
            current_bin_value: int = -100,
            prediction_range_limit: int = None,
            seq2seq: bool = False
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
        self._time_bins = [time_bins] if not isinstance(time_bins, list) else time_bins
        self._bins_column = bins_column
        self._bin_start_column = bin_start_column
        self._bin_end_column = bin_end_column
        self._label_column = label_column
        self._ignore_instruction_column = ignore_instruction_column
        self._negative_code_sampling = negative_code_sampling
        self._position_range_column = position_range_column
        self._seq2seq_column = seq2seq_column
        self._current_bin_value = current_bin_value
        self._idf_column = idf_column
        self._tf_column = tf_column
        self._weights_column = weights_column
        self._seq2seq = seq2seq

        if fixed_position_range:
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

    def process_sample(self, sample, args, ignore_instruction=False):
        """
        Process sample

        Args:
            sample:
            args:
            ignore_instruction:

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
                raise ValueError('This should not happen if code is correct')

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
                    ignore_instruction=ignore_instruction,
                    seq2seq=self._seq2seq
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
            encounter_history = self.get_random_encounters()

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

        return encounter_history[
            (encounter_history[self._position_column] >= prediction_range[0]) &
            (encounter_history[self._position_column] <= prediction_range[1])
            ]

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
            return buffered_time_period[self._code_column]
        else:
            return encounter_history[self._code_column]

    @staticmethod
    def get_sample_sizes(
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
        index = None
        while max_size and sum(sample_limits) > 0:
            # If we exceed the sample limit - re-normalize the probability - so that we only sample
            # from the portion that has data
            index = np.random.choice(indexes, p=probabilities)
            if sample_limits[index] == 0:
                break
            counts[index] += 1
            sample_limits[index] -= 1
            max_size -= 1
        counts[np.abs(index - 1)] += max_size
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
            self,
            codes: Union[List[str], pd.DataFrame, str],
            label_string,
            ignore_instruction,
            seq2seq,
    ):
        """
        Define the start and end times and the label string for the sampled codes

        Args:
            codes:
            label_string:
            ignore_instruction:
            seq2seq

        Returns:

        """
        codes[self._label_column] = label_string
        codes[self._ignore_instruction_column] = ignore_instruction
        codes[self._seq2seq_column] = seq2seq
        return codes

    def convert_samples_to_instructions(
            self,
            instruction_samples: pd.DataFrame,
    ) -> List[Tuple[str, str, bool, bool]]:
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
            [self._code_column, self._label_column, self._ignore_instruction_column, self._seq2seq_column]
        ].itertuples(index=False):
            code, label, ignore_instruction, seq2seq = row[0], row[1], row[2], row[3]
            code_instruct_string = self._code_instructions.get_instruction(
                diagnosis=self._code_convert.transform_code(code=code),
                value=label
            )
            instructions.append(code_instruct_string + (ignore_instruction, seq2seq))
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
            return -range_period[1], self._position_ranges[0][0]
        # Future range
        else:
            return self._position_ranges[0][0], range_period[1]

    @staticmethod
    def sample_from_list(shots):
        """
        Sample an element from a list

        Args:
            shots:

        Returns:

        """
        return np.random.choice(shots)

    def process_time_period(self, start_time, end_time):
        """
        Given a start time and end time in days - return the time range
        in a single month format - to be used in the string prompts

        Args:
            start_time:
            end_time:

        Returns:

        """
        if start_time < self._position_ranges[0][0]:
            return int(np.floor(start_time / 30))
        elif end_time > self._position_ranges[0][0]:
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
        return task_definition, '', True, self._seq2seq

    # THe functions below are mainly used for evaluation
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


class HierarchicalCodeLabelPredictionTask(CodeLabelPredictionTask):
    """
    Make instruction tuning training data from diagnostic codes
    """

    def __init__(
            self,
            encounter_dataframe_process: EncounterDataframeProcess,
            dataframe_sampling: GroupBySampling,
            code_instructions: HierarchicalCodeLabelPredictionInstructionTemplate,
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
            seq2seq_column: str = 'seq2seq',
            position_range_column: str = 'position_range',
            current_bin_value: int = -100,
            prediction_range_limit: int = None,
            seq2seq: bool = True,
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
            seq2seq_column=seq2seq_column,
            current_bin_value=current_bin_value,
            prediction_range_limit=prediction_range_limit,
            fixed_position_range=fixed_position_range,
            seq2seq=seq2seq
        )
        self._trie = negative_code_sampling.get_trie()
        self._tree = negative_code_sampling.get_tree()

    def process_sample(self, sample, args, ignore_instruction=False):
        """
        Process sample

        Args:
            sample:
            args:
            ignore_instruction:

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

        # We will need this to determine whether to ignore instruction for a top level code
        # There can be a scenario where a leaf code never occurs - but say the top
        # code occurs outside the prediction window - in that case we would want to ignore
        # training on that
        full_positive_hash = self.get_positives_hash(codes=encounter_history[self._code_column])

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

            # Positive hash should contain a set of codes. It should contain all the ancestors for a given
            # code - but not their siblings. The ignore_hash is used when the note is labeled with a non leaf
            # code - we want to tell the prompt to not use the leaf codes - because we don't know which leaf code
            # as positive. If there is a code in ignore hash but not in positive hash - don't add that code to the
            # prompt
            positive_hash = self.get_positives_hash(codes=current_time_period[self._code_column])

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

            histories = [negatives, current_time_period]

            # Now that we have sizes - we move on to sampling the codes
            # We try and sample such that we get codes spread across the time range
            # and not just concentrated in one portion of the time range - for both
            # positives and negatives
            sampled_codes = [
                self.sample_codes(encounter_history=encounter_history, max_limit=counts)
                for encounter_history, counts in zip(histories, sample_counts)
            ]

            hierarchies = [
                codes[self._code_column].apply(
                    self.prepare_hierarchy, args=(positive_hash, full_positive_hash)
                ) for codes in sampled_codes
            ]

            # Combine the positive and negative samples
            instruction_samples = pd.concat(hierarchies).sample(frac=1)

            # Convert samples to text instructions (prompt)
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

    def prepare_hierarchy(self, code, positive_hash, full_positive_hash):
        """

        Args:
            code:
            positive_hash:
            full_positive_hash:

        Returns:

        """
        hierarchy = self.get_hierarchy(code=code, positive_hash=positive_hash, full_positive_hash=full_positive_hash)
        hierarchy = self.sample_hierarchy(hierarchy=hierarchy, probabilities=[0.3, 0.7])
        hierarchy = self.process_hierarchy(hierarchy=hierarchy)
        return hierarchy

    def convert_samples_to_instructions(
            self,
            instruction_samples: pd.DataFrame,
    ) -> List[List[Tuple[str, str, bool, bool]]]:
        """
        Return the instruction input and targets
        for the sampled instruction code status classification task with time range

        Args:
            instruction_samples (pd.DataFrame): The samples that will be used as instructions

        Returns:
            (List[Tuple[str, str, bool]]): A list that contains tuples which have the instruction input and target.
            The list will contain only 1 element for zero shot training
        """
        # TODO: Make ignore instruction a bool or a string - if it is a string - we can use as the label?
        #  Or manipulate the label portion of code instruct string
        instructions = list()
        for row in instruction_samples:
            codes, labels, ignore_instructions = zip(*row)
            code_instruct_strings = self._code_instructions.get_instruction(
                diagnosis=codes, value=labels
            )
            instruct_list = [
                code_instruct_string + (ignore_instruction, self._seq2seq)
                for code_instruct_string, ignore_instruction in zip(code_instruct_strings, ignore_instructions)
            ]
            for instruction in instruct_list:
                instructions.append(instruction)
        return instructions

    def get_hierarchy(self, code, positive_hash, full_positive_hash):
        """
        Get twh hierarchy of the codes
        Args:
            code:
            positive_hash:
            full_positive_hash:

        Returns:

        """
        key = '/'.join(self._negative_code_sampling.get_code_representation(code))
        try:
            return [
                (
                    code_obj[1],
                    self.get_true_label_string(code_obj[1] in positive_hash),
                    self.get_ignore_instruction(
                        code=code_obj[1], key=key, positive_hash=positive_hash, full_positive_hash=full_positive_hash
                    )
                )
                for code_obj in self._trie.prefixes(key)
            ]
        except TypeError:
            print(code, positive_hash, full_positive_hash)

    @staticmethod
    def sample_hierarchy(hierarchy, probabilities):
        """
        Sample codes within each hierarchy
        We can get the full trajectory - ancestor to leaf
        or stop at some code in the trajectory in between

        Args:
            hierarchy:
            probabilities:

        Returns:

        """
        hierarchy_length = len(hierarchy)
        # The possible positions we can stop at in the hierarchy
        # We ignore the positions where ignore instruction is True
        indexes = [index for index, code_obj in enumerate(hierarchy) if not code_obj[-1]]
        if indexes:
            end = np.random.choice(indexes)
        else:
            end = hierarchy_length - 1
        end = np.random.choice([end, hierarchy_length - 1], p=probabilities)
        return hierarchy[0:end + 1]

    def process_hierarchy(self, hierarchy):
        """
        Final processing of the hierarchy objects
        Args:
            hierarchy:

        Returns:

        """
        length = len(hierarchy)
        return [
            self.process_code_instruction_tuple(
                code=code_obj[0],
                label=code_obj[1],
                ignore_instruction=code_obj[2],
                add_label_to_input=index == length - 1
            )
            for index, code_obj in enumerate(hierarchy)
        ]

    @staticmethod
    def get_ignore_instruction(code, key, positive_hash, full_positive_hash):
        """
        Check whether to ignore instruction

        Args:
            code:
            key:
            positive_hash:
            full_positive_hash:

        Returns:

        """
        # Adding a bias to ignore top level codes
        # If a code has 5 children - we train (do not ignore instruction)
        # only  1/5 times
        return code not in positive_hash and code in full_positive_hash

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

    def process_code_instruction_tuple(self, code, label, ignore_instruction, add_label_to_input):
        """
        Add label to the code description - modify the label and ignore index accordingly
        Args:
            code:
            label:
            ignore_instruction:
            add_label_to_input:

        Returns:

        """
        code_description = self._code_convert.transform_code(code=code)
        if add_label_to_input:
            code_description = self.add_label_to_input(code=code_description, label=label)
            ignore_instruction = False
        else:
            if np.random.rand() > 1 / max(1, len(self._tree.children(code))):
                ignore_instruction = True
        return code_description, label, ignore_instruction

    @staticmethod
    def add_label_to_input(code, label, sep=' -> '):
        """
        Add the label to the code description

        Args:
            code:
            label:
            sep:

        Returns:

        """
        return f'{code}{sep}{label}'
