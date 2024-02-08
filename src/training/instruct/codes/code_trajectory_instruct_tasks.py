"""Make training data"""
import random
import numpy as np
import pandas as pd
from typing import Union, Tuple, List, Any, Optional, Sequence

from .processing.code_convert import ICDConvert, PHEConvert
from .processing.encounter_dataframe_process import EncounterDataframeProcess
from .processing.dataframe_sampling import GroupBySampling
from .processing.data_bins import AgglomerativeDataBins
from .processing.negative_code_sampling import NegativeCodeCacheSampling

np.random.seed(42)


class CodeTrajectoryPredictionTask(object):
    """
    Make instruction tuning training data from diagnostic codes
    """

    def __init__(
            self,
            encounter_dataframe_process: EncounterDataframeProcess,
            dataframe_sampling: GroupBySampling,
            code_instructions: Any,
            code_convert: Union[ICDConvert, PHEConvert],
            time_bins: Union[AgglomerativeDataBins, Sequence[AgglomerativeDataBins]],
            patient_id_column: str = 'PatientID',
            code_column: str = 'phecode',
            position_column: str = 'position',
            bins_column: str = 'bins',
            bin_start_column: str = 'min',
            bin_end_column: str = 'max',
            label_column: str = 'count',
            ignore_instruction_column: str = 'ignore_instruction'
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

    def process_sample(self, sample, args):
        """
        Process sample

        Args:
            sample:
            args:

        Returns:

        """

        # Get processed encounter history
        encounter_history = self.get_encounter_history_for_task(
            patient_id=sample[args.patient_id_column],
            current_time=sample[args.sample_result_date_column],
            past_time_delta=args.past_time_delta,
            future_time_delta=args.future_time_delta,
            use_log_position=args.use_log_position,
            time_difference_normalize=args.time_difference_normalize
        )

        instruction_samples = self.get_instruction_samples(
            encounter_history=encounter_history,
            shuffle=args.shuffle_bins,
            sampling_weights=args.trajectory_sampling_weights
        )

        return self.convert_samples_to_instructions(instruction_samples)

    def get_encounter_history_for_task(
            self,
            patient_id: Union[str, int],
            current_time: str,
            use_log_position: bool,
            time_difference_normalize: int,
            past_time_delta: Optional[str] = None,
            future_time_delta: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Given the encounter history, filter out any NA codes and apply a time
        filter if passed and return the resulting dataframe

        Args:
            patient_id (Union[str, int]): The unique id of the patient we need to process
            current_time (str): The timestamp of the sample we are processing - used to calculate time deltas
            with respect to other encounters
            use_log_position (bool): Whether to keep time deltas in days or as log value of days
            time_difference_normalize (int): Normalize time difference by this value
            past_time_delta (str, defaults to `None`): Filter out encounters beyond this point in time
            future_time_delta (str, defaults to `None`): Filter out encounters beyond this point in time

        Returns:
            encounter_history (pd.DataFrame): The processed dataframe for the task
        """

        encounter_history = self.get_full_encounter_history(
            patient_id=patient_id,
            current_time=current_time,
            use_log_position=use_log_position,
            time_difference_normalize=time_difference_normalize
        )

        encounter_history = self.transform_encounter_history_for_task(
            encounter_history=encounter_history,
            past_time_delta=past_time_delta,
            future_time_delta=future_time_delta
        )

        return encounter_history

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
            encounter_history = self._encounter_dataframe_process.filter_encounter_history_time_delta(
                encounter_history=encounter_history,
                past_time_delta=past_time_delta,
                future_time_delta=future_time_delta,
            )

        if len(encounter_history) == 0:
            encounter_history = self.get_random_encounters()
            encounter_history.loc[:, self._bins_column] = 0
        elif len(encounter_history) == 1:
            encounter_history.loc[:, self._bins_column] = 0
        else:
            # Get time bins for each encounter
            encounter_history.loc[:, self._bins_column] = self.get_bins(
                bin_value_column=encounter_history[self._position_column]
            )

        return encounter_history

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
            # TODO: A fix for now - because we have missing data - ideally we should not have missing data
            # Sample codes and create a random encounter dataframe
            return self.get_random_encounters()

    def get_instruction_samples(
            self,
            encounter_history: pd.DataFrame,
            shuffle: bool,
            sampling_weights: Optional[Sequence[float]] = None
    ) -> pd.DataFrame:
        """
        Given the encounter history - generate instruction samples that will be
        used to predict the trajectory of the patient in terms of diagnostic codes.

        Args:
            encounter_history (pd.DataFrame): The encounter history of the patient
            shuffle (bool): Whether to shuffle codes within the bin
            sampling_weights (Optional[Sequence[float]], defaults to `None`): Weights for sampling from trajectory

        Returns:
            (pd.DataFrame): The instructions samples - represents diagnostic history
        """

        # Group based sampling - return the encounter history containing only
        # the groups/bins that are sampled
        sampled_encounter_history = self._dataframe_sampling.sample_dataframe(
            dataframe=encounter_history,
            group_by_column=self._bins_column,
            weights=sampling_weights
        )

        instruction_samples = self.get_grouped_encounter_history(encounter_history=sampled_encounter_history)

        if shuffle:
            instruction_samples = (
                instruction_samples
                .groupby(self._bins_column)
                .sample(frac=1)
                .sort_values(by=self._bin_start_column)
            )

        instruction_samples[self._ignore_instruction_column] = self.get_ignore_instructions(
            instruction_samples=instruction_samples
        )

        return instruction_samples

    def get_grouped_encounter_history(self, encounter_history: pd.DataFrame) -> pd.DataFrame:
        """
        Given a dataframe - group the dataframe and return
        the grouped dataframe - it should contain the counts
        of the codes in the group and start and end positions of
        the group

        Args:
            encounter_history (pd.DataFrame): The input dataframe

        Returns:
            (pd.DataFrame): Grouped encounter history
        """

        group_by_object = encounter_history.groupby(self._bins_column, as_index=False)

        group_code_values = group_by_object[self._code_column].value_counts()
        group_positions = group_by_object[self._position_column].agg(['min', 'max'])

        return pd.merge(group_code_values, group_positions, on=self._bins_column)

    def convert_samples_to_instructions(
            self,
            instruction_samples: pd.DataFrame,
            ignore_time_instruction: Optional[bool] = None
    ) -> List[Tuple[str, str, bool]]:
        """
        Return the instruction input and targets
        for the sampled instruction code status classification task with time range

        Args:
            instruction_samples (pd.DataFrame): The samples that will be used as instructions
            ignore_time_instruction(Optional[bool], defaults to `None`): Ignore the time period
            instruction in output/labels

        Returns:
            (List[Tuple[str, str, bool]]): A list that contains tuples which have the instruction input and target.
            The list will contain only 1 element for zero shot training
        """

        # description_instructions = self.get_code_description_instructions(instruction_samples=instruction_samples)
        trajectory_instructions = self.get_code_trajectory_instructions(
            instruction_samples=instruction_samples,
            ignore_time_instruction=ignore_time_instruction
        )

        return trajectory_instructions

    def get_code_trajectory_instructions(
            self,
            instruction_samples: pd.DataFrame,
            ignore_time_instruction: Optional[bool] = None
    ) -> List[Tuple[str, str, bool]]:
        """
        Return the instruction input and targets
        that contain the time period and say counts of the codes in that time period

        Args:
            instruction_samples (pd.DataFrame): The samples that will be used as instructions
            label to the codes. If None is passed it will use the value stored in self._code_value_column
            ignore_time_instruction(Optional[bool], defaults to `None`): Ignore the time period
            instruction in output/labels

        Returns:
            (List[Tuple[str, str, bool]]): A list that contains tuples which have the instruction input and target.
            The list will contain only 1 element for zero shot training
        """
        instructions = list()
        start_flag = None
        for row in instruction_samples[
            [self._code_column, self._label_column, self._bin_start_column,
             self._bin_end_column, self._ignore_instruction_column]
        ].itertuples(index=False):
            code, code_value, bin_start, bin_end, ignore_instruction = row[0], row[1], row[2], row[3], row[4]

            if start_flag is None or bin_start != start_flag:
                start_flag = bin_start
                time_instruct_string =  self._code_instructions.get_time_instruction(
                    start_time=int(bin_start), end_time=int(bin_end)
                )
                instructions.append(
                    time_instruct_string + (
                        ignore_instruction if ignore_time_instruction is None else ignore_time_instruction,
                    )
                )
                instructions.append(self.get_example_separator_instruction())

            code_instruct_string = self._code_instructions.get_code_instruction(
                diagnosis=self._code_convert.transform_code(code=code), value=code_value
            )
            instructions.append(code_instruct_string + (ignore_instruction,))
            instructions.append(self.get_example_separator_instruction())
        return instructions

    @staticmethod
    def get_example_separator_instruction() -> Tuple[str, str, bool]:
        """
        This instruction is added as a separator between instructions

        Returns:
            (List[str, str, bool]): Example separator instruction
        """
        # \n is the separator string, '' is the label (empty - don't train)
        # True indicates ignore this instruction for training loss
        return '\n', '', True

    # def get_code_description_instructions(self, instruction_samples: pd.DataFrame):
    #     """
    #     Return the instruction input and targets
    #     that contain the unique codes and their descriptions
    #
    #     Args:
    #         instruction_samples (pd.DataFrame): The samples that will be used as instructions
    #
    #     Returns:
    #         (List[Tuple[str, str]]): A list that contains tuples which have the instruction input and target.
    #         The list will contain only 1 element for zero shot training
    #     """
    #     unique_codes = instruction_samples[self._code_column].unique()
    #     return [
    #         self._code_instructions.get_code_description_instruction(
    #             diagnosis=code, description=self._code_convert.transform_code(code=code)
    #         )
    #         for code in unique_codes
    #     ]

    def get_bins(self, bin_value_column: pd.Series) -> pd.Series:
        """
        Given the input dataframe - return the dataframe with each row
        put into specific bins

        Args:
            bin_value_column (pd.DataFrame): The column to use to calculate the bins

        Returns:
            (pd.Series): Computed bins
        """
        time_bins_object = np.random.choice(self._time_bins)
        return time_bins_object.get_bins(bin_value_column)

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

        # Sample negative codes
        codes = ['SS_819']
        # Sample position values
        positions = [-3000]
        # Use these to create and return a dataframe
        return pd.DataFrame(
            {self._code_column: codes, self._position_column: positions}
        )


class CodeTrajectoryPredictionTaskEvaluation(CodeTrajectoryPredictionTask):
    """
    Make instruction tuning training data from diagnostic codes
    """

    def __init__(
            self,
            encounter_dataframe_process: EncounterDataframeProcess,
            dataframe_sampling: GroupBySampling,
            code_instructions: Any,
            code_convert: Union[ICDConvert, PHEConvert],
            time_bins: Union[AgglomerativeDataBins, Sequence[AgglomerativeDataBins]],
            patient_id_column: str = 'PatientID',
            code_column: str = 'phecode',
            position_column: str = 'position',
            bins_column: str = 'bins',
            bin_start_column: str = 'min',
            bin_end_column: str = 'max',
            label_column: str = 'count',
            ignore_instruction_column: str = 'ignore_instruction'
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
            patient_id_column=patient_id_column,
            code_column=code_column,
            position_column=position_column,
            bins_column=bins_column,
            bin_start_column=bin_start_column,
            bin_end_column=bin_end_column,
            label_column=label_column,
            ignore_instruction_column=ignore_instruction_column
        )

    def process_sample(self, sample, args):
        """
        Process sample

        Args:
            sample:
            args:

        Returns:

        """
        future_time_delta = self.get_time_filter_delta(
            start_time=args.eval_start_time,
            time_gap=args.eval_time_gap,
            future_time_delta=args.future_time_delta
        )

        # Get processed encounter history
        encounter_history = self.get_encounter_history_for_task(
            patient_id=sample[args.patient_id_column],
            current_time=sample[args.sample_result_date_column],
            past_time_delta=args.past_time_delta,
            future_time_delta=future_time_delta,
            use_log_position=args.use_log_position,
            time_difference_normalize=args.time_difference_normalize
        )

        eval_sample = self.get_eval_sample(
            eval_code=args.eval_code,
            eval_start_time=args.eval_start_time,
            eval_end_time=args.eval_end_time
        )

        # time delta to start time of code
        instruction_samples = self.get_instruction_samples(
            encounter_history=encounter_history,
            shuffle=args.shuffle_bins,
            sampling_weights=args.trajectory_sampling_weights
        )

        eval_instruction_samples = self.get_eval_instruction_samples(
            instruction_samples=instruction_samples, eval_sample=eval_sample
        )

        return self.convert_samples_to_instructions(
            instruction_samples=eval_instruction_samples,
            ignore_time_instruction=True
        )

    @staticmethod
    def get_time_filter_delta(start_time: int, time_gap: int, future_time_delta) -> str:
        """
        Get a time filter - this will filter out all the encounters after this time
        We take the eval start time and subtract any time delta we want from it
        This will be used to get all encounters until this time delta

        Args:
            start_time (int): The start time period of evaluation
            time_gap (int): A time gap between the start time of evaluation
            and a time delta to the rest of the encounters occurring before it

        Returns:
            (str): A time delta string that can be used by pandas
        """
        if time_gap is None:
            return future_time_delta
        else:
            time_delta = start_time - time_gap
            return f'{time_delta}d'

    def get_eval_sample(self, eval_code: str, eval_start_time: int, eval_end_time: int) -> pd.DataFrame:
        """
        Create a row/dataframe that contains details about the evaluation code

        Args:
            eval_code (str): The code to evaluate
            eval_start_time (int): The start time period of the code evaluation prompt
            eval_end_time (int): The end time period of the code evaluation prompt

        Returns:
            (pd.DataFrame): The dataframe that contains the code, start and end times
        """
        eval_sample = {
            self._code_column: [eval_code],
            self._label_column: [''],
            self._bin_start_column: [eval_start_time],
            self._bin_end_column: [eval_end_time],
            self._ignore_instruction_column: [False]
        }
        return pd.DataFrame(eval_sample)

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


class CodeTrajectoryFutureLabelPredictionTask(CodeTrajectoryPredictionTask):
    """
    Make instruction tuning training data from diagnostic codes
    """

    def __init__(
            self,
            encounter_dataframe_process: EncounterDataframeProcess,
            dataframe_sampling: GroupBySampling,
            code_instructions: Any,
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
            current_bin_value: int = -1,
            current_bin_start_time: int = -7,
            label_type: str = 'multiclass'
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
            label_column (str, defaults to `label`): The column that stores some value that represents
            the code in a time bin
            ignore_instruction_column (str, defaults to `ignore_instruction`): Column that stores whether
            to ignore the instruction in that row
            current_bin_value (int, defaults to `-1`): The value assigned to the bin that contains
            the "current encounters"
            current_bin_start_time (int, defaults to `-7`): Anything less than this value is considered outside the
            current bin and belongs to the past
            label_type (str, defaults to 'multiclass'): The label type defines the labels to  train on
        """
        super().__init__(
            encounter_dataframe_process=encounter_dataframe_process,
            dataframe_sampling=dataframe_sampling,
            code_instructions=code_instructions,
            code_convert=code_convert,
            time_bins=time_bins,
            patient_id_column=patient_id_column,
            code_column=code_column,
            position_column=position_column,
            bins_column=bins_column,
            bin_start_column=bin_start_column,
            bin_end_column=bin_end_column,
            label_column=label_column,
            ignore_instruction_column=ignore_instruction_column
        )
        self._negative_code_cache_sampling = negative_code_sampling
        self._position_range_column = position_range_column
        self._current_bin_value = current_bin_value
        self._current_bin_start_time = current_bin_start_time
        self._position_ranges = [(0, 30), (30, 90), (90, 180), (180, 365), (365, np.inf)]
        self._labels = self.get_label_type_labels(label_type=label_type)

    def process_sample(self, sample, args):
        """
        Process sample

        Args:
            sample:
            args:

        Returns:

        """

        sample_size_choices = np.arange(4)

        # Get the full encounter history
        encounter_history = self.get_full_encounter_history(
            patient_id=sample[args.patient_id_column],
            current_time=sample[args.sample_result_date_column],
            use_log_position=args.use_log_position,
            time_difference_normalize=args.time_difference_normalize
        )

        # Get a dataframe that can be used to sample negatives
        encounter_negatives = self._negative_code_cache_sampling.get_code_task_encounter_negatives_from_cache_and_update(
            encounter_history=encounter_history,
            patient_id=sample[args.patient_id_column],
            exclude_codes=self.get_exclude_codes_for_negatives(encounter_history=encounter_history),
            code_column=self._code_column,
            position_column=self._position_column
        )

        # Get processed encounter history - add position ranges and cluster/bin start and columns
        encounter_history = self.transform_encounter_history_for_task(
            encounter_history=encounter_history,
            past_time_delta=args.past_time_delta,
            future_time_delta=args.future_time_delta,
        )

        #  Get the bin period we will be making predictions for
        current_bin_period = self.get_current_bin_period(encounter_history)

        # Assign a different number to all the codes belonging to the current bin
        encounter_history = self.set_current_bins(encounter_history, current_bin_period)

        # Get dataframe that contain encounter in the past, current and future time periods
        past_time_period, current_time_period, future_time_period = self.get_time_periods(
            encounter_history=encounter_history, current_bin_period=current_bin_period
        )

        # Get the encounters that will contain the positive and negative labels
        positive_labels = self.get_positive_labels(
            past_time_period=past_time_period,
            current_time_period=current_time_period,
            future_time_period=future_time_period,
            sample_size_choices=sample_size_choices
        )
        negative_labels = self.get_negative_labels(
            encounter_negatives=encounter_negatives,
            current_bin_period=current_bin_period,
            sample_size_choices=sample_size_choices
        )

        # Filter and sample the encounter history before being used for trajectory
        encounter_history_filtered = self.apply_encounter_history_filters(
            encounter_history=encounter_history,
            current_bin_period=current_bin_period,
            labels=positive_labels[self._code_column],
            sampling_weights=args.trajectory_sampling_weights
        )

        # Convert trajectories, positive and negative labels to instruction samples
        # We've already sampled our trajectory samples - so set weight of full
        # trajectory to 1 - so we don't sample again
        trajectory_samples = self.get_instruction_samples(
            encounter_history=encounter_history_filtered,
            shuffle=args.shuffle_bins,
            sampling_weights=[0.0, 0.0, 1.0]
        )
        positive_label_samples = self.get_label_instruction_samples(
            labels=positive_labels, current_bin_period=current_bin_period
        )
        negative_label_samples = self.get_label_instruction_samples(
            labels=negative_labels, current_bin_period=current_bin_period
        )
        label_samples = self.combine_positive_and_negative_labels(
            positive_labels=positive_label_samples, negative_labels=negative_label_samples
        )

        # Convert the instruction samples to instruction tuples
        trajectory_instructions = self.convert_samples_to_instructions(trajectory_samples)
        label_instructions = self.convert_samples_to_instructions(label_samples, ignore_time_instruction=True)
        prediction_start_instruction = self.get_prediction_start_instruction()

        instructions = trajectory_instructions + prediction_start_instruction + label_instructions

        return instructions

    def get_positive_labels(self, past_time_period, current_time_period, future_time_period, sample_size_choices):

        return self.combine_sampled_positive_labels(
            sampled_codes_for_labels=self.get_labels(
                past_time_period, current_time_period, future_time_period, sample_size_choices
            )
        )

    def get_negative_labels(self, encounter_negatives, current_bin_period, sample_size_choices):

        past_time_period, current_time_period, future_time_period = self.get_time_periods(
            encounter_history=encounter_negatives, current_bin_period=current_bin_period
        )

        negative_labels = self.combine_sampled_negative_labels(
            sampled_codes_for_labels=self.get_labels(
                past_time_period, current_time_period, future_time_period, sample_size_choices
            )
        )

        negative_labels[self._label_column] = 'never'

        return negative_labels
    
    def get_labels(
            self,
            past_time_period,
            current_time_period,
            future_time_period,
            sample_size_choices
    ):
        # Get the set (unique) codes belonging to each time period
        past_time_period_codes = set(past_time_period[self._code_column])
        current_time_period_codes = set(current_time_period[self._code_column])
        future_time_period_codes = set(future_time_period[self._code_column])
        
        # Get the codes belonging to each period according to the labels
        # defined - for example - past only should contain codes that occur
        # in the past and no other period. Similarly current_future should
        # contain codes that occur in the current time period and the future
        # time period and nowhere else.
        (
            past_only_codes,
            current_only_codes,
            future_only_codes,
            past_current_codes,
            current_future_codes,
            past_future_codes,
            past_current_future_codes
        ) = self.get_codes_for_labels(
            past_time_period_codes=past_time_period_codes,
            current_time_period_codes=current_time_period_codes,
            future_time_period_codes=future_time_period_codes
        )

        sampled_codes_for_labels = self.sample_codes_for_labels(
            past_time_period=past_time_period,
            current_time_period=current_time_period,
            future_time_period=future_time_period,
            past_only_codes=past_only_codes,
            current_only_codes=current_only_codes,
            future_only_codes=future_only_codes,
            past_current_codes=past_current_codes,
            current_future_codes=current_future_codes,
            past_future_codes=past_future_codes,
            past_current_future_codes=past_current_future_codes,
            sample_size_choices=sample_size_choices
        )

        return sampled_codes_for_labels

    def combine_sampled_positive_labels(self, sampled_codes_for_labels):
        sampled_codes_for_labels = [
            self.add_label_column(dataframe=dataframe, label=label)
            for label, dataframe in zip(self._labels, sampled_codes_for_labels) if label is not None
        ]

        return pd.concat(sampled_codes_for_labels)

    def combine_sampled_negative_labels(self, sampled_codes_for_labels):
        sampled_codes_for_labels = [
            self.add_label_column(dataframe=dataframe, label=label)
            for label, dataframe in zip(self._labels, sampled_codes_for_labels) if label is not None
        ]

        return pd.concat(sampled_codes_for_labels)

    def combine_positive_and_negative_labels(self, positive_labels, negative_labels):
        return self.concatenate_instruction_samples(
            sampled_positives=positive_labels, sampled_negatives=negative_labels
        ).sample(frac=1)

    def get_instruction_samples(
            self,
            encounter_history: pd.DataFrame,
            shuffle: bool,
            sampling_weights: Optional[Sequence[float]] = None
    ) -> pd.DataFrame:
        """
        Given the encounter history - generate instruction samples that will be
        used to predict the trajectory of the patient in terms of diagnostic codes.

        Args:
            encounter_history (pd.DataFrame): The encounter history of the patient
            shuffle (bool): Whether to shuffle codes within the bin
            sampling_weights (Optional[Sequence[float]], defaults to `None`): Weights for sampling from trajectory

        Returns:
            (pd.DataFrame): The instructions samples - represents diagnostic history
        """

        # Group based sampling - return the encounter history containing only
        # the groups/bins that are sampled
        instruction_samples = super().get_instruction_samples(
            encounter_history=encounter_history,
            shuffle=shuffle,
            sampling_weights=sampling_weights
        )

        return instruction_samples.rename(columns={'count': self._label_column})

    def get_label_instruction_samples(self, labels, current_bin_period):

        instruction_samples = labels.loc[:, [self._code_column, self._label_column]]
        instruction_samples[self._bin_start_column] = current_bin_period[0]
        instruction_samples[self._bin_end_column] = current_bin_period[1]
        instruction_samples[self._ignore_instruction_column] = False
        return instruction_samples.sample(frac=1.0)

    def apply_encounter_history_filters(
            self,
            encounter_history: pd.DataFrame,
            current_bin_period: Tuple[int, int],
            labels: pd.Series,
            sampling_weights
    ) -> pd.DataFrame:
        """
        When training trajectories - we want to first remove encounters that occur in the future.
        The trajectory should only contain encounters that occur in the past - until the start
        of the current encounter period
        
        Args:
            encounter_history (pd.DataFrame):
            current_bin_period: 
            labels:
            sampling_weights:

        Returns:

        """

        # Remove all encounters that occur in the future
        encounter_history = self.filter_future_encounter_history(
            encounter_history=encounter_history,
            current_bin_period=current_bin_period
        )

        # We can sometimes provide the trajectory as context for making predictions in the current
        # bin period - and sometimes we don't provide any context - basically use or don't use past
        # encounters as context/trajectory
        encounter_history = self.filter_trajectory_prediction(
            encounter_history=encounter_history,
            sampling_weights=sampling_weights
        )

        # When using trajectory - we can keep all the label codes in the trajectory - which
        # might make it easier to guess the answer - we can remove all label codes from
        # the trajectory, or we can remove a subset of the label codes from the trajectory
        encounter_history = self.filter_labels_from_encounter_history(
            encounter_history=encounter_history, labels=labels
        )

        # If some cluster is cut across the current and past-time period - we set the bin/cluster end time
        # of the encounters in the past to be 1 less than the start of the ccurrent time period
        (
            encounter_history.loc[encounter_history[self._bin_end_column] > current_bin_period[0], self._bin_end_column]
        ) = (current_bin_period[0] - 1)

        return encounter_history

    def filter_future_encounter_history(self, encounter_history, current_bin_period):
        return encounter_history[encounter_history[self._position_column] < current_bin_period[0]]

    def filter_trajectory_prediction(self, encounter_history, sampling_weights):

        if np.random.random() < sampling_weights[0]:
            return encounter_history.iloc[0:0]
        else:
            weights = np.array([0] + sampling_weights[1:])
            weights = weights/ weights.sum()
            return self._dataframe_sampling.sample_dataframe(
                dataframe=encounter_history,
                group_by_column=self._bins_column,
                weights=weights
            )

    def filter_labels_from_encounter_history(self, encounter_history, labels):
        labels_filter = self.sample_label_filters_for_encounters(labels)
        return encounter_history[~encounter_history[self._code_column].isin(labels_filter)]

    def sample_label_filters_for_encounters(self, labels):
        sampling_percentage = np.random.choice([0, np.random.choice(np.arange(0.1, 1, 0.1)), 1])
        return labels.sample(frac=sampling_percentage)

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

        encounter_history = super().transform_encounter_history_for_task(
            encounter_history=encounter_history,
            past_time_delta=past_time_delta,
            future_time_delta=future_time_delta
        )

        # Assign samples to one of the position ranges defined earlier
        # We use these ranges when sampling codes - to get codes from different ranges
        encounter_history.loc[:, self._position_range_column] = encounter_history[self._position_column].apply(
            self.assign_position_range
        )

        # Assign the start and end positions for each code based on the cluster they belong too
        encounter_history.loc[:, self._bin_start_column] = (
            encounter_history.groupby(self._bins_column)[self._position_column]
            .transform('min')
        )

        encounter_history.loc[:, self._bin_end_column] = (
            encounter_history.groupby(self._bins_column)[self._position_column]
            .transform('max')
        )

        return encounter_history

    def assign_position_range(self, position: Union[int, float]) -> int:
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
            if position_range[0] <= abs(position) <= position_range[1]:
                if position < 0:
                    return -index - 1
                else:
                    return index + 1

    def get_current_bin_period(self, encounter_history: pd.DataFrame) -> Tuple[int, int]:
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
        ranges = self.get_ranges_for_current_bin(position_ranges=encounter_history[self._position_range_column])

        # If there are no samples in the future - return a (0, 30) range
        if len(ranges)  == 0:
            return 0, 30

        # There are many possible current bins - keep sampling until we sample
        # a correct one
        while True:
            sampled_range = random.choice(ranges)
            # Sometimes codes part of the same bin/cluster may be split across two different
            # ranges - to avoid sampling the code that doesn't represent the end of the bin
            # we have the following filter
            bins_in_range = encounter_history[
                (encounter_history[self._position_range_column] == sampled_range)
                & (encounter_history[self._position_column] == encounter_history[self._bin_end_column])
                ]
            if bins_in_range.empty:
                continue
            sampled_bin = (
                bins_in_range[
                    bins_in_range[self._bins_column] == np.random.choice(bins_in_range[self._bins_column].unique())
                    ]
            )
            if not sampled_bin.empty:
                break
        return (
            min(0, max(self._current_bin_start_time, sampled_bin[self._bin_start_column].iloc[0])),
            sampled_bin[self._bin_end_column].iloc[0]
        )

    def get_ranges_for_current_bin(self, position_ranges):
        ranges = (
            position_ranges[position_ranges >= 1]
            .unique()
        )
        return ranges

    def set_current_bins(self, encounter_history: pd.DataFrame, current_bin_period: Tuple[int, int]) -> pd.DataFrame:
        """
        Change the bin value for all the codes/encounters belonging to the current bin period

        Args:
            encounter_history (pd.DataFrame): The encounter dataframe
            current_bin_period (Tuple[int, int]): The start and end positions of the current bin period

        Returns:
            encounter_history (pd.DataFrame): The encounter history with a modified bins column
        """
        encounter_history.loc[
            (encounter_history[self._position_column] >= current_bin_period[0])
            & (encounter_history[self._position_column] <= current_bin_period[1]), self._bins_column
        ] = self._current_bin_value
        return encounter_history

    def get_time_periods(
            self, encounter_history: pd.DataFrame, current_bin_period: Tuple[int, int]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Return three dataframes - one which contains encounters before the current bin period,
        one which contains encounters belonging to the current bin period and one which contains
        encounters after the current bin period

        Args:
            encounter_history (pd.DataFrame): The encounter dataframe
            current_bin_period (Tuple[int, int]): The start and end positions of the current bin period

        Returns:
            (Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]): Dataframes that contains encounter before,
            during and after the current bin
        """
        past_time_period = self.get_past_time_period(encounter_history, current_bin_period)
        current_time_period = self.get_current_time_period(encounter_history, current_bin_period)
        future_time_period = self.get_future_time_period(encounter_history, current_bin_period)
        return past_time_period, current_time_period, future_time_period

    def get_past_time_period(
            self, encounter_history: pd.DataFrame, current_bin_period: Tuple[int, int]
    ) -> pd.DataFrame:
        return (
            encounter_history[encounter_history[self._position_column] < current_bin_period[0]]
        )

    def get_current_time_period(
            self, encounter_history: pd.DataFrame, current_bin_period: Tuple[int, int]
    ) -> pd.DataFrame:
        return (
            encounter_history[
                (encounter_history[self._position_column] >= current_bin_period[0])
                & (encounter_history[self._position_column] <= current_bin_period[1])
                ]
        )

    def get_future_time_period(
            self, encounter_history: pd.DataFrame, current_bin_period
    ) -> pd.DataFrame:
        return (
            encounter_history[encounter_history[self._position_column] > current_bin_period[1]]
        )

    def sample_label_codes_from_dataframe(
            self,
            dataframe: pd.DataFrame,
            codes: set,
            sample_size: int,
            ascending: bool
    ) -> pd.DataFrame:
        """
        Given a dataframe - get only the portion of the dataframe
        that contains the codes - and then sample a subset of rows
        from the resulting dataframe.

        Args:
            dataframe (pd.DataFrame): A dataframe that contains encounters
            codes (set): The set of codes we want to sample
            sample_size (int): The size of the sample
            ascending (bool): Whether to sort position values

        Returns:
            (pd.DataFrame): A sampled version of the dataframe that contains the codes/labels.
        """
        # Drop duplicates of code - remove bias of codes that occur frequently
        # Also by sorting wee always return the latest encounter of that code
        sample_dataframe = (
            dataframe[dataframe[self._code_column].isin(codes)]
            .sort_values(self._position_column, ascending=ascending)
            .drop_duplicates(self._code_column)
        )
        return sample_dataframe.sample(min(sample_size, len(sample_dataframe)))

    def sample_codes_for_labels(
            self,
            past_time_period,
            current_time_period,
            future_time_period,
            past_only_codes,
            current_only_codes,
            future_only_codes,
            past_current_codes,
            current_future_codes,
            past_future_codes,
            past_current_future_codes,
            sample_size_choices
    ):

        sampled_past_only_codes = self.sample_label_codes_from_dataframe(
            dataframe=past_time_period,
            codes=past_only_codes,
            sample_size=np.random.choice(sample_size_choices),
            ascending=True
        )

        sampled_current_only_codes = self.sample_label_codes_from_dataframe(
            dataframe=current_time_period,
            codes=current_only_codes,
            sample_size=np.random.choice(sample_size_choices),
            ascending=False
        )

        sampled_future_only_codes = self.sample_label_codes_from_dataframe(
            dataframe=future_time_period,
            codes=future_only_codes,
            sample_size=np.random.choice(sample_size_choices),
            ascending=False
        )

        sampled_past_current_codes = self.sample_label_codes_from_dataframe(
            dataframe=current_time_period,
            codes=past_current_codes,
            sample_size=np.random.choice(sample_size_choices),
            ascending=False
        )

        sampled_current_future_codes = self.sample_label_codes_from_dataframe(
            dataframe=current_time_period,
            codes=current_future_codes,
            sample_size=np.random.choice(sample_size_choices),
            ascending=False
        )

        sampled_past_future_codes = self.sample_label_codes_from_dataframe(
            dataframe=future_time_period,
            codes=past_future_codes,
            sample_size=np.random.choice(sample_size_choices),
            ascending=False
        )

        sampled_past_current_future_codes = self.sample_label_codes_from_dataframe(
            dataframe=current_time_period,
            codes=past_current_future_codes,
            sample_size=np.random.choice(sample_size_choices),
            ascending=False
        )

        return (
            sampled_past_only_codes,
            sampled_current_only_codes,
            sampled_future_only_codes,
            sampled_past_current_codes,
            sampled_current_future_codes,
            sampled_past_future_codes,
            sampled_past_current_future_codes
        )

    def get_codes_for_labels(
            self,
            past_time_period_codes: set,
            current_time_period_codes: set,
            future_time_period_codes: set
    ) -> Tuple[set, set, set, set, set, set, set]:
        """
        Given codes from different time periods - Get the codes belonging to each period
        according to the labels defined - for example - past only should contain codes that occur
        in the past and no other period. Similarly current_future should contain codes that occur
        in the current time period and the future time period and nowhere else.

        Args:
            past_time_period_codes (set): The unique codes occurring in the past
            current_time_period_codes (set): The unique codes occurring in the current
            future_time_period_codes (set): The unique codes occurring in the future

        Returns:

        """


        # Get all the codes
        past_only_codes = self.get_past_only_codes(
            past_time_period_codes=past_time_period_codes,
            current_time_period_codes=current_time_period_codes,
            future_time_period_codes=future_time_period_codes
        )

        current_only_codes = self.get_current_only_codes(
            past_time_period_codes=past_time_period_codes,
            current_time_period_codes=current_time_period_codes,
            future_time_period_codes=future_time_period_codes
        )

        future_only_codes = self.get_future_only_codes(
            past_time_period_codes=past_time_period_codes,
            current_time_period_codes=current_time_period_codes,
            future_time_period_codes=future_time_period_codes
        )

        past_current_codes = self.get_past_current_codes_codes(
            past_time_period_codes=past_time_period_codes,
            current_time_period_codes=current_time_period_codes,
            future_time_period_codes=future_time_period_codes
        )

        current_future_codes = self.get_current_future_codes(
            past_time_period_codes=past_time_period_codes,
            current_time_period_codes=current_time_period_codes,
            future_time_period_codes=future_time_period_codes
        )

        past_future_codes = self.get_past_future_codes(
            past_time_period_codes=past_time_period_codes,
            current_time_period_codes=current_time_period_codes,
            future_time_period_codes=future_time_period_codes
        )

        past_current_future_codes = self.get_past_current_future_codes(
            past_time_period_codes=past_time_period_codes,
            current_time_period_codes=current_time_period_codes,
            future_time_period_codes=future_time_period_codes
        )

        return (
            past_only_codes,
            current_only_codes,
            future_only_codes,
            past_current_codes,
            current_future_codes,
            past_future_codes,
            past_current_future_codes
        )

    @staticmethod
    def get_past_only_codes(past_time_period_codes, current_time_period_codes, future_time_period_codes):
        return past_time_period_codes.difference(current_time_period_codes).difference(future_time_period_codes)

    @staticmethod
    def get_current_only_codes(past_time_period_codes, current_time_period_codes, future_time_period_codes):
        return current_time_period_codes.difference(past_time_period_codes).difference(future_time_period_codes)

    @staticmethod
    def get_future_only_codes(past_time_period_codes, current_time_period_codes, future_time_period_codes):
        return future_time_period_codes.difference(current_time_period_codes).difference(past_time_period_codes)

    @staticmethod
    def get_past_current_codes_codes(past_time_period_codes, current_time_period_codes, future_time_period_codes):
        return past_time_period_codes.intersection(current_time_period_codes).difference(future_time_period_codes)

    @staticmethod
    def get_current_future_codes(past_time_period_codes, current_time_period_codes, future_time_period_codes):
        return current_time_period_codes.intersection(future_time_period_codes).difference(past_time_period_codes)

    @staticmethod
    def get_past_future_codes(past_time_period_codes, current_time_period_codes, future_time_period_codes):
        return past_time_period_codes.intersection(future_time_period_codes).difference(current_time_period_codes)

    @staticmethod
    def get_past_current_future_codes(past_time_period_codes, current_time_period_codes, future_time_period_codes):
        return past_time_period_codes.intersection(current_time_period_codes).intersection(future_time_period_codes)

    @staticmethod
    def get_label_type_labels(label_type: str) -> List[str]:
        """
        Define the training labels based on the label type

        Args:
            label_type (str): The choices are binary, binary_strict or multiclass

        Returns:
            (List[str]): The list of training labels basd on the label type
        """
        if label_type == 'binary':
            # None would mean to ignore those samples for training
            return [
                None, 'yes', None, 'yes',
                'yes', None, 'yes'
            ]
        elif label_type == 'binary_strict':
            # None would mean to ignore those samples for training
            return [
                'no', 'yes', 'no', 'yes',
                'yes', 'no', 'yes'
            ]
        elif label_type == 'multiclass':
            return [
                'past', 'current', 'future', 'past & current',
                'current & future', 'past & future', 'past & current & future'
            ]

    @staticmethod
    def get_prediction_start_instruction() -> List[Tuple[str, str, bool]]:
        """
        This instruction is added as a separator between the trajectory
        and the start of the predicting/training tokens

        Returns:
            (List[Tuple[str, str, bool]]): Training/Prediction start instruction
        """
        # True indicates ignore this instruction for training loss
        return [('|ECG test|\n', '', True)]

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
        return [True] * len(instruction_samples)

    def add_label_column(self, dataframe, label):
        dataframe[self._label_column] = label
        return dataframe

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
        return encounter_history[self._code_column].unique()


class CodeTrajectoryAnyLabelPredictionTask(CodeTrajectoryFutureLabelPredictionTask):
    """
    Make instruction tuning training data from diagnostic codes
    """

    def __init__(
            self,
            encounter_dataframe_process: EncounterDataframeProcess,
            dataframe_sampling: GroupBySampling,
            code_instructions: Any,
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
            current_bin_value: int = -1,
            current_bin_start_time: int = -7,
            label_type: str = 'multiclass'
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
            label_column (str, defaults to `label`): The column that stores some value that represents
            the code in a time bin
            ignore_instruction_column (str, defaults to `ignore_instruction`): Column that stores whether
            to ignore the instruction in that row
            current_bin_value (int, defaults to `-1`): The value assigned to the bin that contains
            the "current encounters"
            current_bin_start_time (int, defaults to `-7`): Anything less than this value is considered outside the
            current bin and belongs to the past
            label_type (str, defaults to 'multiclass'): The label type defines the labels to  train on
        """
        super().__init__(
            encounter_dataframe_process=encounter_dataframe_process,
            dataframe_sampling=dataframe_sampling,
            code_instructions=code_instructions,
            code_convert=code_convert,
            time_bins=time_bins,
            patient_id_column=patient_id_column,
            code_column=code_column,
            position_column=position_column,
            bins_column=bins_column,
            bin_start_column=bin_start_column,
            bin_end_column=bin_end_column,
            label_column=label_column,
            ignore_instruction_column=ignore_instruction_column,
            negative_code_sampling=negative_code_sampling,
            position_range_column=position_range_column,
            current_bin_value=current_bin_value,
            current_bin_start_time=current_bin_start_time,
            label_type=label_type
        )

    def get_ranges_for_current_bin(self, position_ranges):
        return position_ranges.unique()

