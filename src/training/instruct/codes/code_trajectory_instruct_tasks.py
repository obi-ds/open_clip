"""Make training data"""
import numpy as np
import pandas as pd
from typing import Union, Tuple, List, Any, Optional, Sequence

from .processing.code_convert import ICDConvert, PHEConvert
from .processing.encounter_dataframe_process import EncounterDataframeProcess
from .processing.dataframe_sampling import GroupBySampling
from .processing.data_bins import AgglomerativeDataBins

np.random.seed(42)


class CodeTrajectoryPredictionTask(object):
    """
    Make instruction tuning training data from diagnostic codes
    # TODO: Sampling whole bins v/s within bins,
    # TODO: Pre-train LM
    # TODO: Generative pre-train  v/s instruction pre-train
    """

    def __init__(
            self,
            encounter_dataframe_process: EncounterDataframeProcess,
            dataframe_sampling: GroupBySampling,
            code_instructions: Any,
            code_convert: Union[ICDConvert, PHEConvert],
            time_bins: Union[AgglomerativeDataBins, Sequence[AgglomerativeDataBins]],
            patient_id_column: str = 'PatientID',
            code_column: str = 'ICD10CD',
            position_column: str = 'position',
            bins_column: str = 'bins',
            bin_start_column: str = 'min',
            bin_end_column: str = 'max',
            code_value_column: str = 'count',
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
            code_column (str, defaults to `ICD10CD`): The column that contains the codes
            position_column (str, defaults to `positions`): The column where position values
            (e.g. time difference in days or log days) will be stored
            bins_column (str, defaults to `bins`): Column that stores the assigned bin
            bin_start_column (str, defaults to `min`): Column that stores the start position of each bin
            bin_end_column (str, defaults to `max`): Column that stores the end position of each bin
            code_value_column (str, defaults to `count`): The column that stores some value that represents
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
        self._code_value_column = code_value_column
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
            shuffle=args.shuffle_bins
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

        # Time filter
        if past_time_delta is not None or future_time_delta is not None:
            encounter_history = self._encounter_dataframe_process.filter_encounter_history_time_delta(
                encounter_history=encounter_history,
                past_time_delta=past_time_delta,
                future_time_delta=future_time_delta,
            )

        # Drop NA codes
        encounter_history.dropna(subset=self._code_column, inplace=True)

        if len(encounter_history) == 0:
            encounter_history = self.get_random_encounters()
            encounter_history[self._bins_column] = 0
        elif len(encounter_history) == 1:
            encounter_history[self._bins_column] = 0
        else:
            # Get time bins for each encounter
            encounter_history[self._bins_column] = self.get_bins(
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
            return self._encounter_dataframe_process.get_patient_encounter_history_with_position(
                patient_id=patient_id,
                current_time=current_time,
                use_log_position=use_log_position,
                time_difference_normalize=time_difference_normalize
            )
        except KeyError:
            # TODO: A fix for now - because we have missing data - ideally we should not have missing data
            # Sample codes and create a random encounter dataframe
            return self.get_random_encounters()

    def get_instruction_samples(
            self,
            encounter_history: pd.DataFrame,
            shuffle: bool
    ) -> pd.DataFrame:
        """
        Given the encounter history - generate instruction samples that will be
        used to predict the trajectory of the patient in terms of diagnostic codes.

        Args:
            encounter_history (pd.DataFrame): The encounter history of the patient
            shuffle (bool): Whether to shuffle codes within the bin

        Returns:
            (pd.DataFrame): The instructions samples - represents diagnostic history
        """

        # Group based sampling - return the encounter history containing only
        # the groups/bins that are sampled
        sampled_encounter_history = self._dataframe_sampling.sample_dataframe(
            dataframe=encounter_history,
            group_by_column=self._bins_column
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

    def get_grouped_encounter_history(self, encounter_history):
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
    ) -> List[Tuple[str, str]]:
        """
        Return the instruction input and targets
        for the sampled instruction code status classification task with time range

        Args:
            instruction_samples (pd.DataFrame): The samples that will be used as instructions
            ignore_time_instruction(Optional[bool], defaults to `None`): Ignore the time period
            instruction in output/labels

        Returns:
            (List[Tuple[str, str]]): A list that contains tuples which have the instruction input and target.
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
    ):
        """
        Return the instruction input and targets
        that contain the time period and say counts of the codes in that time period

        Args:
            instruction_samples (pd.DataFrame): The samples that will be used as instructions
            ignore_time_instruction(Optional[bool], defaults to `None`): Ignore the time period
            instruction in output/labels

        Returns:
            (List[Tuple[str, str]]): A list that contains tuples which have the instruction input and target.
            The list will contain only 1 element for zero shot training
        """
        instructions = list()
        start_flag = None
        for row in instruction_samples[
            [self._code_column, self._code_value_column, self._bin_start_column,
             self._bin_end_column, self._ignore_instruction_column]
        ].itertuples(index=False):
            code, code_value, bin_start, bin_end, ignore_instruction = row[0], row[1], row[2], row[3], row[4]

            if start_flag is None or bin_start != start_flag:
                start_flag = bin_start
                # TODO: Round time
                time_instruct_string =  self._code_instructions.get_time_instruction(
                    start_time=int(bin_start), end_time=int(bin_end)
                )
                instructions.append(
                    time_instruct_string + (
                        ignore_instruction if ignore_time_instruction is None else ignore_time_instruction,
                    )
                )

            code_instruct_string = self._code_instructions.get_code_instruction(
                diagnosis=self._code_convert.transform_code(code=code), value=code_value
            )
            instructions.append(code_instruct_string + (ignore_instruction,))
        return instructions

    def get_code_description_instructions(self, instruction_samples: pd.DataFrame):
        """
        Return the instruction input and targets
        that contain the unique codes and their descriptions

        Args:
            instruction_samples (pd.DataFrame): The samples that will be used as instructions

        Returns:
            (List[Tuple[str, str]]): A list that contains tuples which have the instruction input and target.
            The list will contain only 1 element for zero shot training
        """
        unique_codes = instruction_samples[self._code_column].unique()
        return [
            self._code_instructions.get_code_description_instruction(
                diagnosis=code, description=self._code_convert.transform_code(code=code)
            )
            for code in unique_codes
        ]

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
            code_column: str = 'ICD10CD',
            position_column: str = 'position',
            bins_column: str = 'bins',
            bin_start_column: str = 'min',
            bin_end_column: str = 'max',
            code_value_column: str = 'count',
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
            code_column (str, defaults to `ICD10CD`): The column that contains the codes
            position_column (str, defaults to `positions`): The column where position values
            (e.g. time difference in days or log days) will be stored
            bins_column (str, defaults to `bins`): Column that stores the assigned bin
            bin_start_column (str, defaults to `min`): Column that stores the start position of each bin
            bin_end_column (str, defaults to `max`): Column that stores the end position of each bin
            code_value_column (str, defaults to `count`): The column that stores some value that represents
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
            code_value_column=code_value_column,
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

        # time_filter_delta = self.get_time_filter_delta(
        #     start_time=args.eval_start_time,
        #     time_gap=args.eval_time_gap
        # )

        # Get processed encounter history
        encounter_history = self.get_encounter_history_for_task(
            patient_id=sample[args.patient_id_column],
            current_time=sample[args.sample_result_date_column],
            past_time_delta=args.past_time_delta,
            future_time_delta=args.future_time_delta,
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
            shuffle=args.shuffle_bins
        )

        eval_instruction_samples = self.get_eval_instruction_samples(
            instruction_samples=instruction_samples, eval_sample=eval_sample
        )

        return self.convert_samples_to_instructions(
            instruction_samples=eval_instruction_samples,
            ignore_time_instruction=True
        )

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
            self._code_value_column: [''],
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

    @staticmethod
    def get_time_filter_delta(start_time: int, time_gap: int) -> str:
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
        time_delta = start_time - time_gap
        return f'{time_delta}d'

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
