"""Make training data"""
import numpy as np
import pandas as pd
from typing import Union, Tuple, List, Sequence

from .processing.code_convert import ICDConvert, PHEConvert
from .processing.encounter_dataframe_process import EncounterDataframeProcess
from .processing.dataframe_sampling import GroupBySampling
from .processing.data_bins import AgglomerativeDataBins
from .processing.negative_code_sampling import NegativeCodeCacheSampling
from .code_trajectory_instruct_tasks import (
    CodeLabelPredictionTask,
)
from .templates import CodeLabelPredictionInstructionTemplate

np.random.seed(42)


class CodeLabelPredictionTaskEvaluation(CodeLabelPredictionTask):
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
            true_label_column_name: str = 'true_label',
            current_bin_value: int = -100,
            prediction_range_limit: int = None
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
            encounter_dataframe_process,
            dataframe_sampling,
            code_instructions,
            code_convert,
            time_bins,
            negative_code_sampling,
            patient_id_column,
            code_column,
            position_column,
            bins_column,
            bin_start_column,
            bin_end_column,
            label_column,
            ignore_instruction_column,
            position_range_column,
            current_bin_value,
            prediction_range_limit
        )
        self._true_label_column_name = true_label_column_name

    def process_sample(self, sample, args):
        """
        Process sample

        Args:
            sample:
            args:

        Returns:

        """

        regex = False

        (eval_code, eval_start_time, eval_end_time, eval_label) = (
            args.eval_code, args.eval_start_time, args.eval_end_time, args.eval_label
        )

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
        prediction_range = eval_start_time, eval_end_time

        # Get dataframe that contain encounter in the past, current and future time periods
        _, current_time_period, _ = self.get_time_periods(
            encounter_history=encounter_history, prediction_range=prediction_range
        )

        prompt_code, true_label = self.get_prompt_code_and_label(
            encounter_history=current_time_period,
            code=eval_code,
            regex=regex
        )

        ignore_instruction = self.get_eval_ignore_instruction(
            encounter_history=encounter_history, eval_code=eval_code, true_label=true_label,
            eval_strict=args.eval_strict, regex=regex
        )

        eval_sample = self.get_eval_sample(
            eval_code=prompt_code,
            eval_start_time=eval_start_time,
            eval_end_time=eval_end_time,
            eval_label=eval_label if eval_label is not None else self.get_true_label_string(true_label=true_label),
            ignore_instruction=ignore_instruction
        )

        return self.convert_samples_to_instructions(eval_sample)

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
            self.check_eval_code_exists(encounter_history=encounter_history, code=code, regex=regex)
        ][self._code_column]
        if len(matched_codes) == 0:
            return np.random.choice(self.get_eval_sub_codes(code=code)), False
        else:
            return matched_codes.sample(1).iloc[0], True

    def get_eval_sample(
            self,
            eval_code: str,
            eval_start_time: int,
            eval_end_time: int,
            eval_label: str,
            ignore_instruction: bool
    ) -> pd.DataFrame:
        """
        Create a row/dataframe that contains details about the evaluation code

        Args:
            eval_code (str): The code to evaluate
            eval_start_time (int): The start time period of the code evaluation prompt
            eval_end_time (int): The end time period of the code evaluation prompt
            eval_label (str): The label to evaluate
            ignore_instruction (bool): Whether to ignore instruction while computing loss etc

        Returns:
            (pd.DataFrame): The dataframe that contains the code, start and end times
        """
        eval_sample = {
            self._code_column: [eval_code],
            self._label_column: [eval_label],
            self._bin_start_column: [eval_start_time],
            self._bin_end_column: [eval_end_time],
            self._ignore_instruction_column: [ignore_instruction]
        }
        return pd.DataFrame(eval_sample)

    def get_true_label_string(self, true_label):
        if true_label:
            return self.get_positive_labels_string()
        else:
            return self.get_negative_labels_string()

    def check_eval_code_exists(self, encounter_history, code, regex):
        return encounter_history[self._code_column].str.contains(code, regex=regex)

    def get_eval_sub_codes(self, code: str) -> List[str]:
        """
        Given a code - return all sub codes of this. This would
        be all codes that contain this string.
        Args:
            code (str): A given diagnostic code

        Returns:
            (List[str]): All sub codes of the given code
        """
        return [sub_code for sub_code in self._negative_code_sampling._codes if code in sub_code]

    def get_eval_ignore_instruction(self, encounter_history, eval_code, true_label, eval_strict, regex):
        if true_label or eval_strict:
            ignore_instruction = False
        else:
            if sum(self.check_eval_code_exists(encounter_history=encounter_history, code=eval_code, regex=regex)):
                ignore_instruction = True
            else:
                ignore_instruction = False
        return ignore_instruction

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