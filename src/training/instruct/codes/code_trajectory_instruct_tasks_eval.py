"""Make training data"""
import numpy as np
from typing import Union, Sequence

from .processing.code_convert import ICDConvert, PHEConvert
from .processing.encounter_dataframe_process import EncounterDataframeProcess
from .processing.dataframe_sampling import GroupBySampling
from .processing.data_bins import AgglomerativeDataBins
from .processing.negative_code_sampling import NegativeCodeCacheSampling
from .code_trajectory_instruct_tasks import (
    SingleCodeLabelPredictionTask
)
from .templates import CodeLabelPredictionInstructionTemplate

np.random.seed(42)


class CodeLabelPredictionTaskEvaluation(SingleCodeLabelPredictionTask):
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

        (eval_code, eval_start_time, eval_end_time, eval_label) = (
            args.eval_code, args.eval_start_time, args.eval_end_time, args.eval_label
        )

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
        prediction_range = eval_start_time, eval_end_time

        # Store all the prompt elements (input, output)
        all_instructions = list()

        # Get the task instruction - which should indicate we are making prediction for a given
        # time range
        all_instructions.append(self.get_task_instruction(prediction_range=prediction_range))

        # Get dataframe that contain encounter in the past, current and future time periods
        current_time_period = self.get_current_time_period(
            encounter_history=encounter_history, prediction_range=prediction_range
        )
        current_code_occurrences = self.get_code_occurrences(
            encounter_history=current_time_period, code=eval_code, regex=regex
        )

        label = self.get_label(code_occurrences=current_code_occurrences)

        eval_sample = self.get_code_sample(
            code=eval_code,
            start_time=prediction_range[0],
            end_time=prediction_range[1],
            label=self.get_true_label_string(label=label),
            ignore_instruction=ignore_instruction
        )

        # Convert samples to text instructions (prompt)
        all_instructions.extend(
            self.convert_samples_to_instructions(
                instruction_samples=eval_sample,
            )
        )

        return all_instructions
