"""Make training data"""
import sys
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from ..codes.code_trajectory_instruct_tasks import CodeLabelPredictionTask
from .lab_instruct_tasks import LabPredictionTask
sys.path.append('..')

np.random.seed(42)


class LabPredictionPrompt(LabPredictionTask):
    """
    Make instruction tuning training data from diagnostic codes
    """

    def __init__(
            self,
            lab_instructions,
            lab_dataframe_process: Optional = None,
            time_bins: Optional = None,
            fixed_position_range: bool = False,
            patient_id_column: str = 'PatientID',
            lab_name_column: str = 'ExternalNM',
            position_column: str = 'position',
            bins_column: str = 'bins',
            bin_start_column: str = 'min',
            bin_end_column: str = 'max',
            label_column: str = 'ResultTXT',
            reference_range_low_column: str = 'ReferenceRangeLowNBR',
            reference_range_high_column: str = 'ReferenceRangeHighNBR',
            ignore_instruction_column: str = 'ignore_instruction',
            position_range_column: str = 'position_range',
            seq2seq_column: str = 'seq2seq',
            idf_column: str = 'idf',
            tf_column: str = 'tf',
            weights_column: str = 'weights',
            current_bin_value: int = -100,
            prediction_range_limit: int = None,
            seq2seq: bool = False
    ):
        """
        Initialize variables

        Args:
            lab_dataframe_process (EncounterDataframeProcess): Dataframe processing object
            lab_instructions (Any): A list of code instruction tasks. At any point one of these
            tasks is randomly chosen and trained on
            patient_id_column (str, defaults to `PatientID`): The column name for the patient id
            lab_name_column (str, defaults to `phecode`): The column that contains the codes
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

        super().__init__(
            lab_dataframe_process=lab_dataframe_process,
            lab_instructions=lab_instructions,
            time_bins=time_bins,
            lab_name_column=lab_name_column,
            patient_id_column=patient_id_column,
            position_column=position_column,
            fixed_position_range=fixed_position_range,
            bins_column=bins_column,
            bin_start_column=bin_start_column,
            bin_end_column=bin_end_column,
            label_column=label_column,
            ignore_instruction_column=ignore_instruction_column,
            position_range_column=position_range_column,
            seq2seq_column=seq2seq_column,
            idf_column=idf_column,
            tf_column=tf_column,
            weights_column=weights_column,
            current_bin_value=current_bin_value,
            prediction_range_limit=prediction_range_limit,
            seq2seq=seq2seq
        )
        self._reference_range_low_column = reference_range_low_column
        self._reference_range_high_column = reference_range_high_column

    def process_sample(self, sample, args, ignore_instruction=None, lab=None, prediction_range=None):
        """
        Process sample

        Args:
            sample:
            args:
            ignore_instruction:
            lab:
            prediction_range:

        Returns:

        """

        # Store all the prompt elements (input, output)
        all_instructions = list()

        # Get the task instruction - which should indicate we are making prediction for a given
        # time range
        all_instructions.append(self.get_task_instruction(prediction_range=prediction_range))

        # Convert samples to text instructions (prompt)
        all_instructions.append(
            self.get_lab_prompt(
                lab=lab,
            )
        )

        return all_instructions

    def get_lab_prompt(self, lab):
        return self._code_instructions.get_instruction_input(diagnosis=lab), '', True, self._seq2seq
