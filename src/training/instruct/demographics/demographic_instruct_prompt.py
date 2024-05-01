"""Make training data"""
import random
import numpy as np
import pandas as pd
from dateutil import relativedelta
from typing import Union, Tuple, List, Optional

from .processing.demographic_dataframe_process import DemographicDataframeProcess
from .templates import PatientDemographicsTemplate
from .demographic_instruct_tasks import DemographicPredictionTask

np.random.seed(42)


class DemographicPredictionPrompt(DemographicPredictionTask):
    """
    Make instruction tuning training data from diagnostic codes
    """

    def __init__(
            self,
            demographic_instructions: PatientDemographicsTemplate,
            demographic_dataframe_process: Optional[DemographicDataframeProcess] = None,
            patient_id_column: str = 'PatientID',
            birth_date_column: str = 'BirthDTS',
            sex_column: str = 'SexDSC',
            race_column: str = 'PatientRaceDSC',
            height_column: str = 'HeightIN',
            weight_column: str = 'WeightLBS',
            label_column: str = 'label',
            ignore_instruction_column: str = 'ignore_instruction',
            seq2seq_column: str = 'seq2seq',
            seq2seq: bool = False
    ):
        """
        Initialize variables

        Args:
            demographic_dataframe_process (EncounterDataframeProcess): Dataframe processing object
            patient_id_column (str, defaults to `PatientID`): The column name for the patient id
            birth_date_column (str, defaults to `bins`): Column that stores the assigned bin
            sex_column (str, defaults to `min`): Column that stores the start position of each bin
            race_column (str, defaults to `max`): Column that stores the end position of each bin
            label_column (str, defaults to `count`): The column that stores some value that represents
            the code in a time bin
            ignore_instruction_column (str, defaults to `ignore_instruction`): Column that stores whether
            to ignore the instruction in that row
        """
        super().__init__(
            demographic_dataframe_process=demographic_dataframe_process,
            demographic_instructions=demographic_instructions,
            patient_id_column=patient_id_column,
            birth_date_column=birth_date_column,
            sex_column=sex_column,
            race_column=race_column,
            height_column=height_column,
            weight_column=weight_column,
            label_column=label_column,
            ignore_instruction_column=ignore_instruction_column,
            seq2seq_column=seq2seq_column,
            seq2seq=seq2seq
        )

    def process_sample(self, sample, args, ignore_instruction=None, category=None):
        """
        Process sample

        Args:
            sample:
            args:
            ignore_instruction:
            category:

        Returns:

        """

        # Store all the prompt elements (input, output)
        all_instructions = list()

        # Get the task instruction
        all_instructions.append(self.get_task_instruction())

        # Convert samples to text instructions (prompt)
        all_instructions.append(self.get_category_prompt(category=category))

        return all_instructions

    def get_category_prompt(self, category):
        return self._demographic_instructions.get_instruction_input(category=category), '', True, self._seq2seq
