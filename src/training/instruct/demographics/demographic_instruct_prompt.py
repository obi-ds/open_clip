"""Make training data"""
import numpy as np
from typing import Optional

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
        self._attribute_index_map = {'Age': 0, 'Sex': 1, 'Height': 2, 'Weight': 3}

    def process_sample(self, sample, args, ignore_instruction=True, attributes=None):
        """
        Process sample

        Args:
            sample:
            args:
            ignore_instruction:
            attributes:

        Returns:

        """

        # Store all the prompt elements (input, output)
        all_instructions = list()

        # If for some reason we don't have encounter data for this person - we ignore training
        # on this person
        if (
                not self._demographic_dataframe_process.check_patient_id(patient_id=sample[args.patient_id_column])
        ):
            print('Missing demographics for patient: XXX')
            return []

        # Get the full encounter history
        patient_demographics = self.get_patient_demographics(
            patient_id=sample[args.patient_id_column],
        )

        patient_demographics = self.transform_demographics_for_task(
            patient_demographics=patient_demographics,
            sample=sample,
            current_time=sample[args.sample_result_date_column]
        )

        # Get the task instruction
        all_instructions.append(self.get_task_instruction())

        instruction_samples = [
            patient_demographics[self._attribute_index_map[attribute]]
            for attribute in attributes
            if (
                    self._attribute_index_map[attribute] < len(patient_demographics) and
                    attribute == patient_demographics[self._attribute_index_map[attribute]][0]
            )
        ]

        # Convert samples to text instructions (prompt)
        all_instructions.extend(
            self.convert_samples_to_instructions(
                instruction_samples=instruction_samples,
                ignore_instruction=ignore_instruction,
                seq2seq=self._seq2seq
            )
        )

        return all_instructions
