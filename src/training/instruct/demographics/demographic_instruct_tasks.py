"""Make training data"""
import random
import numpy as np
import pandas as pd
from dateutil import relativedelta
from typing import Union, Tuple, List

from .processing.demographic_dataframe_process import DemographicDataframeProcess
from .templates import PatientDemographicsTemplate

np.random.seed(42)


class DemographicPredictionTask(object):
    """
    Make instruction tuning training data from diagnostic codes
    """

    def __init__(
            self,
            demographic_dataframe_process: DemographicDataframeProcess,
            demographic_instructions: PatientDemographicsTemplate,
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
        self._demographic_dataframe_process = demographic_dataframe_process
        self._demographic_instructions = demographic_instructions
        self._patient_id_column = patient_id_column
        self._birth_date_column = birth_date_column
        self._sex_column = sex_column
        self._race_column = race_column
        self._height_column = height_column
        self._weight_column = weight_column
        self._label_column = label_column
        self._ignore_instruction_column = ignore_instruction_column
        self._seq2seq_column = seq2seq_column
        self._seq2seq = seq2seq

    def process_sample(self, sample, args, ignore_instruction=None):
        """
        Process sample

        Args:
            sample:
            args:
            ignore_instruction:

        Returns:

        """
        sample_size = self.sample_from_list(args.k_shot_demographics)

        # If for some reason we don't have encounter data for this person - we ignore training
        # on this person
        if (
                not self._demographic_dataframe_process.check_patient_id(patient_id=sample[args.patient_id_column])
                or not sample_size
        ):
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

        # Store all the prompt elements (input, output)
        all_instructions = list()

        # Get the task instruction
        all_instructions.append(self.get_task_instruction())

        # Sample and shuffle the data
        instruction_samples = random.sample(patient_demographics, k=min(sample_size, len(patient_demographics)))

        ignore_instruction = (
            self.get_ignore_instruction(eval_mode=args.eval_mode) if ignore_instruction is None else ignore_instruction
        )

        # Convert samples to text instructions (prompt)
        all_instructions.extend(
            self.convert_samples_to_instructions(
                instruction_samples=instruction_samples,
                ignore_instruction=ignore_instruction,
                seq2seq=self._seq2seq
            )
        )

        return all_instructions

    def get_patient_demographics(
            self,
            patient_id: Union[str, int],
    ) -> pd.Series:
        """
        Return the demographic details of patient

        Args:
            patient_id (Union[str, int]): The unique id of the patient we need to process
            with respect to other encounters

        Returns:
            (pd.DataFrame): The encounter history
        """
        if self._demographic_dataframe_process.check_patient_id(patient_id=patient_id):
            return self._demographic_dataframe_process.get_patient_demographics(patient_id=patient_id)
        else:
            return pd.Series()

    def transform_demographics_for_task(
            self,
            patient_demographics: pd.Series,
            sample,
            current_time
    ) -> List[Tuple[str, Union[str, int], float]]:
        """
        Get the age and sex of the patient

        Args:
            patient_demographics (pd.Series): Dataframe that contains one row - with patient information
            sample:
            current_time:

        Returns:
            (List[Tuple[str, Union[str, int], float]]): The processed demographics for the task
        """
        age = ('Age', self.get_age(patient_demographics=patient_demographics, current_time=current_time))
        sex = ('Sex', self.get_sex(patient_demographics=patient_demographics))
        height = ('Height', self.get_height(sample=sample))
        weight = ('Weight', self.get_weight(sample=sample))
        return [
            (attribute[0], attribute[1], self.get_weight_for_attribute(attribute=attribute))
            for attribute in [age, sex, height, weight] if attribute[1] is not None
        ]

    @staticmethod
    def get_weight_for_attribute(attribute) -> float:
        """
        TODO: Implement function
        Returns:

        """
        return 1.0

    def convert_samples_to_instructions(
            self,
            instruction_samples: List[Tuple[str, Union[str, int], float]],
            ignore_instruction: bool,
            seq2seq: bool,
    ) -> List[Tuple[str, str, bool, bool, float]]:
        """
        Return the instruction input and targets
        for the sampled instruction code status classification task with time range

        Args:
            instruction_samples (List[Tuple[str, Union[str, int], float]]): Samples that will be used as instructions
            ignore_instruction (bool): Whether to ignore the instruction
            seq2seq (bool)

        Returns:
            (List[Tuple[str, str, bool, float]]): A list that contains tuples which have the instruction input and
            target. The list will contain only 1 element for zero shot training
        """
        instructions = list()
        for category, value, weight in instruction_samples:
            code_instruct_string = self._demographic_instructions.get_instruction(
                category=category,
                value=value
            )
            instructions.append(code_instruct_string + (ignore_instruction, seq2seq) + (weight, ))
        return instructions

    def get_age(self, patient_demographics, current_time):
        """
        Calculate age of patient at current time

        Args:
            patient_demographics:
            current_time:

        Returns:

        """
        time_delta = (
            relativedelta.relativedelta(
                pd.to_datetime(current_time),
                pd.to_datetime(patient_demographics[self._birth_date_column])
            )
        )
        years = time_delta.years
        return years

    def get_sex(self, patient_demographics):
        """
        Get sex of the patient

        Args:
            patient_demographics:

        Returns:

        """
        return patient_demographics[self._sex_column]

    def get_height(self, sample):
        """
        Return the height

        Args:
            sample:

        Returns:

        """
        height = sample.get(self._height_column, 0)
        if height is not None and float(height) > 0:
            return height
        else:
            return None

    def get_weight(self, sample):
        """
        Return the weight

        Args:
            sample:

        Returns:

        """
        weight = sample.get(self._weight_column, 0)
        if weight is not None and float(weight) > 0:
            return weight
        else:
            return None

    @staticmethod
    def get_ignore_instruction(eval_mode):
        """
        Ignore demographic token loss when doing eval
        Args:
            eval_mode:

        Returns:

        """
        if eval_mode:
            return True
        else:
            return False

    @staticmethod
    def shuffle_data(patient_demographics):
        """
        Shuffle the data

        Args:
            patient_demographics:

        Returns:

        """
        random.shuffle(patient_demographics)
        return patient_demographics

    def get_task_instruction(self):
        """
        Return a task instruction based on the task definition and prediction range

        Returns:

        """
        task_definition = self._demographic_instructions.get_task_definition()
        return task_definition, '', True, self._seq2seq, -100

    @staticmethod
    def sample_from_list(shots):
        """
        Sample an element from a list

        Args:
            shots:

        Returns:

        """
        return np.random.choice(shots)
