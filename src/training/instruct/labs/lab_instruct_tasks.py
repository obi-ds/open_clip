"""Make training data"""
import sys
sys.path.append('..')

import numpy as np
import pandas as pd
from typing import Tuple, List

from ..codes.code_trajectory_instruct_tasks import CodeLabelPredictionTask


np.random.seed(42)


class LabPredictionTask(CodeLabelPredictionTask):
    """
    Make instruction tuning training data from diagnostic codes
    """

    def __init__(
            self,
            lab_dataframe_process,
            lab_instructions,
            time_bins,
            fixed_position_range: bool,
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
            idf_column: str = 'idf',
            tf_column: str = 'tf',
            weights_column: str = 'weights',
            current_bin_value: int = -100,
            prediction_range_limit: int = None,
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
            encounter_dataframe_process=lab_dataframe_process,
            dataframe_sampling=None,
            code_instructions=lab_instructions,
            time_bins=time_bins,
            code_convert=None,
            negative_code_sampling=None,
            patient_id_column=patient_id_column,
            code_column=lab_name_column,
            position_column=position_column,
            fixed_position_range=fixed_position_range,
            bins_column=bins_column,
            bin_start_column=bin_start_column,
            bin_end_column=bin_end_column,
            label_column=label_column,
            ignore_instruction_column=ignore_instruction_column,
            position_range_column=position_range_column,
            idf_column=idf_column,
            tf_column=tf_column,
            weights_column=weights_column,
            current_bin_value=current_bin_value,
            prediction_range_limit=prediction_range_limit
        )
        self._reference_range_low_column = reference_range_low_column
        self._reference_range_high_column = reference_range_high_column

    def process_sample(self, sample, args):
        """
        Process sample

        Args:
            sample:
            args:

        Returns:

        """

        # If for some reason we don't have encounter data for this person - we ignore training
        # on this person
        if not self._encounter_dataframe_process.check_patient_id(patient_id=sample[args.patient_id_column]):
            return []

        # Get the full encounter history
        lab_history = self.get_full_encounter_history(
            patient_id=sample[args.patient_id_column],
            current_time=sample[args.sample_result_date_column],
            use_log_position=args.use_log_position,
            time_difference_normalize=args.time_difference_normalize
        )

        # Filter encounters by time if the time delta values are not None
        # Compute which time bins the encounters belong too - based on some clustering
        # Also compute the position range buckets they fall into
        lab_history = self.transform_encounter_history_for_task(
            encounter_history=lab_history,
            past_time_delta=args.past_time_delta,
            future_time_delta=args.future_time_delta,
        )

        # Get the bin period we will be making predictions for.
        # The list can contain multiple elements - we can make prediction for
        # multiple time bins
        prediction_ranges = self.get_prediction_ranges(
            past_only=args.past_only_labs,
            number_of_ranges=self.sample_from_list(args.number_of_instructions),
            fixed_range=args.time_period_range
        )

        ignore_instruction = self.get_ignore_instruction(eval_mode=args.eval_mode)

        # Store all the prompt elements (input, output)
        all_instructions = list()

        for prediction_range in prediction_ranges:

            # Get dataframe that contain encounter in the current time period
            current_time_period = self.get_current_time_period(
                encounter_history=lab_history, prediction_range=prediction_range
            )

            k_shot = min(self.sample_from_list(args.k_shot_labs), len(current_time_period))

            if k_shot == 0 or current_time_period.empty:
                continue

            # Get the task instruction - which should indicate we are making prediction for a given
            # time range
            all_instructions.append(self.get_task_instruction(prediction_range=prediction_range))

            # Now that we have sizes - we move on to sampling the codes
            # We try and sample such that we get codes spread across the time range
            # and not just concentrated in one portion of the time range - for both
            # positives and negatives
            sampled_labs = self.sample_labs(encounter_history=current_time_period, max_limit=k_shot)

            instruction_samples = self.prepare_sampled_codes(
                codes=sampled_labs,
                label_string=None,
                prediction_range=prediction_range,
                ignore_instruction=ignore_instruction,
            ).sample(frac=1)

            # Convert samples to text instructions (prompt)
            all_instructions.extend(
                self.convert_samples_to_instructions(
                    instruction_samples=instruction_samples,
                    include_reference_range=args.include_reference_range
                )
            )

        return all_instructions

    def get_prediction_ranges(
            self,
            past_only: bool,
            number_of_ranges: int,
            fixed_range: Tuple[int, int] = None
    ) -> List[Tuple[int, int]]:
        """
        Returns a list of tuples that contain the start and end times of the bin
        we will be using as the current bin period - this is the period we
        will make predictions for.

        Args:
            fixed_range (Tuple[int, int], defaults to `None`): Return the fixed range
            past_only (bool): Use ranges occurring only in the future
            number_of_ranges (int): The number of prediction ranges to sample

        Returns:
            (List[Tuple[int, int]]): The start and end position of the current bin period
        """
        prediction_ranges = super().get_prediction_ranges(
            future_only=past_only,
            number_of_ranges=number_of_ranges,
            fixed_range=fixed_range
        )
        if past_only:
            prediction_ranges = self.switch_prediction_ranges(prediction_ranges=prediction_ranges)
        return prediction_ranges


    def sample_labs(self, encounter_history, max_limit):
        """
        Sample codes from the dataframe - sample such that we have good
        representation from different time ranges - for example

        Args:
            encounter_history:
            max_limit:

        Returns:

        """
        labs = super().sample_codes(encounter_history=encounter_history, max_limit=max_limit)[self._code_column]
        labs_sampled = encounter_history[encounter_history[self._code_column].isin(labs)]
        return labs_sampled.groupby(self._code_column).sample(n=1)

    def prepare_sampled_codes(
            self, codes, label_string, prediction_range, ignore_instruction
    ):
        """
        Define the start and end times and the label string for the sampled codes

        Args:
            codes:
            label_string:
            prediction_range:
            ignore_instruction:

        Returns:

        """
        codes[self._bin_start_column] = prediction_range[0]
        codes[self._bin_end_column] = prediction_range[1]
        codes[self._ignore_instruction_column] = ignore_instruction
        return codes

    def convert_samples_to_instructions(
            self,
            instruction_samples: pd.DataFrame,
            include_reference_range: bool = False
    ) -> List[Tuple[str, str, bool]]:
        """
        Return the instruction input and targets
        for the sampled instruction code status classification task with time range

        Args:
            instruction_samples (pd.DataFrame): The samples that will be used as instructions
            include_reference_range (bool, defaults to False): Include high and low ranges

        Returns:
            (List[Tuple[str, str, bool]]): A list that contains tuples which have the instruction input and target.
            The list will contain only 1 element for zero shot training
        """
        if include_reference_range:
            return self.convert_samples_to_instructions_with_reference_range(
                instruction_samples=instruction_samples
            )
        else:
            return self.convert_samples_to_instructions_without_reference_range(
                instruction_samples=instruction_samples
            )

    def convert_samples_to_instructions_with_reference_range(
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
            [
                self._code_column,
                self._label_column,
                self._reference_range_low_column,
                self._reference_range_high_column,
                self._ignore_instruction_column
            ]
        ].itertuples(index=False):
            (lab, label, reference_range_low, reference_range_high,  ignore_instruction) = (
                row[0], row[1], row[2], row[3], row[4]
            )
            lab_string_with_reference_ranges = self.get_lab_string_with_reference_ranges(
                lab=lab,
                reference_range_low=reference_range_low,
                reference_range_high=reference_range_high
            )
            lab_instruct_string = self._code_instructions.get_instruction(
                diagnosis=lab_string_with_reference_ranges,
                value=label
            )
            instructions.append(lab_instruct_string + (ignore_instruction,))
        return instructions

    def convert_samples_to_instructions_without_reference_range(
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
            [
                self._code_column,
                self._label_column,
                self._ignore_instruction_column
            ]
        ].itertuples(index=False):
            lab, label, ignore_instruction = row[0], row[1], row[2]
            lab_instruct_string = self._code_instructions.get_instruction(
                diagnosis=lab,
                value=label
            )
            instructions.append(lab_instruct_string + (ignore_instruction,))
        return instructions


    @staticmethod
    def get_lab_string_with_reference_ranges(lab, reference_range_low, reference_range_high):
        """
        Add reference ranges to prompt

        Args:
            lab:
            reference_range_low:
            reference_range_high:

        Returns:

        """
        lab_string = f'{lab}'
        if reference_range_low is not None:
            lab_string += f' (Low= {reference_range_low}'
            if reference_range_high is not None:
                lab_string += f', High= {reference_range_high})'
            else:
                lab_string += ')'
        else:
            if reference_range_high is not None:
                lab_string += f' (High= {reference_range_high})'
        return lab_string

    @staticmethod
    def switch_prediction_ranges(prediction_ranges):
        """

        Args:
            prediction_ranges:

        Returns:

        """
        return [(-prediction_range[1], -prediction_range[0]) for prediction_range in prediction_ranges]

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
