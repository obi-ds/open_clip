"""Make training data"""
import sys
import numpy as np

from .lab_instruct_tasks import LabPredictionTask
sys.path.append('..')

np.random.seed(42)


class LabPredictionPrompt(LabPredictionTask):
    """
    Make instruction tuning training data from diagnostic codes
    """

    def __init__(
            self,
            lab_dataframe_process,
            lab_instructions,
            fixed_position_range: bool,
            patient_id_column: str = 'PatientID',
            lab_name_column: str = 'ExternalNM_lower',
            lab_name_normalized_column: str = 'ExternalNM',
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
            seq2seq: bool = False,
            update_lab_counts: bool = False
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
            patient_id_column=patient_id_column,
            lab_name_column=lab_name_column,
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
            seq2seq=seq2seq,
            update_lab_counts=update_lab_counts,
            lab_name_normalized_column=lab_name_normalized_column,
            reference_range_high_column=reference_range_high_column,
            reference_range_low_column=reference_range_low_column
        )

    def process_sample(self, sample, args, ignore_instruction=True, labs=None):
        """
        Process sample

        Args:
            sample:
            args:
            ignore_instruction:
            labs:

        Returns:

        """

        # Store all the prompt elements (input, output)
        all_instructions = list()

        # If for some reason we don't have encounter data for this person - we ignore training
        # on this person
        if (
                not self._encounter_dataframe_process.check_patient_id(patient_id=sample[args.patient_id_column])
        ):
            print('Missing labs for patient: XXX')
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

        # Group 3 elements together (Lab Name, Start Time, End Time)
        labs = list(zip(*[iter(labs.split(', '))] * 3))

        start_flag = None
        end_flag = None
        for lab_name, start_time, end_time in labs:
            start_time = int(start_time)
            end_time = int(end_time)
            prediction_range = start_time, end_time

            # Get dataframe that contain encounter in the current time period
            current_time_period = self.get_current_time_period(
                encounter_history=lab_history, prediction_range=prediction_range
            )

            lab_details = self.get_lab_details(lab_history=current_time_period, lab_name=lab_name)

            instruction_samples = self.prepare_sampled_codes(
                codes=lab_details,
                label_string=None,
                ignore_instruction=ignore_instruction,
                seq2seq=self._seq2seq
            )

            # Convert samples to text instructions (prompt)
            instructions = self.convert_samples_to_instructions(
                instruction_samples=instruction_samples,
                include_reference_range=args.include_reference_range
            )

            if len(instructions):
                # Get the task instruction
                if start_time != start_flag or end_time != end_flag:
                    # Add end instruction before adding start instruction of the next time period
                    if start_flag is not None and end_flag is not None:
                        all_instructions.append(self._diagnosis_instructions.get_task_separator_instruction())
                    # Start an instruction when time period changes
                    # Get the task instruction - which should indicate we are making prediction for a given
                    # time range
                    all_instructions.append(self.get_task_instruction(prediction_range=prediction_range))
                    start_flag = start_time
                    end_flag = end_time

                all_instructions.extend(
                    instructions
                )

        if len(all_instructions):
            all_instructions.append(self._diagnosis_instructions.get_task_separator_instruction())

        return all_instructions

    def get_lab_details(self, lab_history, lab_name):
        """
        Get the value of the lab from the dataframe

        Args:
            lab_history:
            lab_name:

        Returns:

        """

        lab_history = lab_history[lab_history[self._code_column] == lab_name.lower()].sort_values(
            by=self._position_column, key=abs
        )
        if lab_history.empty:
            return lab_history
        else:
            return lab_history.iloc[[0]]
