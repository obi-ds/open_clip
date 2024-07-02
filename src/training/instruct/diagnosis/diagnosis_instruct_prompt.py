from typing import Union

from .processing.code_convert import ICDConvert, PHEConvert
from .processing.encounter_dataframe_process import EncounterDataframeProcess
from .processing.negative_code_sampling import NegativeCodeCacheSampling
from .diagnosis_instruct_tasks_eval import DiagnosisLabelPredictionTaskEvaluation
from .templates import DiagnosisLabelPredictionInstructionTemplate


class DiagnosisLabelPredictionPrompt(DiagnosisLabelPredictionTaskEvaluation):
    """
    Make instruction tuning training data from diagnostic codes
    """

    def __init__(
            self,
            encounter_dataframe_process: EncounterDataframeProcess,
            diagnosis_instructions: DiagnosisLabelPredictionInstructionTemplate,
            code_convert: Union[ICDConvert, PHEConvert],
            negative_code_sampling: NegativeCodeCacheSampling,
            patient_id_column: str = 'PatientID',
            code_column: str = 'phecode',
            position_column: str = 'position',
            bins_column: str = 'bins',
            bin_start_column: str = 'min',
            bin_end_column: str = 'max',
            label_column: str = 'label',
            weights_column: str = 'weights',
            ignore_instruction_column: str = 'ignore_instruction',
            position_range_column: str = 'position_range',
            seq2seq_column: str = 'seq2seq',
            current_bin_value: int = -100,
            prediction_range_limit: int = None,
            seq2seq: bool = False,
    ):
        super().__init__(
            encounter_dataframe_process=encounter_dataframe_process,
            diagnosis_instructions=diagnosis_instructions,
            code_convert=code_convert,
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
            seq2seq_column=seq2seq_column,
            current_bin_value=current_bin_value,
            prediction_range_limit=prediction_range_limit,
            seq2seq=seq2seq,
            weights_column=weights_column,
        )

    def process_sample(self, sample, args, ignore_instruction=True, diagnoses=None):
        """
        Process sample

        Args:
            sample:
            args:
            ignore_instruction:
            diagnoses:

        Returns:

        """
        # Store all the prompt elements (input, output)
        all_instructions = list()

        # If for some reason we don't have encounter data for this person - we ignore training
        # on this person
        if (
                not self._encounter_dataframe_process.check_patient_id(patient_id=sample[args.patient_id_column])
        ):
            print('Missing encounter history for patient: XXX')
            return []

        # Get the full encounter history
        encounter_history = self.get_full_encounter_history(
            patient_id=sample[args.patient_id_column],
            current_time=sample[args.sample_result_date_column],
            use_log_position=args.use_log_position,
            time_difference_normalize=args.time_difference_normalize
        )

        # Filter encounters by time if the time delta values are not None
        # Compute which time bins the encounters belong too - based on some clustering
        # Also compute the position range buckets they fall into
        encounter_history = self.transform_encounter_history_for_task(
            encounter_history=encounter_history,
            past_time_delta=args.past_time_delta,
            future_time_delta=args.future_time_delta,
        )

        # Group 3 elements together (Code, Start Time, End Time)
        diagnoses = list(zip(*[iter(diagnoses.split(', '))] * 3))

        start_flag = None
        end_flag = None

        for code, start_time, end_time in diagnoses:
            start_time = int(start_time)
            end_time = int(end_time)
            prediction_range = start_time, end_time

            # Get dataframe that contain encounter in the current time period
            current_time_period = self.get_current_time_period(
                encounter_history=encounter_history, prediction_range=prediction_range
            )

            # TODO: What if code is not present in the given time period but is present in a different
            #  time period
            current_code_occurrences = self.get_code_occurrences(
                encounter_history=current_time_period, code=code, regex=False
            )

            label = self.get_label(code_occurrences=current_code_occurrences)

            eval_sample = self.get_code_sample(
                code=code,
                label=self.get_true_label_string(label=label),
                ignore_instruction=ignore_instruction,
                seq2seq=self._seq2seq,
            )

            # Convert samples to text instructions (prompt)
            instructions = self.convert_samples_to_instructions(
                instruction_samples=eval_sample,
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
