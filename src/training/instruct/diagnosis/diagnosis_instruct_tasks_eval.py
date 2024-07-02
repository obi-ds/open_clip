"""Make training data"""
import numpy as np
import pandas as pd
from typing import Union, Sequence, List, Tuple

from .processing.code_convert import ICDConvert, PHEConvert
from .processing.encounter_dataframe_process import EncounterDataframeProcess
from .processing.negative_code_sampling import NegativeCodeCacheSampling
from .diagnosis_instruct_tasks import (
    DiagnosisLabelPredictionTask
)
from .templates import DiagnosisLabelPredictionInstructionTemplate

np.random.seed(42)


class DiagnosisLabelPredictionTaskEvaluation(DiagnosisLabelPredictionTask):
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
            seq2seq: bool = False
    ):
        """
        Initialize variables

        Args:
            encounter_dataframe_process (EncounterDataframeProcess): Dataframe processing object
            diagnosis_instructions (Any): A list of code instruction tasks. At any point one of these
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
            fixed_position_range=False,
            weights_column=weights_column
        )

    def process_sample(self, sample, args, ignore_instruction=False):
        """
        Process sample

        Args:
            sample:
            args:
            ignore_instruction:

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
            label=self.get_true_label_string(label=label),
            ignore_instruction=ignore_instruction,
            seq2seq=self._seq2seq,
        )

        # Convert samples to text instructions (prompt)
        instructions = self.convert_samples_to_instructions(
            instruction_samples=eval_sample,
        )

        if len(instructions):
            # Get the task instruction - which should indicate we are making prediction for a given
            # time range
            all_instructions.append(self.get_task_instruction(prediction_range=prediction_range))
            all_instructions.extend(
                instructions
            )
            all_instructions.append(self._diagnosis_instructions.get_task_separator_instruction())

        return all_instructions

    @staticmethod
    def get_label(code_occurrences: pd.DataFrame) -> bool:
        """
        Check if a given code exists in a given encounter history
        Args:
            code_occurrences (pd.DataFrame): The dataframe containing all the occurrences of the code

        Returns:
            (str): The input code or sub code
            (bool): Whether the code or it's sub code is in the encounter history
        """
        return not code_occurrences.empty

    def get_code_sample(
            self,
            code: str,
            label: str,
            ignore_instruction: bool,
            seq2seq: bool
    ) -> pd.DataFrame:
        """
        Create a row/dataframe that contains details about the code

        Args:
            code (str): The code to evaluate
            label (str): The label to evaluate
            ignore_instruction (bool): Whether to ignore instruction while computing loss etc.
            seq2seq (bool)

        Returns:
            (pd.DataFrame): The dataframe that contains the code, start and end times
        """
        code_sample = {
            self._code_column: [code],
            self._label_column: [label],
            self._ignore_instruction_column: [ignore_instruction],
            self._seq2seq_column: [seq2seq],
            self._weights_column: [1]
        }
        return pd.DataFrame(code_sample)

    def get_true_label_string(self, label):
        """
        Get the label string
        Args:
            label:

        Returns:

        """
        if label:
            return self.get_positive_labels_string()
        else:
            return self.get_negative_labels_string()


class HierarchicalDiagnosisLabelPredictionTaskEvaluation(DiagnosisLabelPredictionTaskEvaluation):
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
            ignore_instruction_column: str = 'ignore_instruction',
            position_range_column: str = 'position_range',
            seq2seq_column: str = 'seq2seq',
            current_bin_value: int = -100,
            prediction_range_limit: int = None,
            seq2seq: bool = False,
            sep: str = ' -> '
    ):
        """
        Initialize variables

        Args:
            encounter_dataframe_process (EncounterDataframeProcess): Dataframe processing object
            diagnosis_instructions (Any): A list of code instruction tasks. At any point one of these
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
        )
        self._trie = negative_code_sampling.get_trie()
        self._tree = negative_code_sampling.get_tree()
        self._sep = sep

    def convert_samples_to_instructions(
            self,
            instruction_samples: pd.DataFrame,
    ) -> List[Tuple[str, str, bool, bool]]:
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
            [self._code_column, self._label_column, self._ignore_instruction_column, self._seq2seq_column]
        ].itertuples(index=False):
            code, label, ignore_instruction, seq2seq = row[0], row[1], row[2], row[3]
            code_instruct_string = self._diagnosis_instructions.get_instruction(
                diagnosis=self.get_code_description_with_hierarchy(code=code),
                value=label
            )
            instructions.append(code_instruct_string + (ignore_instruction, seq2seq))
        return instructions

    def get_code_description_with_hierarchy(self, code):
        """

        Args:
            code:

        Returns:

        """
        key = '/'.join(self._negative_code_sampling.get_code_representation(code))
        code_description_with_hierarchy = ''
        for code_obj in self._trie.prefixes(key):
            code_description = self._code_convert.transform_code(code=code_obj[1])
            code_description_with_hierarchy += f'{code_description}{self._sep}'
        return code_description_with_hierarchy.strip()
