"""Make training data"""
import random
import numpy as np
from typing import Union, Tuple, List

from ..demographics.templates import PatientDemographicsTemplate

np.random.seed(42)


class ECGAttributePredictionTask(object):
    """
    Make instruction tuning training data from diagnostic codes
    """

    def __init__(
            self,
            ecg_attribute_instructions: PatientDemographicsTemplate,
            patient_id_column: str = 'PatientID',
            label_column: str = 'label',
            ignore_instruction_column: str = 'ignore_instruction',
            seq2seq_column: str = 'seq2seq',
            seq2seq: bool = False
    ):
        """
        Initialize variables

        Args:
            patient_id_column (str, defaults to `PatientID`): The column name for the patient id
            label_column (str, defaults to `count`): The column that stores some value that represents
            the code in a time bin
            ignore_instruction_column (str, defaults to `ignore_instruction`): Column that stores whether
            to ignore the instruction in that row
        """
        self._ecg_attribute_instructions = ecg_attribute_instructions
        self._patient_id_column = patient_id_column
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
        sample_size = self.sample_from_list(args.k_shot_ecg_attributes)

        if not sample_size:
            return []

        # Get the full encounter history
        ecg_attributes = self.get_ecg_attributes_for_task(
            sample=sample,
        )

        # Store all the prompt elements (input, output)
        all_instructions = list()

        # Get the task instruction
        all_instructions.append(self.get_task_instruction())

        # Sample and shuffle the data
        instruction_samples = random.sample(ecg_attributes, k=min(sample_size, len(ecg_attributes)))

        ignore_instruction = (
            self.get_ignore_instruction(eval_mode=args.eval_mode) if ignore_instruction is None else ignore_instruction
        )

        # Convert samples to text instructions (prompt)
        instructions = self.convert_samples_to_instructions(
            instruction_samples=instruction_samples,
            ignore_instruction=ignore_instruction,
            seq2seq=self._seq2seq
        )
        all_instructions.extend(
            instructions
        )
        if len(instructions):
            all_instructions.append(self._ecg_attribute_instructions.get_task_separator_instruction())

        return all_instructions

    def get_ecg_attributes_for_task(
            self,
            sample,
    ) -> List[Tuple[str, Union[str, int], float]]:
        """
        Get the age and sex of the patient

        Args:
            sample:

        Returns:
            (List[Tuple[str, Union[str, int], float]]): The processed demographics for the task
        """
        ventricular_rate = self.get_ventricular_rate(sample=sample)
        atrial_rate = self.get_atrial_rate(sample=sample)
        pr_interval = self.get_pr_interval(sample=sample)
        qrs_duration = self.get_qrs_duration(sample=sample)
        qt_interval = self.get_qt_interval(sample=sample)
        qt_corrected = self.get_qt_corrected(sample=sample)
        p_axis = self.get_p_axis(sample=sample)
        r_axis = self.get_r_axis(sample=sample)
        t_axis = self.get_t_axis(sample=sample)
        qrs_count = self.get_qrs_count(sample=sample)
        q_onset = self.get_q_onset(sample=sample)
        q_offset = self.get_q_offset(sample=sample)
        p_onset = self.get_p_onset(sample=sample)
        p_offset = self.get_p_offset(sample=sample)
        t_offset = self.get_t_offset(sample=sample)
        qtc_frederica = self.get_qtc_frederica(sample=sample)
        hr = self.get_hr(sample=sample)

        ecg_attributes = [
            ventricular_rate,
            atrial_rate,
            pr_interval,
            qrs_duration,
            qt_interval,
            qt_corrected,
            p_axis,
            r_axis,
            t_axis,
            qrs_count,
            q_onset,
            q_offset,
            p_onset,
            p_offset,
            t_offset,
            qtc_frederica,
            hr
        ]

        return [
            (attribute[0], attribute[1], self.get_weight_for_attribute(attribute=attribute))
            for attribute in ecg_attributes if attribute[1] is not None
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
            code_instruct_string = self._ecg_attribute_instructions.get_instruction(
                category=category,
                value=value
            )
            instructions.append(code_instruct_string + (ignore_instruction, seq2seq) + (weight, ))
        return instructions

    @staticmethod
    def get_ventricular_rate(sample):
        return 'Ventricular Rate', sample.get('VentricularRate', None)

    @staticmethod
    def get_atrial_rate(sample):
        return 'Atrial Rate', sample.get('AtrialRate', None)

    @staticmethod
    def get_pr_interval(sample):
        return 'PR Interval', sample.get('PRInterval', None)

    @staticmethod
    def get_qrs_duration(sample):
        return 'QRS Duration', sample.get('QRSDuration', None)

    @staticmethod
    def get_qt_interval(sample):
        return 'QT Interval', sample.get('QTInterval', None)

    @staticmethod
    def get_qt_corrected(sample):
        return 'QT Corrected', sample.get('QTCorrected', None)

    @staticmethod
    def get_p_axis(sample):
        return 'P Axis', sample.get('PAxis', None)

    @staticmethod
    def get_r_axis(sample):
        return 'R Axis', sample.get('RAxis', None)

    @staticmethod
    def get_t_axis(sample):
        return 'T Axis', sample.get('TAxis', None)

    @staticmethod
    def get_qrs_count(sample):
        return 'QRS Count', sample.get('QRSCount', None)

    @staticmethod
    def get_q_onset(sample):
        return 'Q Onset', sample.get('QOnset', None)

    @staticmethod
    def get_q_offset(sample):
        return 'Q Offset', sample.get('QOffset', None)

    @staticmethod
    def get_p_onset(sample):
        return 'P Onset', sample.get('POnset', None)

    @staticmethod
    def get_p_offset(sample):
        return 'P Offset', sample.get('POffset', None)

    @staticmethod
    def get_t_offset(sample):
        return 'T Offset', sample.get('TOffset', None)

    @staticmethod
    def get_qtc_frederica(sample):
        return 'QTc Frederica', sample.get('QTcFrederica', None)

    @staticmethod
    def get_hr(sample):
        return 'HR', sample.get('HR', None)

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

    def get_task_instruction(self):
        """
        Return a task instruction based on the task definition and prediction range

        Returns:

        """
        task_definition = self._ecg_attribute_instructions.get_task_definition()
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
