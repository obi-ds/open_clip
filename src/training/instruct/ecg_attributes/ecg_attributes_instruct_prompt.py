"""Make training data"""
import numpy as np

from .templates import ECGAttributesTemplate
from .ecg_attributes_instruct_tasks import ECGAttributePredictionTask

np.random.seed(42)


class ECGAttributePredictionPrompt(ECGAttributePredictionTask):
    """
    Make instruction tuning training data from diagnostic codes
    """

    def __init__(
            self,
            ecg_attribute_instructions: ECGAttributesTemplate,
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
        super().__init__(
            ecg_attribute_instructions=ecg_attribute_instructions,
            patient_id_column=patient_id_column,
            label_column=label_column,
            ignore_instruction_column=ignore_instruction_column,
            seq2seq_column=seq2seq_column,
            seq2seq=seq2seq
        )

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

        # Get the full encounter history
        ecg_attributes = self.get_ecg_attributes_for_task(
            sample=sample,
        )

        # Get the task instruction
        all_instructions.append(self.get_task_instruction())

        attributes = set(attributes)
        instruction_samples = [
            ecg_attribute for ecg_attribute in ecg_attributes if ecg_attribute[0] in attributes
        ]

        # TODO: What do we do in this case when evaluating?
        # if len(attributes) != len(instruction_samples):
        #     # Don't compute metrics on samples where we don't have all the
        #     # required information in the prompt
        #     return []

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
