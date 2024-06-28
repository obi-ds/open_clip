"""Util functions"""
from .diagnosis.templates import (
    DiagnosisLabelPredictionInstructionTemplate,
    HierarchicalDiagnosisLabelPredictionInstructionTemplate
)
from .demographics.templates import PatientDemographicsTemplate
from .ecg_attributes.templates import ECGAttributesTemplate


def get_diagnosis_label_prediction_instruction_template(
        task_definition='Diagnoses in the next {time} months',
        inputs='* {diagnosis}',
        x_y_delimiter=': ',
        example_separator='\n'
) -> DiagnosisLabelPredictionInstructionTemplate:
    """
    Return the object can use the given data and instruction templates
    and return an instruction string for the input and output

    Returns:
        (CodeTrajectoryPredictionInstructions): Object to build instruction strings (inputs and targets)
    """
    return DiagnosisLabelPredictionInstructionTemplate(
        inputs=inputs,
        x_y_delimiter=x_y_delimiter,
        task_definition=task_definition,
        example_separator=example_separator
    )


def get_hierarchical_diagnosis_label_prediction_instruction_template(
        task_definition='Diagnoses in the next {time} months',
        inputs='{diagnosis}',
        inputs_prefix='* ',
        targets_prefix=':  ',
        x_y_delimiter=' -> ',
        example_separator='\n'
) -> HierarchicalDiagnosisLabelPredictionInstructionTemplate:
    """
    Return the object can use the given data and instruction templates
    and return an instruction string for the input and output

    Returns:
        (CodeTrajectoryPredictionInstructions): Object to build instruction strings (inputs and targets)
    """
    return HierarchicalDiagnosisLabelPredictionInstructionTemplate(
        inputs=inputs,
        inputs_prefix=inputs_prefix,
        targets_prefix=targets_prefix,
        x_y_delimiter=x_y_delimiter,
        task_definition=task_definition,
        example_separator=example_separator
    )


def get_patient_demographics_instruction_template(
        task_definition='Patient attributes',
        inputs='* {category}:',
        x_y_delimiter=' ',
        example_separator='\n'
) -> PatientDemographicsTemplate:
    """
    Return the object can use the given data and instruction templates
    and return an instruction string for the input and output

    Returns:
        (CodeTrajectoryPredictionInstructions): Object to build instruction strings (inputs and targets)
    """
    return PatientDemographicsTemplate(
        inputs=inputs,
        x_y_delimiter=x_y_delimiter,
        task_definition=task_definition,
        example_separator=example_separator
    )


def get_patient_labs_instruction_template(
        task_definition='Labs in the next {time} months',
        inputs='* {diagnosis}:',
        x_y_delimiter=' ',
        example_separator='\n'
) -> DiagnosisLabelPredictionInstructionTemplate:
    """
    Return the object can use the given data and instruction templates
    and return an instruction string for the input and output

    Returns:
        (CodeTrajectoryPredictionInstructions): Object to build instruction strings (inputs and targets)
    """
    return DiagnosisLabelPredictionInstructionTemplate(
        inputs=inputs,
        x_y_delimiter=x_y_delimiter,
        task_definition=task_definition,
        example_separator=example_separator
    )

def get_ecg_attributes_instruction_template(
        task_definition='ECG attributes',
        inputs='* {category}:',
        x_y_delimiter=' ',
        example_separator='\n'
) -> ECGAttributesTemplate:
    """
    Return the object can use the given data and instruction templates
    and return an instruction string for the input and output

    Returns:
        (ECGAttributesTemplate): Object to build instruction strings (inputs and targets)
    """
    return ECGAttributesTemplate(
        inputs=inputs,
        x_y_delimiter=x_y_delimiter,
        task_definition=task_definition,
        example_separator=example_separator
    )
