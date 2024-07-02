"""Util functions"""
from .diagnosis.templates import (
    DiagnosisLabelPredictionInstructionTemplate,
    HierarchicalDiagnosisLabelPredictionInstructionTemplate
)
from .demographics.templates import PatientDemographicsTemplate


def get_diagnosis_label_prediction_instruction_template(
        task_definition='Diagnoses in the next {time} months:',
        inputs='\t* {diagnosis}',
        x_y_delimiter=': ',
        example_separator='\n',
        task_separator=None
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
        example_separator=example_separator,
        task_separator=task_separator
    )


def get_hierarchical_diagnosis_label_prediction_instruction_template(
        task_definition='Diagnoses in the next {time} months:',
        inputs='{diagnosis}',
        inputs_prefix='\t* ',
        targets_prefix=':  ',
        x_y_delimiter=' -> ',
        example_separator='\n',
        task_separator=None
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
        example_separator=example_separator,
        task_separator=task_separator
    )


def get_patient_demographics_instruction_template(
        task_definition='Patient attributes:',
        inputs='\t* {category}:',
        x_y_delimiter=' ',
        example_separator='\n',
        task_separator=None
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
        example_separator=example_separator,
        task_separator=task_separator
    )


def get_patient_labs_instruction_template(
        task_definition='Labs in the next {time} months:',
        inputs='\t* {diagnosis}:',
        x_y_delimiter=' ',
        example_separator='\n',
        task_separator=None
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
        example_separator=example_separator,
        task_separator=task_separator
    )

def get_ecg_attributes_instruction_template(
        task_definition='ECG attributes:',
        inputs='\t* {category}:',
        x_y_delimiter=' ',
        example_separator='\n',
        task_separator=None
) -> PatientDemographicsTemplate:
    """
    Return the object can use the given data and instruction templates
    and return an instruction string for the input and output

    Returns:
        (ECGAttributesTemplate): Object to build instruction strings (inputs and targets)
    """
    return PatientDemographicsTemplate(
        inputs=inputs,
        x_y_delimiter=x_y_delimiter,
        task_definition=task_definition,
        example_separator=example_separator,
        task_separator=task_separator
    )
