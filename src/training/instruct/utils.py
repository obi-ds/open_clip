"""Util functions"""
from .codes.templates import (
    CodeLabelPredictionInstructionTemplate,
    HierarchicalCodeLabelPredictionInstructionTemplate
)
from .demographics.templates import PatientDemographicsTemplate


def get_code_label_prediction_instruction_template(
        task_definition='Diagnoses in the next {time} months',
        inputs='* {diagnosis}',
        x_y_delimiter=': ',
        example_separator='\n'
) -> CodeLabelPredictionInstructionTemplate:
    """
    Return the object can use the given data and instruction templates
    and return an instruction string for the input and output

    Returns:
        (CodeTrajectoryPredictionInstructions): Object to build instruction strings (inputs and targets)
    """
    return CodeLabelPredictionInstructionTemplate(
        inputs=inputs,
        x_y_delimiter=x_y_delimiter,
        task_definition=task_definition,
        example_separator=example_separator
    )


def get_hierarchical_code_label_prediction_instruction_template(
        task_definition='Diagnoses in the next {time} months',
        inputs='{diagnosis}',
        inputs_prefix='* ',
        targets_prefix=':  ',
        x_y_delimiter=' -> ',
        example_separator='\n'
) -> HierarchicalCodeLabelPredictionInstructionTemplate:
    """
    Return the object can use the given data and instruction templates
    and return an instruction string for the input and output

    Returns:
        (CodeTrajectoryPredictionInstructions): Object to build instruction strings (inputs and targets)
    """
    return HierarchicalCodeLabelPredictionInstructionTemplate(
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
) -> CodeLabelPredictionInstructionTemplate:
    """
    Return the object can use the given data and instruction templates
    and return an instruction string for the input and output

    Returns:
        (CodeTrajectoryPredictionInstructions): Object to build instruction strings (inputs and targets)
    """
    return CodeLabelPredictionInstructionTemplate(
        inputs=inputs,
        x_y_delimiter=x_y_delimiter,
        task_definition=task_definition,
        example_separator=example_separator
    )
