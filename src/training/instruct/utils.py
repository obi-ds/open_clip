"""Util functions"""
from .codes.templates import (
    CodeLabelPredictionInstructionTemplate,
)

def get_code_label_prediction_instruction_template() -> CodeLabelPredictionInstructionTemplate:
    """
    Return the object can use the given data and instruction templates
    and return an instruction string for the input and output

    Returns:
        (CodeTrajectoryPredictionInstructions): Object to build instruction strings (inputs and targets)
    """
    past_input = 'Was the patient diagnosed with {diagnosis} in the past {time} months?'
    future_input = 'Will the patient be diagnosed with {diagnosis} in next {time} months?'
    task_definition = ''
    x_y_delimiter = ' '

    return CodeLabelPredictionInstructionTemplate(
        past_input=past_input,
        future_input=future_input,
        x_y_delimiter=x_y_delimiter,
        task_definition=task_definition
    )


