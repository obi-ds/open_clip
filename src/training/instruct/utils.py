"""Util functions"""
from .codes.templates import (
    TimeBinCodeTemplate,
    TimeBinPeriodTemplate,
    CodeTrajectoryPredictionInstructions,
    DescriptionInstructionTemplate
)

def get_code_trajectory_prediction_instructions() -> CodeTrajectoryPredictionInstructions:
    """
    Return the object can use the given data and instruction templates
    and return an instruction string for the input and output

    Returns:
        (CodeTrajectoryPredictionInstructions): Object to build instruction strings (inputs and targets)
    """
    time_bin_period_instruction_template = TimeBinPeriodTemplate(inputs='\nDay {start_time} to {end_time}')
    time_bin_code_instruction_template = TimeBinCodeTemplate(inputs='- {diagnosis}')
    code_description_instruction_template = DescriptionInstructionTemplate(
        inputs='* {diagnosis}',
        targets='{description}'
    )
    return CodeTrajectoryPredictionInstructions(
        time_bin_period_instruction_template=time_bin_period_instruction_template,
        time_bin_code_instruction_template=time_bin_code_instruction_template,
        code_description_instruction_template=code_description_instruction_template
    )

