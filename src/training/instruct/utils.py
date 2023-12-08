"""Util functions"""
import numpy as np
from .codes.templates import (
    CodeStatusInstructionTemplate,
    CodeStatusClassificationInstructions,
    CodeT2EInstructionTemplate,
    CodeT2EPredictionInstructions
)

# ICD utils

def get_code_status_classification_instructions(
        task_definition: str,
        future_input: str,
        future_target: str,
        past_input: str,
        past_target: str,
        positive_answer_target: str,
        negative_answer_target: str
):
    instruction_template_future = CodeStatusInstructionTemplate(
        inputs=future_input,
        targets=future_target
    )

    instruction_template_past = CodeStatusInstructionTemplate(
        inputs=past_input,
        targets=past_target
    )

    return CodeStatusClassificationInstructions(
        task_definition=task_definition,
        instruction_template_future=instruction_template_future,
        instruction_template_past=instruction_template_past,
        positive_answer_target=positive_answer_target,
        negative_answer_target=negative_answer_target
    )


def get_code_t2e_prediction_instructions(
        task_definition: str,
        inputs: str,
        targets: str,
        negative_answer_target: float = np.inf,
):
    instruction_template = CodeT2EInstructionTemplate(
        inputs=inputs,
        targets=targets
    )

    return CodeT2EPredictionInstructions(
        task_definition=task_definition,
        instruction_template=instruction_template,
        negative_answer_target=negative_answer_target,
    )

