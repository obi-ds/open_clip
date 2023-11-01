"""Util functions"""
from .icd.templates import (
    ICDStatusInstructionTemplate,
    ICDStatusRangeInstructionTemplate,
    ICDStatusClassificationInstructions,
    ICDStatusRangeClassificationInstructions
)

# ICD utils

def get_icd_status_classification_instructions(
        task_definition: str,
        future_input: str,
        future_target: str,
        past_input: str,
        past_target: str,
        positive_answer_target: str,
        negative_answer_target: str
):
    instruction_template_future = ICDStatusInstructionTemplate(
        inputs=future_input,
        targets=future_target
    )

    instruction_template_past = ICDStatusInstructionTemplate(
        inputs=past_input,
        targets=past_target
    )

    return ICDStatusClassificationInstructions(
        task_definition=task_definition,
        instruction_template_future=instruction_template_future,
        instruction_template_past=instruction_template_past,
        positive_answer_target=positive_answer_target,
        negative_answer_target=negative_answer_target
    )

def get_icd_status_range_classification_instructions(
        task_definition: str,
        future_input: str,
        future_target: str,
        past_input: str,
        past_target: str,
        positive_answer_target: str,
        negative_answer_target: str,
        lock_range: bool
):
    instruction_template_future = ICDStatusRangeInstructionTemplate(
        inputs=future_input,
        targets=future_target
    )

    instruction_template_past = ICDStatusRangeInstructionTemplate(
        inputs=past_input,
        targets=past_target
    )

    return ICDStatusRangeClassificationInstructions(
        task_definition=task_definition,
        instruction_template_future=instruction_template_future,
        instruction_template_past=instruction_template_past,
        positive_answer_target=positive_answer_target,
        negative_answer_target=negative_answer_target,
        lock_range=lock_range
    )
