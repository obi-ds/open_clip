"""Util functions to set up web dataset dataloader"""
import pandas as pd
from .instruct.diagnosis import (
    DiagnosisLabelPredictionTask,
    DiagnosisLabelPredictionPrompt,
    DiagnosisLabelPredictionTaskEvaluation,
    HierarchicalDiagnosisLabelPredictionTask,
    HierarchicalDiagnosisLabelPredictionTaskEvaluation
)
from .instruct.demographics import DemographicPredictionTask, DemographicPredictionPrompt
from .instruct.ecg_attributes import ECGAttributePredictionTask, ECGAttributePredictionPrompt
from .instruct.labs import LabPredictionTask, LabPredictionPrompt
from .instruct.diagnosis.descriptions import ICDDescription, PHEDescription
from .instruct.diagnosis.processing import (
    EncounterDataframeProcess,
    NegativeCodeCacheSampling,
    ICDConvert,
    PHEConvert,
)
from .instruct.demographics.processing import DemographicDataframeProcess
from .instruct.labs.processing import LabsDataframeProcess
from .instruct.utils import (
    get_diagnosis_label_prediction_instruction_template,
    get_patient_demographics_instruction_template,
    get_patient_labs_instruction_template,
    get_ecg_attributes_instruction_template,
    get_hierarchical_diagnosis_label_prediction_instruction_template
)
from .instruct import (
    InstructTokenizer
)


def get_diagnosis_label_prediction_task(args):
    """

    Args:
        args:

    Returns:

    """
    (
        encounter_dataframe,
        encounter_dataframe_process,
        negative_code_sampling,
        code_convert,
        diagnosis_label_prediction_instructions,
    ) = get_diagnosis_label_task_objects(args)

    return DiagnosisLabelPredictionTask(
        encounter_dataframe_process=encounter_dataframe_process,
        diagnosis_instructions=diagnosis_label_prediction_instructions,
        code_convert=code_convert,
        negative_code_sampling=negative_code_sampling,
        patient_id_column=args.patient_id_column,
        code_column=args.code_column,
        position_column=args.position_column,
        fixed_position_range=args.fixed_position_range,
        update_code_counts=args.update_code_counts,
    )


def get_hierarchical_diagnosis_label_prediction_task(args):
    """

    Args:
        args:

    Returns:

    """
    (
        encounter_dataframe,
        encounter_dataframe_process,
        negative_code_sampling,
        code_convert,
        _,
    ) = get_diagnosis_label_task_objects(args)

    diagnosis_label_prediction_instructions = get_hierarchical_diagnosis_label_prediction_instruction_template(
        task_separator=get_task_separator(args=args)
    )

    return HierarchicalDiagnosisLabelPredictionTask(
        encounter_dataframe_process=encounter_dataframe_process,
        diagnosis_instructions=diagnosis_label_prediction_instructions,
        code_convert=code_convert,
        negative_code_sampling=negative_code_sampling,
        patient_id_column=args.patient_id_column,
        code_column=args.code_column,
        position_column=args.position_column,
        fixed_position_range=args.fixed_position_range
    )


def get_diagnosis_label_prediction_task_eval(args):
    """

    Args:
        args:

    Returns:

    """
    (
        encounter_dataframe,
        encounter_dataframe_process,
        negative_code_sampling,
        code_convert,
        diagnosis_label_prediction_instructions,
    ) = get_diagnosis_label_task_objects(args)

    return DiagnosisLabelPredictionTaskEvaluation(
        encounter_dataframe_process=encounter_dataframe_process,
        diagnosis_instructions=diagnosis_label_prediction_instructions,
        code_convert=code_convert,
        negative_code_sampling=negative_code_sampling,
        patient_id_column=args.patient_id_column,
        code_column=args.code_column,
        position_column=args.position_column,
    )


def get_hierarchical_diagnosis_label_prediction_task_eval(args):
    """

    Args:
        args:

    Returns:

    """
    (
        encounter_dataframe,
        encounter_dataframe_process,
        negative_code_sampling,
        code_convert,
        _,
    ) = get_diagnosis_label_task_objects(args)

    diagnosis_label_prediction_instructions = get_diagnosis_label_prediction_instruction_template(
        task_separator=get_task_separator(args=args),
        x_y_delimiter=' '
    )

    return HierarchicalDiagnosisLabelPredictionTaskEvaluation(
        encounter_dataframe_process=encounter_dataframe_process,
        diagnosis_instructions=diagnosis_label_prediction_instructions,
        code_convert=code_convert,
        negative_code_sampling=negative_code_sampling,
        patient_id_column=args.patient_id_column,
        code_column=args.code_column,
        position_column=args.position_column,
    )


def get_demographic_task(args):
    """

    Args:
        args:

    Returns:

    """
    demographic_dataframe = get_demographic_dataframe(filepath=args.demographic_file)
    demographic_instructions = get_patient_demographics_instruction_template(
        task_separator=get_task_separator(args=args)
    )
    demographic_dataframe_process = DemographicDataframeProcess(
        demographic_dataframe=demographic_dataframe
    )
    demographic_prediction_task = DemographicPredictionTask(
        demographic_dataframe_process=demographic_dataframe_process,
        demographic_instructions=demographic_instructions
    )
    return demographic_prediction_task


def get_ecg_attributes_task(args):
    """

    Args:
        args:

    Returns:

    """
    ecg_attribute_instructions = get_ecg_attributes_instruction_template(
        task_separator=get_task_separator(args=args)
    )
    return ECGAttributePredictionTask(
        ecg_attribute_instructions=ecg_attribute_instructions,
    )


def get_lab_task(args):
    """

    Args:
        args:

    Returns:

    """
    lab_dataframes = get_lab_dataframes(labs_folder=args.labs_folder)
    lab_prediction_instructions = get_patient_labs_instruction_template(
        task_separator=get_task_separator(args=args)
    )
    lab_dataframe_process = get_lab_dataframe_process(lab_dataframes=lab_dataframes, args=args)
    return LabPredictionTask(
        lab_dataframe_process=lab_dataframe_process,
        lab_instructions=lab_prediction_instructions,
        patient_id_column=args.patient_id_column,
        lab_name_column=args.lab_name_column,
        position_column=args.position_column,
        fixed_position_range=args.fixed_position_range,
        update_lab_counts=args.update_lab_counts
    )


def get_demographic_prompt(args):
    """

    Args:
        args:

    Returns:

    """
    demographic_dataframe = get_demographic_dataframe(filepath=args.demographic_file)
    demographic_instructions = get_patient_demographics_instruction_template(
        task_separator=get_task_separator(args=args)
    )
    demographic_dataframe_process = DemographicDataframeProcess(
        demographic_dataframe=demographic_dataframe
    )
    return DemographicPredictionPrompt(
        demographic_dataframe_process=demographic_dataframe_process,
        demographic_instructions=demographic_instructions,
    )

def get_ecg_attributes_prompt(args):
    """

    Args:
        args:

    Returns:

    """
    ecg_attribute_instructions = get_ecg_attributes_instruction_template(
        task_separator=get_task_separator(args=args)
    )
    return ECGAttributePredictionPrompt(
        ecg_attribute_instructions=ecg_attribute_instructions,
    )


def get_lab_prompt(args):
    """

    Args:
        args:

    Returns:

    """

    lab_dataframes = get_lab_dataframes(labs_folder=args.labs_folder)
    lab_prediction_instructions = get_patient_labs_instruction_template(
        task_separator=get_task_separator(args=args)
    )
    lab_dataframe_process = get_lab_dataframe_process(lab_dataframes=lab_dataframes, args=args)
    return LabPredictionPrompt(
        lab_dataframe_process=lab_dataframe_process,
        lab_instructions=lab_prediction_instructions,
        patient_id_column=args.patient_id_column,
        lab_name_column=args.lab_name_column,
        position_column=args.position_column,
        fixed_position_range=args.fixed_position_range,
    )

def get_diagnosis_prompt(args):
    (
        encounter_dataframe,
        encounter_dataframe_process,
        negative_code_sampling,
        code_convert,
        diagnosis_label_prediction_instructions,
    ) = get_diagnosis_label_task_objects(args)

    return DiagnosisLabelPredictionPrompt(
        encounter_dataframe_process=encounter_dataframe_process,
        diagnosis_instructions=diagnosis_label_prediction_instructions,
        code_convert=code_convert,
        negative_code_sampling=negative_code_sampling,
        patient_id_column=args.patient_id_column,
        code_column=args.code_column,
        position_column=args.position_column,
    )


def get_instruct_tokenizer(tokenizer, ignore_index, args):
    """

    Args:
        tokenizer:
        ignore_index:
        args:

    Returns:

    """
    return InstructTokenizer(
        tokenizer=tokenizer,
        pad_id=args.pad_id,
        padding_side=args.padding_side,
        max_seq_length=args.max_seq_length,
        token_loss_weighting=args.token_loss_weighting,
        ignore_index=ignore_index,
    )


def get_diagnosis_label_task_objects(args):
    """

    Args:
        args:

    Returns:

    """
    encounter_dataframe = get_encounter_dataframe(encounter_file=args.encounter_file)
    encounter_dataframe_process = get_encounter_dataframe_process(encounter_dataframe, args)
    negative_code_sampling = get_negative_code_sampling(encounter_dataframe_process, args)
    code_convert = get_code_convert(args=args)
    diagnosis_label_prediction_instructions = get_diagnosis_label_prediction_instruction_template(
        task_separator=get_task_separator(args=args)
    )
    return (
        encounter_dataframe,
        encounter_dataframe_process,
        negative_code_sampling,
        code_convert,
        diagnosis_label_prediction_instructions,
    )


def get_encounter_dataframe(encounter_file):
    """

    Args:
        encounter_file:

    Returns:

    """
    return pd.read_parquet(
        encounter_file, columns=['PatientID', 'ContactDTS', 'ICD10CD', 'phecode', 'idf']
    )


def get_encounter_dataframe_process(encounter_dataframe, args):
    """

    Args:
        encounter_dataframe:
        args:

    Returns:

    """
    return EncounterDataframeProcess(
        encounter_dataframe=encounter_dataframe,
        patient_id_column=args.patient_id_column,
        contact_date_column=args.contact_date_column,
        time_difference_column=args.time_difference_column,
        code_column=args.code_column,
        position_column=args.position_column,
    )


def get_negative_code_sampling(encounter_dataframe_process, args):
    """

    Args:
        encounter_dataframe_process:
        args:

    Returns:

    """
    return NegativeCodeCacheSampling(
        encounter_dataframe_process=encounter_dataframe_process,
        negatives_type=args.negatives_type,
        code_task_negative_cache_size=100,
    )


def get_code_convert(args):
    """

    Args:
        args:

    Returns:

    """
    if 'phe' in args.code_column:
        return PHEConvert(
            descriptions=PHEDescription(),
            lowercase=False
        )
    else:
        return ICDConvert(
            descriptions=ICDDescription(),
            billable_probability=args.billable_probability,
            top_non_probability=args.top_non_probability,
            mixed_non_probability=args.mixed_non_probability,
            lowercase=False
        )


def get_demographic_dataframe(filepath):
    """

    Args:
        filepath:

    Returns:

    """
    return pd.read_parquet(filepath)


def get_lab_dataframes(labs_folder):
    """

    Args:
        labs_folder:

    Returns:

    """
    lab_suffixes = [
        'mgh_2020_labs_0_pd.parquet',
        'mgh_2020_labs_1_pd.parquet',
        'mgh_2020_labs_2_pd.parquet',
        'mgh_2020_labs_3_pd.parquet',
        'mgh_2020_labs_4_pd.parquet',
        'mgh_2020_labs_5_pd.parquet',
        'mgh_2020_labs_6_pd.parquet',
        'mgh_2020_labs_7_pd.parquet',
        'mgh_2020_labs_8_pd.parquet',
        'mgh_2020_labs_9_pd.parquet'
    ]
    return [pd.read_parquet(f'{labs_folder}/{lab_suffix}') for lab_suffix in lab_suffixes]


def get_lab_dataframe_process(lab_dataframes, args):
    """

    Args:
        lab_dataframes:
        args:

    Returns:

    """
    return LabsDataframeProcess(
        lab_dataframes=lab_dataframes,
        patient_id_column=args.patient_id_column,
        time_difference_column=args.time_difference_column,
        code_column=args.code_column,
        position_column=args.position_column,
    )


def get_example_separator(args):
    """

    Args:
        args:

    Returns:

    """
    return '\n'

def get_task_separator(args):
    """

    Args:
        args:

    Returns:

    """
    if 'biogpt' in args.model or 'bio_gpt' in args.model:
        return '</s>\n'
    else:
        return '<end_of_text>\n'
