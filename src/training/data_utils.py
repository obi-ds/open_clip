"""Util functions to set up web dataset dataloader"""
import pandas as pd
from .instruct.codes import (
    CodeLabelPredictionTask,
    SingleCodeLabelPredictionTask,
    CodeLabelPredictionTaskEvaluation,
    HierarchicalCodeLabelPredictionTask
)
from .instruct.demographics import DemographicPredictionTask
from .instruct.labs import LabPredictionTask
from .instruct.codes.descriptions import ICDDescription, PHEDescription
from .instruct.codes.processing import (
    EncounterDataframeProcess,
    NegativeCodeCacheSampling,
    GroupBySampling,
    ICDConvert,
    PHEConvert,
)
from .instruct.demographics.processing import DemographicDataframeProcess
from .instruct.labs.processing import LabsDataframeProcess
from .instruct.utils import (
    get_code_label_prediction_instruction_template,
    get_patient_demographics_instruction_template
)
from .instruct.codes.processing.data_bins import AgglomerativeDataBins
from .instruct import (
    GPT2InstructTokenizer,
    InstructTokenizer
)

def get_all_code_label_prediction_task(args):
    """

    Args:
        args:

    Returns:

    """
    (
        encounter_dataframe,
        encounter_dataframe_process,
        negative_code_sampling,
        dataframe_sampling,
        code_convert,
        code_label_prediction_instructions,
        time_bins
    ) = get_code_label_task_objects(args)

    return CodeLabelPredictionTask(
        encounter_dataframe_process=encounter_dataframe_process,
        dataframe_sampling=dataframe_sampling,
        code_instructions=code_label_prediction_instructions,
        time_bins=time_bins,
        code_convert=code_convert,
        negative_code_sampling=negative_code_sampling,
        patient_id_column=args.patient_id_column,
        code_column=args.code_column,
        position_column=args.position_column,
        fixed_position_range=args.fixed_position_range
    )


def get_single_code_label_prediction_task(args):
    """

    Args:
        args:

    Returns:

    """
    (
        encounter_dataframe,
        encounter_dataframe_process,
        negative_code_sampling,
        dataframe_sampling,
        code_convert,
        code_label_prediction_instructions,
        time_bins
    ) = get_code_label_task_objects(args)

    return SingleCodeLabelPredictionTask(
        encounter_dataframe_process=encounter_dataframe_process,
        dataframe_sampling=dataframe_sampling,
        code_instructions=code_label_prediction_instructions,
        time_bins=time_bins,
        code_convert=code_convert,
        negative_code_sampling=negative_code_sampling,
        patient_id_column=args.patient_id_column,
        code_column=args.code_column,
        position_column=args.position_column,
        fixed_position_range=args.fixed_position_range
    )


def get_tree_code_label_prediction_task(args):
    """

    Args:
        args:

    Returns:

    """
    (
        encounter_dataframe,
        encounter_dataframe_process,
        negative_code_sampling,
        dataframe_sampling,
        code_convert,
        code_label_prediction_instructions,
        time_bins
    ) = get_code_label_task_objects(args)

    return HierarchicalCodeLabelPredictionTask(
        encounter_dataframe_process=encounter_dataframe_process,
        dataframe_sampling=dataframe_sampling,
        code_instructions=code_label_prediction_instructions,
        time_bins=time_bins,
        code_convert=code_convert,
        negative_code_sampling=negative_code_sampling,
        patient_id_column=args.patient_id_column,
        code_column=args.code_column,
        position_column=args.position_column,
        fixed_position_range=args.fixed_position_range
    )


def get_code_label_prediction_task_eval(args):
    """

    Args:
        args:

    Returns:

    """
    (
        encounter_dataframe,
        encounter_dataframe_process,
        negative_code_sampling,
        dataframe_sampling,
        code_convert,
        code_label_prediction_instructions,
        time_bins
    ) = get_code_label_task_objects(args)

    return CodeLabelPredictionTaskEvaluation(
        encounter_dataframe_process=encounter_dataframe_process,
        dataframe_sampling=dataframe_sampling,
        code_instructions=code_label_prediction_instructions,
        time_bins=time_bins,
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
    demographic_instructions = get_patient_demographics_instruction_template()
    demographic_dataframe_process = DemographicDataframeProcess(
        demographic_dataframe=demographic_dataframe
    )
    demographic_prediction_task = DemographicPredictionTask(
        demographic_dataframe_process=demographic_dataframe_process,
        demographic_instructions=demographic_instructions
    )
    return demographic_prediction_task

def get_lab_task(args):
    """

    Args:
        args:

    Returns:

    """
    lab_dataframes = get_lab_dataframes(labs_folder=args.labs_folder)
    lab_prediction_instructions = get_code_label_prediction_instruction_template()
    lab_dataframe_process = get_lab_dataframe_process(lab_dataframes=lab_dataframes, args=args)
    time_bins = get_time_bins(args)
    return LabPredictionTask(
        lab_dataframe_process=lab_dataframe_process,
        lab_instructions=lab_prediction_instructions,
        time_bins=time_bins,
        patient_id_column=args.patient_id_column,
        lab_name_column=args.lab_name_column,
        position_column=args.position_column,
        fixed_position_range=args.fixed_position_range
    )


def get_instruct_tokenizer(tokenizer, ignore_index, args):
    """

    Args:
        tokenizer:
        ignore_index:
        args:

    Returns:

    """
    if 'gpt' in args.model:
        return GPT2InstructTokenizer(
            tokenizer=tokenizer, pad_id=args.pad_id, max_seq_length=args.max_seq_length, ignore_index=ignore_index
        )
    else:
        return InstructTokenizer(
            tokenizer=tokenizer, pad_id=args.pad_id, max_seq_length=args.max_seq_length, ignore_index=ignore_index
        )

def get_code_label_task_objects(args):
    """

    Args:
        args:

    Returns:

    """
    encounter_dataframe = get_encounter_dataframe(encounter_file=args.encounter_file)
    encounter_dataframe_process = get_encounter_dataframe_process(encounter_dataframe, args)
    negative_code_sampling = get_negative_code_sampling(encounter_dataframe_process, args)
    dataframe_sampling = GroupBySampling()
    code_convert = get_code_convert(args=args)
    code_label_prediction_instructions = get_code_label_prediction_instruction_template()
    time_bins = get_time_bins(args)
    return (
        encounter_dataframe,
        encounter_dataframe_process,
        negative_code_sampling,
        dataframe_sampling,
        code_convert,
        code_label_prediction_instructions,
        time_bins
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

def get_time_bins(args):
    """

    Args:
        args:

    Returns:

    """
    return [
        AgglomerativeDataBins(distance_threshold=distance_threshold)
        for distance_threshold in args.distance_threshold
    ]

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
        'labs_1k_2401_suffix_0.parquet',
        'labs_1k_2401_suffix_1.parquet',
        'labs_1k_2401_suffix_2.parquet',
        'labs_1k_2401_suffix_3.parquet',
        'labs_1k_2401_suffix_4.parquet',
        'labs_1k_2401_suffix_5.parquet',
        'labs_1k_2401_suffix_6.parquet',
        'labs_1k_2401_suffix_7.parquet',
        'labs_1k_2401_suffix_8.parquet',
        'labs_1k_2401_suffix_9.parquet'
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
