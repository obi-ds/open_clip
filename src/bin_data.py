import os
import sys

import blosc
import pandas as pd
import webdataset as wds

from bin_args import get_args
from open_clip import get_cast_dtype
from training.data import get_sample_keys
from training.instruct.codes.processing import (
    EncounterDataframeProcess,
)
from training.params import parse_args
from training.precision import get_autocast, get_input_dtype

start, end = sys.argv[1], sys.argv[2]


def get_encounter_history(patient_id, current_time, encounter_dataframe_process):
    return encounter_dataframe_process.get_patient_encounter_history_with_position(
        patient_id=patient_id,
        current_time=current_time,
        use_log_position=args.use_log_position,
        time_difference_normalize=args.time_difference_normalize
    )


def binned_sample_status(sample_history, eval_code):
    # set unmapped phecodes to 'NA'
    sample_history.loc[sample_history.phecode.isna(), 'phecode'] = 'NOT_MAPPED'
    pos_diagnoses = sample_history.loc[sample_history.phecode.str.contains(eval_code, regex=False)].copy()

    time_bins = [-10000, -1080, -720, -360, -180, -90, -30, -1, 30, 90, 180, 360, 720, 1080, 10000]
    bins = time_bins
    pos_diagnoses['position_bin'] = pd.cut(pos_diagnoses['position'], bins, include_lowest=True)
    # count the number of occurrences in each bin
    bin_counts = pos_diagnoses['position_bin'].value_counts(sort=False).reindex(pd.IntervalIndex.from_breaks(bins),
                                                                                fill_value=0)
    bin_counts_df = pd.DataFrame([bin_counts.tolist()], columns=bin_counts.index.astype(str))
    bin_counts_df['all'] = pos_diagnoses.shape[0]

    return bin_counts_df


def get_eval_dataloader(args):
    image_key, text_key = get_sample_keys(args)
    eval_dataset_image = (
        wds.WebDataset(args.val_data)
        .decode(wds.handle_extension("blosc", blosc.unpack_array))
        .rename(image=image_key, text=text_key)
        .to_tuple("image", "text")
        .batched(args.batch_size)
    )
    return eval_dataset_image


def evaluate_label(dataloader, eval_code, args):
    all_sample_history = []
    all_metadata = []

    for batch in dataloader:
        _, sample_metadata = batch

        sample_encounter_history = [
            get_encounter_history(sample[args.patient_id_column], sample[args.sample_result_date_column],
                                  encounter_dataframe_process)
            for sample in sample_metadata if
            encounter_dataframe_process.check_patient_id(patient_id=sample[args.patient_id_column])
        ]
        all_sample_history.extend(sample_encounter_history)

        all_metadata.extend([
            {
                args.patient_id_column: metadata[args.patient_id_column],
                args.sample_result_date_column: metadata[args.sample_result_date_column],
                'TestTime': metadata['TestTime'] if 'TestTime' in metadata else 'NA',
                'DATE_TIME': metadata['DATE_TIME'] if 'DATE_TIME' in metadata else 'NA',
                'file': metadata['file']
            }
            for metadata in sample_metadata if
            encounter_dataframe_process.check_patient_id(patient_id=metadata[args.patient_id_column])
        ])

    print('Forward pass complete')

    # binned_status_df = Parallel(n_jobs=128)(delayed(binned_sample_status)(x, eval_code) for x in all_sample_history)

    binned_status_df = [binned_sample_status(x, eval_code) for x in all_sample_history]

    print('Bin complete')

    binned_status_df = pd.concat(binned_status_df)

    metadata_df = pd.DataFrame(all_metadata)

    print('Meta complete')

    return pd.concat([metadata_df, binned_status_df.reset_index(drop=True)], axis=1)


tokenizer = None

for file_suffix, args_str in get_args():
    args_str = args_str.replace('"', '')
    args = parse_args(args_str.split())

    encounter_dataframe = pd.read_parquet(
        args.encounter_file, columns=['PatientID', 'ContactDTS', 'ICD10CD', 'phecode']
    )

    encounter_dataframe_process = EncounterDataframeProcess(
        encounter_dataframe=encounter_dataframe,
        patient_id_column=args.patient_id_column,
        contact_date_column=args.contact_date_column,
        time_difference_column=args.time_difference_column,
        code_column=args.code_column,
        position_column=args.position_column,
    )

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    input_dtype = get_input_dtype(args.precision)

    phecodes_file = '/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/phecode/phecodeX_info.csv'
    test_phecodes_df = pd.read_csv(phecodes_file, encoding='ISO-8859-1')
    test_phecodes = test_phecodes_df['phecode']

    # phecodes_file = './eval_code_list.csv'
    # test_phecodes = pd.read_csv(phecodes_file)['0']

    filepath = f'/mnt/obi0/phi/ehr_projects/bloodcell_clip/evaluation/ecg/time_bins/{file_suffix}'

    os.makedirs(filepath, exist_ok=True)

    dataloader = get_eval_dataloader(args=args)

    for phecode in test_phecodes[int(start): int(end)]:
        print('PHECODE: ', phecode)
        binned_status_df = evaluate_label(dataloader=dataloader, eval_code=phecode, args=args)
        binned_status_df.to_parquet(f'{filepath}/{phecode}.parquet')