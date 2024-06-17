import sys
import pandas as pd
from training.data import get_wds_dataset_icd_prompt
from training.params import parse_args

from training.instruct.demographics.processing import DemographicDataframeProcess
from training.data_utils import (
    get_demographic_dataframe,
    get_encounter_dataframe,
    get_encounter_dataframe_process,
    get_lab_dataframe_process,
    get_lab_dataframes,
)

start, end = sys.argv[1], sys.argv[2]

def get_lab_history(
        patient_id,
        current_time,
        code_column='ExternalNM_lower',
        use_log_position=False,
        time_difference_normalize=1
):
    lab_history = lab_dataframe_process.get_patient_encounter_history_with_position(
        patient_id=patient_id,
        current_time=current_time,
        use_log_position=use_log_position,
        time_difference_normalize=time_difference_normalize
    )
    lab_history.dropna(subset=code_column, inplace=True)
    return lab_history


def get_labels(dataloader, eval_attribute, args):
    eval_lab_labels = list()
    for i, batch in enumerate(dataloader):
        images, metadata_list = batch
        for metadata in metadata_list:
            if not lab_dataframe_process.check_patient_id(patient_id=metadata[args.patient_id_column]):
                continue
            patient_id, current_time = metadata[args.patient_id_column], metadata[args.sample_result_date_column]
            lab_history = get_lab_history(patient_id=patient_id, current_time=current_time)
            eval_lab_history = lab_history[lab_history['ExternalNM_lower'] == eval_attribute.lower()]
            closest_past_value = eval_lab_history['position'][eval_lab_history['position'] < 0].max()
            closest_future_value = eval_lab_history['position'][eval_lab_history['position'] >= 0].min()
            closest = eval_lab_history[
                (eval_lab_history['position'] == closest_future_value) | (
                            eval_lab_history['position'] == closest_past_value)
                ]
            closest.loc[:, ['SampleDate']] = current_time
            closest.loc[:, ['SampleTime']] = metadata['TestTime']
            closest.loc[:, ['SampleFile']] = metadata['file']
            eval_lab_labels.append(closest)
    return pd.concat(eval_lab_labels)


args_str = f'--train-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_train_2403/shard_{{' \
           f'0000..0078}}.tar"  \
    --val-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2403/shard_{{0000..0010}}.tar"  \
    --train-num-samples 252800 \
    --val-num-samples 35200 \
    --dataset-type icddataset \
    --name generate \
    --workers 8 \
    --batch-size 3200 \
    --encounter-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/all_encounters_2308_with_phecodes_with_na.parquet.check" \
    --seed 0'.replace('"', '')

args = parse_args(args_str.split())
args.model = 'ecg'

encounter_file = "/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/all_encounters_2308_with_phecodes_with_na" \
                 ".parquet.check"
demographic_file = "/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/demographics_2404.parquet"
labs_folder = "/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/labs"

encounter_dataframe = get_encounter_dataframe(encounter_file=encounter_file)
encounter_dataframe_process = get_encounter_dataframe_process(encounter_dataframe, args)

demographic_dataframe = get_demographic_dataframe(filepath=demographic_file)
demographic_dataframe_process = DemographicDataframeProcess(
    demographic_dataframe=demographic_dataframe
)

# Uncomment if using labs
lab_dataframes = get_lab_dataframes(labs_folder=labs_folder)
lab_dataframe_process = get_lab_dataframe_process(lab_dataframes=lab_dataframes, args=args)

# Load dataloader
eval_dataset_image = get_wds_dataset_icd_prompt(
    args,
    preprocess_img=lambda x: x,
    is_train=False,
    tokenizer=None,
    return_sample=False,
)
eval_dataloader = eval_dataset_image.dataloader

attribute_file = '/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/labs/test_labs_with_name.csv'
attribute_name_column = 'PromptName'
eval_attributes = pd.read_csv(attribute_file)[attribute_name_column]

filepath = '/mnt/obi0/phi/ehr_projects/bloodcell_clip/evaluation/ecg/model_generations/24_03_mgh_val/labels'

for attribute in eval_attributes[int(start): int(end)]:
    print(f'Start - Evaluate {attribute}: ')
    attribute_labels = get_labels(eval_dataloader, attribute, args)
    attribute_labels.to_parquet(f'{filepath}/{attribute}.parquet')
    print(f'Finish - Evaluate {attribute}: ')
