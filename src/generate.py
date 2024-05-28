import os
import sys
import argparse
from pathlib import Path
from dateutil import relativedelta

import pandas as pd
import torch

from generate_args import get_args_for_generation
from open_clip import get_cast_dtype
from open_clip.factory import get_tokenizer, create_model_and_transforms
from training.data import get_wds_dataset_icd_prompt
from training.params import parse_args
from training.precision import get_autocast, get_input_dtype
from open_clip.tokenizer import HFTokenizer, decode
from training.instruct.codes.processing import EncounterDataframeProcess
from training.instruct.demographics.processing import DemographicDataframeProcess
from training.data_utils import (
    get_demographic_dataframe,
    get_encounter_dataframe,
    get_instruct_tokenizer
)

from main import get_eos_token_id

parser = argparse.ArgumentParser()
parser.add_argument(
    "--gpu",
    type=str,
    required=True,
    help="The GPU to run eval on",
)
parser.add_argument(
    "--batch-size",
    type=int,
    required=True,
    help="The batch size",
)
parser.add_argument(
    "--category",
    type=str,
    required=True,
    choices=['Age', 'Sex', 'Weight', 'Height', 'QRS Duration', 'QT Interval', 'Ventricular Rate', 'T Axis'],
    help="Evaluate a specific demographic category",
)
parser.add_argument(
    "--eval-data",
    type=str,
    required=True,
    help="The path where the shards are located",
)
parser.add_argument(
    "--num-samples",
    type=str,
    required=True,
    help="The number of samples in the shards",
)
parser.add_argument(
    "--model-type",
    type=str,
    required=True,
    help="The config file used to train the model",
)
parser.add_argument(
    "--model-folder",
    type=str,
    required=True,
    help="The path to the model we are evaluating",
)
parser.add_argument(
    "--eval-every-epoch",
    type=int,
    required=True,
    help="Evaluate the models at every epoch",
)
parser.add_argument(
    "--epoch-start",
    type=int,
    required=True,
    help="Start evaluating from this epoch",
)
parser.add_argument(
    "--code-column",
    type=str,
    default='phecode',
    help="Column containing the codes",
)
parser.add_argument(
    "--result-date-column",
    type=str,
    default='TestDate',
    help="Column containing the codes",
)
parser.add_argument(
    "--output-folder",
    type=str,
    required=True,
    help="Where to write the binned data",
)
parser.add_argument(
    "--file-suffix",
    type=str,
    required=True,
    help="A suffix to distinguish between different dataset",
)

eval_args = parser.parse_args(sys.argv[1:])

os.environ["CUDA_VISIBLE_DEVICES"] = str(eval_args.gpu)


def get_eval_dataloader(gen_args, gen_tokenizer):
    eval_dataset_image = get_wds_dataset_icd_prompt(
        gen_args,
        preprocess_img=lambda x: x,
        is_train=False,
        tokenizer=gen_tokenizer,
        return_sample=False,
    )
    return eval_dataset_image.dataloader


# Function to convert prompt into tokens
def get_custom_prompt(gen_tokenizer, text, batch_size):
    input_tokens = gen_tokenizer.get_input_tokens(text)
    return torch.tensor(input_tokens).tile(batch_size, 1)


# Get the true value of demographics
def get_true_category(gen_category, patient_demographics, current_time, metadata):
    if gen_category.lower() == 'age':
        return get_true_age(patient_demographics, current_time)
    elif gen_category.lower() == 'sex':
        return get_true_sex(patient_demographics)
    elif gen_category.lower() == 'height':
        return get_true_height(metadata)
    elif gen_category.lower() == 'weight':
        return get_true_weight(metadata)
    elif gen_category.lower().replace(' ', '') == 'qrsduration':
        return get_true_qrs_duration(metadata)
    elif gen_category.lower().replace(' ', '') == 'qtinterval':
        return get_true_qt_interval(metadata)
    elif gen_category.lower().replace(' ', '') == 'ventricularrate':
        return get_true_ventricular_rate(metadata)
    elif gen_category.lower().replace(' ', '') == 'taxis':
        return get_true_t_axis(metadata)


def get_true_age(patient_demographics, current_time):
    time_delta = (
        relativedelta.relativedelta(
            pd.to_datetime(current_time),
            pd.to_datetime(patient_demographics['BirthDTS'])
        )
    )
    years = time_delta.years
    return years


def get_true_sex(patient_demographics):
    return patient_demographics['SexDSC']


def get_true_height(metadata):
    return metadata.get('HeightIN', None)


def get_true_weight(metadata):
    return metadata.get('WeightLBS', None)


def get_true_qrs_duration(metadata):
    return metadata.get('QRSDuration', None)


def get_true_qt_interval(metadata):
    return metadata.get('QTInterval', None)


def get_true_ventricular_rate(metadata):
    return metadata.get('VentricularRate', None)


def get_true_t_axis(metadata):
    return metadata.get('TAxis', None)


# Convert the predicted tokens to string
def get_predicted_string(predicted_token):
    if isinstance(tokenizer, HFTokenizer):
        predicted_string = tokenizer.tokenizer.decode(predicted_token)
        return predicted_string
    else:
        predicted_string = decode(predicted_token)
        return predicted_string.replace(' ', '')


# The final output contains the initial prompt and the predicted tokens
# Use this function to extract only the predicted tokens
def get_predicted_token(generated_token, prompt_token, gen_eos_token_id, gen_pad_token_id):
    predicted_token = generated_token[len(prompt_token):]
    predicted_token = predicted_token[(predicted_token != gen_eos_token_id) & (predicted_token != gen_pad_token_id)]
    return predicted_token


def generate_text(gen_dataloader, gen_category, prompt_tokens_full, gen_pad_token_id, gen_eos_token_id, gen_args, gen_model):
    generated_dataset = list()
    with torch.no_grad():
        for i, batch in enumerate(gen_dataloader):
            images, batch_metadata = batch
            images = images.to(device=device, dtype=input_dtype, non_blocking=True)
            prompt_tokens = prompt_tokens_full[:images.size()[0]]

            try:
                generated_tokens = gen_model.generate(
                    images,
                    prompt_tokens,
                    generation_type='top_k',
                    pad_token_id=gen_pad_token_id,
                    eos_token_id=gen_eos_token_id
                )
            except:
                print('Fail')
                print(images.shape)
                print(prompt_tokens.shape)
                raise ValueError()

            for index, (sample_metadata, generated_token) in enumerate(zip(batch_metadata, generated_tokens)):
                patient_demographics = demographic_dataframe_process.get_patient_demographics(
                    patient_id=sample_metadata[gen_args.patient_id_column],
                )
                true_category = get_true_category(
                    gen_category=gen_category,
                    patient_demographics=patient_demographics,
                    current_time=sample_metadata[gen_args.sample_result_date_column],
                    metadata=sample_metadata
                )
                predicted_token = get_predicted_token(
                    generated_token, prompt_tokens[index], eos_token_id, pad_token_id
                )
                predicted_string = get_predicted_string(generated_token)
                predicted_value = get_predicted_string(predicted_token=predicted_token)

                generated_sample = {
                    'PatientID': sample_metadata[gen_args.patient_id_column],
                    'TrueValue': true_category,
                    'PredictedValue': predicted_value,
                    'PredictedString': predicted_string,
                    'GeneratedTokens': generated_token.detach().cpu().tolist()
                }

                generated_dataset.append(generated_sample)

    return generated_dataset


encounter_file = "/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/all_encounters_2308_with_phecodes_with_na" \
                 ".parquet.check"
demographic_file = "/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/demographics_2404.parquet"
labs_folder = "/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/labs"

encounter_dataframe = get_encounter_dataframe(encounter_file=encounter_file)
demographic_dataframe = get_demographic_dataframe(filepath=demographic_file)

category = eval_args.category
if category.lower() in ['age', 'sex', 'height', 'weight']:
    prompt = f'Patient attributes\n* {category}: '
else:
    prompt = f'Labs in the next 6 months\n* {category}: '
print('Prompt:\n', prompt)


for file_suffix, args_str, model_type, model_path in get_args_for_generation(
        model_type=eval_args.model_type,
        model_folder=eval_args.model_folder,
        eval_every_epoch=eval_args.eval_every_epoch,
        eval_data=eval_args.eval_data,
        num_samples=eval_args.num_samples,
        batch_size=eval_args.batch_size,
        epoch_start=eval_args.epoch_start,
        code_column=eval_args.code_column,
        file_suffix=eval_args.file_suffix,
        result_date_column=eval_args.result_date_column
):

    args_str = args_str.replace('"', '')
    args = parse_args(args_str.split())

    print('Dataset: ', args.val_data)
    print('Model: ', model_path)

    prefix = Path(model_path).parent.parent.name
    epoch = Path(model_path).name.split('.')[0]

    filepath = f'{eval_args.output_folder}{file_suffix}/{prefix}/' \
               f'{epoch}/' \
               f'{args.eval_start_time}-{args.eval_end_time}'

    # if os.path.exists(filepath):
    #     print('Skipping: This result already exists')
    #     continue

    os.makedirs(filepath, exist_ok=True)

    encounter_dataframe_process = EncounterDataframeProcess(
        encounter_dataframe=encounter_dataframe,
        patient_id_column=args.patient_id_column,
        contact_date_column=args.contact_date_column,
        time_difference_column=args.time_difference_column,
        code_column=args.code_column,
        position_column=args.position_column,
    )

    demographic_dataframe_process = DemographicDataframeProcess(
        demographic_dataframe=demographic_dataframe
    )

    model, _, preprocess_val = create_model_and_transforms(
        model_type,
        pretrained=model_path,
    )

    tokenizer = get_tokenizer(model_type, context_length=args.max_seq_length)
    instruct_tokenizer = get_instruct_tokenizer(tokenizer, -100, args)
    eos_token_id = get_eos_token_id(model_name=model_type, tokenizer=tokenizer)
    pad_token_id = 1 if 'gpt' in model_type else 0

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    input_dtype = get_input_dtype(args.precision)

    device = torch.device('cuda')
    model = model.eval()
    model = model.to(device)

    print('Category: ', category)
    dataloader = get_eval_dataloader(gen_args=args, gen_tokenizer=tokenizer)
    eos_token_id = get_eos_token_id(model_name=args.model, tokenizer=tokenizer)
    print('EOS Token ID: ', eos_token_id)

    # This is the prompt that was used in training
    prompt_tokens_all = get_custom_prompt(gen_tokenizer=instruct_tokenizer, text=prompt, batch_size=args.batch_size)
    prompt_tokens_all = prompt_tokens_all.to(device=device, non_blocking=True)
    print('Generation Start')
    predicted_generated_dataset = generate_text(
        gen_dataloader=dataloader,
        gen_category=category,
        prompt_tokens_full=prompt_tokens_all,
        gen_eos_token_id=eos_token_id,
        gen_pad_token_id=pad_token_id,
        gen_args=args,
        gen_model=model
    )
    print('Generation Finish')
    pd.DataFrame(predicted_generated_dataset).to_parquet(f'{filepath}/{category}.parquet')
