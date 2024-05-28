import copy
import os
import sys
import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.nn import functional as F

from eval_args import get_model_details_for_eval
from open_clip import get_cast_dtype
from open_clip.factory import get_tokenizer, create_model_and_transforms
from training.data import get_wds_dataset_icd_instruct
from training.instruct.codes.processing import (
    EncounterDataframeProcess,
)
from training.params import parse_args
from training.precision import get_autocast, get_input_dtype

from main import get_token_id, get_eos_token_id

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
    "--phecode-file",
    type=str,
    required=True,
    help="The location to the phecode file",
)
parser.add_argument(
    "--start",
    type=int,
    required=True,
    help="The start position of the phecode file",
)
parser.add_argument(
    "--end",
    type=int,
    required=True,
    help="The end position of the phecode file",
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
parser.add_argument(
    "--overwrite",
    type=bool,
    default=False,
    help="Overwrite existing results",
)
parser.add_argument(
    "--demographic-prompt-attributes",
    nargs='+',
    default=None,
    help="The demographic attributes to use in the prompt"
)


eval_args = parser.parse_args(sys.argv[1:])

os.environ["CUDA_VISIBLE_DEVICES"] = str(eval_args.gpu)

def get_eval_dataloader(args, eval_code, tokenizer):
    args = copy.deepcopy(args)
    args.eval_code = eval_code
    eval_dataset_image = get_wds_dataset_icd_instruct(
        args,
        preprocess_img=lambda x: x,
        is_train=False,
        tokenizer=tokenizer,
        return_sample=True,
        eval_mode=True
    )
    return eval_dataset_image.dataloader


def compute_generative_loss(logits, labels, ignore_index=-100, reduction='mean'):
    if logits.dim() == 2:
        logits = logits.unsqueeze(1)
        labels = labels.unsqueeze(1)
    return F.cross_entropy(logits.permute(0, 2, 1), labels, ignore_index=ignore_index, reduction=reduction)


def compute_probability(loss):
    mask = loss != 0
    loss_mean = (loss * mask).sum(dim=1) / mask.sum(dim=1)
    return torch.exp(-loss_mean)


def get_token_scores(model_name, logits, tokens):
    return {token: logits[:, get_token_id(model_name, tokenizer, token)] for token in tokens}


def convert_instructions_list_to_string(instructions):
    full_string = ''
    for instruction in instructions:
        full_string += instruction[0] + instruction[1]
    return full_string


def evaluate_label(dataloader, tokens, eos_token_id, args, ignore_index=-100):
    all_labels, all_metadata = [], []
    all_scores = {token: [] for token in tokens}
    all_scores['probability'] = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            images, texts, sample_metadata = batch
            images = images.to(device=device, dtype=input_dtype, non_blocking=True)
            texts = texts.to(device=device, non_blocking=True)

            # with autocast():
            model_out = model(images, texts)
            model_logits = model_out['logits']
            model_labels = model_out['labels']

            # Ignore loss/probability of pad and eos tokens
            # labels[labels == eos_token_id] = ignore_index
            # masked_logits = logits[labels != ignore_index]

            mask = (model_labels != ignore_index) & (model_labels != eos_token_id)
            labels = model_labels[mask]
            logits = model_logits[mask]

            loss = compute_generative_loss(
                logits=logits,
                labels=labels,
                reduction='none'
            )

            token_scores = get_token_scores(model_name=args.model, logits=logits, tokens=tokens)
            probability = compute_probability(loss)

            all_scores['probability'].extend(probability.cpu().tolist())
            for token, scores in token_scores.items():
                all_scores[token].extend(scores.cpu().tolist())

            all_labels.extend(labels.cpu().tolist())

            all_metadata.extend([
                {
                    args.patient_id_column: metadata[args.patient_id_column],
                    args.sample_result_date_column: metadata[args.sample_result_date_column],
                    'TestTime': metadata['TestTime'] if 'TestTime' in metadata else 'NA',
                    'DATE_TIME': metadata['DATE_TIME'] if 'DATE_TIME' in metadata else 'NA',
                    'ResultDTS': metadata['ResultDTS'] if 'ResultDTS' in metadata else 'NA',
                    'file': metadata['file'] if 'file' in metadata else 'NA',
                }
                for metadata in sample_metadata if
                encounter_dataframe_process.check_patient_id(patient_id=metadata[args.patient_id_column])
            ])

            if len(all_metadata) != len(all_scores[tokens[0]]) or len(all_metadata) != len(all_labels):
                raise ValueError()

    return all_metadata, all_scores, all_labels


def get_eval_dataframe(metadata, scores, labels):
    metadata_df = pd.DataFrame(metadata)
    scores_df = pd.DataFrame(scores)
    eval_df = pd.concat([metadata_df, scores_df], axis=1)
    eval_df['labels'] = labels
    return eval_df

for file_suffix, args_str, model_type, model_path in get_model_details_for_eval(
        model_type=eval_args.model_type,
        model_folder=eval_args.model_folder,
        eval_every_epoch=eval_args.eval_every_epoch,
        eval_data=eval_args.eval_data,
        num_samples=eval_args.num_samples,
        batch_size=eval_args.batch_size,
        epoch_start=eval_args.epoch_start,
        code_column=eval_args.code_column,
        file_suffix=eval_args.file_suffix,
        result_date_column=eval_args.result_date_column,
        demographic_prompt_attributes=eval_args.demographic_prompt_attributes
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

    # if os.path.exists(filepath) and not args.overwrite:
    #     print('Skipping: This result already exists')
    #     continue

    os.makedirs(filepath, exist_ok=True)

    encounter_dataframe = pd.read_parquet(
        args.encounter_file,
        columns=['PatientID', 'ContactDTS', 'ICD10CD', 'phecode']
    )

    encounter_dataframe_process = EncounterDataframeProcess(
        encounter_dataframe=encounter_dataframe,
        patient_id_column=args.patient_id_column,
        contact_date_column=args.contact_date_column,
        time_difference_column=args.time_difference_column,
        code_column=args.code_column,
        position_column=args.position_column,
    )

    model, _, preprocess_val = create_model_and_transforms(
        model_type,
        pretrained=model_path,
    )

    tokenizer = get_tokenizer(model_type, context_length=args.max_seq_length)

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    input_dtype = get_input_dtype(args.precision)

    device = torch.device('cuda')
    model = model.eval()
    model = model.to(device)

    #test_phecodes_df = pd.read_csv(eval_args.phecode_file, encoding='ISO-8859-1')
    if eval_args.phecode_file.endswith('phecodeX_info.csv'):
        test_phecodes_df = pd.read_csv(eval_args.phecode_file, encoding='ISO-8859-1')
    else:
        test_phecodes_df = pd.read_csv(eval_args.phecode_file, sep='\t')
    
    test_phecodes = test_phecodes_df[eval_args.code_column]

    print('Number of codes: ', len(test_phecodes))

    for phecode in test_phecodes[int(eval_args.start): int(eval_args.end)]:
        print('PHECODE: ', phecode)
        if os.path.exists(f'{filepath}/{phecode}.parquet'):
            print('Skipping: This result already exists')
            continue

        # create empty placeholder file
        open(f'{filepath}/{phecode}.parquet', 'a').close()
        dataloader = get_eval_dataloader(args=args, eval_code=phecode, tokenizer=tokenizer)
        eos_token_id = get_eos_token_id(model_name=args.model, tokenizer=tokenizer)
        print('EOS Token ID: ', eos_token_id)
        metadata, scores, labels = evaluate_label(dataloader=dataloader, tokens=['yes', 'no'],
                                                  eos_token_id=eos_token_id, args=args)
        eval_df = get_eval_dataframe(metadata, scores, labels)
        eval_df.to_parquet(f'{filepath}/{phecode}.parquet')
