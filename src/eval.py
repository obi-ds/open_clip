import copy
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.nn import functional as F

from code_eval.arguments.eval_args import parse_args
from open_clip.factory import get_tokenizer, create_model_and_transforms
from training.data import get_wds_dataset_moca_instruct
from training.instruct.diagnosis.processing import (
    EncounterDataframeProcess,
)
from training.precision import get_input_dtype

from main import get_token_id, get_eos_token_id


def get_eval_dataloader(args, eval_code, tokenizer):
    args = copy.deepcopy(args)
    args.eval_code = eval_code
    eval_dataset_image = get_wds_dataset_moca_instruct(
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


def get_token_scores(model_name, logits, tokens, tokenizer):
    return {token: logits[:, get_token_id(model_name, tokenizer, token)] for token in tokens}


def convert_instructions_list_to_string(instructions):
    full_string = ''
    for instruction in instructions:
        full_string += instruction[0] + instruction[1]
    return full_string


def evaluate_label(
        model,
        dataloader,
        tokenizer,
        device,
        input_dtype,
        tokens,
        eos_token_id,
        args,
        encounter_dataframe_process,
        ignore_index=-100
):
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

            token_scores = get_token_scores(model_name=args.model, logits=logits, tokens=tokens, tokenizer=tokenizer)
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


def main(eval_arguments):
    print('STARTING')
    args = parse_args(eval_arguments)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print('Dataset: ', args.val_data)
    print('Model: ', args.pretrained)

    prefix = Path(args.pretrained).parent.parent.name
    epoch = Path(args.pretrained).name.split('.')[0]

    filepath = f'{args.output_folder}{args.file_suffix}/{prefix}/' \
               f'{epoch}/' \
               f'{args.eval_start_time}-{args.eval_end_time}'

    print(filepath)

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
        args.model,
        pretrained=args.pretrained,
    )

    tokenizer = get_tokenizer(args.model, context_length=args.max_seq_length)

    input_dtype = get_input_dtype(args.precision)

    device = torch.device('cuda')
    model = model.eval()
    model = model.to(device)

    if args.phecode_file.endswith('phecodeX_info.csv'):
        test_phe_codes_df = pd.read_csv(args.phecode_file, encoding='ISO-8859-1')
    else:
        test_phe_codes_df = pd.read_csv(args.phecode_file, sep='\t')

    test_phe_codes = test_phe_codes_df[args.code_column]

    print('Number of codes: ', len(test_phe_codes))

    for phecode in test_phe_codes[int(args.start): int(args.end)]:
        print('PHECODE: ', phecode)
        if os.path.exists(f'{filepath}/{phecode}.parquet'):
            print('Skipping: This result already exists')
            continue

        # create empty placeholder file
        open(f'{filepath}/{phecode}.parquet', 'a').close()
        dataloader = get_eval_dataloader(args=args, eval_code=phecode, tokenizer=tokenizer)
        eos_token_id = get_eos_token_id(model_name=args.model, tokenizer=tokenizer)
        print('EOS Token ID: ', eos_token_id)
        metadata, scores, labels = evaluate_label(
            model=model,
            tokenizer=tokenizer,
            dataloader=dataloader,
            device=device,
            input_dtype=input_dtype,
            tokens=['yes', 'no'],
            eos_token_id=eos_token_id,
            encounter_dataframe_process=encounter_dataframe_process,
            args=args
        )
        eval_df = get_eval_dataframe(metadata, scores, labels)
        eval_df.to_parquet(f'{filepath}/{phecode}.parquet')


if __name__ == "__main__":
    main(sys.argv[1:])
