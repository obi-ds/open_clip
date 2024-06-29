import os
import sys
from pathlib import Path
import pandas as pd
import torch

from code_eval.arguments.generation_args import parse_args
from open_clip.factory import get_tokenizer, create_model_and_transforms
from training.data import get_wds_dataset_moca_prompt
from training.precision import get_input_dtype
from training.data_utils import (
    get_instruct_tokenizer,
    get_diagnosis_label_prediction_instruction_template,
    get_patient_demographics_instruction_template,
    get_example_separator,
    get_task_separator
)

from main import get_eos_token_id


def get_eval_dataloader(args, tokenizer):
    eval_dataset_image = get_wds_dataset_moca_prompt(
        args,
        preprocess_img=lambda x: x,
        is_train=False,
        tokenizer=tokenizer,
        return_sample=False,
    )
    return eval_dataset_image.dataloader


def get_prompt(args, attribute, tokenizer, add_img_token, time=None):
    if attribute.lower() in ['age', 'sex', 'height', 'weight']:
        demographic_instructions = get_patient_demographics_instruction_template(
            example_separator=get_example_separator(args=args),
            task_separator=get_task_separator(args)
        )
        task_definition = demographic_instructions.get_task_definition()
        instruction_input = demographic_instructions.get_instruction_input(category=attribute)
    elif attribute.lower() in ['xxxx']:
        # TODO: Implement
        raise NotImplementedError('ECG attributes not implemented')
    else:
        diagnosis_label_prediction_instructions = get_diagnosis_label_prediction_instruction_template(
            example_separator=get_example_separator(args=args),
            task_separator=get_task_separator(args)
        )
        task_definition = diagnosis_label_prediction_instructions.get_task_definition(time=time)
        instruction_input = diagnosis_label_prediction_instructions.get_instruction_input(diagnosis=attribute)
    if add_img_token:
        bos_token = tokenizer.tokenizer.bos_token + '\n'
        prompt_string = f'{bos_token}{task_definition}{instruction_input}'
    else:
        prompt_string = f'{task_definition}{instruction_input}'
    print('Prompt:\n', prompt_string)
    return prompt_string


# Function to convert prompt into tokens
def get_custom_prompt(tokenizer, text, batch_size):
    input_tokens = tokenizer.get_input_tokens(text)
    return torch.tensor(input_tokens).tile(batch_size, 1)


# Convert the predicted tokens to string
def get_predicted_string(tokenizer, predicted_token):
    predicted_string = tokenizer.tokenizer.decode(predicted_token)
    return predicted_string


# The final output contains the initial prompt and the predicted tokens
# Use this function to extract only the predicted tokens
def get_predicted_token(generated_token, prompt_token, eos_token_id, pad_token_id):
    predicted_token = generated_token[len(prompt_token):]
    predicted_token = predicted_token[(predicted_token != eos_token_id) & (predicted_token != pad_token_id)]
    return predicted_token


def generate_text(
        dataloader,
        prompt_tokens_full,
        pad_token_id,
        eos_token_id,
        args,
        model,
        tokenizer,
        device,
        input_dtype
):
    generated_dataset = list()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            images, batch_metadata = batch
            images = images.to(device=device, dtype=input_dtype, non_blocking=True)
            prompt_tokens = prompt_tokens_full[:images.size()[0]]

            try:
                generated_tokens = model.generate(
                    images,
                    prompt_tokens,
                    generation_type='top_k',
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id
                )
            except:
                print('Fail')
                print(images.shape)
                print(prompt_tokens.shape)
                raise ValueError()

            for index, (sample_metadata, generated_token) in enumerate(zip(batch_metadata, generated_tokens)):
                predicted_token = get_predicted_token(
                    generated_token, prompt_tokens[index], eos_token_id, pad_token_id
                )
                predicted_string = get_predicted_string(tokenizer=tokenizer, predicted_token=generated_token)
                predicted_value = get_predicted_string(predicted_token=predicted_token, tokenizer=tokenizer)

                generated_sample = {
                    'PatientID': sample_metadata[args.patient_id_column],
                    'SampleDate': sample_metadata[args.sample_result_date_column],
                    'SampleTime': sample_metadata['TestTime'],
                    'SampleFile': sample_metadata['file'],
                    'PredictedValue': predicted_value,
                    'PredictedString': predicted_string,
                    'GeneratedTokens': generated_token.detach().cpu().tolist()
                }

                generated_dataset.append(generated_sample)

    return generated_dataset


def main(generation_arguments):
    print('STARTING')
    args = parse_args(generation_arguments)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print('Dataset: ', args.val_data)
    print('Model: ', args.pretrained)

    prefix = Path(args.pretrained).parent.parent.name
    epoch = Path(args.pretrained).name.split('.')[0]

    filepath = f'{args.output_folder}{args.file_suffix}/{prefix}/' \
               f'{epoch}/' \
               f'{args.eval_start_time}-{args.eval_end_time}'

    print('Output filepath: ', filepath)

    # if os.path.exists(filepath):
    #     print('Skipping: This result already exists')
    #     continue

    os.makedirs(filepath, exist_ok=True)

    eval_attributes = pd.read_csv(args.attribute_file)[args.attribute_name_column]

    model, _, preprocess_val = create_model_and_transforms(
        args.model,
        pretrained=args.pretrained,
    )

    tokenizer = get_tokenizer(args.model, context_length=args.max_seq_length)
    instruct_tokenizer = get_instruct_tokenizer(tokenizer, -100, args)
    eos_token_id = get_eos_token_id(model_name=args.model, tokenizer=tokenizer)

    input_dtype = get_input_dtype(args.precision)

    device = torch.device('cuda')
    model = model.eval()
    model = model.to(device)

    print('EOS Token ID: ', eos_token_id)

    for eval_attribute in eval_attributes[args.start: args.end]:
        dataloader = get_eval_dataloader(args=args, tokenizer=tokenizer)
        print('Attribute: ', eval_attribute)
        prompt = get_prompt(
            attribute=eval_attribute,
            tokenizer=tokenizer,
            add_img_token=args.add_img_token,
            time=None,
            args=args
        )
        # This is the prompt that was used in training
        prompt_tokens_all = get_custom_prompt(tokenizer=instruct_tokenizer, text=prompt, batch_size=args.batch_size)
        prompt_tokens_all = prompt_tokens_all.to(device=device, non_blocking=True)

        print('Generation Start')
        predicted_generated_dataset = generate_text(
            dataloader=dataloader,
            prompt_tokens_full=prompt_tokens_all,
            eos_token_id=eos_token_id,
            pad_token_id=args.pad_token_id,
            args=args,
            model=model,
            tokenizer=tokenizer,
            device=device,
            input_dtype=input_dtype
        )
        print('Generation Finish')
        pd.DataFrame(predicted_generated_dataset).to_parquet(f'{filepath}/{eval_attribute}.parquet')


if __name__ == "__main__":
    main(sys.argv[1:])
