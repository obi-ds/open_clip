model_name = 'bin'
#
# args_str_val = f'--train-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_train_2403/shard_{{0000..0082}}.tar"  \
#     --val-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2403/shard_{{0000..0010}}.tar"  \
#     --train-num-samples 1062400 \
#     --val-num-samples 35200 \
#     --dataset-type icddataset \
#     --name {model_name} \
#     --workers 4 \
#     --batch-size 3200 \
#     --epochs 32 \
#     --lr 2e-4 \
#     --beta1 0.9 \
#     --beta1 0.98 \
#     --eps 1e-6 \
#     --wd 0.01 \
#     --warmup 4000 \
#     --lr-scheduler cosine \
#     --lr-cooldown-end 5e-5 \
#     --coca-caption-loss-weight 1.0 \
#     --coca-contrastive-loss-weight 0.0 \
#     --precision amp \
#     --save-frequency 1 \
#     --val-frequency 1 \
#     --zeroshot-frequency 0 \
#     --local-loss \
#     --gather-with-grad \
#     --model {model_type} \
#     --report-to wandb \
#     --billable-probability 0.0 \
#     --top-non-probability 1.0 \
#     --code-column phecode \
#     --encounter-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/all_encounters_2308_with_phecodes_with_na.parquet" \
#     --time-difference-normalize 1 \
#     --number-of-instructions 1 \
#     --k-shot 1 \
#     --max_seq_length 76 \
#     --pad_id 0 \
#     --distance-threshold 60 \
#     --negatives-type random \
#     --eval-mode \
#     --eval-start-time 0 \
#     --eval-end-time 180 \
#     --seed 0'.replace('"', '')
#
# args_str_test = f'--train-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_train_2403/shard_{{0000..0082}}.tar"  \
#     --val-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_test_2403/shard_{{0000..0021}}.tar"  \
#     --train-num-samples 1062400 \
#     --val-num-samples 70400 \
#     --dataset-type icddataset \
#     --name {model_name} \
#     --workers 4 \
#     --batch-size 3200 \
#     --epochs 32 \
#     --lr 2e-4 \
#     --beta1 0.9 \
#     --beta1 0.98 \
#     --eps 1e-6 \
#     --wd 0.01 \
#     --warmup 4000 \
#     --lr-scheduler cosine \
#     --lr-cooldown-end 5e-5 \
#     --coca-caption-loss-weight 1.0 \
#     --coca-contrastive-loss-weight 0.0 \
#     --precision amp \
#     --save-frequency 1 \
#     --val-frequency 1 \
#     --zeroshot-frequency 0 \
#     --local-loss \
#     --gather-with-grad \
#     --model {model_type} \
#     --report-to wandb \
#     --billable-probability 0.0 \
#     --top-non-probability 1.0 \
#     --code-column phecode \
#     --encounter-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/all_encounters_2308_with_phecodes_with_na.parquet" \
#     --time-difference-normalize 1 \
#     --number-of-instructions 1 \
#     --k-shot 1 \
#     --max_seq_length 76 \
#     --pad_id 0 \
#     --distance-threshold 60 \
#     --negatives-type random \
#     --eval-mode \
#     --eval-start-time 0 \
#     --eval-end-time 180 \
#     --seed 0'.replace('"', '')

def get_args(bin_data, num_samples, model_type, result_date_column, code_column):
    args_str = f'--train-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_train_2403/shard_{{0000..0082}}.tar"  \
        --val-data="{bin_data}"  \
        --train-num-samples 1062400 \
        --val-num-samples {num_samples} \
        --dataset-type icddataset \
        --name {model_name} \
        --workers 4 \
        --batch-size 3200 \
        --epochs 32 \
        --lr 2e-4 \
        --beta1 0.9 \
        --beta1 0.98 \
        --eps 1e-6 \
        --wd 0.01 \
        --warmup 4000 \
        --lr-scheduler cosine \
        --lr-cooldown-end 5e-5 \
        --coca-caption-loss-weight 1.0 \
        --coca-contrastive-loss-weight 0.0 \
        --precision amp \
        --save-frequency 1 \
        --val-frequency 1 \
        --zeroshot-frequency 0 \
        --local-loss \
        --gather-with-grad \
        --model {model_type} \
        --report-to wandb \
        --billable-probability 0.0 \
        --top-non-probability 1.0 \
        --code-column {code_column} \
        --sample-result-date-column {result_date_column} \
        --encounter-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/all_encounters_2308_with_phecodes_with_na.parquet" \
        --time-difference-normalize 1 \
        --number-of-instructions 1 \
        --k-shot 1 \
        --max_seq_length 76 \
        --pad_id 0 \
        --distance-threshold 60 \
        --negatives-type random \
        --eval-mode \
        --eval-start-time 0 \
        --eval-end-time 180 \
        --seed 0'.replace('"', '')

    return args_str

def get_args_for_binning(file_suffix, bin_data, num_samples, model_type, result_date_column, code_column):
    args_str = get_args(
        bin_data=bin_data, num_samples=num_samples, model_type=model_type,
        result_date_column=result_date_column, code_column=code_column
    )
    return [(file_suffix, args_str)]
