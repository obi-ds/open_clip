def get_args_str_old(model_name, model_type):
    return f'--train-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_train_23_10_23/shard_{{0000..0082}}.tar"  \
        --val-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_23_10_23/shard_{{0000..0002}}.tar"  \
        --train-num-samples 1062400 \
        --val-num-samples 38400 \
        --dataset-type icddataset \
        --name {model_name} \
        --workers 4 \
        --batch-size 1024 \
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
        --code-column phecode \
        --encounter-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/all_encounters_2308_with_phecodes_with_na.parquet.check" \
        --time-difference-normalize 1 \
        --number-of-instructions 1 \
        --k-shot 1 \
        --max_seq_length 64 \
        --pad_id 0 \
        --distance-threshold 60 \
        --negatives-type random \
        --tasks eval \
        --eval-mode \
        --eval-start-time 0 \
        --eval-end-time 180 \
        --seed 0'.replace('"', '')

def get_args_str_old_demographics(model_name, model_type):
    return f'--train-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_train_23_10_23/shard_{{0000..0082}}.tar"  \
        --val-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_23_10_23/shard_{{0000..0002}}.tar"  \
        --train-num-samples 1062400 \
        --val-num-samples 38400 \
        --dataset-type icddataset \
        --name {model_name} \
        --workers 4 \
        --batch-size 1024 \
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
        --code-column phecode \
        --encounter-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/all_encounters_2308_with_phecodes_with_na.parquet.check" \
        --demographic-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/demographics_2401.parquet" \
        --time-difference-normalize 1 \
        --number-of-instructions 1 \
        --k-shot 1 \
        --k-shot-demographics 2 \
        --max_seq_length 64 \
        --pad_id 0 \
        --distance-threshold 60 \
        --negatives-type random \
        --tasks demographics eval \
        --eval-mode \
        --eval-start-time 0 \
        --eval-end-time 180 \
        --seed 0'.replace('"', '')


def get_args_str_new(model_name, model_type):
    if 'gpt' in model_type:
        pad_id = 1
    else:
        pad_id = 0
    return f'--train-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_train_2403/shard_{{0000..0078}}.tar"  \
        --val-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2403/shard_{{0000..0010}}.tar"  \
        --train-num-samples 252800 \
        --val-num-samples 35200 \
        --dataset-type icddataset \
        --name {model_name} \
        --workers 4 \
        --batch-size 1024 \
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
        --code-column phecode \
        --encounter-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/all_encounters_2308_with_phecodes_with_na.parquet.check" \
        --time-difference-normalize 1 \
        --number-of-instructions 1 \
        --k-shot 1 \
        --max_seq_length 64 \
        --pad_id {pad_id} \
        --distance-threshold 60 \
        --negatives-type random \
        --tasks eval \
        --eval-mode \
        --eval-start-time 0 \
        --eval-end-time 180 \
        --seed 0'.replace('"', '')

def get_args_str_new_test(model_name, model_type):
    if 'gpt' in model_type:
        pad_id = 1
    else:
        pad_id = 0
    return f'--train-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_train_2403/shard_{{0000..0078}}.tar" \
        --val-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_test_2403/shard_{{0000..0021}}.tar" \
        --train-num-samples 252800 \
        --val-num-samples 70400 \
        --dataset-type icddataset \
        --name {model_name} \
        --workers 4 \
        --batch-size 1024 \
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
        --code-column phecode \
        --encounter-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/all_encounters_2308_with_phecodes_with_na.parquet.check" \
        --time-difference-normalize 1 \
        --number-of-instructions 1 \
        --k-shot 1 \
        --max_seq_length 64 \
        --pad_id 0 \
        --distance-threshold 60 \
        --negatives-type random \
        --tasks eval \
        --eval-mode \
        --eval-start-time 0 \
        --eval-end-time 180 \
        --seed 0'.replace('"', '')

def get_args_str_new_bwh(model_name, model_type):
    return f'--train-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_train_2403/shard_{{0000..0078}}.tar" \
        --val-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/bwh/bwh_all_23_10_23/shard_{{0000..0005}}.tar" \
        --train-num-samples 1062400 \
        --val-num-samples 76800 \
        --dataset-type icddataset \
        --name {model_name} \
        --workers 4 \
        --batch-size 1024 \
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
        --code-column phecode \
        --encounter-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/all_encounters_2308_with_phecodes_with_na.parquet.check" \
        --time-difference-normalize 1 \
        --number-of-instructions 1 \
        --k-shot 1 \
        --max_seq_length 64 \
        --pad_id 0 \
        --distance-threshold 60 \
        --negatives-type random \
        --tasks eval \
        --eval-mode \
        --eval-start-time 0 \
        --eval-end-time 180 \
        --sample-result-date-column TestDate_x \
        --seed 0'.replace('"', '')

def get_args_str_new_tree(model_name, model_type):
    return f'--train-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_train_2403/shard_{{0000..0078}}.tar"  \
        --val-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2403/shard_{{0000..0010}}.tar"  \
        --train-num-samples 252800 \
        --val-num-samples 35200 \
        --dataset-type icddataset \
        --name {model_name} \
        --workers 4 \
        --batch-size 1024 \
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
        --code-column phecode \
        --encounter-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/all_encounters_2308_with_phecodes_with_na.parquet.check" \
        --time-difference-normalize 1 \
        --number-of-instructions 1 \
        --k-shot 1 \
        --max_seq_length 64 \
        --pad_id 0 \
        --distance-threshold 60 \
        --negatives-type random \
        --tasks tree_eval \
        --eval-mode \
        --eval-start-time 0 \
        --eval-end-time 180 \
        --seed 0'.replace('"', '')

def get_args_str_new_demographics(model_name, model_type):
    return f'--train-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_train_2403/shard_{{0000..0078}}.tar"  \
        --val-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2403/shard_{{0000..0010}}.tar"  \
        --train-num-samples 252800 \
        --val-num-samples 35200 \
        --dataset-type icddataset \
        --name {model_name} \
        --workers 4 \
        --batch-size 1024 \
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
        --code-column phecode \
        --encounter-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/all_encounters_2308_with_phecodes_with_na.parquet.check" \
        --demographic-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/demographics_2401.parquet" \
        --time-difference-normalize 1 \
        --number-of-instructions 1 \
        --k-shot 1 \
        --k-shot-demographics 2 \
        --max_seq_length 64 \
        --pad_id 0 \
        --distance-threshold 60 \
        --negatives-type random \
        --tasks demographics eval \
        --eval-mode \
        --eval-start-time 0 \
        --eval-end-time 180 \
        --seed 0'.replace('"', '')

def get_model_path(model_name, epoch):
    model_path_prefix = '/mnt/obi0/pk621/projects/med_instruct/vision/open_clip/src/logs/'
    model_path_suffix = '/checkpoints/'
    return f'{model_path_prefix}{model_name}{model_path_suffix}{epoch}'

model_details_old = [
    ('23_10_mgh_val', 'scatter_base', 'ecg_phe_instruct_k_1_random_32_old_fpr_trial_2', 'epoch_27.pt'),
    ('23_10_mgh_val', 'scatter_base', 'ecg_phe_instruct_k_1_random_32_fpr_demographics_trial_2', 'epoch_27.pt'),
    ('23_10_mgh_val', 'scatter_base', 'ecg_phe_instruct_k_1_random_32_fpr_demographics_shuffled_trial_2', 'epoch_27.pt'),
]

# model_details_new = [
#     ('24_03_mgh_val', 'scatter_base', 'ecg_phe_instruct_k_1_random_150_new_fpr_demographics_shuffled_future_trial_5', 'epoch_150.pt'),
#     ('24_03_mgh_val', 'scatter_base', 'ecg_phe_instruct_k_1_random_150_new_fpr_demographics_labs_shuffled_future_trial_5', 'epoch_150.pt'),
# ]

# model_details_new = [
#     ('24_03_mgh_test', 'scatter_base', 'ecg_phe_instruct_k_1_random_150_new_fpr_future_trial_1', 'epoch_150.pt'),
# ]

# model_details_new = [
#     ('24_03_mgh_val', 'scatter_base', 'ecg_phe_instruct_k_4_random_325_new_fpr_demographics_labs_all_future_focal_trial_3', 'epoch_310.pt'),
# ]

# model_details_new = [
#     ('24_03_mgh_val', 'ecg_scatter_global', 'ecg_k_1_random_small_all_scatter_windowed_future_trial_6', 'epoch_194.pt'),
# ]

# model_details_new = [
#     ('24_03_mgh_val', 'ecg_scatter_global', 'ecg_k_1_all_random_scatter_global_future_250_trial_16', 'epoch_220.pt'),
#     ('24_03_mgh_val', 'ecg_scatter_global', 'ecg_k_1_all_random_cached_scatter_global_future_250_trial_16', 'epoch_200.pt'),
# ]

# model_details_new = [
#     ('24_03_mgh_val', 'ecg_cnn_windowed_biogpt', 'ecg_k_1_diagnosis_random_cnn_windowed_biogpt_frozen_future_250_trial_1', 'epoch_240.pt'),
#     ('24_03_mgh_val', 'ecg_cnn_windowed_biogpt', 'ecg_k_1_diagnosis_random_cnn_windowed_biogpt_frozen_future_250_trial_1', 'epoch_160.pt'),
#     ('24_03_mgh_val', 'ecg_scatter_global_biogpt', 'ecg_k_1_diagnosis_random_scatter_global_biogpt_frozen_future_250_trial_3', 'epoch_190.pt'),
#     ('24_03_mgh_val', 'ecg_scatter_global_biogpt', 'ecg_k_1_diagnosis_random_weighted_scatter_global_biogpt_frozen_future_250_trial_6', 'epoch_200.pt'),
# ]

# TODO:
#   I usually comment out the models I've evaluated before
#   1. First argument is the dataset we want to evaluate on - currently I still need to merge some stuff
#           - It will work for the validation dataset - for the MGH test set and BWH test set - you need to do some
#           additional steps which I plan on making one single step
#   2. Second argument is the model type to evaluate - this should point to the config file we used to train the model
#   3. Third argument - The model we are evaluating - this is the name that shows up on wandb - then go to
#       the "get_model_path" function and modify the directory where the models are stored - the
#       "model_path_prefix" variable
#   4. The epoch we are evaluating
#   5. There are examples above


model_details_new = [
    ('', '', '', '.pt'),
]


model_details_new_test = [
    ('24_03_mgh_test', 'scatter_base', 'ecg_phe_instruct_k_4_random_325_new_fpr_demographics_labs_all_future_trial_1', 'epoch_193.pt'),
]

model_details_new_bwh = [
    ('23_10_bwh_all', 'scatter_base', 'ecg_phe_instruct_k_4_random_325_new_fpr_demographics_labs_all_future_trial_1', 'epoch_193.pt')
]

model_details_new_tree = [
    ('24_03_mgh_val', 'scatter_base', 'ecg_phe_instruct_k_1_random_3000_new_fpr_future_tree_trial_1', 'epoch_247.pt'),
]

model_details_old_demographics = [
    ('23_10_mgh_val_demographic', 'scatter_base', 'ecg_phe_instruct_k_1_random_32_fpr_demographics_trial_2', 'epoch_27.pt'),
    ('23_10_mgh_val_demographic', 'scatter_base', 'ecg_phe_instruct_k_1_random_32_fpr_demographics_shuffled_trial_2', 'epoch_27.pt'),
]


model_details_new_demographics = [
    ('24_03_mgh_val_demographic', 'scatter_base', 'ecg_phe_instruct_k_1_random_32_fpr_new_demographics_trial_1', 'epoch_30.pt'),
    ('24_03_mgh_val_demographic', 'scatter_base', 'ecg_phe_instruct_k_1_random_32_new_fpr_demographics_shuffled_trial_1', 'epoch_30.pt'),
]

model_details_old_debug = [
    ('24_03_mgh_val', 'scatter_base', 'ecg_phe_instruct_k_1_random_32_old_fpr_trial_2', 'epoch_27.pt'),
]

model_details_new_debug = [
    ('23_10_mgh_val', 'scatter_base', 'ecg_phe_instruct_k_1_random_32_new_fpr_trial_1', 'epoch_30.pt'),
]

eval_attributes_old = [
    (file_suffix, get_args_str_old(model_name, model_type), model_type, get_model_path(model_name, epoch))
    for file_suffix, model_type, model_name, epoch in model_details_old
]

eval_attributes_new = [
    (file_suffix, get_args_str_new(model_name, model_type), model_type, get_model_path(model_name, epoch))
    for file_suffix, model_type, model_name, epoch in model_details_new
]

eval_attributes_new_test = [
    (file_suffix, get_args_str_new_test(model_name, model_type), model_type, get_model_path(model_name, epoch))
    for file_suffix, model_type, model_name, epoch in model_details_new_test
]

eval_attributes_new_bwh = [
    (file_suffix, get_args_str_new_bwh(model_name, model_type), model_type, get_model_path(model_name, epoch))
    for file_suffix, model_type, model_name, epoch in model_details_new_bwh
]

eval_attributes_new_tree = [
    (file_suffix, get_args_str_new_tree(model_name, model_type), model_type, get_model_path(model_name, epoch))
    for file_suffix, model_type, model_name, epoch in model_details_new_tree
]

eval_attributes_old_demographics = [
    (file_suffix, get_args_str_old_demographics(model_name, model_type), model_type, get_model_path(model_name, epoch))
    for file_suffix, model_type, model_name, epoch in model_details_old_demographics
]

eval_attributes_new_demographics = [
    (file_suffix, get_args_str_new_demographics(model_name, model_type), model_type, get_model_path(model_name, epoch))
    for file_suffix, model_type, model_name, epoch in model_details_new_demographics
]

eval_attributes_old_debug = [
    (file_suffix, get_args_str_new_demographics(model_name, model_type), model_type, get_model_path(model_name, epoch))
    for file_suffix, model_type, model_name, epoch in model_details_old_debug
]

eval_attributes_new_debug = [
    (file_suffix, get_args_str_old_demographics(model_name, model_type), model_type, get_model_path(model_name, epoch))
    for file_suffix, model_type, model_name, epoch in model_details_new_debug
]

def get_eval_attributes():
    return eval_attributes_new
