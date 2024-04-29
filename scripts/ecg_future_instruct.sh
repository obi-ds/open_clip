# TEXT TRANSFORMER

# 1. Random Negatives
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
torchrun \
    --nnodes=1 --nproc_per_node=8 --master_addr=localhost --master_port=2126 \
    -m main \
    --train-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_train_2402/shard_{0000..0084}.tar"  \
    --val-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2402/shard_{0000..0011}.tar"  \
    --train-num-samples 272000 \
    --val-num-samples 38400 \
    --dataset-type icddataset \
    --name ecg_phe_future_instruct_k_1_random_32_trial_6 \
    --workers 4 \
    --batch-size 360 \
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
    --model scatter_base \
    --report-to wandb \
    --code-column phecode \
    --wandb-project-name="open-clip-phe-test-runs" \
    --encounter-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/all_encounters_2308_with_phecodes_with_na.parquet" \
    --time-difference-normalize 1 \
    --number-of-instructions 1 \
    --k-shot 1 \
    --max_seq_length 76 \
    --pad_id 0 \
    --distance-threshold 60 \
    --negatives-type random \
    --training-type all \
    --future-only \
    --seed 0

# 2. Random & Cached Negatives
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
torchrun \
    --nnodes=1 --nproc_per_node=8 --master_addr=localhost --master_port=2126 \
    -m main \
    --train-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_train_2402/shard_{0000..0084}.tar"  \
    --val-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2402/shard_{0000..0011}.tar"  \
    --train-num-samples 272000 \
    --val-num-samples 38400 \
    --dataset-type icddataset \
    --name ecg_phe_future_instruct_k_1_random_cached_32_trial_6 \
    --workers 4 \
    --batch-size 360 \
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
    --model scatter_base \
    --report-to wandb \
    --code-column phecode \
    --wandb-project-name="open-clip-phe-test-runs" \
    --encounter-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/all_encounters_2308_with_phecodes_with_na.parquet" \
    --time-difference-normalize 1 \
    --number-of-instructions 1 \
    --k-shot 1 \
    --max_seq_length 76 \
    --pad_id 0 \
    --distance-threshold 60 \
    --negatives-type random_cached \
    --training-type all \
    --future-only \
    --seed 0


# GPT

# 1. Random Negatives
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
torchrun \
    --nnodes=1 --nproc_per_node=8 --master_addr=localhost --master_port=2126 \
    -m main \
    --train-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_train_2402/shard_{0000..0084}.tar"  \
    --val-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2402/shard_{0000..0011}.tar"  \
    --train-num-samples 272000 \
    --val-num-samples 38400 \
    --dataset-type icddataset \
    --name ecg_phe_future_instruct_k_1_random_32_gpt_trial_2 \
    --workers 4 \
    --batch-size 360 \
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
    --model scatter_base_gpt \
    --report-to wandb \
    --code-column phecode \
    --wandb-project-name="open-clip-phe-test-runs" \
    --encounter-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/all_encounters_2308_with_phecodes_with_na.parquet" \
    --time-difference-normalize 1 \
    --number-of-instructions 1 \
    --k-shot 1 \
    --max_seq_length 76 \
    --pad_id 50256 \
    --distance-threshold 60 \
    --negatives-type random \
    --training-type all \
    --future-only \
    --seed 0

# 2. Random & Cached Negatives
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
torchrun \
    --nnodes=1 --nproc_per_node=8 --master_addr=localhost --master_port=2126 \
    -m main \
    --train-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_train_2402/shard_{0000..0084}.tar"  \
    --val-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2402/shard_{0000..0011}.tar"  \
    --train-num-samples 272000 \
    --val-num-samples 38400 \
    --dataset-type icddataset \
    --name ecg_phe_future_instruct_k_1_random_cached_32_gpt_trial_1 \
    --workers 4 \
    --batch-size 360 \
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
    --model scatter_base_gpt \
    --report-to wandb \
    --code-column phecode \
    --wandb-project-name="open-clip-phe-test-runs" \
    --encounter-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/all_encounters_2308_with_phecodes_with_na.parquet" \
    --time-difference-normalize 1 \
    --number-of-instructions 1 \
    --k-shot 1 \
    --max_seq_length 76 \
    --pad_id 50256 \
    --distance-threshold 60 \
    --negatives-type random_cached \
    --training-type all \
    --future-only \
    --seed 0

# 3. Tree
# Change seq length in model config too

export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
torchrun \
    --nnodes=1 --nproc_per_node=8 --master_addr=localhost --master_port=2126 \
    -m main \
    --train-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_train_2402/shard_{0000..0084}.tar"  \
    --val-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2402/shard_{0000..0011}.tar"  \
    --train-num-samples 272000 \
    --val-num-samples 38400 \
    --dataset-type icddataset \
    --name ecg_phe_future_instruct_k_1_tree_32_gpt_trial_1 \
    --workers 4 \
    --batch-size 16 \
    --accum-freq 16 \
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
    --model scatter_base_gpt \
    --report-to wandb \
    --code-column phecode \
    --wandb-project-name="open-clip-phe-test-runs" \
    --encounter-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/all_encounters_2308_with_phecodes_with_na.parquet" \
    --time-difference-normalize 1 \
    --number-of-instructions 1 \
    --k-shot 1 \
    --max_seq_length 1024 \
    --pad_id 50256 \
    --distance-threshold 60 \
    --negatives-type random_cached \
    --training-type tree \
    --future-only \
    --seed 0