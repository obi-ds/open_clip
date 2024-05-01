# METHOD 1: Trajectory with counts - Generative modeling

export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
torchrun \
    --nnodes=1 --nproc_per_node=8 --master_addr=localhost --master_port=2126 \
    -m main \
    --train-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_train_2402/shard_{0000..0084}.tar"  \
    --val-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2402/shard_{0000..0002}.tar"  \
    --train-num-samples 272000 \
    --val-num-samples 9600 \
    --dataset-type icddataset \
    --name ecg_phe_trajectory_gpt_1 \
    --workers 4 \
    --batch-size 16 \
    --epochs 20 \
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
    --billable-probability 0.0 \
    --top-non-probability 1.0 \
    --code-column phecode \
    --wandb-project-name="open-clip-phe-test-runs" \
    --encounter-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/all_encounters_2308_with_phecodes_with_na.parquet" \
    --time-difference-normalize 1 \
    --max_seq_length 1024 \
    --pad_id 50256 \
    --distance-threshold 7 30 60 120 180 365 \
    --shuffle-bins \
    --training-type trajectory \
    --trajectory-sampling-weights 0.4 0.2 0.4 \
    --seed 0


# METHOD 2: Future Instruct Trajectory - Instruction tuning
# Binary/Multiclass labels - can specify via a parameter
# Predicted on future time periods (time period is always Day 0 to Day X)
# Trajectory can contain all codes, no codes (only predict on time period without context) or a sampled trajectory
# Sampled  trajectory can contain or not contain the codes we are predicting on

# METHOD 2A: Multi-class labels
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
torchrun \
    --nnodes=1 --nproc_per_node=8 --master_addr=localhost --master_port=2126 \
    -m main \
    --train-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_train_2402/shard_{0000..0084}.tar"  \
    --val-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2402/shard_{0000..0002}.tar"  \
    --train-num-samples 272000 \
    --val-num-samples 9600 \
    --dataset-type icddataset \
    --name ecg_phe_future_instruct_trajectory_multiclass_gpt_2 \
    --workers 4 \
    --batch-size 16 \
    --epochs 7 \
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
    --billable-probability 0.0 \
    --top-non-probability 1.0 \
    --code-column phecode \
    --wandb-project-name="open-clip-phe-test-runs" \
    --encounter-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/all_encounters_2308_with_phecodes_with_na.parquet" \
    --time-difference-normalize 1 \
    --max_seq_length 1024 \
    --pad_id 50256 \
    --distance-threshold 7 30 60 120 180 365 \
    --shuffle-bins \
    --training-type future_instruct_trajectory \
    --label-type multiclass \
    --trajectory-sampling-weights 0.2 0.0 0.8 \
    --seed 0


# METHOD 2B: Binary labels
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
torchrun \
    --nnodes=1 --nproc_per_node=8 --master_addr=localhost --master_port=2126 \
    -m main \
    --train-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_train_2402/shard_{0000..0084}.tar"  \
    --val-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2402/shard_{0000..0002}.tar"  \
    --train-num-samples 272000 \
    --val-num-samples 9600 \
    --dataset-type icddataset \
    --name ecg_phe_future_instruct_trajectory_binary_gpt_1 \
    --workers 4 \
    --batch-size 16 \
    --epochs 7 \
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
    --billable-probability 0.0 \
    --top-non-probability 1.0 \
    --code-column phecode \
    --wandb-project-name="open-clip-phe-test-runs" \
    --encounter-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/all_encounters_2308_with_phecodes_with_na.parquet" \
    --time-difference-normalize 1 \
    --max_seq_length 1024 \
    --pad_id 50256 \
    --distance-threshold 7 30 60 120 180 365 \
    --shuffle-bins \
    --training-type future_instruct_trajectory \
    --label-type binary \
    --trajectory-sampling-weights 0.2 0.0 0.8 \
    --seed 0

# METHOD 2C: Binary Strict labels
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
torchrun \
    --nnodes=1 --nproc_per_node=8 --master_addr=localhost --master_port=2126 \
    -m main \
    --train-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_train_2402/shard_{0000..0084}.tar"  \
    --val-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2402/shard_{0000..0002}.tar"  \
    --train-num-samples 272000 \
    --val-num-samples 9600 \
    --dataset-type icddataset \
    --name ecg_phe_future_instruct_trajectory_binary_strict_gpt_1 \
    --workers 4 \
    --batch-size 16 \
    --epochs 20 \
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
    --billable-probability 0.0 \
    --top-non-probability 1.0 \
    --code-column phecode \
    --wandb-project-name="open-clip-phe-test-runs" \
    --encounter-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/all_encounters_2308_with_phecodes_with_na.parquet" \
    --time-difference-normalize 1 \
    --max_seq_length 1024 \
    --pad_id 50256 \
    --distance-threshold 7 30 60 120 180 365 \
    --shuffle-bins \
    --training-type future_instruct_trajectory \
    --label-type binary_strict \
    --trajectory-sampling-weights 0.5 0.0 0.5 \
    --seed 0

# METHOD 2D: Multi-class labels without trajectory
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
torchrun \
    --nnodes=1 --nproc_per_node=8 --master_addr=localhost --master_port=2126 \
    -m main \
    --train-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_train_2402/shard_{0000..0084}.tar"  \
    --val-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2402/shard_{0000..0002}.tar"  \
    --train-num-samples 272000 \
    --val-num-samples 9600 \
    --dataset-type icddataset \
    --name ecg_phe_future_instruct_multiclass_gpt_1 \
    --workers 4 \
    --batch-size 16 \
    --epochs 7 \
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
    --billable-probability 0.0 \
    --top-non-probability 1.0 \
    --code-column phecode \
    --wandb-project-name="open-clip-phe-test-runs" \
    --encounter-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/all_encounters_2308_with_phecodes_with_na.parquet" \
    --time-difference-normalize 1 \
    --max_seq_length 1024 \
    --pad_id 50256 \
    --distance-threshold 7 30 60 120 180 365 \
    --shuffle-bins \
    --training-type future_instruct_trajectory \
    --label-type multiclass \
    --trajectory-sampling-weights 1.0 0.0 0.0 \
    --seed 0


# METHOD 2E: Binary labels without trajectory
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
torchrun \
    --nnodes=1 --nproc_per_node=8 --master_addr=localhost --master_port=2126 \
    -m main \
    --train-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_train_2402/shard_{0000..0084}.tar"  \
    --val-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2402/shard_{0000..0002}.tar"  \
    --train-num-samples 272000 \
    --val-num-samples 9600 \
    --dataset-type icddataset \
    --name ecg_phe_future_instruct_binary_gpt_1 \
    --workers 4 \
    --batch-size 16 \
    --epochs 7 \
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
    --billable-probability 0.0 \
    --top-non-probability 1.0 \
    --code-column phecode \
    --wandb-project-name="open-clip-phe-test-runs" \
    --encounter-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/all_encounters_2308_with_phecodes_with_na.parquet" \
    --time-difference-normalize 1 \
    --max_seq_length 1024 \
    --pad_id 50256 \
    --distance-threshold 7 30 60 120 180 365 \
    --shuffle-bins \
    --training-type future_instruct_trajectory \
    --label-type binary \
    --trajectory-sampling-weights 1.0 0.0 0.0 \
    --seed 0

# METHOD 2F: Binary Strict labels without trajectory
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
torchrun \
    --nnodes=1 --nproc_per_node=8 --master_addr=localhost --master_port=2126 \
    -m main \
    --train-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_train_2402/shard_{0000..0084}.tar"  \
    --val-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2402/shard_{0000..0002}.tar"  \
    --train-num-samples 272000 \
    --val-num-samples 9600 \
    --dataset-type icddataset \
    --name ecg_phe_future_instruct_trajectory_binary_strict_gpt_1 \
    --workers 4 \
    --batch-size 16 \
    --epochs 20 \
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
    --billable-probability 0.0 \
    --top-non-probability 1.0 \
    --code-column phecode \
    --wandb-project-name="open-clip-phe-test-runs" \
    --encounter-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/all_encounters_2308_with_phecodes_with_na.parquet" \
    --time-difference-normalize 1 \
    --max_seq_length 1024 \
    --pad_id 50256 \
    --distance-threshold 7 30 60 120 180 365 \
    --shuffle-bins \
    --training-type future_instruct_trajectory \
    --label-type binary_strict \
    --trajectory-sampling-weights 1.0 0.0 0.0 \
    --seed 0


# METHOD 3: Month Instruct Trajectory - Instruction tuning
# Binary labels only - can specify via a parameter
# Predicted on ant time periods (time period is always Month 0 to Month Y)
# Trajectory can contain all codes, no codes (only predict on time period without context) or a sampled trajectory
# Sampled  trajectory can contain or not contain the codes we are predicting on

# METHOD 3A: Binary labels
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
torchrun \
    --nnodes=1 --nproc_per_node=8 --master_addr=localhost --master_port=2126 \
    -m main \
    --train-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_train_2402/shard_{0000..0084}.tar"  \
    --val-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2402/shard_{0000..0002}.tar"  \
    --train-num-samples 272000 \
    --val-num-samples 9600 \
    --dataset-type icddataset \
    --name ecg_phe_month_instruct_trajectory_binary_gpt_1 \
    --workers 4 \
    --batch-size 16 \
    --epochs 7 \
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
    --billable-probability 0.0 \
    --top-non-probability 1.0 \
    --code-column phecode \
    --wandb-project-name="open-clip-phe-test-runs" \
    --encounter-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/all_encounters_2308_with_phecodes_with_na.parquet" \
    --time-difference-normalize 1 \
    --max_seq_length 1024 \
    --pad_id 50256 \
    --distance-threshold 7 30 60 120 180 365 \
    --shuffle-bins \
    --training-type month_instruct_trajectory \
    --label-type binary \
    --trajectory-sampling-weights 0.2 0.0 0.8 \
    --seed 0

# METHOD 3B: Binary Strict labels
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
torchrun \
    --nnodes=1 --nproc_per_node=8 --master_addr=localhost --master_port=2126 \
    -m main \
    --train-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_train_2402/shard_{0000..0084}.tar"  \
    --val-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2402/shard_{0000..0002}.tar"  \
    --train-num-samples 272000 \
    --val-num-samples 9600 \
    --dataset-type icddataset \
    --name ecg_phe_month_instruct_trajectory_binary_strict_gpt_1 \
    --workers 4 \
    --batch-size 16 \
    --epochs 20 \
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
    --billable-probability 0.0 \
    --top-non-probability 1.0 \
    --code-column phecode \
    --wandb-project-name="open-clip-phe-test-runs" \
    --encounter-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/all_encounters_2308_with_phecodes_with_na.parquet" \
    --time-difference-normalize 1 \
    --max_seq_length 1024 \
    --pad_id 50256 \
    --distance-threshold 7 30 60 120 180 365 \
    --shuffle-bins \
    --training-type month_instruct_trajectory \
    --label-type binary_strict \
    --trajectory-sampling-weights 0.5 0.0 0.5 \
    --seed 0


# METHOD 3C: Binary labels without trajectory
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
torchrun \
    --nnodes=1 --nproc_per_node=8 --master_addr=localhost --master_port=2126 \
    -m main \
    --train-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_train_2402/shard_{0000..0084}.tar"  \
    --val-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2402/shard_{0000..0002}.tar"  \
    --train-num-samples 272000 \
    --val-num-samples 9600 \
    --dataset-type icddataset \
    --name ecg_phe_month_instruct_binary_gpt_2 \
    --workers 4 \
    --batch-size 64 \
    --epochs 100 \
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
    --billable-probability 0.0 \
    --top-non-probability 1.0 \
    --code-column phecode \
    --wandb-project-name="open-clip-phe-test-runs" \
    --encounter-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/all_encounters_2308_with_phecodes_with_na.parquet" \
    --time-difference-normalize 1 \
    --max_seq_length 256 \
    --pad_id 50256 \
    --distance-threshold 7 30 60 120 180 365 \
    --shuffle-bins \
    --training-type month_instruct \
    --label-type binary \
    --trajectory-sampling-weights 1.0 0.0 0.0 \
    --seed 0

# METHOD 3D: Binary Strict labels without trajectory
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
torchrun \
    --nnodes=1 --nproc_per_node=8 --master_addr=localhost --master_port=2126 \
    -m main \
    --train-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_train_2402/shard_{0000..0084}.tar"  \
    --val-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2402/shard_{0000..0002}.tar"  \
    --train-num-samples 272000 \
    --val-num-samples 9600 \
    --dataset-type icddataset \
    --name ecg_phe_month_instruct_binary_strict_gpt_1 \
    --workers 4 \
    --batch-size 16 \
    --epochs 20 \
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
    --billable-probability 0.0 \
    --top-non-probability 1.0 \
    --code-column phecode \
    --wandb-project-name="open-clip-phe-test-runs" \
    --encounter-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/all_encounters_2308_with_phecodes_with_na.parquet" \
    --time-difference-normalize 1 \
    --max_seq_length 1024 \
    --pad_id 50256 \
    --distance-threshold 7 30 60 120 180 365 \
    --shuffle-bins \
    --training-type month_instruct_trajectory \
    --label-type binary_strict \
    --trajectory-sampling-weights 1.0 0.0 1.0 \
    --seed 0