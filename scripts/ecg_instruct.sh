export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
torchrun \
    --nnodes=1 --nproc_per_node=8 --master_addr=localhost --master_port=2126 \
    -m main \
    --train-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_train_23_10_23/shard_{0000..0082}.tar"  \
    --val-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_23_10_23/shard_{0000..0002}.tar"  \
    --train-num-samples 1062400 \
    --val-num-samples 38400 \
    --dataset-type icddataset \
    --name ecg_phe_gpt_instruct_k_1_trial_3 \
    --workers 4 \
    --batch-size 360 \
    --epochs 128 \
    --lr 5e-4 \
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
    --number-of-instructions 1 \
    --max_seq_length 76 \
    --pad_id 50256 \
    --distance-threshold 60 \
    --seed 0

# TEXT TRANSFORMER

# 1. Easy Negatives
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
torchrun \
    --nnodes=1 --nproc_per_node=8 --master_addr=localhost --master_port=2127 \
    -m main \
    --train-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_train_23_10_23/shard_{0000..0082}.tar"  \
    --val-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_23_10_23/shard_{0000..0002}.tar"  \
    --train-num-samples 1062400 \
    --val-num-samples 38400 \
    --dataset-type icddataset \
    --name="ecg_phe_trj_run_scatter2_2" \
    --workers 4 \
    --batch-size 16 \
    --epochs 18 \
    --lr 2.5e-4 \
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
    --billable-probability 0.0 \
    --top-non-probability 1.0 \
    --code-column phecode \
    --wandb-project-name="open-clip-phe-test-runs" \
    --encounter-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/all_encounters_2308_with_phecodes_with_na.parquet" \
    --time-difference-normalize 1 \
    --number-of-instructions 1 \
    --max_seq_length 76 \
    --pad_id 0 \
    --distance-threshold 60 \
    --negatives-type random \
    --training-type all \
    --seed 0



# debug run for CNN
#     #--name="ecg_phe_cnn_run_1" \
#     --name="ecg_phe_cnn_run_21" \
#export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export CUDA_VISIBLE_DEVICES='1'
torchrun \
    --nnodes=1 --nproc_per_node=1 --master_addr=localhost --master_port=2126 \
    -m main \
    --train-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_train_23_10_23/shard_{0000..0082}.tar" \
    --val-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_23_10_23/shard_{0000..0002}.tar" \
    --train-num-samples 1062400 \
    --val-num-samples 38400 \
    --dataset-type icddataset \
    --workers 4 \
    --batch-size 4 \
    --epochs 20 \
    --lr 1e-4 \
    --beta1 0.9 \
    --beta1 0.98 \
    --eps 1e-6 \
    --wd 0.01 \
    --warmup 4000 \
    --lr-scheduler cosine \
    --lr-scheduler="cosine" \
    --lr-cooldown-end 5e-5 \
    --coca-caption-loss-weight 1.0 \
    --coca-contrastive-loss-weight 0.0 \
    --precision amp \
    --save-frequency 1 \
    --val-frequency 1 \
    --zeroshot-frequency 0 \
    --local-loss \
    --gather-with-grad \
    --model scatter_cnn \
    --report-to wandb \
    --billable-probability 0.0 \
    --top-non-probability 1.0 \
    --code-column phecode \
    --wandb-project-name="open-clip-phe-test-runs" \
    --encounter-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/all_encounters_2308_with_phecodes_with_na.parquet" \
    --time-difference-normalize 1 \
    --number-of-instructions 1 \
    --max_seq_length 76 \
    --pad_id 0 \
    --distance-threshold 60 \
    --negatives-type cached \
    --training-type all \
    --seed 0

# windowed scattering transformer
export CUDA_VISIBLE_DEVICES='0'
torchrun \
    --nnodes=1 --nproc_per_node=1 --master_addr=localhost --master_port=2127 \
    -m main \
    --train-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_train_23_10_23/shard_{0000..0007}.tar" \
    --val-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_23_10_23/shard_{0000..0002}.tar" \
    --train-num-samples 102400 \
    --val-num-samples 38400 \
    --dataset-type icddataset \
    --workers 4 \
    --batch-size 4 \
    --epochs 20 \
    --lr 1e-5 \
    --beta1 0.9 \
    --beta1 0.98 \
    --eps 1e-6 \
    --wd 0.01 \
    --warmup 4000 \
    --lr-scheduler="cosine" \
    --lr-cooldown-end 5e-5 \
    --coca-caption-loss-weight 1.0 \
    --coca-contrastive-loss-weight 0.0 \
    --precision amp \
    --save-frequency 1 \
    --val-frequency 1 \
    --zeroshot-frequency 0 \
    --local-loss \
    --gather-with-grad \
    --model ecg_scatter_windowed \
    --report-to wandb \
    --billable-probability 0.0 \
    --top-non-probability 1.0 \
    --code-column phecode \
    --wandb-project-name="open-clip-phe-test-runs" \
    --encounter-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/all_encounters_2308_with_phecodes_with_na.parquet" \
    --time-difference-normalize 1 \
    --max_seq_length 1024 \
    --distance-threshold 7 30 60 120 180 365 \
    --shuffle-bins \
    --force-patch-dropout 0.1 \
    --seed 0

# windowed CNN transformer
export CUDA_VISIBLE_DEVICES='1'
torchrun \
    --nnodes=1 --nproc_per_node=1 --master_addr=localhost --master_port=2126 \
    -m main \
    --train-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_train_23_10_23/shard_{0000..0007}.tar" \
    --val-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_23_10_23/shard_{0000..0002}.tar" \
    --train-num-samples 102400 \
    --val-num-samples 38400 \
    --dataset-type icddataset \
    --workers 4 \
    --batch-size 4 \
    --epochs 20 \
    --lr 1e-4 \
    --beta1 0.9 \
    --beta1 0.98 \
    --eps 1e-6 \
    --wd 0.01 \
    --warmup 4000 \
    --lr-scheduler="cosine" \
    --lr-cooldown-end 5e-5 \
    --coca-caption-loss-weight 1.0 \
    --coca-contrastive-loss-weight 0.0 \
    --precision amp \
    --save-frequency 1 \
    --val-frequency 1 \
    --zeroshot-frequency 0 \
    --local-loss \
    --gather-with-grad \
    --model ecg_cnn_windowed \
    --report-to wandb \
    --billable-probability 0.0 \
    --top-non-probability 1.0 \
    --code-column phecode \
    --wandb-project-name="open-clip-phe-test-runs" \
    --encounter-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/all_encounters_2308_with_phecodes_with_na.parquet" \
    --time-difference-normalize 1 \
    --max_seq_length 1024 \
    --distance-threshold 7 30 60 120 180 365 \
    --shuffle-bins \
    --force-patch-dropout 0.1 \
    --seed 0

# global scattering transformer
export CUDA_VISIBLE_DEVICES='1'
torchrun \
    --nnodes=1 --nproc_per_node=1 --master_addr=localhost --master_port=2126 \
    -m main \
    --train-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_train_23_10_23/shard_{0000..0007}.tar" \
    --val-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_23_10_23/shard_{0000..0002}.tar" \
    --train-num-samples 102400 \
    --val-num-samples 38400 \
    --dataset-type icddataset \
    --workers 4 \
    --batch-size 4 \
    --epochs 20 \
    --lr 1e-5 \
    --beta1 0.9 \
    --beta1 0.98 \
    --eps 1e-6 \
    --wd 0.01 \
    --warmup 4000 \
    --lr-scheduler="cosine" \
    --lr-cooldown-end 5e-5 \
    --coca-caption-loss-weight 1.0 \
    --coca-contrastive-loss-weight 0.0 \
    --precision amp \
    --save-frequency 1 \
    --val-frequency 1 \
    --zeroshot-frequency 0 \
    --local-loss \
    --gather-with-grad \
    --model ecg_scatter_global \
    --report-to wandb \
    --billable-probability 0.0 \
    --top-non-probability 1.0 \
    --code-column phecode \
    --wandb-project-name="open-clip-phe-test-runs" \
    --encounter-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/all_encounters_2308_with_phecodes_with_na.parquet" \
    --time-difference-normalize 1 \
    --max_seq_length 1024 \
    --distance-threshold 7 30 60 120 180 365 \
    --shuffle-bins \
    --force-patch-dropout 0.1 \
    --seed 0
