#export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
#torchrun \
#    --nnodes=1 --nproc_per_node=8 --master_addr=localhost --master_port=2126 \
#    -m main \
#    --train-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_train_23_10_23/shard_{0000..0082}.tar"  \
#    --val-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_23_10_23/shard_{0000..0002}.tar"  \
#    --train-num-samples 1062400 \
#    --val-num-samples 38400 \
#    --dataset-type icddataset \
#    --name="ecg_icd_test_run_v97" \
#    --workers 4 \
#    --batch-size 256 \
#    --epochs 32 \
#    --lr 1e-4 \
#    --beta1 0.9 \
#    --beta1 0.98 \
#    --eps 1e-6 \
#    --wd 0.2 \
#    --warmup 4000 \
#    --lr-scheduler="cosine" \
#    --lr-cooldown-end 1e-5 \
#    --grad-clip-norm 1.0 \
#    --coca-caption-loss-weight 1.0 \
#    --coca-contrastive-loss-weight 1.0 \
#    --precision amp \
#    --save-frequency 1 \
#    --zeroshot-frequency 1 \
#    --local-loss \
#    --gather-with-grad \
#    --model scatter_base \
#    --report-to wandb \
#    --past-time-delta="355d" \
#    --future-time-delta="355d" \
#    --wandb-project-name="open-clip-test-runs" \
#    --encounter-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/earliest_encounters_2308_with_score.parquet" \
#    --k-shot 0 \
#    --seed 0


export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
torchrun \
    --nnodes=1 --nproc_per_node=8 --master_addr=localhost --master_port=2126 \
    -m main \
    --train-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_train_23_10_23/shard_{0000..0082}.tar"  \
    --val-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_23_10_23/shard_{0000..0002}.tar"  \
    --train-num-samples 1062400 \
    --val-num-samples 38400 \
    --dataset-type icddataset \
    --name="ecg_phe_test_run_9" \
    --workers 4 \
    --batch-size 256 \
    --epochs 200 \
    --lr 5e-4 \
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
    --model scatter_base \
    --report-to wandb \
    --billable-probability 0.0 \
    --top-non-probability 1.0 \
    --random-negative-probability 1.0 \
    --code-column phecode \
    --past-time-delta="720d" \
    --future-time-delta="720d" \
    --wandb-project-name="open-clip-phe-test-runs" \
    --encounter-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/all_encounters_2308_with_phecodes_with_na.parquet" \
    --number-of-instructions 1 2 3 4 5 \
    --seed 0


export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
torchrun \
    --nnodes=1 --nproc_per_node=8 --master_addr=localhost --master_port=2126 \
    -m main \
    --train-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_train_23_10_23/shard_{0000..0082}.tar"  \
    --val-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_23_10_23/shard_{0000..0002}.tar"  \
    --train-num-samples 1062400 \
    --val-num-samples 38400 \
    --dataset-type icddataset \
    --name="ecg_phe_trj_run_12" \
    --workers 4 \
    --batch-size 32 \
    --epochs 18 \
    --lr 2.5e-4 \
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
    --model scatter_base \
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
    --seed 0


