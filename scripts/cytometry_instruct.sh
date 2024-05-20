export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
torchrun \
    --nnodes=1 --nproc_per_node=8 --master_addr=localhost --master_port=2126 \
    -m main \
    --train-data="/mnt/obi0/phi/ehr_projects/sysmex_datasets/processed_23-06-02/mgh_train_enc180d_v1/shard_{000..082}.tar"  \
    --val-data="/mnt/obi0/phi/ehr_projects/sysmex_datasets/processed_23-06-02/mgh_val_enc180d_v1/shard_{000..002}.tar"  \
    --train-num-samples 1062400 \
    --val-num-samples 38400 \
    --dataset-type icddataset \
    --name="cytometry_phe_test_run_5" \
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
    --model coca_cyto_base \
    --billable-probability 0.0 \
    --top-non-probability 1.0 \
    --random-negative-probability 1.0 \
    --code-column phecode \
    --sample-result-date-column "ResultDTS" \
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
    --train-data="/mnt/obi0/phi/ehr_projects/sysmex_datasets/processed_2402/mgh_train_v1/shard_{000..055}.tar"  \
    --val-data="/mnt/obi0/phi/ehr_projects/sysmex_datasets/processed_2402/mgh_val_v1/shard_{000..002}.tar"  \
    --train-num-samples 716800 \
    --val-num-samples 38400 \
    --dataset-type icddataset \
    --name="cytometry_phe_test_run_6" \
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
    --model coca_cyto_base \
    --billable-probability 0.0 \
    --top-non-probability 1.0 \
    --code-column phecode \
    --sample-result-date-column "ResultDTS" \
    --past-time-delta="720d" \
    --future-time-delta="720d" \
    --wandb-project-name="open-clip-phe-test-runs" \
    --encounter-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/all_encounters_2308_with_phecodes_with_na.parquet" \
    --number-of-instructions 1 2 3 4 5 \
    --time-difference-normalize 1 \
    --max_seq_length 1024 \
    --distance-threshold 7 30 60 120 180 365 \
    --shuffle-bins \
    --seed 0


export CUDA_VISIBLE_DEVICES='1'
torchrun \
    --nnodes=1 --nproc_per_node=1 --master_addr=localhost --master_port=2131 \
    -m main \
    --train-data="/mnt/obi0/phi/ehr_projects/sysmex_datasets/processed_23-06-02/mgh_train_enc180d_v1/shard_{000..082}.tar"  \
    --val-data="/mnt/obi0/phi/ehr_projects/sysmex_datasets/processed_23-06-02/mgh_val_enc180d_v1/shard_{000..002}.tar"  \
    --train-num-samples 1062400 \
    --val-num-samples 38400 \
    --dataset-type icddataset \
    --workers 4 \
    --batch-size 128 \
    --epochs 50 \
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
    --report-to wandb \
    --code-column phecode \
    --sample-result-date-column "ResultDTS" \
    --wandb-project-name="open-clip-phe-test-runs" \
    --encounter-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/all_encounters_2308_with_phecodes_with_na.parquet.check" \
    --demographic-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/demographics_2404.parquet" \
    --labs-folder="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/labs" \
    --time-difference-normalize 1 \
    --number-of-instructions 1 \
    --k-shot 1 \
    --max_seq_length 64 \
    --pad_id 1 \
    --distance-threshold 60 \
    --negatives-type random \
    --tasks diagnosis \
    --task-shuffle \
    --fixed-position-range \
    --future-only \
    --seed 0 \
    --model coca_cyto_base


export CUDA_VISIBLE_DEVICES='7'
torchrun \
    --nnodes=1 --nproc_per_node=1 --master_addr=localhost --master_port=2131 \
    -m main \
    --train-data="/mnt/obi0/phi/ehr_projects/sysmex_datasets/processed_2402/mgh_train_enc_v2/shard_{000..004}.tar"  \
    --val-data="/mnt/obi0/phi/ehr_projects/sysmex_datasets/processed_2402/mgh_train_enc_v2/shard_{002..008}.tar"  \
    --train-num-samples 1062400 \
    --val-num-samples 38400 \
    --dataset-type icddataset \
    --workers 2 \
    --batch-size 128 \
    --epochs 50 \
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
    --report-to wandb \
    --code-column phecode \
    --sample-result-date-column "ResultDTS" \
    --wandb-project-name="open-clip-phe-test-runs" \
    --encounter-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/all_encounters_2308_with_phecodes_with_na.parquet.check" \
    --demographic-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/demographics_2404.parquet" \
    --labs-folder="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/labs" \
    --time-difference-normalize 1 \
    --number-of-instructions 1 \
    --k-shot 1 \
    --max_seq_length 64 \
    --pad_id 1 \
    --distance-threshold 60 \
    --negatives-type random \
    --tasks diagnosis \
    --task-shuffle \
    --fixed-position-range \
    --future-only \
    --seed 0 \
    --model coca_cyto_base


#export CUDA_VISIBLE_DEVICES='7'

export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
torchrun \
    --nnodes=1 --nproc_per_node=8 --master_addr=localhost --master_port=2130 \
    -m main \
    --train-data="/mnt/obi0/phi/ehr_projects/sysmex_datasets/processed_23-06-02/mgh_train_enc180d_v2/shard_{000..058}.tar"  \
    --val-data="/mnt/obi0/phi/ehr_projects/sysmex_datasets/processed_23-06-02/mgh_train_enc180d_v2/shard_{000..004}.tar"  \
    --train-num-samples 742400 \
    --val-num-samples 64000 \
    --dataset-type icddataset \
    --workers 6 \
    --batch-size 196 \
    --accum-freq 4 \
    --epochs 250 \
    --lr 5e-5 \
    --beta1 0.9 \
    --beta2 0.98 \
    --eps 1e-6 \
    --wd 0.1 \
    --grad-clip-norm 1.0 \
    --warmup 4000 \
    --lr-scheduler cosine \
    --lr-cooldown-end 5e-6 \
    --coca-caption-loss-weight 1.0 \
    --coca-contrastive-loss-weight 0.0 \
    --precision amp \
    --save-frequency 10 \
    --val-frequency 10 \
    --zeroshot-frequency 0 \
    --local-loss \
    --gather-with-grad \
    --report-to wandb \
    --code-column phecode \
    --sample-result-date-column "ResultDTS" \
    --wandb-project-name="open-clip-phe-test-runs" \
    --encounter-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/all_encounters_2308_with_phecodes_with_na.parquet.check" \
    --demographic-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/demographics_2404.parquet" \
    --labs-folder="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/labs" \
    --time-difference-normalize 1 \
    --number-of-instructions 1 \
    --k-shot 1 1 1 1 1 2 2 2 3 4 \
    --k-shot-demographics 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 2 2 2 3 4 \
    --max_seq_length 64 \
    --pad_id 0 \
    --distance-threshold 60 \
    --negatives-type random \
    --tasks demographics diagnosis \
    --task-shuffle \
    --fixed-position-range \
    --future-only \
    --seed 0 \
    --training-eval-codes MS_700.11 BI_170.1 ID_092.2 CA_121.1 \
    --model coca_cyto_base \
    --name cyto-diagnosis_demographic-coca_cyto_base-future_250_trial_6


    # ID_092.2  Sepsis
    # CA_121.1  AML
    # MS_700.11 SLE
    # BI_170.1  Neutropenia
