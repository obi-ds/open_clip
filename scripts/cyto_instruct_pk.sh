# 1. Scratch - Diagnosis - Random - Fixed - Future
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6'
torchrun \
    --nnodes=1 --nproc_per_node=7 --master_addr=localhost --master_port=2120 \
    -m main \
    --train-data=""  \
    --val-data=""  \
    --train-num-samples  \
    --val-num-samples  \
    --dataset-type icddataset \
    --workers 8 \
    --batch-size 100 \
    --accum-freq 3 \
    --epochs 250 \
    --lr-scheduler cosine \
    --lr 2.5e-5 \
    --lr-cooldown-end 2.5e-6 \
    --beta1 0.9 \
    --beta2 0.999 \
    --eps 1e-8 \
    --wd 0.1 \
    --grad-clip-norm 1.0 \
    --warmup 10000 \
    --precision amp \
    --save-frequency 10 \
    --val-frequency 10 \
    --loss-function lm \
    --report-to wandb \
    --code-column phecode \
    --wandb-project-name="open-clip-phe-test-runs" \
    --encounter-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/all_encounters_2308_with_phecodes_with_na.parquet.check" \
    --demographic-file=/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/demographics_2404.parquet \
    --labs-folder=/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/labs \
    --time-difference-normalize 1 \
    --number-of-instructions 1 \
    --k-shot 1 \
    --max_seq_length 64 \
    --pad_id 1 \
    --add-img-token \
    --negatives-type random \
    --tasks diagnosis \
    --training-eval-codes CV_424.4 EM_249 GU_627.2 NS_324.11 CV_416.214 \
    --fixed-position-range \
    --future-only \
    --name  \
    --model  \
    --seed 0









export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6'
torchrun \
    --nnodes=1 --nproc_per_node=7 --master_addr=localhost --master_port=2130 \
    -m main \
    --train-data="/mnt/obi0/phi/ehr_projects/sysmex_datasets/processed_23-06-02/mgh_train_enc180d_v2/shard_{000..058}.tar"  \
    --val-data="/mnt/obi0/phi/ehr_projects/sysmex_datasets/processed_23-06-02/mgh_train_enc180d_v2/shard_{000..004}.tar"  \
    --train-num-samples 742400 \
    --val-num-samples 64000 \
    --dataset-type icddataset \
    --workers 6 \
    --batch-size 50 \
    --accum-freq 13 \
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
    --k-shot-labs 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 2 2 2 3 4 \
    --max_seq_length 100 \
    --pad_id 1 \
    --distance-threshold 60 \
    --negatives-type random \
    --tasks labs demographics diagnosis \
    --task-shuffle \
    --fixed-position-range \
    --future-only \
    --loss-function lm \
    --seed 0 \
    --training-eval-codes MS_700.11 BI_170.1 ID_092.2 CA_121.1 \
    --model coca_cyto_biogpt \
    --name cyto-labs_diagnosis_demographic-coca_cyto_biogpt-future_250_trial_5