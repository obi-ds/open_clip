# 1. Scratch - Diagnosis - Random - Fixed - Future
# ecg_moca_scratch_diagnosis_k_1_random_fixed_future_250_trial_16
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6'
torchrun \
    --nnodes=1 --nproc_per_node=7 --master_addr=localhost --master_port=2120 \
    -m main \
    --train-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_train_2403/shard_{0000..0078}.tar"  \
    --val-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2403/shard_{0000..0010}.tar"  \
    --train-num-samples 252800 \
    --val-num-samples 35200 \
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
    --name ecg_moca_scratch_diagnosis_k_1_random_fixed_future_250_trial_16 \
    --model ecg_moca_biogpt_scratch \
    --seed 0


# 2. Scratch - QFormer - Diagnosis - Random - Fixed - Future
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6'
torchrun \
    --nnodes=1 --nproc_per_node=7 --master_addr=localhost --master_port=2120 \
    -m main \
    --train-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_train_2403/shard_{0000..0078}.tar"  \
    --val-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2403/shard_{0000..0010}.tar"  \
    --train-num-samples 252800 \
    --val-num-samples 35200 \
    --dataset-type icddataset \
    --workers 8 \
    --batch-size 60 \
    --accum-freq 5 \
    --epochs 250 \
    --lr 2.5e-5 \
    --beta1 0.9 \
    --beta2 0.999 \
    --eps 1e-8 \
    --wd 0.1 \
    --grad-clip-norm 1.0 \
    --warmup 10000 \
    --lr-scheduler cosine \
    --lr-cooldown-end 5e-6 \
    --precision amp \
    --save-frequency 10 \
    --val-frequency 10 \
    --zeroshot-frequency 0 \
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
    --distance-threshold 60 \
    --negatives-type random \
    --tasks diagnosis \
    --training-eval-codes CV_424.4 EM_249 GU_627.2 NS_324.11 CV_416.214 \
    --fixed-position-range \
    --future-only \
    --name ecg_moca_scratch_q_former_diagnosis_k_1_random_fixed_future_250_trial_5 \
    --model ecg_moca_biogpt_scratch_q_former \
    --seed 0



# 3. Pre-trained Text - Diagnosis - Random - Fixed - Future - Frozen
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6'
torchrun \
    --nnodes=1 --nproc_per_node=7 --master_addr=localhost --master_port=2120 \
    -m main \
    --train-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_train_2403/shard_{0000..0078}.tar"  \
    --val-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2403/shard_{0000..0010}.tar"  \
    --train-num-samples 252800 \
    --val-num-samples 35200 \
    --dataset-type icddataset \
    --workers 8 \
    --batch-size 100 \
    --accum-freq 3 \
    --epochs 250 \
    --lr 2.5e-5 \
    --beta1 0.9 \
    --beta2 0.999 \
    --eps 1e-8 \
    --wd 0.1 \
    --grad-clip-norm 1.0 \
    --warmup 10000 \
    --lr-scheduler cosine \
    --lr-cooldown-end 5e-6 \
    --precision amp \
    --save-frequency 10 \
    --val-frequency 10 \
    --zeroshot-frequency 0 \
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
    --distance-threshold 60 \
    --negatives-type random \
    --tasks diagnosis \
    --training-eval-codes CV_424.4 EM_249 GU_627.2 NS_324.11 CV_416.214 \
    --fixed-position-range \
    --future-only \
    --name ecg_moca_pretrained_frozen_diagnosis_k_1_random_fixed_future_250_trial_4 \
    --model ecg_moca_biogpt \
    --lock-text \
    --seed 0



# 4. Pre-trained Text - Diagnosis - Random - Fixed - Future - Fine-tuned
# FIXME: Unable to match performance of models trained from scratch
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6'
torchrun \
    --nnodes=1 --nproc_per_node=7 --master_addr=localhost --master_port=2120 \
    -m main \
    --train-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_train_2403/shard_{0000..0078}.tar"  \
    --val-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2403/shard_{0000..0010}.tar"  \
    --train-num-samples 252800 \
    --val-num-samples 35200 \
    --dataset-type icddataset \
    --workers 8 \
    --batch-size 100 \
    --accum-freq 3 \
    --epochs 250 \
    --lr 4.5e-5 \
    --text-decoder-lr 5e-6 \
    --beta1 0.9 \
    --beta2 0.999 \
    --eps 1e-8 \
    --wd 0.1 \
    --grad-clip-norm 1.0 \
    --warmup 10000 \
    --lr-scheduler cosine \
    --lr-cooldown-end 5e-6 \
    --precision amp \
    --save-frequency 10 \
    --val-frequency 10 \
    --zeroshot-frequency 0 \
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
    --distance-threshold 60 \
    --negatives-type random \
    --tasks diagnosis \
    --fixed-position-range \
    --future-only \
    --name ecg_moca_pretrained_diagnosis_k_1_random_fixed_future_250_trial_7 \
    --model ecg_moca_biogpt \
    --seed 0



# 5. Pre-trained Text - QFormer - Diagnosis - Random - Fixed - Future - Fine-tuned
# FIXME: Unable to match performance of models trained from scratch
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6'
torchrun \
    --nnodes=1 --nproc_per_node=7 --master_addr=localhost --master_port=2120 \
    -m main \
    --train-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_train_2403/shard_{0000..0078}.tar"  \
    --val-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2403/shard_{0000..0010}.tar"  \
    --train-num-samples 252800 \
    --val-num-samples 35200 \
    --dataset-type icddataset \
    --workers 8 \
    --batch-size 60 \
    --accum-freq 5 \
    --epochs 250 \
    --lr 4e-5 \
    --text-decoder-lr 1e-5 \
    --beta1 0.9 \
    --beta2 0.999 \
    --eps 1e-8 \
    --wd 0.1 \
    --grad-clip-norm 1.0 \
    --warmup 10000 \
    --lr-scheduler cosine \
    --lr-cooldown-end 5e-6 \
    --precision amp \
    --save-frequency 10 \
    --val-frequency 10 \
    --zeroshot-frequency 0 \
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
    --distance-threshold 60 \
    --negatives-type random \
    --tasks diagnosis \
    --fixed-position-range \
    --future-only \
    --name ecg_moca_pretrained_q_former_diagnosis_k_1_random_fixed_future_250_trial_2 \
    --model ecg_moca_biogpt_q_former \
    --seed 0



# 6. Scratch - Labs - Demographics - Diagnosis - Random - Fixed - Future
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6'
torchrun \
    --nnodes=1 --nproc_per_node=7 --master_addr=localhost --master_port=2120 \
    -m main \
    --train-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_train_2403/shard_{0000..0078}.tar"  \
    --val-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2403/shard_{0000..0010}.tar"  \
    --train-num-samples 252800 \
    --val-num-samples 35200 \
    --dataset-type icddataset \
    --workers 8 \
    --batch-size 75 \
    --accum-freq 4 \
    --epochs 250 \
    --lr 2.5e-5 \
    --beta1 0.9 \
    --beta2 0.999 \
    --eps 1e-8 \
    --wd 0.1 \
    --grad-clip-norm 1.0 \
    --warmup 10000 \
    --lr-scheduler cosine \
    --lr-cooldown-end 5e-6 \
    --precision amp \
    --save-frequency 10 \
    --val-frequency 10 \
    --zeroshot-frequency 0 \
    --loss-function lm \
    --report-to wandb \
    --code-column phecode \
    --wandb-project-name="open-clip-phe-test-runs" \
    --encounter-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/all_encounters_2308_with_phecodes_with_na.parquet.check" \
    --demographic-file=/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/demographics_2404.parquet \
    --labs-folder=/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/labs \
    --time-difference-normalize 1 \
    --number-of-instructions 1 \
    --k-shot 1 1 1 1 1 2 2 2 3 4 \
    --k-shot-demographics 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 2 2 2 3 4 \
    --k-shot-labs 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 2 2 2 3 4 \
    --max_seq_length 128 \
    --pad_id 1 \
    --add-img-token \
    --distance-threshold 60 \
    --negatives-type random \
    --tasks demographics diagnosis labs \
    --task-shuffle \
    --training-eval-codes CV_424.4 EM_249 GU_627.2 NS_324.11 CV_416.214 \
    --fixed-position-range \
    --future-only \
    --name ecg_moca_scratch_demographics_diagnosis_labs_random_fixed_future_250_trial_6 \
    --model ecg_moca_biogpt_scratch \
    --seed 0



# 7. Scratch - Labs - Demographics - Diagnosis - Random - Fixed - Future - Higher Learning Rate
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6'
torchrun \
    --nnodes=1 --nproc_per_node=7 --master_addr=localhost --master_port=2120 \
    -m main \
    --train-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_train_2403/shard_{0000..0078}.tar"  \
    --val-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2403/shard_{0000..0010}.tar"  \
    --train-num-samples 252800 \
    --val-num-samples 35200 \
    --dataset-type icddataset \
    --workers 8 \
    --batch-size 75 \
    --accum-freq 4 \
    --epochs 250 \
    --lr 4e-5 \
    --beta1 0.9 \
    --beta2 0.999 \
    --eps 1e-8 \
    --wd 0.1 \
    --grad-clip-norm 1.0 \
    --warmup 10000 \
    --lr-scheduler cosine \
    --lr-cooldown-end 5e-6 \
    --precision amp \
    --save-frequency 10 \
    --val-frequency 10 \
    --zeroshot-frequency 0 \
    --loss-function lm \
    --report-to wandb \
    --code-column phecode \
    --wandb-project-name="open-clip-phe-test-runs" \
    --encounter-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/all_encounters_2308_with_phecodes_with_na.parquet.check" \
    --demographic-file=/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/demographics_2404.parquet \
    --labs-folder=/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/labs \
    --time-difference-normalize 1 \
    --number-of-instructions 1 \
    --k-shot 1 1 1 1 1 2 2 2 3 4 \
    --k-shot-demographics 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 2 2 2 3 4 \
    --k-shot-labs 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 2 2 2 3 4 \
    --max_seq_length 128 \
    --pad_id 1 \
    --add-img-token \
    --distance-threshold 60 \
    --negatives-type random \
    --tasks demographics diagnosis labs \
    --task-shuffle \
    --training-eval-codes CV_424.4 EM_249 GU_627.2 NS_324.11 CV_416.214 \
    --fixed-position-range \
    --future-only \
    --name ecg_moca_scratch_demographics_diagnosis_labs_random_fixed_future_250_trial_8 \
    --model ecg_moca_biogpt_scratch \
    --seed 0



# 8. Scratch - Labs - Demographics - Diagnosis - Random Cached - Fixed - Future
# TODO: Choose LR based on previous run - The lower learning rate worked better
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6'
torchrun \
    --nnodes=1 --nproc_per_node=7 --master_addr=localhost --master_port=2120 \
    -m main \
    --train-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_train_2403/shard_{0000..0078}.tar"  \
    --val-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2403/shard_{0000..0010}.tar"  \
    --train-num-samples 252800 \
    --val-num-samples 35200 \
    --dataset-type icddataset \
    --workers 8 \
    --batch-size 75 \
    --accum-freq 4 \
    --epochs 250 \
    --lr 2.5e-5 \
    --beta1 0.9 \
    --beta2 0.999 \
    --eps 1e-8 \
    --wd 0.1 \
    --grad-clip-norm 1.0 \
    --warmup 10000 \
    --lr-scheduler cosine \
    --lr-cooldown-end 5e-6 \
    --precision amp \
    --save-frequency 10 \
    --val-frequency 10 \
    --zeroshot-frequency 0 \
    --loss-function lm \
    --report-to wandb \
    --code-column phecode \
    --wandb-project-name="open-clip-phe-test-runs" \
    --encounter-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/all_encounters_2308_with_phecodes_with_na.parquet.check" \
    --demographic-file=/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/demographics_2404.parquet \
    --labs-folder=/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/labs \
    --time-difference-normalize 1 \
    --number-of-instructions 1 \
    --k-shot 1 1 1 1 1 2 2 2 3 4 \
    --k-shot-demographics 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 2 2 2 3 4 \
    --k-shot-labs 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 2 2 2 3 4 \
    --max_seq_length 128 \
    --pad_id 1 \
    --add-img-token \
    --distance-threshold 60 \
    --negatives-type random_cached \
    --tasks demographics diagnosis labs \
    --task-shuffle \
    --training-eval-codes CV_424.4 EM_249 GU_627.2 NS_324.11 CV_416.214 \
    --fixed-position-range \
    --future-only \
    --name ecg_moca_scratch_demographics_diagnosis_labs_random_cached_fixed_future_250_trial_1 \
    --model ecg_moca_biogpt_scratch \
    --seed 0



# 9. Scratch - Diagnosis - Random - Fixed - Future - Token Loss Weighting
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6'
torchrun \
    --nnodes=1 --nproc_per_node=7 --master_addr=localhost --master_port=2120 \
    -m main \
    --train-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_train_2403/shard_{0000..0078}.tar"  \
    --val-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2403/shard_{0000..0010}.tar"  \
    --train-num-samples 252800 \
    --val-num-samples 35200 \
    --dataset-type icddataset \
    --workers 8 \
    --batch-size 100 \
    --accum-freq 3 \
    --epochs 250 \
    --lr 2.5e-5 \
    --beta1 0.9 \
    --beta2 0.999 \
    --eps 1e-8 \
    --wd 0.1 \
    --grad-clip-norm 1.0 \
    --warmup 10000 \
    --lr-scheduler cosine \
    --lr-cooldown-end 5e-6 \
    --precision amp \
    --save-frequency 10 \
    --val-frequency 10 \
    --zeroshot-frequency 0 \
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
    --distance-threshold 60 \
    --negatives-type random \
    --tasks diagnosis \
    --training-eval-codes CV_424.4 EM_249 GU_627.2 NS_324.11 CV_416.214 \
    --fixed-position-range \
    --future-only \
    --token-loss-weighting \
    --name ecg_moca_scratch_weighted_diagnosis_k_1_random_fixed_future_250_trial_6 \
    --model ecg_moca_biogpt_scratch \
    --seed 0



# 10. Scratch - Diagnosis - Random - Fixed - Future - Token Loss Weighting - Focal
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6'
torchrun \
    --nnodes=1 --nproc_per_node=7 --master_addr=localhost --master_port=2120 \
    -m main \
    --train-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_train_2403/shard_{0000..0078}.tar"  \
    --val-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2403/shard_{0000..0010}.tar"  \
    --train-num-samples 252800 \
    --val-num-samples 35200 \
    --dataset-type icddataset \
    --workers 8 \
    --batch-size 95 \
    --accum-freq 3 \
    --epochs 250 \
    --lr 2.5e-5 \
    --beta1 0.9 \
    --beta2 0.999 \
    --eps 1e-8 \
    --wd 0.1 \
    --grad-clip-norm 1.0 \
    --warmup 10000 \
    --lr-scheduler cosine \
    --lr-cooldown-end 5e-6 \
    --precision amp \
    --save-frequency 10 \
    --val-frequency 10 \
    --zeroshot-frequency 0 \
    --loss-function focal \
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
    --distance-threshold 60 \
    --negatives-type random \
    --tasks diagnosis \
    --training-eval-codes CV_424.4 EM_249 GU_627.2 NS_324.11 CV_416.214 \
    --fixed-position-range \
    --future-only \
    --token-loss-weighting \
    --name ecg_moca_scratch_weighted_focal_diagnosis_k_1_random_fixed_future_250_trial_5 \
    --model ecg_moca_biogpt_scratch \
    --seed 0



# 12. Scratch - QFormer - Labs - Demographics - Diagnosis - Random - Fixed - Future
# TODO: Choose negatives type and LR based on previous run
#  Negatives: random negatives worked better than random_cached
#  LR: Lower learning rate worked better
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6'
torchrun \
    --nnodes=1 --nproc_per_node=7 --master_addr=localhost --master_port=2120 \
    -m main \
    --train-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_train_2403/shard_{0000..0078}.tar"  \
    --val-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2403/shard_{0000..0010}.tar"  \
    --train-num-samples 252800 \
    --val-num-samples 35200 \
    --dataset-type icddataset \
    --workers 8 \
    --batch-size 50 \
    --accum-freq 6 \
    --epochs 250 \
    --lr 2.5e-5 \
    --beta1 0.9 \
    --beta2 0.999 \
    --eps 1e-8 \
    --wd 0.1 \
    --grad-clip-norm 1.0 \
    --warmup 10000 \
    --lr-scheduler cosine \
    --lr-cooldown-end 5e-6 \
    --precision amp \
    --save-frequency 10 \
    --val-frequency 10 \
    --zeroshot-frequency 0 \
    --loss-function lm \
    --report-to wandb \
    --code-column phecode \
    --wandb-project-name="open-clip-phe-test-runs" \
    --encounter-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/all_encounters_2308_with_phecodes_with_na.parquet.check" \
    --demographic-file=/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/demographics_2404.parquet \
    --labs-folder=/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/labs \
    --time-difference-normalize 1 \
    --number-of-instructions 1 \
    --k-shot 1 1 1 1 1 2 2 2 3 4 \
    --k-shot-demographics 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 2 2 2 3 4 \
    --k-shot-labs 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 2 2 2 3 4 \
    --max_seq_length 128 \
    --pad_id 1 \
    --add-img-token \
    --distance-threshold 60 \
    --negatives-type random \
    --tasks demographics diagnosis labs \
    --task-shuffle \
    --training-eval-codes CV_424.4 EM_249 GU_627.2 NS_324.11 CV_416.214 \
    --fixed-position-range \
    --future-only \
    --name ecg_moca_scratch_q_former_demographics_diagnosis_labs_random_fixed_future_250_trial_4 \
    --model ecg_moca_biogpt_scratch_q_former \
    --seed 0



# 11. Scratch - ? - Labs - Demographics - Diagnosis - ? - Fixed - Future - Z Loss
# TODO: Choose negatives type, architecture, LR, Loss Function based on previous run
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6'
torchrun \
    --nnodes=1 --nproc_per_node=7 --master_addr=localhost --master_port=2120 \
    -m main \
    --train-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_train_2403/shard_{0000..0078}.tar"  \
    --val-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2403/shard_{0000..0010}.tar"  \
    --train-num-samples 252800 \
    --val-num-samples 35200 \
    --dataset-type icddataset \
    --workers 8 \
    --batch-size  \
    --accum-freq  \
    --epochs 250 \
    --lr  \
    --beta1 0.9 \
    --beta2 0.999 \
    --eps 1e-8 \
    --wd 0.1 \
    --grad-clip-norm 1.0 \
    --warmup 10000 \
    --lr-scheduler cosine \
    --lr-cooldown-end 5e-6 \
    --precision amp \
    --save-frequency 10 \
    --val-frequency 10 \
    --zeroshot-frequency 0 \
    --loss-function lm_z \
    --report-to wandb \
    --code-column phecode \
    --wandb-project-name="open-clip-phe-test-runs" \
    --encounter-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/all_encounters_2308_with_phecodes_with_na.parquet.check" \
    --demographic-file=/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/demographics_2404.parquet \
    --labs-folder=/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/labs \
    --time-difference-normalize 1 \
    --number-of-instructions 1 \
    --k-shot 1 1 1 1 1 2 2 2 3 4 \
    --k-shot-demographics 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 2 2 2 3 4 \
    --k-shot-labs 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 2 2 2 3 4 \
    --max_seq_length 128 \
    --pad_id 1 \
    --add-img-token \
    --distance-threshold 60 \
    --negatives-type \
    --tasks demographics diagnosis labs \
    --task-shuffle \
    --training-eval-codes CV_424.4 EM_249 GU_627.2 NS_324.11 CV_416.214 \
    --fixed-position-range \
    --future-only \
    --name  \
    --model  \
    --seed 0