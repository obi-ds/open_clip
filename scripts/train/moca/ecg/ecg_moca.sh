# 1. Scratch - Diagnosis - Random - Fixed - Future
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
    --name ecg_moca_scratch_diagnosis_k_1_random_fixed_future_250_trial_18 \
    --model ecg_moca_biogpt_scratch \
    --seed 0



# 2. Scratch - Labs - Demographics - Diagnosis - Random - Fixed - Past/Future - Multi
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
torchrun \
    --nnodes=1 --nproc_per_node=8 --master_addr=localhost --master_port=2120 \
    -m main \
    --train-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_train_2403/shard_{0000..0078}.tar"  \
    --val-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2403/shard_{0000..0010}.tar"  \
    --train-num-samples 252800 \
    --val-num-samples 35200 \
    --dataset-type icddataset \
    --workers 8 \
    --batch-size 60 \
    --accum-freq 5 \
    --epochs 300 \
    --lr-scheduler cosine \
    --lr 2.5e-5 \
    --lr-cooldown-end 2.5e-6 \
    --beta1 0.9 \
    --beta2 0.999 \
    --eps 1e-8 \
    --wd 0.1 \
    --grad-clip-norm 1.0 \
    --warmup 12000 \
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
    --number-of-instructions 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 \
    --k-shot 1 1 1 1 1 2 2 2 3 4 \
    --k-shot-demographics 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 2 2 2 3 4 \
    --k-shot-labs 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 2 2 2 3 4 \
    --k-shot-ecg-attributes 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 2 2 2 3 4 \
    --max_seq_length 196 \
    --pad_id 1 \
    --add-img-token \
    --negatives-type random \
    --tasks demographics diagnosis ecg_attributes labs \
    --task-shuffle \
    --training-eval-codes CV_424.4 EM_249 GU_627.2 NS_324.11 CV_416.214 \
    --fixed-position-range \
    --name ecg_moca_scratch_demographics_diagnosis_ecg_attributes_labs_random_fixed_multi_300_trial_2 \
    --model ecg_moca_biogpt_scratch \
    --seed 0


# 3. Pre-trained - Diagnosis - Random - Fixed - Future
# TODO: Trial 15  was learning rate of 1e-5, Trial 16 is 5e-6
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
torchrun \
    --nnodes=1 --nproc_per_node=8 --master_addr=localhost --master_port=2120 \
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
    --lr 5e-6 \
    --lr-cooldown-end 5e-7 \
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
    --name ecg_moca_pretrained_diagnosis_k_1_random_fixed_future_250_trial_16 \
    --model ecg_moca_biogpt \
    --seed 0



# 4. Pre-trained - LoRA - Diagnosis - Random - Fixed - Future
# TODO: 2.5e-5 LR seems to be okay, maybe we can try 2e-5 later
# TODO: Trial 9 had 128 query tokens, we should do another trial with 64 query tokens - change model config file
# TODO: If the performance is similar we can stick with 64 - computationally better
# TODO: (I think we can increase batch size back to 100)
# TODO: Train without Q-Former? - Trial 10
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
torchrun \
    --nnodes=1 --nproc_per_node=8 --master_addr=localhost --master_port=2120 \
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
    --name ecg_moca_pretrained_lora_diagnosis_k_1_random_fixed_future_250_trial_12 \
    --model ecg_moca_biogpt_lora \
    --seed 0
















# 3. Scratch - Diagnosis - Random - Fixed - Future - Frozen
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
    --name ecg_moca_pretrained_frozen_diagnosis_k_1_random_fixed_future_250_trial_5 \
    --model ecg_moca_biogpt \
    --lock-text \
    --seed 0



# 4. Scratch - Labs - Demographics - Diagnosis - Random - Fixed - Future - Frozen
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
    --name ecg_moca_pretrained_frozen_demographics_diagnosis_labs_random_fixed_future_250_trial_1 \
    --model ecg_moca_biogpt \
    --lock-text \
    --seed 0
