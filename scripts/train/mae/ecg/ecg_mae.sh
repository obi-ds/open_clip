export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
torchrun \
    --nnodes=1 --nproc_per_node=8 --master_addr=localhost --master_port=2120 \
    -m main_mae \
    --train-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mae_pre_train_24_07_16/shard_{0000..0065}.tar"  \
    --val-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mae_pre_val_24_07_16/shard_{0000..0002}.tar"  \
    --train-num-samples 832000 \
    --val-num-samples 38400 \
    --dataset-type mae \
    --workers 8 \
    --batch-size 1024 \
    --accum-freq 1 \
    --epochs 500 \
    --lr-scheduler cosine \
    --lr 3.5e-4 \
    --lr-cooldown-end 3.5e-5 \
    --beta1 0.9 \
    --beta2 0.999 \
    --eps 1e-8 \
    --wd 0.1 \
    --grad-clip-norm 1.0 \
    --warmup 15000 \
    --precision amp \
    --save-frequency 10 \
    --val-frequency 10 \
    --loss-function mae \
    --report-to wandb \
    --wandb-project-name="mae-test-runs" \
    --name ecg_mae_scratch_trial_8 \
    --model ecg_mae_biogpt_scratch \
    --seed 0


# DGX-1

export CUDA_VISIBLE_DEVICES='0,3'
torchrun \
    --nnodes=1 --nproc_per_node=2 --master_addr=localhost --master_port=2120 \
    -m main_mae \
    --train-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mae_pre_train_24_07_16/shard_{0000..0065}.tar"  \
    --val-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mae_pre_val_24_07_16/shard_{0000..0002}.tar"  \
    --train-num-samples 832000 \
    --val-num-samples 38400 \
    --dataset-type mae \
    --workers 8 \
    --batch-size 300 \
    --accum-freq 7 \
    --epochs 250 \
    --lr-scheduler cosine \
    --lr 1e-4 \
    --lr-cooldown-end 1e-5 \
    --beta1 0.9 \
    --beta2 0.999 \
    --eps 1e-8 \
    --wd 0.1 \
    --grad-clip-norm 1.0 \
    --warmup 30000 \
    --precision amp \
    --save-frequency 10 \
    --val-frequency 10 \
    --loss-function mae \
    --report-to wandb \
    --wandb-project-name="mae-test-runs" \
    --name ecg_mae_scratch_trial_4 \
    --model ecg_mae_biogpt_scratch \
    --seed 0