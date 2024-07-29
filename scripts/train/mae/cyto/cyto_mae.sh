# DGX-2
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
torchrun \
    --nnodes=1 --nproc_per_node=8 --master_addr=localhost --master_port=2121 \
    -m main_mae \
    --train-data="/mnt/obi0/phi/ehr_projects/sysmex_datasets/processed_23-06-02/mgh_train_enc180d_v2/shard_{000..058}.tar"  \
    --val-data="/mnt/obi0/phi/ehr_projects/sysmex_datasets/processed_23-06-02/mgh_train_enc180d_v2/shard_{000..002}.tar"  \
    --train-num-samples 742400 \
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
    --name cyto_mae_scratch_trial_1 \
    --model cyto_mae_biogpt_scratch \
    --seed 0


# GPU 2
export CUDA_VISIBLE_DEVICES='0,1'
torchrun \
    --nnodes=1 --nproc_per_node=2 --master_addr=localhost --master_port=2121 \
    -m main_mae \
    --train-data="/mnt/obi0/phi/ehr_projects/sysmex_datasets/processed_23-06-02/mgh_train_enc180d_v2/shard_{000..058}.tar"  \
    --val-data="/mnt/obi0/phi/ehr_projects/sysmex_datasets/processed_23-06-02/mgh_train_enc180d_v2/shard_{000..002}.tar"  \
    --train-num-samples 742400 \
    --val-num-samples 38400 \
    --dataset-type mae \
    --workers 8 \
    --batch-size 128 \
    --accum-freq 32 \
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
    --name cyto_mae_scratch_trial_5 \
    --model cyto_mae_biogpt_scratch \
    --seed 0

