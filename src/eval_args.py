from glob import glob
from pathlib import Path
def get_args_str(model_type, batch_size):
    if 'gpt' in model_type:
        pad_id = 1
    else:
        pad_id = 0
    return f'--train-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_train_2403/shard_{{0000..0078}}.tar"  \
        --val-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2403/shard_{{0000..0010}}.tar"  \
        --train-num-samples 252800 \
        --val-num-samples 35200 \
        --dataset-type icddataset \
        --name eval \
        --workers 4 \
        --batch-size {batch_size} \
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
        --model {model_type} \
        --report-to wandb \
        --billable-probability 0.0 \
        --top-non-probability 1.0 \
        --code-column phecode \
        --encounter-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/all_encounters_2308_with_phecodes_with_na.parquet.check" \
        --time-difference-normalize 1 \
        --number-of-instructions 1 \
        --k-shot 1 \
        --max_seq_length 64 \
        --pad_id {pad_id} \
        --distance-threshold 60 \
        --negatives-type random \
        --tasks eval \
        --eval-mode \
        --eval-start-time 0 \
        --eval-end-time 180 \
        --seed 0'.replace('"', '')

def get_model_details_for_eval(model_type, model_folder, eval_every_epoch, batch_size):
    model_details = list()
    for file in glob(model_folder + '*pt'):
        file = Path(file)
        epoch = int(file.name.split('_')[1].split('.')[0])
        if epoch % eval_every_epoch == 0:
            model_details.append(
                ['24_03_mgh_val', get_args_str(model_type, batch_size), model_type, str(file)]
            )
    return model_details
