python plot_mae_reconstructions.py \
--dataset-type mae \
--model ecg_mae_biogpt_scratch \
--pretrained /mnt/obi0/pk621/projects/med_instruct/vision/open_clip/src/logs/ecg_mae_scratch_trial_8/checkpoints/epoch_500.pt \
--val-data "/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mae_pre_val_24_07_16/shard_{0000..0002}.tar" \
--val-num-samples 38400 \
--workers 8 \
--num-samples 5


python plot_mae_reconstructions.py \
--dataset-type mae \
--model ecg_mae_biogpt_scratch \
--pretrained /mnt/obi0/pk621/projects/med_instruct/vision/open_clip/src/logs/ecg_mae_scratch_trial_8/checkpoints/epoch_500.pt \
--val-data "/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mae_pre_train_24_07_16/shard_{0000..0002}.tar" \
--val-num-samples 38400 \
--workers 8 \
--num-samples 10