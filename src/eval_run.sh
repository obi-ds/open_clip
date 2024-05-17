# Specify GPT in model type if using GPT based model - and remove it if not using - also specify ecg or cyto in model type

python eval.py \
--gpu 0 \
--batch-size 256 \
--model-type ecg_scatter_global_biogpt2 \
--model-folder /home/mhomilius/projects/bloodcell_clip/vision/open_clip/scripts/logs/2024_05_15-22_18_50-model_ecg_scatter_global_biogpt2-lr_5e-05-b_100-j_8-p_amp/checkpoints/ \
--phecode-file /mnt/obi0/phi/ehr_projects/bloodcell_clip/data/phecode/phecodeX_info.csv \
--start 0 \
--end 10 \
--code-column phecode \
--eval-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2403/shard_{0000..0010}.tar" \
--num-samples 35200 \
--epoch-start 30 \
--eval-every-epoch 10 \
--file-suffix 24_03_mgh_val \
--output-folder /mnt/obi0/phi/ehr_projects/bloodcell_clip/evaluation/ecg/forward_pass/

#python eval.py \
#--gpu 3 \
#--batch-size 256 \
#--model-type ecg_scatter_global_biogpt2 \
#--model-folder /home/mhomilius/projects/bloodcell_clip/vision/open_clip/scripts/logs/2024_05_15-22_18_50-model_ecg_scatter_global_biogpt2-lr_5e-05-b_100-j_8-p_amp/checkpoints/ \
#--eval-every-epoch 10 \
#--start 0 \
#--end 10 \
#--epoch-start 50
#
#
#python eval.py \
#--gpu 1 \
#--batch-size 256 \
#--model-type ecg_cnn_windowed_biogpt5 \
#--model-folder /home/mhomilius/projects/bloodcell_clip/vision/open_clip/scripts/logs/ecg_labs_diagnosis_demographic_random_cnn_windowed_biogpt5_frozen_future_250_trial_1/checkpoints/ \
#--eval-every-epoch 10 \
#--start 0 \
#--end 10 \
#--epoch-start 50
#
#python eval.py \
#--gpu 2 \
#--batch-size 256 \
#--model-type ecg_cnn_windowed_biogpt5 \
#--model-folder /home/mhomilius/projects/bloodcell_clip/vision/open_clip/scripts/logs/ecg_labs_diagnosis_demographic_random_cnn_windowed_biogpt5_frozen_future_250_trial_1/checkpoints/ \
#--eval-every-epoch 10 \
#--start 0 \
#--end 10 \
#--epoch-start 50
#
#python eval.py \
#--gpu 4 \
#--batch-size 256 \
#--model-type ecg_cnn_windowed_biogpt5 \
#--model-folder /home/mhomilius/projects/bloodcell_clip/vision/open_clip/scripts/logs/ecg_labs_diagnosis_demographic_random_cnn_windowed_biogpt5_frozen_future_250_trial_1/checkpoints/ \
#--eval-every-epoch 10 \
#--start 0 \
#--end 10 \
#--epoch-start 50
#
#python eval.py \
#--gpu 0 \
#--batch-size 350 \
#--model-type ecg_vision_biogpt2 \
#--model-folder /home/mhomilius/projects/bloodcell_clip/vision/open_clip/scripts/logs/ecg-diagnosis_demographic-ecg_vision_biogpt2-frozen_future_250_trial_1/checkpoints/ \
#--eval-every-epoch 10 \
#--start 0 \
#--end 10 \
#--epoch-start 50
