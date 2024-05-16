python eval.py \
--gpu 0 \
--batch-size 256 \
--model-type ecg_scatter_global_biogpt2 \
--model-folder /home/mhomilius/projects/bloodcell_clip/vision/open_clip/scripts/logs/2024_05_15-22_18_50-model_ecg_scatter_global_biogpt2-lr_5e-05-b_100-j_8-p_amp/checkpoints/ \
--eval-every-epoch 10 \
--start 0 \
--end 10

python eval.py \
--gpu 3 \
--batch-size 256 \
--model-type ecg_scatter_global_biogpt2 \
--model-folder /home/mhomilius/projects/bloodcell_clip/vision/open_clip/scripts/logs/2024_05_15-22_18_50-model_ecg_scatter_global_biogpt2-lr_5e-05-b_100-j_8-p_amp/checkpoints/ \
--eval-every-epoch 10 \
--start 0 \
--end 10


python eval.py \
--gpu 1 \
--batch-size 256 \
--model-type ecg_cnn_windowed_biogpt5 \
--model-folder /home/mhomilius/projects/bloodcell_clip/vision/open_clip/scripts/logs/ecg_labs_diagnosis_demographic_random_cnn_windowed_biogpt5_frozen_future_250_trial_1/checkpoints/ \
--eval-every-epoch 10 \
--start 0 \
--end 10

python eval.py \
--gpu 2 \
--batch-size 256 \
--model-type ecg_cnn_windowed_biogpt5 \
--model-folder /home/mhomilius/projects/bloodcell_clip/vision/open_clip/scripts/logs/ecg_labs_diagnosis_demographic_random_cnn_windowed_biogpt5_frozen_future_250_trial_1/checkpoints/ \
--eval-every-epoch 10 \
--start 0 \
--end 10

python eval.py \
--gpu 4 \
--batch-size 256 \
--model-type ecg_cnn_windowed_biogpt5 \
--model-folder /home/mhomilius/projects/bloodcell_clip/vision/open_clip/scripts/logs/ecg_labs_diagnosis_demographic_random_cnn_windowed_biogpt5_frozen_future_250_trial_1/checkpoints/ \
--eval-every-epoch 10 \
--start 0 \
--end 10
