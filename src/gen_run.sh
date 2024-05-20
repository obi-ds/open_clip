# Specify GPT in model type if using GPT based model - and remove it if not using - also specify ecg or cyto in model type

python generate.py \
--gpu 0 \
--batch-size 2048 \
--model-type ecg_cnn_windowed_biogpt5 \
--model-folder /home/mhomilius/projects/bloodcell_clip/vision/open_clip/scripts/logs/ecg_labs_diagnosis_demographic_random_cnn_windowed_biogpt5_frozen_future_250_trial_1/checkpoints/ \
--category="Age" \
--code-column phecode \
--eval-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2403/shard_{0000..0010}.tar" \
--num-samples 35200 \
--epoch-start 150 \
--eval-every-epoch 150 \
--file-suffix 24_03_mgh_val \
--output-folder /mnt/obi0/phi/ehr_projects/bloodcell_clip/evaluation/ecg/model_generations/

python generate.py \
--gpu 1 \
--batch-size 2048 \
--model-type ecg_cnn_windowed_biogpt5 \
--model-folder /home/mhomilius/projects/bloodcell_clip/vision/open_clip/scripts/logs/ecg_labs_diagnosis_demographic_random_cnn_windowed_biogpt5_frozen_future_250_trial_1/checkpoints/ \
--category="Sex" \
--code-column phecode \
--eval-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2403/shard_{0000..0010}.tar" \
--num-samples 35200 \
--epoch-start 150 \
--eval-every-epoch 150 \
--file-suffix 24_03_mgh_val \
--output-folder /mnt/obi0/phi/ehr_projects/bloodcell_clip/evaluation/ecg/model_generations/



python generate.py \
--gpu 2 \
--batch-size 2048 \
--model-type ecg_cnn_windowed_biogpt5 \
--model-folder /home/mhomilius/projects/bloodcell_clip/vision/open_clip/scripts/logs/ecg_labs_diagnosis_demographic_random_cnn_windowed_biogpt5_frozen_future_250_trial_1/checkpoints/ \
--category="Weight" \
--code-column phecode \
--eval-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2403/shard_{0000..0010}.tar" \
--num-samples 35200 \
--epoch-start 150 \
--eval-every-epoch 150 \
--file-suffix 24_03_mgh_val \
--output-folder /mnt/obi0/phi/ehr_projects/bloodcell_clip/evaluation/ecg/model_generations/


python generate.py \
--gpu 3 \
--batch-size 2048 \
--model-type ecg_cnn_windowed_biogpt5 \
--model-folder /home/mhomilius/projects/bloodcell_clip/vision/open_clip/scripts/logs/ecg_labs_diagnosis_demographic_random_cnn_windowed_biogpt5_frozen_future_250_trial_1/checkpoints/ \
--category="Height" \
--code-column phecode \
--eval-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2403/shard_{0000..0010}.tar" \
--num-samples 35200 \
--epoch-start 150 \
--eval-every-epoch 150 \
--file-suffix 24_03_mgh_val \
--output-folder /mnt/obi0/phi/ehr_projects/bloodcell_clip/evaluation/ecg/model_generations/


python generate.py \
--gpu 4 \
--batch-size 2048 \
--model-type ecg_cnn_windowed_biogpt5 \
--model-folder /home/mhomilius/projects/bloodcell_clip/vision/open_clip/scripts/logs/ecg_labs_diagnosis_demographic_random_cnn_windowed_biogpt5_frozen_future_250_trial_1/checkpoints/ \
--category="QRS Duration" \
--code-column phecode \
--eval-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2403/shard_{0000..0010}.tar" \
--num-samples 35200 \
--epoch-start 150 \
--eval-every-epoch 150 \
--file-suffix 24_03_mgh_val \
--output-folder /mnt/obi0/phi/ehr_projects/bloodcell_clip/evaluation/ecg/model_generations/


python generate.py \
--gpu 5 \
--batch-size 2048 \
--model-type ecg_cnn_windowed_biogpt5 \
--model-folder /home/mhomilius/projects/bloodcell_clip/vision/open_clip/scripts/logs/ecg_labs_diagnosis_demographic_random_cnn_windowed_biogpt5_frozen_future_250_trial_1/checkpoints/ \
--category="QT Interval" \
--code-column phecode \
--eval-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2403/shard_{0000..0010}.tar" \
--num-samples 35200 \
--epoch-start 150 \
--eval-every-epoch 150 \
--file-suffix 24_03_mgh_val \
--output-folder /mnt/obi0/phi/ehr_projects/bloodcell_clip/evaluation/ecg/model_generations/


python generate.py \
--gpu 6 \
--batch-size 2048 \
--model-type ecg_cnn_windowed_biogpt5 \
--model-folder /home/mhomilius/projects/bloodcell_clip/vision/open_clip/scripts/logs/ecg_labs_diagnosis_demographic_random_cnn_windowed_biogpt5_frozen_future_250_trial_1/checkpoints/ \
--category="Ventricular Rate" \
--code-column phecode \
--eval-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2403/shard_{0000..0010}.tar" \
--num-samples 35200 \
--epoch-start 150 \
--eval-every-epoch 150 \
--file-suffix 24_03_mgh_val \
--output-folder /mnt/obi0/phi/ehr_projects/bloodcell_clip/evaluation/ecg/model_generations/


python generate.py \
--gpu 7 \
--batch-size 2048 \
--model-type ecg_cnn_windowed_biogpt5 \
--model-folder /home/mhomilius/projects/bloodcell_clip/vision/open_clip/scripts/logs/ecg_labs_diagnosis_demographic_random_cnn_windowed_biogpt5_frozen_future_250_trial_1/checkpoints/ \
--category="T Axis" \
--code-column phecode \
--eval-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2403/shard_{0000..0010}.tar" \
--num-samples 35200 \
--epoch-start 150 \
--eval-every-epoch 150 \
--file-suffix 24_03_mgh_val \
--output-folder /mnt/obi0/phi/ehr_projects/bloodcell_clip/evaluation/ecg/model_generations/
