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




python generate.py \
--gpu 0 \
--batch-size 2048 \
--model-type ecg_cnn_windowed_biogpt5 \
--model-folder /home/mhomilius/projects/bloodcell_clip/vision/open_clip/scripts/logs/ecg_labs_diagnosis_demographic_random_cnn_windowed_biogpt5_frozen_future_250_trial_1/checkpoints/ \
--attribute-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/labs/test_labs_with_name.csv" \
--attribute-name-column PromptName \
--start 0 \
--end 2 \
--code-column phecode \
--eval-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2403/shard_{0000..0010}.tar" \
--num-samples 35200 \
--epoch-start 150 \
--eval-every-epoch 150 \
--file-suffix 24_03_mgh_val \
--output-folder /mnt/obi0/phi/ehr_projects/bloodcell_clip/evaluation/ecg/model_generations/

parallel -j 7 --progress --eta --delay 1 "
python generate.py \
--gpu {1} \
--start {2} \
--end {3} \
--batch-size 2048 \
--model-type ecg_cnn_windowed_biogpt5 \
--model-folder /home/mhomilius/projects/bloodcell_clip/vision/open_clip/scripts/logs/ecg_labs_diagnosis_demographic_random_cnn_windowed_biogpt5_frozen_future_250_trial_1/checkpoints/ \
--attribute-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/labs/test_labs_with_name.csv" \
--attribute-name-column PromptName \
--code-column phecode \
--eval-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2403/shard_{0000..0010}.tar" \
--num-samples 35200 \
--epoch-start 150 \
--eval-every-epoch 150 \
--file-suffix 24_03_mgh_val \
--output-folder /mnt/obi0/phi/ehr_projects/bloodcell_clip/evaluation/ecg/model_generations/" ::: $(seq 0 6) :::+ $(seq 0 10 60) :::+ $(seq 10 10 70)



python eval.py \
--gpu {1} \
--start {2} \
--end {3} \
--batch-size 256 \
--model-type ecg_cnn_windowed_biogpt5 \
--model-folder /home/mhomilius/projects/bloodcell_clip/vision/open_clip/scripts/logs/ecg_labs_diagnosis_demographic_random_cnn_windowed_biogpt5_frozen_future_250_trial_1/checkpoints/ \
--phecode-file /mnt/obi0/phi/ehr_projects/bloodcell_clip/data/phecode/phecodeX_info_subset_ecg_178.tsv \
--code-column phecode \
--eval-data='/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/bwh/bwh_all_23_10_23/shard_{0000..0033}.tar' \
--num-samples 105600 \
--epoch-start 150 \
--eval-every-epoch 150 \
--file-suffix 23_10_bwh_all_with_demographics \
--result-date-column TestDate_x \
--demographic-prompt-attributes Age Sex \
--output-folder /mnt/obi0/phi/ehr_projects/bloodcell_clip/evaluation/ecg/forward_pass/" ::: $(seq 0 7) :::+ $(seq 0 23 180) :::+ $(seq 23 23 190)
