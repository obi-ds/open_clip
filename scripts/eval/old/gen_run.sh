# Specify GPT in model type if using GPT based model - and remove it if not using - also specify ecg or cyto in model type

python generate.py \
--gpu 0 \
--val-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2403/shard_{0000..0010}.tar" \
--val-num-samples 35200 \
--dataset-type icddataset \
--batch-size 512 \
--workers 8 \
--precision amp \
--attribute-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/labs/test_labs_with_name.csv" \
--attribute-name-column PromptName \
--start 0 \
--end 2 \
--model ecg_moca_biogpt_scratch \
--pretrained="/mnt/obi0/pk621/projects/med_instruct/vision/open_clip/src/logs/ecg_moca_scratch_diagnosis_k_1_random_fixed_future_250_trial_16/checkpoints/epoch_200.pt" \
--code-column phecode \
--sample-result-date-column TestDate \
--encounter-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/all_encounters_2308_with_phecodes_with_na.parquet.check" \
--demographic-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/demographics_2404.parquet" \
--labs-folder="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/labs" \
--file-suffix 24_03_mgh_val_test_refactor \
--time-difference-normalize 1 \
--number-of-instructions 1 \
--k-shot 1 \
--add-img-token \
--max_seq_length 64 \
--pad_id 1 \
--eval-mode \
--eval-start-time 0 \
--eval-end-time 180 \
--tasks eval \
--output-folder /mnt/obi0/phi/ehr_projects/bloodcell_clip/evaluation/ecg/model_generations/








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
--attribute-file='/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/labs/test_labs_with_name.csv' \
--attribute-name-column PromptName \
--code-column phecode \
--eval-data='/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2403/shard_{0000..0010}.tar' \
--num-samples 35200 \
--epoch-start 150 \
--eval-every-epoch 150 \
--file-suffix 24_03_mgh_val \
--output-folder /mnt/obi0/phi/ehr_projects/bloodcell_clip/evaluation/ecg/model_generations/" ::: $(seq 0 6) :::+ $(seq 0 10 60) :::+ $(seq 10 10 70)


parallel -j 7 --progress --eta --delay 1 "
python lab_eval_labels.py {1} {2} " ::: $(seq 0 10 60) :::+ $(seq 10 10 70)



parallel -j 7 --progress --eta --delay 1 "
python generate.py \
--gpu {1} \
--start {2} \
--end {3} \
--batch-size 2048 \
--model-type ecg_cnn_windowed_biogpt5 \
--model-folder /home/mhomilius/projects/bloodcell_clip/vision/open_clip/scripts/logs/ecg_labs_diagnosis_demographic_random_cnn_windowed_biogpt5_frozen_future_250_trial_1/checkpoints/ \
--attribute-file='/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/labs/test_labs_with_name.csv' \
--attribute-name-column PromptName \
--code-column phecode \
--eval-data='/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2403/shard_{0000..0010}.tar' \
--num-samples 35200 \
--epoch-start 150 \
--eval-every-epoch 150 \
--file-suffix 24_03_mgh_val \
--output-folder /mnt/obi0/phi/ehr_projects/bloodcell_clip/evaluation/ecg/model_generations/" ::: $(seq 0 6) :::+ $(seq 0 1 6) :::+ $(seq 1 1 7)





python generate.py \
--gpu 0 \
--val-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2403/shard_{0000..0010}.tar" \
--val-num-samples 35200 \
--dataset-type icddataset \
--batch-size 512 \
--workers 8 \
--precision amp \
--attribute-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/labs/test_labs_with_name.csv" \
--attribute-name-column PromptName \
--start 0 \
--end 2 \
--model ecg_moca_biogpt_scratch \
--pretrained="/mnt/obi0/pk621/projects/med_instruct/vision/open_clip/src/logs/ecg_moca_scratch_diagnosis_k_1_random_fixed_future_250_trial_16/checkpoints/epoch_200.pt" \
--code-column phecode \
--sample-result-date-column TestDate \
--encounter-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/all_encounters_2308_with_phecodes_with_na.parquet.check" \
--demographic-file="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/demographics_2404.parquet" \
--labs-folder="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/labs" \
--file-suffix 24_03_mgh_val_test_refactor \
--time-difference-normalize 1 \
--number-of-instructions 1 \
--k-shot 1 \
--add-img-token \
--max_seq_length 64 \
--pad_id 1 \
--eval-mode \
--eval-start-time 0 \
--eval-end-time 180 \
--tasks eval \
--output-folder /mnt/obi0/phi/ehr_projects/bloodcell_clip/evaluation/ecg/model_generations/
