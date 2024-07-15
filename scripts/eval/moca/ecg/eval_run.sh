# Specify GPT in model type if using GPT based model - and remove it if not using - also specify ecg or cyto in model type

# 1. Scratch - Diagnosis - Random - Fixed - Future

# 1a - MGH Val - Full
parallel -j 7 --progress --eta --delay 1 "
python eval.py \
--gpu {1} \
--val-data='/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2403/shard_{0000..0010}.tar'  \
--val-num-samples 35200 \
--dataset-type icddataset \
--batch-size 512 \
--workers 8 \
--precision amp \
--phecode-file='/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/phecode/phecodeX_info.csv' \
--start {2} \
--end {3} \
--model ecg_moca_biogpt_scratch \
--pretrained='/mnt/obi0/pk621/projects/med_instruct/vision/open_clip/src/logs/ecg_moca_scratch_diagnosis_k_1_random_fixed_future_250_trial_18/checkpoints/epoch_170.pt' \
--code-column phecode \
--sample-result-date-column TestDate \
--encounter-file='/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/all_encounters_2308_with_phecodes_with_na.parquet.check' \
--demographic-file='/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/demographics_2404.parquet' \
--labs-folder='/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/labs' \
--tasks eval \
--file-suffix 24_03_mgh_val \
--time-difference-normalize 1 \
--number-of-instructions 1 \
--k-shot 1 \
--add-img-token \
--max_seq_length 64 \
--pad_id 1 \
--eval-mode \
--eval-start-time 0 \
--eval-end-time 180 \
--output-folder /mnt/obi0/phi/ehr_projects/bloodcell_clip/evaluation/ecg/forward_pass/" ::: $(seq 0 6) :::+ $(seq 0 516 3097) :::+ $(seq 516 516 3613)





# 2. Scratch - Labs - Demographics - Diagnosis - Random - Fixed - Past/Future - Multi

# 2a - MGH Val - Full
parallel -j 7 --progress --eta --delay 1 "
python eval.py \
--gpu {1} \
--val-data='/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2403/shard_{0000..0010}.tar'  \
--val-num-samples 35200 \
--dataset-type icddataset \
--batch-size 512 \
--workers 8 \
--precision amp \
--phecode-file='/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/phecode/phecodeX_info.csv' \
--start {2} \
--end {3} \
--model ecg_moca_biogpt_scratch \
--pretrained='/mnt/obi0/pk621/projects/med_instruct/vision/open_clip/src/logs/ecg_moca_scratch_demographics_diagnosis_ecg_attributes_labs_random_fixed_multi_300_trial_2/checkpoints/epoch_270.pt' \
--code-column phecode \
--sample-result-date-column TestDate \
--encounter-file='/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/all_encounters_2308_with_phecodes_with_na.parquet.check' \
--demographic-file='/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/demographics_2404.parquet' \
--labs-folder='/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/labs' \
--tasks eval \
--file-suffix 24_03_mgh_val \
--time-difference-normalize 1 \
--number-of-instructions 1 \
--k-shot 1 \
--add-img-token \
--max_seq_length 64 \
--pad_id 1 \
--eval-mode \
--eval-start-time 0 \
--eval-end-time 180 \
--output-folder /mnt/obi0/phi/ehr_projects/bloodcell_clip/evaluation/ecg/forward_pass/" ::: $(seq 0 6) :::+ $(seq 0 516 3097) :::+ $(seq 516 516 3613)


# 2b - MGH Val - Full - With demographics
parallel -j 7 --progress --eta --delay 1 "
python eval.py \
--gpu {1} \
--val-data='/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2403/shard_{0000..0010}.tar'  \
--val-num-samples 35200 \
--dataset-type icddataset \
--batch-size 512 \
--workers 8 \
--precision amp \
--phecode-file='/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/phecode/phecodeX_info.csv' \
--start {2} \
--end {3} \
--model ecg_moca_biogpt_scratch \
--pretrained='/mnt/obi0/pk621/projects/med_instruct/vision/open_clip/src/logs/ecg_moca_scratch_demographics_diagnosis_ecg_attributes_labs_random_fixed_multi_300_trial_2/checkpoints/epoch_270.pt' \
--code-column phecode \
--sample-result-date-column TestDate \
--encounter-file='/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/all_encounters_2308_with_phecodes_with_na.parquet.check' \
--demographic-file='/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/demographics_2404.parquet' \
--labs-folder='/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/labs' \
--tasks demographics_prompt eval \
--demographic-prompt-attributes Age Sex \
--file-suffix 24_03_mgh_val_with_demographics \
--time-difference-normalize 1 \
--number-of-instructions 1 \
--k-shot 1 \
--add-img-token \
--max_seq_length 76 \
--pad_id 1 \
--eval-mode \
--eval-start-time 0 \
--eval-end-time 180 \
--output-folder /mnt/obi0/phi/ehr_projects/bloodcell_clip/evaluation/ecg/forward_pass/" ::: $(seq 0 6) :::+ $(seq 0 516 3097) :::+ $(seq 516 516 3613)







#python eval.py \
#--gpu 0 \
#--val-data='/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2403/shard_{0000..0010}.tar'  \
#--val-num-samples 35200 \
#--dataset-type icddataset \
#--batch-size 512 \
#--workers 8 \
#--precision amp \
#--phecode-file='/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/phecode/phecodeX_info.csv' \
#--start 0 \
#--end 2 \
#--model ecg_moca_biogpt_scratch \
#--pretrained='/mnt/obi0/pk621/projects/med_instruct/vision/open_clip/src/logs/ecg_moca_scratch_demographics_diagnosis_ecg_attributes_labs_random_fixed_multi_300_trial_2/checkpoints/epoch_270.pt' \
#--code-column phecode \
#--sample-result-date-column TestDate \
#--encounter-file='/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/all_encounters_2308_with_phecodes_with_na.parquet.check' \
#--demographic-file='/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/demographics_2404.parquet' \
#--labs-folder='/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/labs' \
#--tasks eval \
#--file-suffix 24_03_mgh_val \
#--time-difference-normalize 1 \
#--number-of-instructions 1 \
#--k-shot 1 \
#--add-img-token \
#--max_seq_length 64 \
#--pad_id 1 \
#--eval-mode \
#--eval-start-time 0 \
#--eval-end-time 180 \
#--output-folder /mnt/obi0/phi/ehr_projects/bloodcell_clip/evaluation/ecg/forward_pass_validate/