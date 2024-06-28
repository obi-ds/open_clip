# Specify GPT in model type if using GPT based model - and remove it if not using - also specify ecg or cyto in model type

# ECG Eval

# 1. Model: ecg vision base - MGH Validation
parallel -j 8 --progress --eta --delay 1 "
python eval.py \
--gpu {1} \
--start {2} \
--end {3} \
--batch-size 256 \
--model-type ecg_vision_base \
--model-folder /home/mhomilius/projects/bloodcell_clip/vision/open_clip/scripts/logs/ecg-diagnosis_demographic-ecg_vision_base-future_250_trial_6/checkpoints/ \
--phecode-file /mnt/obi0/phi/ehr_projects/bloodcell_clip/data/phecode/phecodeX_info_subset_ecg_178.tsv \
--code-column phecode \
--eval-data='/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2403/shard_{0000..0010}.tar' \
--num-samples 35200 \
--epoch-start 170 \
--eval-every-epoch 170 \
--file-suffix 24_03_mgh_val \
--output-folder /mnt/obi0/phi/ehr_projects/bloodcell_clip/evaluation/ecg/forward_pass/" ::: $(seq 0 7) :::+ $(seq 0 23 180) :::+ $(seq 23 23 190)



# 2. Model: BioGPT - CNN Windowed - With co-variates dataset - MGH Validation
parallel -j 8 --progress --eta --delay 1 "
python eval.py \
--gpu {1} \
--start {2} \
--end {3} \
--batch-size 256 \
--model-type ecg_cnn_windowed_biogpt5 \
--model-folder /home/mhomilius/projects/bloodcell_clip/vision/open_clip/scripts/logs/ecg_labs_diagnosis_demographic_random_cnn_windowed_biogpt5_frozen_future_250_trial_1/checkpoints/ \
--phecode-file /mnt/obi0/phi/ehr_projects/bloodcell_clip/data/phecode/phecodeX_info_subset_ecg_178.tsv \
--code-column phecode \
--eval-data='/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2403/shard_{0000..0010}.tar' \
--num-samples 35200 \
--epoch-start 150 \
--eval-every-epoch 150 \
--file-suffix 24_03_mgh_val_with_demographics \
--demographic-prompt-attributes Age Sex \
--output-folder /mnt/obi0/phi/ehr_projects/bloodcell_clip/evaluation/ecg/forward_pass/" ::: $(seq 0 7) :::+ $(seq 0 23 180) :::+ $(seq 23 23 190)



# 3. Model: BioGPT - CNN Windowed - With co-variates dataset - BWH Validation
parallel -j 8 --progress --eta --delay 1 "
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



# 4. Model: ECGMoca - Scratch - MGH Validation
parallel -j 7 --progress --eta --delay 1 "
python eval.py \
--gpu {1} \
--start {2} \
--end {3} \
--batch-size 512 \
--model-type ecg_moca_biogpt_scratch \
--model-folder /mnt/obi0/pk621/projects/med_instruct/vision/open_clip/src/logs/ecg_moca_scratch_diagnosis_k_1_random_fixed_future_250_trial_11/checkpoints/ \
--phecode-file /mnt/obi0/phi/ehr_projects/bloodcell_clip/data/phecode/phecodeX_info_subset_ecg_178.tsv \
--code-column phecode \
--eval-data='/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2403/shard_{0000..0010}.tar' \
--num-samples 35200 \
--epoch-start 140 \
--eval-every-epoch 90 \
--file-suffix 24_03_mgh_val \
--output-folder /mnt/obi0/phi/ehr_projects/bloodcell_clip/evaluation/ecg/forward_pass/" ::: $(seq 0 6) :::+ $(seq 0 26 180) :::+ $(seq 26 26 190)



# 5. Model: ECGMoca - Scratch - QFormer - MGH Validation
parallel -j 7 --progress --eta --delay 1 "
python eval.py \
--gpu {1} \
--start {2} \
--end {3} \
--batch-size 512 \
--model-type ecg_moca_biogpt_scratch_q_former \
--model-folder /mnt/obi0/pk621/projects/med_instruct/vision/open_clip/src/logs/ecg_moca_scratch_q_former_diagnosis_k_1_random_fixed_future_250_trial_4/checkpoints/ \
--phecode-file /mnt/obi0/phi/ehr_projects/bloodcell_clip/data/phecode/phecodeX_info_subset_ecg_178.tsv \
--code-column phecode \
--eval-data='/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2403/shard_{0000..0010}.tar' \
--num-samples 35200 \
--epoch-start 200 \
--eval-every-epoch 30 \
--file-suffix 24_03_mgh_val \
--output-folder /mnt/obi0/phi/ehr_projects/bloodcell_clip/evaluation/ecg/forward_pass/" ::: $(seq 0 6) :::+ $(seq 0 26 180) :::+ $(seq 26 26 190)


python eval.py \
--gpu 0 \
--start 0 \
--end 2 \
--batch-size 512 \
--model-type ecg_moca_biogpt_scratch_q_former \
--model-folder /mnt/obi0/pk621/projects/med_instruct/vision/open_clip/src/logs/ecg_moca_scratch_q_former_diagnosis_k_1_random_fixed_future_250_trial_4/checkpoints/ \
--phecode-file /mnt/obi0/phi/ehr_projects/bloodcell_clip/data/phecode/phecodeX_info_subset_ecg_178.tsv \
--code-column phecode \
--eval-data='/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2403/shard_{0000..0010}.tar' \
--num-samples 35200 \
--epoch-start 200 \
--eval-every-epoch 30 \
--file-suffix 24_03_mgh_val \
--output-folder /mnt/obi0/phi/ehr_projects/bloodcell_clip/evaluation/ecg/forward_pass/







# cyto vision base - MGH test set
parallel -j 8 --progress --eta --delay 1 "
python eval.py \
--gpu {1} \
--start {2} \
--end {3} \
--batch-size 1600 \
--model-type coca_cyto_base \
--model-folder /home/mhomilius/projects/bloodcell_clip/vision/open_clip/scripts/logs/cyto-diagnosis_demographic-coca_cyto_base-future_250_trial_6/checkpoints/ \
--phecode-file /mnt/obi0/phi/ehr_projects/bloodcell_clip/data/phecode/phecodeX_info_subset_800.tsv \
--code-column phecode \
--eval-data='/mnt/obi0/phi/ehr_projects/sysmex_datasets/processed_23-06-02/mgh_test_enc180d_v2_sampled/shard_{000..005}.tar' \
--num-samples 76800 \
--epoch-start 210 \
--eval-every-epoch 10 \
--result-date ResultDTS \
--file-suffix 23_06_mgh_test_subset_800 \
--output-folder /mnt/obi0/phi/ehr_projects/bloodcell_clip/evaluation/cyto/forward_pass/ \
" ::: $(seq 0 7) :::+ $(seq 0 105 830) :::+ $(seq 105 105 850)


# python eval.py \
# --gpu 1 \
# --batch-size 512 \
# --model-type coca_cyto_base \
# --model-folder /home/mhomilius/projects/bloodcell_clip/vision/open_clip/scripts/logs/cyto-diagnosis_demographic-coca_cyto_base-future_250_trial_6/checkpoints/ \
# --phecode-file /mnt/obi0/phi/ehr_projects/bloodcell_clip/data/phecode/phecodeX_info_subset_800.tsv \
# --start 0 \
# --end 20 \
# --code-column phecode \
# --eval-data="/mnt/obi0/phi/ehr_projects/sysmex_datasets/processed_23-06-02/mgh_test_enc180d_v2_sampled/shard_{000..005}.tar" \
# --num-samples 76800 \
# --epoch-start 210 \
# --eval-every-epoch 10 \
# --result-date ResultDTS \
# --file-suffix 23_06_mgh_test_subset_800 \
# --output-folder /mnt/obi0/phi/ehr_projects/bloodcell_clip/evaluation/cyto/forward_pass/


# cyto vision base - BWH test set
parallel -j 8 --progress --eta --delay 1 "
python eval.py \
--gpu {1} \
--start {2} \
--end {3} \
--batch-size 1600 \
--model-type coca_cyto_base \
--model-folder /home/mhomilius/projects/bloodcell_clip/vision/open_clip/scripts/logs/cyto-diagnosis_demographic-coca_cyto_base-future_250_trial_6/checkpoints/ \
--phecode-file /mnt/obi0/phi/ehr_projects/bloodcell_clip/data/phecode/phecodeX_info_subset_800.tsv \
--code-column phecode \
--eval-data='/mnt/obi0/phi/ehr_projects/sysmex_datasets/processed_23-06-02/bwh_all_enc180d_v2_sampled/shard_{000..006}.tar' \
--num-samples 89600 \
--epoch-start 210 \
--eval-every-epoch 10 \
--result-date ResultDTS \
--file-suffix 23_06_bwh_all_subset_800 \
--output-folder /mnt/obi0/phi/ehr_projects/bloodcell_clip/evaluation/cyto/forward_pass/ \
" ::: $(seq 0 7) :::+ $(seq 0 105 830) :::+ $(seq 105 105 850)



parallel -j 7 --progress --eta --delay 1 "
python eval.py \
--gpu {1} \
--start {2} \
--end {3} \
--batch-size 1600 \
--model-type coca_cyto_biogpt \
--model-folder /mnt/obi0/pk621/projects/med_instruct/vision/open_clip/src/logs/cyto-labs_diagnosis_demographic-coca_cyto_biogpt-future_250_trial_5/checkpoints/ \
--phecode-file /mnt/obi0/phi/ehr_projects/bloodcell_clip/data/phecode/phecodeX_info_subset_800.tsv \
--code-column phecode \
--eval-data='/mnt/obi0/phi/ehr_projects/sysmex_datasets/processed_23-06-02/bwh_all_enc180d_v2_sampled/shard_{000..006}.tar' \
--num-samples 89600 \
--epoch-start 240 \
--eval-every-epoch 240 \
--result-date ResultDTS \
--file-suffix 23_06_bwh_all_subset_800 \
--output-folder /mnt/obi0/phi/ehr_projects/bloodcell_clip/evaluation/cyto/forward_pass/" ::: $(seq 0 6) :::+ $(seq 0 120 830) :::+ $(seq 105 120 850)


python eval.py \
--gpu 0 \
--start 0 \
--end 1 \
--batch-size 1600 \
--model-type coca_cyto_biogpt \
--model-folder /mnt/obi0/pk621/projects/med_instruct/vision/open_clip/src/logs/cyto-labs_diagnosis_demographic-coca_cyto_biogpt-future_250_trial_5/checkpoints/ \
--phecode-file /mnt/obi0/phi/ehr_projects/bloodcell_clip/data/phecode/phecodeX_info_subset_800.tsv \
--code-column phecode \
--eval-data="/mnt/obi0/phi/ehr_projects/sysmex_datasets/processed_23-06-02/bwh_all_enc180d_v2_sampled/shard_{000..006}.tar" \
--num-samples 89600 \
--epoch-start 240 \
--eval-every-epoch 240 \
--result-date ResultDTS \
--file-suffix 23_06_bwh_all_subset_800 \
--output-folder /mnt/obi0/phi/ehr_projects/bloodcell_clip/evaluation/cyto/forward_pass/


parallel -j 8 --progress --eta --delay 1 "
python eval.py \
--gpu {1} \
--start {2} \
--end {3} \
--batch-size 256 \
--model-type ecg_cnn_windowed_biogpt5 \
--model-folder /home/mhomilius/projects/bloodcell_clip/vision/open_clip/scripts/logs/ecg_labs_diagnosis_demographic_random_cnn_windowed_biogpt5_frozen_future_250_trial_1/checkpoints/ \
--phecode-file /mnt/obi0/phi/ehr_projects/bloodcell_clip/data/phecode/phecodeX_info_subset_ecg_178.tsv \
--code-column phecode \
--eval-data='/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2403/shard_{0000..0010}.tar' \
--num-samples 35200 \
--epoch-start 150 \
--eval-every-epoch 150 \
--file-suffix 24_03_mgh_val_with_demographics \
--demographic-prompt-attributes Age Sex \
--output-folder /mnt/obi0/phi/ehr_projects/bloodcell_clip/evaluation/ecg/forward_pass/" ::: $(seq 0 7) :::+ $(seq 0 23 180) :::+ $(seq 23 23 190)







python eval.py \
--gpu 4 \
--batch-size 1600 \
--model-type coca_cyto_base \
--model-folder /home/mhomilius/projects/bloodcell_clip/vision/open_clip/scripts/logs/cyto-diagnosis_demographic-coca_cyto_base-future_250_trial_6/checkpoints/ \
--phecode-file /mnt/obi0/phi/ehr_projects/bloodcell_clip/data/phecode/phecodeX_info_subset_800.tsv \
--start 400 \
--end 500 \
--code-column phecode \
--eval-data="/mnt/obi0/phi/ehr_projects/sysmex_datasets/processed_23-06-02/mgh_val_enc180d_v2_sampled/shard_{000..001}.tar" \
--num-samples 35200 \
--epoch-start 200 \
--eval-every-epoch 10 \
--result-date ResultDTS \
--file-suffix 23_06_mgh_val_subset_800 \
--overwrite True \
--output-folder /mnt/obi0/phi/ehr_projects/bloodcell_clip/evaluation/cyto/forward_pass/

python eval.py \
--gpu 5 \
--batch-size 1600 \
--model-type coca_cyto_base \
--model-folder /home/mhomilius/projects/bloodcell_clip/vision/open_clip/scripts/logs/cyto-diagnosis_demographic-coca_cyto_base-future_250_trial_6/checkpoints/ \
--phecode-file /mnt/obi0/phi/ehr_projects/bloodcell_clip/data/phecode/phecodeX_info_subset_800.tsv \
--start 500 \
--end 600 \
--code-column phecode \
--eval-data="/mnt/obi0/phi/ehr_projects/sysmex_datasets/processed_23-06-02/mgh_val_enc180d_v2_sampled/shard_{000..001}.tar" \
--num-samples 35200 \
--epoch-start 200 \
--eval-every-epoch 10 \
--result-date ResultDTS \
--file-suffix 23_06_mgh_val_subset_800 \
--overwrite True \
--output-folder /mnt/obi0/phi/ehr_projects/bloodcell_clip/evaluation/cyto/forward_pass/

python eval.py \
--gpu 6 \
--batch-size 1600 \
--model-type coca_cyto_base \
--model-folder /home/mhomilius/projects/bloodcell_clip/vision/open_clip/scripts/logs/cyto-diagnosis_demographic-coca_cyto_base-future_250_trial_6/checkpoints/ \
--phecode-file /mnt/obi0/phi/ehr_projects/bloodcell_clip/data/phecode/phecodeX_info_subset_800.tsv \
--start 600 \
--end 700 \
--code-column phecode \
--eval-data="/mnt/obi0/phi/ehr_projects/sysmex_datasets/processed_23-06-02/mgh_val_enc180d_v2_sampled/shard_{000..001}.tar" \
--num-samples 35200 \
--epoch-start 200 \
--eval-every-epoch 10 \
--result-date ResultDTS \
--file-suffix 23_06_mgh_val_subset_800 \
--overwrite True \
--output-folder /mnt/obi0/phi/ehr_projects/bloodcell_clip/evaluation/cyto/forward_pass/

python eval.py \
--gpu 7 \
--batch-size 1600 \
--model-type coca_cyto_base \
--model-folder /home/mhomilius/projects/bloodcell_clip/vision/open_clip/scripts/logs/cyto-diagnosis_demographic-coca_cyto_base-future_250_trial_6/checkpoints/ \
--phecode-file /mnt/obi0/phi/ehr_projects/bloodcell_clip/data/phecode/phecodeX_info_subset_800.tsv \
--start 700 \
--end 800 \
--code-column phecode \
--eval-data="/mnt/obi0/phi/ehr_projects/sysmex_datasets/processed_23-06-02/mgh_val_enc180d_v2_sampled/shard_{000..001}.tar" \
--num-samples 35200 \
--epoch-start 200 \
--eval-every-epoch 10 \
--result-date ResultDTS \
--file-suffix 23_06_mgh_val_subset_800 \
--overwrite True \
--output-folder /mnt/obi0/phi/ehr_projects/bloodcell_clip/evaluation/cyto/forward_pass/

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



python eval.py \
--gpu 0 \
--start 0 \
--end 13 \
--batch-size 64 \
--model-type ecg_cnn_windowed_biogpt5 \
--model-folder /home/mhomilius/projects/bloodcell_clip/vision/open_clip/scripts/logs/ecg_labs_diagnosis_demographic_random_cnn_windowed_biogpt5_frozen_future_250_trial_1/checkpoints/ \
--phecode-file /mnt/obi0/phi/ehr_projects/bloodcell_clip/data/phecode/phecodeX_info_subset_ecg_26.tsv \
--code-column phecode \
--eval-data='/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/bwh/bwh_all_23_10_23/shard_{0000..0033}.tar' \
--num-samples 105600 \
--epoch-start 150 \
--eval-every-epoch 150 \
--file-suffix 23_10_bwh_all \
--result-date-column TestDate_x \
--demographic-prompt-attributes Age Sex \
--output-folder /mnt/obi0/phi/ehr_projects/bloodcell_clip/evaluation/ecg/forward_pass/