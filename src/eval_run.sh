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


python eval.py \
--gpu 0 \
--batch-size 256 \
--model-type ecg_vision_base \
--model-folder /home/mhomilius/projects/bloodcell_clip/vision/open_clip/scripts/logs/ecg-diagnosis_demographic-ecg_vision_base-future_250_trial_6/checkpoints/ \
--phecode-file /mnt/obi0/phi/ehr_projects/bloodcell_clip/data/phecode/phecodeX_info_subset_ecg_178.tsv \
--start 0 \
--end 1 \
--code-column phecode \
--eval-data='/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2403/shard_{0000..0010}.tar' \
--num-samples 35200 \
--epoch-start 170 \
--eval-every-epoch 170 \
--file-suffix 24_03_mgh_val \
--output-folder /mnt/obi0/phi/ehr_projects/bloodcell_clip/evaluation/ecg/forward_pass/

# ecg vision base
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


# biogpt 5 on bwh dataset
# this runs the full dataset
sleep 6h && \
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
--file-suffix 23_10_bwh_all \
--result-date-column TestDate_x \
--output-folder /mnt/obi0/phi/ehr_projects/bloodcell_clip/evaluation/ecg/forward_pass/" ::: $(seq 0 7) :::+ $(seq 0 23 180) :::+ $(seq 23 23 190)

# --phecode-file /mnt/obi0/phi/ehr_projects/bloodcell_clip/data/phecode/phecodeX_info_subset_ecg_178.tsv \
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
--output-folder /mnt/obi0/phi/ehr_projects/bloodcell_clip/evaluation/ecg/forward_pass/

python eval.py \
--gpu 1 \
--start 13 \
--end 27 \
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
--output-folder /mnt/obi0/phi/ehr_projects/bloodcell_clip/evaluation/ecg/forward_pass/

# debug on 1080
export WORLD_SIZE=1
python eval.py \
--gpu 0 \
--start 20 \
--end 27 \
--batch-size 32 \
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
--output-folder /mnt/obi0/phi/ehr_projects/bloodcell_clip/evaluation/ecg/forward_pass/


# ecg biogpt5 24_03_mgh_val on GPU2
#ecg_labs_diagnosis_demographic_random_cnn_windowed_biogpt5_frozen_future_250_trial_1_epoch_150
parallel -j 2 --progress --eta --delay 1 "
python eval.py \
--gpu {1} \
--start {2} \
--end {3} \
--batch-size 64 \
--model-type ecg_cnn_windowed_biogpt5 \
--model-folder /home/mhomilius/projects/bloodcell_clip/vision/open_clip/scripts/logs/ecg_labs_diagnosis_demographic_random_cnn_windowed_biogpt5_frozen_future_250_trial_1/checkpoints/ \
--phecode-file /mnt/obi0/phi/ehr_projects/bloodcell_clip/data/phecode/phecodeX_info_subset_ecg_178.tsv \
--code-column phecode \
--eval-data='/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2403/shard_{0000..0010}.tar' \
--num-samples 35200 \
--epoch-start 150 \
--eval-every-epoch 150 \
--file-suffix 24_03_mgh_val \
--output-folder /mnt/obi0/phi/ehr_projects/bloodcell_clip/evaluation/ecg/forward_pass/" ::: 0 1 :::+ 0 90 :::+ 90 180 


# ecg biogpt5 24_03_mgh_val on GPU3,4,6
parallel -j 4 --progress --eta --delay 1 "
python eval.py \
--gpu {1} \
--start {2} \
--end {3} \
--batch-size 48 \
--model-type ecg_cnn_windowed_biogpt5 \
--model-folder /home/mhomilius/projects/bloodcell_clip/vision/open_clip/scripts/logs/ecg_labs_diagnosis_demographic_random_cnn_windowed_biogpt5_frozen_future_250_trial_1/checkpoints/ \
--phecode-file /mnt/obi0/phi/ehr_projects/bloodcell_clip/data/phecode/phecodeX_info_subset_ecg_178.tsv \
--code-column phecode \
--eval-data='/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2403/shard_{0000..0010}.tar' \
--num-samples 35200 \
--epoch-start 150 \
--eval-every-epoch 150 \
--file-suffix 24_03_mgh_val \
--output-folder /mnt/obi0/phi/ehr_projects/bloodcell_clip/evaluation/ecg/forward_pass/" ::: 0 1 2 3 :::+ 0 45 90 125 :::+ 90 125 150 180 


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






# cyto eval on gpu2
python eval.py \
--gpu 0 \
--batch-size 256 \
--model-type coca_cyto_base \
--model-folder /home/mhomilius/projects/bloodcell_clip/vision/open_clip/scripts/logs/cyto-diagnosis_demographic-coca_cyto_base-future_250_trial_6/checkpoints/ \
--phecode-file /mnt/obi0/phi/ehr_projects/bloodcell_clip/data/phecode/phecodeX_info_subset_800.tsv \
--start 820 \
--end 845 \
--code-column phecode \
--eval-data="/mnt/obi0/phi/ehr_projects/sysmex_datasets/processed_23-06-02/mgh_val_enc180d_v2_sampled/shard_{000..001}.tar" \
--num-samples 35200 \
--epoch-start 210 \
--eval-every-epoch 10 \
--result-date ResultDTS \
--file-suffix 23_06_mgh_val_subset_800 \
--overwrite True \
--output-folder /mnt/obi0/phi/ehr_projects/bloodcell_clip/evaluation/cyto/forward_pass/

python eval.py \
--gpu 1 \
--batch-size 512 \
--model-type coca_cyto_base \
--model-folder /home/mhomilius/projects/bloodcell_clip/vision/open_clip/scripts/logs/cyto-diagnosis_demographic-coca_cyto_base-future_250_trial_6/checkpoints/ \
--phecode-file /mnt/obi0/phi/ehr_projects/bloodcell_clip/data/phecode/phecodeX_info_subset_800.tsv \
--start 820 \
--end 845 \
--code-column phecode \
--eval-data="/mnt/obi0/phi/ehr_projects/sysmex_datasets/processed_23-06-02/mgh_val_enc180d_v2_sampled/shard_{000..001}.tar" \
--num-samples 35200 \
--epoch-start 210 \
--eval-every-epoch 10 \
--result-date ResultDTS \
--file-suffix 23_06_mgh_val_subset_800 \
--overwrite True \
--output-folder /mnt/obi0/phi/ehr_projects/bloodcell_clip/evaluation/cyto/forward_pass/


# cyto eval on DGX-2
python eval.py \
--gpu 0 \
--batch-size 1600 \
--model-type coca_cyto_base \
--model-folder /home/mhomilius/projects/bloodcell_clip/vision/open_clip/scripts/logs/cyto-diagnosis_demographic-coca_cyto_base-future_250_trial_6/checkpoints/ \
--phecode-file /mnt/obi0/phi/ehr_projects/bloodcell_clip/data/phecode/phecodeX_info_subset_800.tsv \
--start 0 \
--end 100 \
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
--gpu 1 \
--batch-size 1600 \
--model-type coca_cyto_base \
--model-folder /home/mhomilius/projects/bloodcell_clip/vision/open_clip/scripts/logs/cyto-diagnosis_demographic-coca_cyto_base-future_250_trial_6/checkpoints/ \
--phecode-file /mnt/obi0/phi/ehr_projects/bloodcell_clip/data/phecode/phecodeX_info_subset_800.tsv \
--start 100 \
--end 200 \
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
--gpu 2 \
--batch-size 1600 \
--model-type coca_cyto_base \
--model-folder /home/mhomilius/projects/bloodcell_clip/vision/open_clip/scripts/logs/cyto-diagnosis_demographic-coca_cyto_base-future_250_trial_6/checkpoints/ \
--phecode-file /mnt/obi0/phi/ehr_projects/bloodcell_clip/data/phecode/phecodeX_info_subset_800.tsv \
--start 200 \
--end 300 \
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
--gpu 3 \
--batch-size 1600 \
--model-type coca_cyto_base \
--model-folder /home/mhomilius/projects/bloodcell_clip/vision/open_clip/scripts/logs/cyto-diagnosis_demographic-coca_cyto_base-future_250_trial_6/checkpoints/ \
--phecode-file /mnt/obi0/phi/ehr_projects/bloodcell_clip/data/phecode/phecodeX_info_subset_800.tsv \
--start 300 \
--end 400 \
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
