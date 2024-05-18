python bin_data.py \
--phecode-file /mnt/obi0/phi/ehr_projects/bloodcell_clip/data/phecode/phecodeX_info.csv \
--start 0 \
--end 10 \
--bin-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2403/shard_{0000..0010}.tar" \
--num-samples 35200 \
--output-folder /mnt/obi0/phi/ehr_projects/bloodcell_clip/evaluation/ecg/time_bins \
--file-suffix 24_03_mgh_val \
--dataset-type ecg \
--result-date TestDate \
--code-column phecode


python bin_data.py \
--phecode-file /mnt/obi0/phi/ehr_projects/bloodcell_clip/data/phecode/phecodeX_info.csv \
--start 0 \
--end 10 \
--bin-data="/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/mgh/mgh_val_2403/shard_{0000..0010}.tar" \
--num-samples 35200 \
--output-folder /mnt/obi0/phi/ehr_projects/bloodcell_clip/evaluation/ecg/time_bins \
--file-suffix test \
--dataset-type ecg \
--result-date TestDate \
--code-column phecode


# cyto data binning test
python bin_data.py \
--phecode-file /mnt/obi0/phi/ehr_projects/bloodcell_clip/data/phecode/phecodeX_info_subset_800.tsv \
--start 0 \
--end 10 \
--bin-data="/mnt/obi0/phi/ehr_projects/sysmex_datasets/processed_23-06-02/mgh_train_enc180d_v2/shard_{000..004}.tar" \
--num-samples 64000 \
--output-folder /mnt/obi0/phi/ehr_projects/bloodcell_clip/evaluation/cyto/time_bins \
--file-suffix 26_06_mgh_val \
--dataset-type cyto \
--result-date ResultDTS \
--code-column phecode


# run 90 jobs in parallel, 10 codes each, up to 845 codes in file
parallel -j 90 --progress --eta --delay 1 "/home/mhomilius/mambaforge/envs/med_instruct/bin/python bin_data.py \
--start {1} \
--end {2}  \
--phecode-file /mnt/obi0/phi/ehr_projects/bloodcell_clip/data/phecode/phecodeX_info_subset_800.tsv \
--bin-data='/mnt/obi0/phi/ehr_projects/sysmex_datasets/processed_23-06-02/mgh_val_enc180d_v2_sampled/shard_{000..001}.tar' \
--num-samples 25600 \
--output-folder /mnt/obi0/phi/ehr_projects/bloodcell_clip/evaluation/cyto/time_bins \
--file-suffix 26_06_mgh_val_subset \
--dataset-type cyto \
--result-date ResultDTS \
--code-column phecode" ::: $(seq 0 10 835) :::+ $(seq 10 10 845)
