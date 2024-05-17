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