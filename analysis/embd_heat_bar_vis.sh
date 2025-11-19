# for dataset in "Tabula_Muris_brain_filtered"; do
# for dataset in "Tabula_Muris_limb_muscle_filtered" "Tabula_Muris_liver_filtered" "Tabula_Muris_lung_filtered" "Tabula_Sapiens_ear_crista_ampullaris_filtered" "Tabula_Sapiens_ear_utricle_filtered" "Tabula_Sapiens_liver_10percent_filtered" "Tabula_Sapiens_lung_10percent_filtered" "Tabula_Sapiens_testis_filtered" "Tabula_Sapiens_trachea_filtered" "Tabula_Muris_brain_filtered"; do
for dataset in "Tabula_Muris_brain_filtered" "Tabula_Muris_kidney_filtered" "Tabula_Muris_limb_muscle_filtered" "Tabula_Muris_liver_filtered" "Tabula_Muris_lung_filtered" "Tabula_Sapiens_ear_crista_ampullaris_filtered" "Tabula_Sapiens_ear_utricle_filtered" "Tabula_Sapiens_liver_10percent_filtered" "Tabula_Sapiens_lung_10percent_filtered" "Tabula_Sapiens_testis_filtered" "Tabula_Sapiens_trachea_filtered"; do
# for dataset in "Mauro_human_Pancreas_cell" "Sonya_HumanLiver_counts_top5000"; do
# for dataset in "Tabula_Muris_kidney_filtered" "Tabula_Muris_limb_muscle_filtered" "Tabula_Sapiens_ear_utricle_filtered"; do
    echo "Running for dataset: $dataset"
    CUDA_VISIBLE_DEVICES="0" python embd_heat_bar_vis_new_data.py --dataset "$dataset"
done

# for dataset in "meuro" "sonya"; do
# for dataset in "sonya"; do
# for dataset in "adam" "amit" "cbmc" "pbmc" "junyue" "klein" "maayanh1" "maayanh2" "maayanh3" "maayanh4" "maayanm1" "maayanm2" "macosko" "roman" "shekhar" "qsdiaph" "qslimb" "qslung" "qstrach" "qxblad" "qxlimb" "qxspleen" "xiaoping" "younghkidney"; do
# for dataset in "meuro" "sonya" "adam" "amit" "cbmc" "pbmc" "junyue" "klein" "maayanh1" "maayanh2" "maayanh3" "maayanh4" "maayanm1" "maayanm2" "macosko" "roman" "shekhar" "qsdiaph" "qslimb" "qslung" "qstrach" "qxblad" "qxlimb" "qxspleen" "xiaoping" "younghkidney"; do
    # echo "Running for dataset: $dataset"
    # CUDA_VISIBLE_DEVICES="5" python embd_heat_bar_vis_old_data.py --dataset "$dataset"
# done
