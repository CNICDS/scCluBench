for dataset in "Tabula_Muris_brain_filtered" "Tabula_Muris_kidney_filtered" "Tabula_Muris_limb_muscle_filtered" "Tabula_Muris_liver_filtered" "Tabula_Muris_lung_filtered" "Tabula_Sapiens_ear_crista_ampullaris_filtered" "Tabula_Sapiens_ear_utricle_filtered" "Tabula_Sapiens_liver_10percent_filtered" "Tabula_Sapiens_lung_10percent_filtered" "Tabula_Sapiens_testis_filtered" "Tabula_Sapiens_trachea_filtered"; do
# for dataset in "Mauro_human_Pancreas_cell" "Sonya_HumanLiver_counts_top5000"; do
# for dataset in "Tabula_Sapiens_trachea_filtered"; do
# for dataset in "Tabula_Muris_kidney_filtered" "Tabula_Muris_limb_muscle_filtered" "Tabula_Sapiens_ear_utricle_filtered"; do
    echo "Running for dataset: $dataset"
    python sankey_calculate_json_new_data.py --dataset "$dataset"
done

# for dataset in "meuro" "sonya"; do
#     echo "Running for dataset: $dataset"
#     python sankey_calculate_json_old_data.py --dataset "$dataset"
# done
