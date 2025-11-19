import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

baselines = [
    "DEC","scDeepCluster","DESC","scziDesk","scDCC","scNAME","scMAE",
    "scGNN","scDSC","AttentionAE-sc",
    "scCDCG"
]
# baselines = [
#     "DEC","DESC","scDCC","scDeepCluster","scNAME","scziDesk","scMAE",
#     "AttentionAE-sc","scDSC","scGAE","scGNN",
#     "scCDCG"
# ]
datasets = [
    "Tabula_Muris_brain_filtered", "Tabula_Muris_kidney_filtered", "Tabula_Muris_limb_muscle_filtered", "Tabula_Muris_liver_filtered", "Tabula_Muris_lung_filtered", 
    "Tabula_Sapiens_ear_crista_ampullaris_filtered", "Tabula_Sapiens_ear_utricle_filtered", "Tabula_Sapiens_liver_10percent_filtered", "Tabula_Sapiens_lung_10percent_filtered", "Tabula_Sapiens_testis_filtered", "Tabula_Sapiens_trachea_filtered",
    "Mauro_human_Pancreas_cell", "Sonya_HumanLiver_counts_top5000"
]
all_data = []

# subplot 11 columns (baselines) and 11 rows (datasets)
fig, axs = plt.subplots(13, 11, figsize=(33, 39))
# fig, axs = plt.subplots(13, 12, figsize=(36, 39))
fig.subplots_adjust(hspace=0.2)
axs = axs.flatten()
# Set the title for each method
for i, method in enumerate(baselines):
    axs[i].set_title(method, fontsize=24)
    # axs[i].set_xlabel("Clusters")
    # axs[i].set_ylabel("Normalized Frequency")

# Plot the overlap for each method
for i, dataset in enumerate(datasets):
    print(dataset)
    for j, baseline in enumerate(baselines):
        # print(i, j)
        ax = axs[i * 11 + j]
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.patch.set_edgecolor('black')
        # ax.patch.set_linewidth(1)
        # if "Tabula_Sapiens_trachea_filtered_AttentionAE-sc" == f"{dataset}_{baseline}":
        #     continue
        # if "Tabula_Muris_brain_filtered_scziDesk" == f"{dataset}_{baseline}":
        #     continue
        # if "meuro_scziDesk_score" == f"{dataset}_{baseline}":
        #     continue
        # if "sonya_AttentionAE-sc_score" == f"{dataset}_{baseline}":
        #     continue
        try:
            df = pd.read_csv(f"scores/{dataset}_{baseline}_score.csv", index_col=0)
            # rearrange columns by the order of their argmax (try to make the highest score of each column on the diagonal)
            max_row_positions = df.apply(lambda col: np.argmax(col.values))
            sorted_columns = max_row_positions.sort_values().index.tolist()
            # move columns whose max value is zero (not max is at zero positon) to the end 
            zero_columns = [col for col in sorted_columns if df[col].max() == 0]
            sorted_columns = [col for col in sorted_columns if col not in zero_columns] + list(zero_columns)
            # reorder the dataframe
            df = df[sorted_columns]
            # reassign headers from 0 to number of columns
            df.columns = [str(i) for i in range(len(df.columns))]
            # drop index
            df = df.reset_index(drop=True)
        except Exception as e:
            continue

        # plot the heatmap
        lw = 0 if dataset in ["Sonya_HumanLiver_counts_top5000", "Mauro_human_Pancreas_cell"] and baseline=="scGAE" else 0.5
        # print(lw)
        sns.heatmap(df, ax=ax, annot=False, cmap="YlGnBu", cbar=False, linewidths=lw, yticklabels=False, xticklabels=False)
        
# Set the title for each dataset
for i, dataset in enumerate(datasets):
    axs[i * 11].set_ylabel(dataset.replace("_counts_top5000", "").replace("_filtered", "").replace("_10percent", "").replace("Tabula_", "").replace("_cell", "").replace("_", " ").title(), fontsize=16)

plt.tight_layout()
# Save the figure
fig.savefig("overlap_new/heatmap_all_data.pdf", bbox_inches='tight')
plt.close()

