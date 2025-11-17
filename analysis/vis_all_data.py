import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

baselines = ["DEC","DESC","scDCC","scDeepCluster","scNAME","scziDesk","scMAE","AttentionAE-sc","scDSC","scGNN","scCDCG"]
# baselines = ["DEC","DESC","scDCC","scDeepCluster","scNAME","scziDesk","scMAE","AttentionAE-sc","scDSC","scGAE","scGNN","scCDCG"]
baselines_sorted = [
    "DEC","scDeepCluster","DESC","scziDesk","scDCC","scNAME","scMAE","scGNN","scDSC","AttentionAE-sc","scCDCG"
]
# baselines_sorted = baselines
d_b = ["DEC","DESC","scDCC","scDeepCluster","scNAME","scziDesk","scMAE"]
g_b = ["AttentionAE-sc","scDSC","scGAE","scGNN","scCDCG"]

use_digest = False
if use_digest:
    datasets = ["Tabula_Muris_kidney_filtered", "Tabula_Muris_limb_muscle_filtered",  
            "Tabula_Sapiens_ear_utricle_filtered", "Tabula_Sapiens_lung_10percent_filtered"]
else:
    datasets = ["Tabula_Muris_brain_filtered", "Tabula_Muris_kidney_filtered", "Tabula_Muris_limb_muscle_filtered", "Tabula_Muris_liver_filtered", "Tabula_Muris_lung_filtered", 
            "Tabula_Sapiens_ear_crista_ampullaris_filtered", "Tabula_Sapiens_ear_utricle_filtered", "Tabula_Sapiens_liver_10percent_filtered", "Tabula_Sapiens_lung_10percent_filtered", "Tabula_Sapiens_testis_filtered", "Tabula_Sapiens_trachea_filtered",
            "Mauro_human_Pancreas_cell", "Sonya_HumanLiver_counts_top5000"]

tsne_all = []
y_all = []
for dataset in datasets:
    tsne_file = f"vis_new/_{dataset}_tsne.npy"
    tsne = np.load(tsne_file)
    tsne = np.delete(tsne, 9, axis=0)
    tsne_all.append(tsne)
    y_file = f"vis_new/_{dataset}_y.npy"
    y = np.load(y_file)
    y = np.delete(y, 9, axis=0)
    y_all.append(y)
    print(f"Loaded {dataset} with shape {tsne.shape} and labels {y.shape}")

# subplot 11 columns (baselines) and 11 rows (datasets)
if use_digest:
    fig, axs = plt.subplots(4, 11, figsize=(33, 12))
else:
    fig, axs = plt.subplots(13, 11, figsize=(33, 39))
fig.subplots_adjust(hspace=0.2)
axs = axs.flatten()
# Set the title for each method
for ii, method in enumerate(baselines):
    i = baselines_sorted.index(method)
    axs[i].set_title(method, fontsize=24)
# Set the title for each dataset
for i, dataset in enumerate(datasets):
    axs[i * 11].set_ylabel(dataset.replace("_counts_top5000", "").replace("_filtered", "").replace("_10percent", "").replace("Tabula_", "").replace("_cell", "").replace("_", " ").title(), fontsize=16)
# Plot the visualization for each method
for i, dataset in enumerate(datasets):
    for jj, method in enumerate(baselines):
        j = baselines_sorted.index(method)
        ax = axs[i * 11 + j]
        ax.set_xticks([])
        ax.set_yticks([])
        # if "Tabula_Sapiens_trachea_filtered_AttentionAE-sc" == f"{dataset}_{method}":
        #     continue
        # if "Tabula_Muris_brain_filtered_scziDesk" == f"{dataset}_{method}":
        #     continue
        # if "meuro_DESC" == f"{dataset}_{method}":
        #     continue
        # if "sonya_DESC" == f"{dataset}_{method}":
        #     continue
        # if "meuro_scDCC" == f"{dataset}_{method}":
        #     continue
        # if "sonya_scDCC" == f"{dataset}_{method}":
        #     continue
        x = np.array(tsne_all[i][jj])
        y = np.array(y_all[i][jj])
        classes = np.unique(y)
        # print(method, len(classes))
        color_list =['#54c4c2','#0d8a8c','#70c17f','#4583b3','#f78e26','#f172ad','#f7aab9','#c63596','#be86ba','#8b66b8','#4068b2','#512a93','#223271','#060606','#f56c00','#b03d26']
        point_colors = [color_list[k % len(color_list)] for k in range(len(classes))]
        for k, c in zip(range(len(classes)), point_colors):
            b = y == classes[k]
            ax.scatter(x[b, 0], x[b, 1], color=c, label=str(classes[k]), s=1.6)
        # ax.set_axis_off()
        # plt.savefig(f"vis_new/{dataset}_{baseline}.pdf", transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)

        if method in d_b:
            # set border color
            for spine in ax.spines.values():
                spine.set_edgecolor('lightblue')
                spine.set_linewidth(4)
        elif method in g_b:
            # set border color
            for spine in ax.spines.values():
                spine.set_edgecolor('lightgreen')
                spine.set_linewidth(4)
                
plt.tight_layout()
# Save the figure
if use_digest:
    fig.savefig("vis_new/vis_digest.png", bbox_inches='tight')
else:
    fig.savefig("vis_new/vis_all_data.png", bbox_inches='tight')    
plt.close()