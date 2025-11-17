import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

baselines = ["DEC","DESC","scDCC","scDeepCluster","scNAME","scziDesk","scMAE","AttentionAE-sc","scDSC","scGNN","scCDCG"]
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

all_data = []
for dataset in datasets:
    file = f"bar_new/_{dataset}_histogram_normalized.npy"
    data = np.load(file).reshape(12, 20)
    # drop the number 10 data
    data = np.delete(data, 9, axis=0)
    all_data.append(data)

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
# Plot the histogram for each method
bins = np.linspace(0, 1, 21)  # 等分为20个区间
bin_centers = (bins[:-1] + bins[1:]) / 2  # 计算每个区间的中心位置
for i, dataset in enumerate(datasets):
    for jj, method in enumerate(baselines):
        j = baselines_sorted.index(method)
        ax = axs[i * 11 + j]
        ax.set_xticks([])
        ax.set_yticks([])

        if method in d_b:
            ax.set_facecolor('lightblue')
        elif method in g_b:
            ax.set_facecolor('lightgreen')
        ax.patch.set_alpha(0.5)

        # if "Tabula_Sapiens_trachea_filtered_AttentionAE-sc" == f"{dataset}_{method}":
        #     continue
        if "Tabula_Muris_brain_filtered_scziDesk" == f"{dataset}_{method}":
            continue
        
        hist_normalized = all_data[i][jj]
        custom_colors = [(0, "#005F73"), (0.5, "white"), (1, "#BB3E03")]
        custom_cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', custom_colors)
        colors = custom_cmap(np.linspace(0, 1, len(hist_normalized)))
        ax.bar(bin_centers, hist_normalized, width=np.diff(bins), color=colors, edgecolor='black', align='center')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, hist_normalized.max() * 1.1)
        ax.tick_params(bottom=False, left=False)
        
plt.tight_layout()
# Save the figure
if use_digest:
    fig.savefig("bar_new/bar_digest.pdf", bbox_inches='tight')
else:
    fig.savefig("bar_new/bar_all_data.pdf", bbox_inches='tight')
plt.close()




