import numpy as np
from preprocess import *
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import torch
from scipy.stats import norm

from argparse import ArgumentParser
parser = ArgumentParser(description='Process some integers.')
parser.add_argument('--dataset', type=str, default='Tabula_Muris_limb_muscle_filtered', help='Dataset name')

args = parser.parse_args()

dataset = args.dataset
baselines = [
    "DEC","DESC","scDCC","scDeepCluster","scNAME","scziDesk","scMAE",
    "AttentionAE-sc", "scDSC", 
    "scGAE", "scGNN",
    "scCDCG", 
    # "leiden",
]

def get_best_seed(dataset, method):
    if method == "AttentionAE-sc":
        if dataset == "Tabula_Muris_brain_filtered":
            return 2050
        elif dataset == "Tabula_Muris_kidney_filtered":
            return 3047
        elif dataset == "Tabula_Muris_limb_muscle_filtered":
            return 3041
        elif dataset == "Tabula_Muris_liver_filtered":
            return 2021
        elif dataset == "Tabula_Muris_lung_filtered":
            return 2022
        elif dataset == "Tabula_Sapiens_ear_crista_ampullaris_filtered":
            return 3041
        elif dataset == "Tabula_Sapiens_ear_utricle_filtered":
            return 2022
        elif dataset == "Tabula_Sapiens_liver_10percent_filtered":
            return 3041
        elif dataset == "Tabula_Sapiens_lung_10percent_filtered":
            return 3041
        elif dataset == "Tabula_Sapiens_testis_filtered":
            return 3047
        elif dataset == "Tabula_Sapiens_trachea_filtered":
            return 3041
    if method == "scDSC":
        if dataset == "Tabula_Muris_brain_filtered":
            return 2050
        elif dataset == "Tabula_Muris_kidney_filtered":
            return 2050
        elif dataset == "Tabula_Muris_limb_muscle_filtered":
            return 2022
        elif dataset == "Tabula_Muris_liver_filtered":
            return 2022
        elif dataset == "Tabula_Muris_lung_filtered":
            return 3047
        elif dataset == "Tabula_Sapiens_ear_crista_ampullaris_filtered":
            return 2050
        elif dataset == "Tabula_Sapiens_ear_utricle_filtered":
            return 2022
        elif dataset == "Tabula_Sapiens_liver_10percent_filtered":
            return 2022
        elif dataset == "Tabula_Sapiens_lung_10percent_filtered":
            return 2022
        elif dataset == "Tabula_Sapiens_testis_filtered":
            return 2050
        elif dataset == "Tabula_Sapiens_trachea_filtered":
            return 2022
    if method == "scGNN":
        if dataset == "Tabula_Muris_brain_filtered":
            return 2050
        elif dataset == "Tabula_Muris_kidney_filtered":
            return 2050
        elif dataset == "Tabula_Muris_limb_muscle_filtered":
            return 3047
        elif dataset == "Tabula_Muris_liver_filtered":
            return 3047
        elif dataset == "Tabula_Muris_lung_filtered":
            return 3047
        elif dataset == "Tabula_Sapiens_ear_crista_ampullaris_filtered":
            return 3047
        elif dataset == "Tabula_Sapiens_ear_utricle_filtered":
            return 2021
        elif dataset == "Tabula_Sapiens_liver_10percent_filtered":
            return 3047
        elif dataset == "Tabula_Sapiens_lung_10percent_filtered":
            return 3047
        elif dataset == "Tabula_Sapiens_testis_filtered":
            return 2050
        elif dataset == "Tabula_Sapiens_trachea_filtered":
            return 3047
    if method == "scGAE":
        if dataset == "Tabula_Muris_brain_filtered":
            return 2021
        elif dataset == "Tabula_Muris_kidney_filtered":
            return 2050
        elif dataset == "Tabula_Muris_limb_muscle_filtered":
            return 2050
        elif dataset == "Tabula_Muris_liver_filtered":
            return 3047
        elif dataset == "Tabula_Muris_lung_filtered":
            return 3047
        elif dataset == "Tabula_Sapiens_ear_crista_ampullaris_filtered":
            return 2050
        elif dataset == "Tabula_Sapiens_ear_utricle_filtered":
            return 2022
        elif dataset == "Tabula_Sapiens_liver_10percent_filtered":
            return 2022
        elif dataset == "Tabula_Sapiens_lung_10percent_filtered":
            return 2021
        elif dataset == "Tabula_Sapiens_testis_filtered":
            return 2021
        elif dataset == "Tabula_Sapiens_trachea_filtered":
            return 3047
    return 3047

def get_file_name(method, dataset):
    if method == "leiden":
        return f"Summary/R/leiden/{dataset}.h5"
    if dataset in ["Mauro_human_Pancreas_cell", "Sonya_HumanLiver_counts_top5000"]:
        return f"Summary_2/{dataset}/{method}.h5"
    dir_part_1 = "Summary"
    dir_part_2 = "DL" if method in ["DEC","DESC","scDCC","scDeepCluster","scDSSC","scNAME","scziDesk","scMAE"] else "GNN"
    dir_part_3 = method
    dir_part_4 = "embedding" if method in ["AttentionAE-sc", "scDSC", "scGAE", "scGNN"] else ""
    dir = f"{dir_part_1}/{dir_part_2}/{dir_part_3}/{dir_part_4}"
    if method in ["DESC", "scCDCG"]:
        file = f"{dataset}.h5"
    elif method in ["AttentionAE-sc", "scDSC", "scGNN"]:
        seed = get_best_seed(dataset, method)
        file = f"{dataset}/{seed}/embedding.h5"
    elif method in ["scGAE"]:
        seed = get_best_seed(dataset, method)
        file = f"{dataset}/{seed}/embedding_0.h5"
    else:
        file = f"{method}_{dataset}.h5"
    return os.path.join(dir, file)

# %% 
# Ploting embedding bar chart and embedding heatmap
print("Plotting embedding bar chart and embedding heatmap")
hist_normalized_all = []
for baseline in baselines:
    file_path = get_file_name(baseline, dataset)
    try:
        with h5py.File(file_path, 'r') as file:
            print(f"found results in {file_path}")
            Y = file['Y'][()]
            X = file['X'][()]
            file.close()
    except Exception as e:
        print(f"Error: {e}")
        continue
    # print(X.shape)
    # continue
    # print(Y.shape)
    cat = np.concatenate([X, Y.reshape(-1, 1)], axis=1)
    arg_sort = np.argsort(Y)
    cat = cat[arg_sort]
    embedding_sample = cat[:, :-1]

    # cosine similarity
    # cosine_sim = cosine_similarity(embedding_sample, embedding_sample)

    # embedding_sample = torch.tensor(embedding_sample).cuda()
    # cosine_sim = torch.nn.functional.cosine_similarity(embedding_sample.unsqueeze(1), embedding_sample.unsqueeze(0), dim=2).cpu().numpy()

    # Move embeddings to GPU (assuming `embedding_sample` is a numpy array)
    embeddings = torch.tensor(embedding_sample).cuda()  
    n = embeddings.shape[0]
    batch_size = 128  # Adjust based on GPU memory
    cosine_sim = torch.zeros((n, n), device='cuda')

    # Compute in batches
    for i in range(0, n, batch_size):
        for j in range(0, n, batch_size):
            batch_i = embeddings[i:i+batch_size].unsqueeze(1)  # [bs, 1, dim]
            batch_j = embeddings[j:j+batch_size].unsqueeze(0)  # [1, bs, dim]
            cosine_sim[i:i+batch_size, j:j+batch_size] = torch.nn.functional.cosine_similarity(batch_i, batch_j, dim=2)

    cosine_sim = cosine_sim.cpu().numpy()  # Move back to CPU if needed

    # Plot the heatmap
    # plt.figure(figsize=(10, 10))
    plt.imshow(cosine_sim, cmap="RdBu_r", interpolation='nearest', vmin=-1, vmax=1)
    plt.colorbar()

    # Hide axis
    plt.gca().set_axis_off()
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(f'heat/{dataset}_{baseline}.pdf', transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

    # Plot the barplot
    cosine_sim_flat = cosine_sim[np.triu_indices_from(cosine_sim, k=1)]
    bins = np.linspace(0, 1, 21)  # 等分为20个区间
    bin_centers = (bins[:-1] + bins[1:]) / 2  # 计算每个区间的中心位置
    hist, bin_edges = np.histogram(cosine_sim_flat, bins=bins, density=False)
    hist_normalized = hist / hist.sum()  # Sum of probabilities = 1
    mu = np.mean(cosine_sim_flat)
    sigma = np.std(cosine_sim_flat)
    # Calculate probability for each bin using the CDF
    prob_pdf = norm.cdf(bin_edges[1:], mu, sigma) - norm.cdf(bin_edges[:-1], mu, sigma)
    # Ensure the normal distribution sums to 1 over your bins (optional)
    prob_pdf_normalized = prob_pdf / prob_pdf.sum()
    custom_colors = [(0, "#005F73"), (0.5, "white"), (1, "#BB3E03")]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', custom_colors)
    colors = custom_cmap(np.linspace(0, 1, len(hist_normalized)))

    # 绘制柱状图（横纵坐标调换）
    plt.figure(figsize=(5, 5))
    plt.bar(bin_centers, hist_normalized, width=np.diff(bins), color=colors, edgecolor='black', align='center')
    # plt.plot(bin_centers, prob_pdf_normalized, '--', linewidth=2, label="Normal Distribution", color='lightgreen')
    # plt.xticks(fontsize=12, color='blue', rotation=45)  # 横轴刻度字体和倾斜角度
    plt.xticks(fontsize=12, color='green')  # 横轴刻度字体和倾斜角度
    plt.yticks(fontsize=12, color='blue')             # 纵轴刻度字体大小和颜色
    plt.xlabel('Cosine Similarity', fontsize=12)       # 原来的纵轴改为横轴
    plt.ylabel('Probability', fontsize=12)                   # 原来的横轴改为纵轴
    plt.ylim(0, max(hist_normalized) * 1.1)  # 纵轴从0开始
    plt.xlim(0, 1)                # 横轴从0到1
    plt.savefig(f"bar_new/{dataset}_{baseline}.pdf", transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

    hist_normalized_all.append(hist_normalized)
# save the histogram data
hist_normalized_all = np.array(hist_normalized_all)
np.save(f"bar_new/_{dataset}_histogram_normalized.npy", hist_normalized_all)


# exit(0)

# %% 
# Plotting embedding visualization
print("Plotting embedding visualization")
tsne_all = []
y_all = []
for baseline in baselines:
    if dataset == "Tabula_Muris_brain_filtered" and baseline == "scziDesk":
        tsne_all.append(np.zeros((13417, 2)))
        y_all.append(np.zeros((13417,)))
        continue
    # if dataset == "Tabula_Sapiens_trachea_filtered" and baseline == "AttentionAE-sc":
    #     tsne_all.append(np.zeros((22592, 2)))
    #     y_all.append(np.zeros((22592,)))
    #     continue
    file_path = get_file_name(baseline, dataset)
    try:
        with h5py.File(file_path, 'r') as file:
            print(f"found results in {file_path}")
            Y = file['Y'][()]
            X = file['X'][()]
            file.close()
    except Exception as e:
        print(f"Error: {e}")
        continue
    # Plot the scatter plot
    tsne = TSNE(n_components=2, random_state=42)
    embedded_data = tsne.fit_transform(X)
    classes = np.unique(Y)
    color_list =['#54c4c2','#0d8a8c','#70c17f','#4583b3','#f78e26','#f172ad','#f7aab9','#c63596','#be86ba','#8b66b8','#4068b2','#512a93','#223271','#060606','#f56c00','#b03d26']
    point_colors = [color_list[i % len(color_list)] for i in range(len(classes))]
    fig, ax = plt.subplots(figsize=(5, 5))
    for i, c in zip(range(len(classes)), point_colors):
        b = Y == classes[i]
        ax.scatter(embedded_data[b, 0], embedded_data[b, 1], color=c, label=str(classes[i]), s=1.6)
    ax.set_axis_off()
    plt.savefig(f"vis_new/{dataset}_{baseline}.pdf", transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

    print(embedded_data.shape, Y.shape)
    if dataset == "Sonya_HumanLiver_counts_top5000" and baseline == "scGNN":
        # Sonya_HumanLiver_counts_top5000 scGNN missing last point, so we add it manually by repeating the last point
        print("Sonya_HumanLiver_counts_top5000 scGNN missing last point, so we add it manually by repeating the last point")
        embedded_data = np.append(embedded_data, [embedded_data[-1]], axis=0)
        Y = np.append(Y, Y[-1])
        print("After adding last point, embedded_data.shape:", embedded_data.shape, "Y.shape:", Y.shape)

    tsne_all.append(embedded_data)
    y_all.append(Y)
# save the tsne data
# check if all tsne_all have the same shape
if len(set([x.shape[0] for x in tsne_all])) != 1 or len(set([y.shape[0] for y in y_all])) != 1:
    print("Error: not all tsne_all have the same shape")
    for i, x in enumerate(tsne_all):
        print(f"tsne_all[{i}].shape: {x.shape}, y_all[{i}].shape: {y_all[i].shape}")
    exit(1)
tsne_all = np.array(tsne_all)
np.save(f"vis_new/_{dataset}_tsne.npy", tsne_all)
y_all = np.array(y_all)
np.save(f"vis_new/_{dataset}_y.npy", y_all)
