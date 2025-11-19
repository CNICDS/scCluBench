import numpy as np
import pandas as pd
from preprocess import *
import h5py
import anndata
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import scanpy as sc
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib as mpl
from copy import deepcopy
# argparse
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--dataset', type=str, default='Tabula_Muris_limb_muscle_filtered', help='Dataset name')

args = parser.parse_args()

dataset = args.dataset
baselines = [
    "DEC","DESC","scDCC","scDeepCluster","scNAME","scziDesk","scMAE",
    "AttentionAE-sc", "scDSC", "scGAE", "scGNN",
    "scCDCG", 
    # "leiden","scMAE","scCDCG",
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

# %% load gold type
if dataset in ["Mauro_human_Pancreas_cell", "Sonya_HumanLiver_counts_top5000"]:
    gold_file = f"/home/wangzaitian/work/2501/scbench/Summary_2/{dataset}.h5ad"
else: 
    gold_file = f"/home/xuping/scRNA-seq_GraphClustering/our_data/Tabula_dataset/filter/{dataset}.h5ad"
gold = anndata.read_h5ad(gold_file)
# print(gold)
X = gold.to_df()
y = gold.obs['cell_type'].values
y_set = set(y)
y_id = [list(y_set).index(i) for i in y]
y_id = [str(i) for i in y_id]
adata = anndata.AnnData(X=X)
adata.var_names = pd.Categorical(gold.var['feature_name'].values)
# print(adata.var_names)
adata.obs['cell.type'] = pd.Categorical(y)
adata.obs['cell.type_id'] = pd.Categorical(y_id)
# print(adata.obs['cell.type'])
# print(adata.obs['cell.type_id'])
# exit()  
# %% load baseline type
for baseline in baselines:
    file_path = get_file_name(baseline, dataset)
    try:
        with h5py.File(file_path, 'r') as file:
            print(f"found results in {file_path}")
            Y = file['Y'][()]
            # y to str
            Y = [str(i) for i in Y]
            X = file['X'][()]
            file.close()
    except Exception as e:
        print(f"Error: {e}")
        continue
    model_label_categorical = pd.Categorical(Y)
    # print(model_label_categorical)
    # print(adata.obs['cell.type'])
    # print(baseline)
    if baseline == "scGNN" and dataset == "Sonya_HumanLiver_counts_top5000":
        model_label_categorical = np.concatenate((model_label_categorical, [model_label_categorical[-1]]))
    adata.obs[baseline] = model_label_categorical
adata.obs.to_csv(f"obs/{dataset}_obs.csv", index=False)
adata.write_h5ad(f"obs/{dataset}.h5ad")
adata = anndata.read_h5ad(f"obs/{dataset}.h5ad")
baselines = adata.obs.columns[2:] # skip gold cell type and cell type id

# exit()


sc.set_figure_params(dpi=1000, color_map = 'viridis')
sc.settings.verbosity = 2
marker_gene_method = "wilcoxon"

# %% marker genes
print("finding marker genes...")
sc.tl.rank_genes_groups(adata,'cell.type',method=marker_gene_method,n_genes=100)
sc.tl.filter_rank_genes_groups(adata, 
                                min_in_group_fraction=0.5, 
                                max_out_group_fraction=1.1, 
                                min_fold_change=0.5)
ranked_filtered_genes = adata.uns['rank_genes_groups_filtered']['names']
ranked_filtered_genes_df = pd.DataFrame({group: ranked_filtered_genes[group] for group in ranked_filtered_genes.dtype.names})
csv_file_path = f"genes/{dataset}_gold_100genes_filtered.csv"
ranked_filtered_genes_df.to_csv(csv_file_path, index=False)

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['figure.figsize'] = 4,4
mpl.rcParams['axes.grid']=False
figsize = (15, 6)
for baseline in baselines:
    # if os.path.exists(f"genes/{dataset}_{baseline}_100genes.csv"):
    #     print(f"found {baseline} marker genes in {dataset}, skipping...")
    #     continue
    if f"{dataset}_{baseline}" in ["Tabula_Muris_brain_filtered_scziDesk"]:
        print(f"Error: found fake file, skipping for {dataset}_{baseline}")
        continue
    print(f"finding marker genes for {baseline} in {dataset}...")
    try:
        problematic_clusters = []
        while adata.obs[baseline].value_counts().min() == 1:
            top_cluster = adata.obs[baseline].value_counts().idxmax()
            bot_cluster = adata.obs[baseline].value_counts().idxmin()
            adata.obs.loc[adata.obs[baseline] == bot_cluster, baseline] = top_cluster
            adata.obs[baseline] = pd.Categorical(adata.obs[baseline].values.tolist()) # refresh the categories to remove 0-sample clusters
            problematic_clusters.append(bot_cluster)
        if len(problematic_clusters) > 0:
            print(f"Replaced clusters {problematic_clusters} in {baseline} to {top_cluster} due to low (1) cell count.")
        sc.tl.rank_genes_groups(adata, baseline, method='wilcoxon', n_genes=100)
        try:
            sc.pl.rank_genes_groups_dotplot(adata, groupby=baseline, n_genes=3, save=f'/{dataset}_{baseline}', dendrogram=True, show=False)
            sc.pl.rank_genes_groups_stacked_violin(adata, groupby=baseline, n_genes=3, save=f'/{dataset}_{baseline}', dendrogram=True, show=False)
            sc.pl.rank_genes_groups_tracksplot(adata, groupby=baseline, n_genes=3, save=f'/{dataset}_{baseline}', dendrogram=True, show=False)
            sc.pl.rank_genes_groups_tracksplot(adata, groupby=baseline, n_genes=10, save=f'_10/{dataset}_{baseline}', dendrogram=True, show=False)
            sc.pl.rank_genes_groups_heatmap(adata, groupby=baseline, n_genes=3, save=f'/{dataset}_{baseline}', show=False)
            sc.pl.rank_genes_groups_matrixplot(adata, groupby=baseline, n_genes=3, save=f'/{dataset}_{baseline}', dendrogram=True, show=False)
        except Exception as e:
            print(f"Error while plotting marker genes for {baseline} in {dataset}: {e}")
    except Exception as e:
        print(f"Error: {e}")
        continue
    ranked_genes = adata.uns['rank_genes_groups']['names']
    ranked_genes_df = pd.DataFrame({group: ranked_genes[group] for group in ranked_genes.dtype.names})
    csv_file_path = f"genes/{dataset}_{baseline}_100genes.csv"
    ranked_genes_df.to_csv(csv_file_path, index=False)
# exit()

# %% plot marker genes
print("calculating overlap...")
gold_deg = pd.read_csv(f"genes/{dataset}_gold_100genes_filtered.csv")
for baseline in baselines:
    try:
        current_deg = pd.read_csv(f"genes/{dataset}_{baseline}_100genes.csv")
    except Exception as e:
        print(f"Error: {e}, creating all-zero overlap matrix for {dataset}_{baseline}")
        # create an all-zero overlap matrix for this baseline
        # overlap_matrix = pd.DataFrame(index=gold_deg.columns, columns=current_deg.columns)
        overlap_matrix = pd.DataFrame(index=gold_deg.index, columns=gold_deg.columns)
        overlap_matrix.fillna(0, inplace=True)
        overlap_matrix.to_csv(f"scores/{dataset}_{baseline}_score.csv")
        continue
    if len(current_deg.columns) < len(gold_deg.columns):
        # add null genes to current_deg to match the number of columns in gold_deg
        for i in range(len(gold_deg.columns) - len(current_deg.columns)):
            current_deg[f"empty_clu_{i}"] = np.nan
    # 创建一个空的数据框用于存放重叠度矩阵
    overlap_matrix = pd.DataFrame(index=gold_deg.columns, columns=current_deg.columns)
    # 计算重叠度
    for gold_col in gold_deg.columns:
        for current_col in current_deg.columns:
            overlap_count = len(set(gold_deg[gold_col]) & set(current_deg[current_col]))
            overlap_matrix.at[gold_col, current_col] = overlap_count / 100

    overlap_matrix.to_csv(f"scores/{dataset}_{baseline}_score.csv")

print("plotting overlap...")
for baseline in baselines:
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
        # df.columns = [str(i) for i in range(len(df.columns))]
    except Exception as e:
        print(f"Error: {e}")
        continue
    plt.figure(figsize=(12, 8))
    font_scale = 10/len(df.columns)
    sns.set_theme(font_scale=font_scale)
    sns.heatmap(df, annot=True, cmap="YlGnBu", cbar=True, linewidths=0.5)

    plt.savefig(f"overlap_new/{dataset}_{baseline}_heatmap.pdf", bbox_inches='tight')
    plt.close()
