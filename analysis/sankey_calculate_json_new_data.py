import numpy as np
import pandas as pd
from preprocess import *
import h5py
import anndata
from scipy.optimize import linear_sum_assignment as linear_assignment
import json
from collections import defaultdict
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--dataset', type=str, default='Tabula_Muris_limb_muscle_filtered', help='Dataset name')

args = parser.parse_args()

dataset = args.dataset

methods = [
    "DEC","DESC","scDCC","scDeepCluster","scNAME","scziDesk","scMAE",
    "AttentionAE-sc", "scDSC", "scGAE", "scGNN",
    "scCDCG", 
    # "AttentionAE-sc",
    # "leiden",

]


for method in methods:
    try:
        obs = pd.read_csv(f'./obs/{dataset}_obs.csv')
        score = pd.read_csv(f'./scores/{dataset}_{method}_score.csv')
        # drop from score where column name starts with 'empty_clu'
        score = score.loc[:, ~score.columns.str.startswith('empty_clu')]
    except FileNotFoundError:
        print(f"File not found for {dataset}_{method}. Skipping...")
        continue

    unique_mappings = obs[['cell.type_id', 'cell.type']].drop_duplicates()
    label_mapping = dict(zip(unique_mappings['cell.type_id'], unique_mappings['cell.type']))

    id2name_annotation = {}
    for col in score.columns[1:]:
        max_celltype_id = score[col].idxmax()
        id2name_annotation[int(col)] = score.iloc[max_celltype_id, 0]

    # print(id2name_annotation)

    def best_map(y_true, y_pred):
        if len(y_true) != len(y_pred):
            print("y_true.shape must == y_pred.shape")
            exit(0)
        label_set = np.unique(y_true)
        num_class = len(label_set)
        G = np.zeros((num_class, num_class))
        for i in range(0, num_class):
            for j in range(0, num_class):
                s = y_true == label_set[i]
                t = y_pred == label_set[j]
                G[i, j] = np.count_nonzero(s & t)
        A = linear_assignment(-G)
        new_y_pred = np.zeros(y_pred.shape)
        for i in range(0, num_class):
            new_y_pred[y_pred == label_set[A[1][i]]] = label_set[A[0][i]]
        return new_y_pred.astype(int), label_set[A[1]], label_set[A[0]]

    y_true = obs['cell.type_id'].values
    y_pred = obs[method].values
    y_pred_bm, _, _ = best_map(y_true, y_pred)

    gold_type = obs['cell.type'].values.tolist()

    # print(id2name_annotation)
    # print(label_mapping)
    # print(y_pred)
    # print(y_pred_bm)
    annotation_type = []
    for i in range(len(y_pred)):
        if y_pred[i] in id2name_annotation:
            annotation_type.append(id2name_annotation[y_pred[i]])
        else: # due to my manipulation when handling 1-sample clusters (0-sample clusters should be ok)
            # set to the most frequent annotation type
            most_frequent_annotation = max(id2name_annotation.values(), key=list(id2name_annotation.values()).count)
            print(f"Annotation type for {y_pred[i]} not found in id2name_annotation. Using most frequent annotation: {most_frequent_annotation}")
            annotation_type.append(most_frequent_annotation)

    cluster_type = []
    for i in range(len(y_pred_bm)):
        cluster_type.append(label_mapping[y_pred_bm[i]])

    # print(gold_type[0:5])
    # print(cluster_type[0:5])
    # print(annotation_type[0:5])

    unique_nodes = list(set(gold_type))
    nodes = []
    for i in range(len(unique_nodes)):
        nodes.append({'name': unique_nodes[i]})

    def generate_sankey_links(source_list, target_list, source_suffix, target_suffix):
        """Generate links between two layers with proper naming convention"""
        counter = defaultdict(int)
        for src, tgt in zip(source_list, target_list):
            counter[(f"{src}_{source_suffix}", f"{tgt}_{target_suffix}")] += 1
        
        return [
            {"source": src, "target": tgt, "value": count}
            for (src, tgt), count in counter.items()
        ]

    # Generate both link layers
    # cluster_to_gold = generate_sankey_links(cluster_type, gold_type, "cluster", "gold")
    # gold_to_annotation = generate_sankey_links(gold_type, annotation_type, "gold", "annotation")
    gold_to_cluster = generate_sankey_links(gold_type, cluster_type, "gold1", "cluster")
    cluster_to_annotation = generate_sankey_links(cluster_type, annotation_type, "cluster", "annotation")
    annotation_to_gold = generate_sankey_links(annotation_type, gold_type, "annotation", "gold2")

    # Combine all links
    # combined_links = cluster_to_gold + gold_to_annotation
    combined_links = gold_to_cluster + cluster_to_annotation + annotation_to_gold

    # Generate unique nodes with proper suffixes
    def get_unique_nodes(list_of_lists):
        nodes = set()
        for lst, suffix in list_of_lists:
            nodes.update({f"{cell}_{suffix}" for cell in lst})
        return [{"name": n} for n in sorted(nodes)]

    nodes = get_unique_nodes([
        # (gold_type, "gold"),
        (cluster_type, "cluster"),
        (gold_type, "gold1"),
        (gold_type, "gold2"),
        (annotation_type, "annotation")
    ])

    # save
    with open(f"./sankey/{dataset}_{method}_node.json", "w") as f:
        json.dump(nodes, f, indent=4)
    with open(f"./sankey/{dataset}_{method}_link.json", "w") as f:
        json.dump(combined_links, f, indent=4)
    print(f"Sankey data for {dataset}-{method} saved successfully.")