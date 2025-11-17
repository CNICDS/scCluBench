import json
import pandas as pd

datasets = [
    "Tabula_Muris_brain_filtered", "Tabula_Muris_kidney_filtered", "Tabula_Muris_limb_muscle_filtered", "Tabula_Muris_liver_filtered", "Tabula_Muris_lung_filtered", 
    "Tabula_Sapiens_ear_crista_ampullaris_filtered", "Tabula_Sapiens_ear_utricle_filtered", "Tabula_Sapiens_liver_10percent_filtered", "Tabula_Sapiens_lung_10percent_filtered", "Tabula_Sapiens_testis_filtered", "Tabula_Sapiens_trachea_filtered",
    "Mauro_human_Pancreas_cell", "Sonya_HumanLiver_counts_top5000"
    ]
methods = [
    "DEC","DESC","scDCC","scDeepCluster","scNAME","scziDesk","scMAE",
    "AttentionAE-sc", "scDSC", "scGAE", "scGNN",
    "scCDCG", 
]

cols = []
for method in methods:
    cols.append(method)
    cols.append(method + '+DEG')
df = pd.DataFrame(columns=cols, index=datasets)


for dataset in datasets:
    for method in methods:
        try:
            with open(f'sankey/{dataset}_{method}_link.json', 'r') as f:
                links = json.load(f)
        except FileNotFoundError:
            print(f"File not found for {dataset}-{method}. Skipping...")
            continue
        cluster_total = 0
        cluster_correct = 0
        annotation_total = 0
        annotation_correct = 0
        for i, link in enumerate(links):
            source = link['source']
            target = link['target']
            if source.endswith('_gold1') and target.endswith('_cluster'):
                cluster_total += link['value']
                if source.replace('_gold1', '') == target.replace('_cluster', ''):
                    cluster_correct += link['value']
            elif source.endswith('_annotation') and target.endswith('_gold2'):
                annotation_total += link['value']
                if source.replace('_annotation', '') == target.replace('_gold2', ''):
                    annotation_correct += link['value']
        cluster_acc = cluster_correct / cluster_total if cluster_total > 0 else 0
        annotation_acc = annotation_correct / annotation_total if annotation_total > 0 else 0
        df.loc[dataset, method] = round(cluster_acc * 100, 2)
        df.loc[dataset, method + '+DEG'] = round(annotation_acc * 100, 2)
    mean_clu, mean_anno = df.loc[dataset, methods].mean(), df.loc[dataset, [m + '+DEG' for m in methods]].mean()
    df.loc[dataset, 'mean'] = round(mean_clu, 2)
    df.loc[dataset, 'mean+DEG'] = round(mean_anno, 2)
mean = df.mean()
for m in mean.index:
    mean[m] = round(mean[m], 2)
df.loc['mean'] = mean
gain = []
for method in methods+['mean']:
    gain.append(0)
    gain.append(round(df.loc['mean', method + '+DEG'] - df.loc['mean', method], 2))
print(gain)
df.loc['gain'] = gain
df.to_csv('deg_acc.csv')
print(df)
