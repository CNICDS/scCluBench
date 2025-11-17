from snapshot_selenium import snapshot
from pyecharts import options as opts
from pyecharts.charts import Sankey
from pyecharts.render import make_snapshot
import json

# datasets = ["Mauro_human_Pancreas_cell", "Sonya_HumanLiver_counts_top5000"]
datasets = ["Tabula_Muris_brain_filtered", "Tabula_Muris_kidney_filtered", "Tabula_Muris_limb_muscle_filtered", "Tabula_Muris_liver_filtered", "Tabula_Muris_lung_filtered", "Tabula_Sapiens_ear_crista_ampullaris_filtered", "Tabula_Sapiens_ear_utricle_filtered", "Tabula_Sapiens_liver_10percent_filtered", "Tabula_Sapiens_lung_10percent_filtered", "Tabula_Sapiens_testis_filtered", "Tabula_Sapiens_trachea_filtered"]
# datasets = ["Tabula_Muris_kidney_filtered", "Tabula_Muris_limb_muscle_filtered", "Tabula_Sapiens_ear_utricle_filtered"]
methods = [
    "DEC","DESC","scDCC","scDeepCluster","scNAME","scziDesk","scMAE",
    "AttentionAE-sc", "scDSC", "scGAE", "scGNN",
    "scCDCG", 
    # "leiden",
]

def sankey(nodes, links, dataset='meuro', method='scCDCG'):
    c = (
        Sankey()
        .add(
            "",
            nodes,
            links,
            pos_left="10%",
            is_draggable=False,
            linestyle_opt=opts.LineStyleOpts(opacity=0.2, curve=0.5, color="source"),
            label_opts=opts.LabelOpts(position="inside"),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(is_show=False, title="meuro_scCDCG"),
            legend_opts=opts.LegendOpts(is_show=False),
        )
        .render(f"sankey/sankey_{dataset}_{method}.html")
    )
    return c

for dataset in datasets:
    for method in methods:
        try:
            with open(f'sankey/{dataset}_{method}_node.json', 'r') as f:
                nodes = json.load(f)
            with open(f'sankey/{dataset}_{method}_link.json', 'r') as f:
                links = json.load(f)
        except FileNotFoundError:
            print(f"File not found for {dataset}-{method}. Skipping...")
            continue

        for i, node in enumerate(nodes):
            nodes[i]['name'] = nodes[i]['name'].replace('_gold1', ' ').replace('_gold2', '  ').replace('_cluster', '   ').replace('_annotation', '    ')
        for i, link in enumerate(links):
            links[i]['source'] = links[i]['source'].replace('_gold1', ' ').replace('_gold2', '  ').replace('_cluster', '   ').replace('_annotation', '    ')
            links[i]['target'] = links[i]['target'].replace('_gold1', ' ').replace('_gold2', '  ').replace('_cluster', '   ').replace('_annotation', '    ')

        sankey(nodes, links, dataset, method)
        # make_snapshot(snapshot, f"sankey/sankey_{dataset}_{method}.html", f"sankey_new/sankey_{dataset}_{method}.png")
