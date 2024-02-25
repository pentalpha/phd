import pandas as pd
from tqdm import tqdm
import networkx as nx
import numpy as np
from gene_ontology import load_go_graph
from sklearn import metrics

def annotations_dict_to_df(ann: dict, goids: list):
    lines = []
    for proteinid, goprobs in ann.items():
        line = {goids[i]: goprobs[i] for i in range(len(goids))}
        line['id'] = proteinid
        lines.append(line)
    return pd.DataFrame(lines)

validation_df_path = "experiments/validation_full.tsv"
validation_df = pd.read_csv(validation_df_path, sep='\t', index_col=False)
all_goids = [col for col in validation_df.columns if col.startswith('GO:')]
go_graph = load_go_graph()

labels_validation_path = 'input/validation/labels.tsv'
annotations_true = {}
for rawline in open(labels_validation_path, 'r'):
    cells = rawline.rstrip('\n').split('\t')
    proteinid = cells[0]+'\t'+cells[1]
    annotated = cells[2].split(',')
    probs = [1.0 if goid in annotated else 0.0 for goid in all_goids]
    annotations_true[proteinid] = probs

true_probs = annotations_dict_to_df(annotations_true, all_goids)
goid_freqs = [(goid, true_probs[goid].sum()) for goid in all_goids]
goid_freqs.sort(key=lambda tp: tp[1])
top_600_goids = [goid for goid, freq in goid_freqs[-1500:]]

true_probs_values = true_probs[top_600_goids]
true_probs_binary = (true_probs_values == 1.0).astype(int)

#%%
def uncorrected(df: pd.DataFrame, goidsubset):
    annotations = {}
    print(df.head())
    print(df.shape)
    for index, row in tqdm(df.iterrows()):
        proteinid = row['protein']+'\t'+row['taxid']
        annotations[proteinid] = [row[goid] for goid in goidsubset]
    return annotations

def find_children(term: str, term_subset: list, go_graph: nx.MultiDiGraph):
    if term in go_graph:
        all_children = set(nx.ancestors(go_graph, term))
        in_subset = all_children.intersection(term_subset)
        return in_subset
    else:
        return set()

def find_parents(term: str, term_subset: list, go_graph: nx.MultiDiGraph):
    if term in go_graph:
        all_children = set(nx.descendants(go_graph, term))
        in_subset = all_children.intersection(term_subset)
        return in_subset
    else:
        return set()

def children_parent_avg(term, parent_probs, children_probs):
    total = term
    n = 1
    if len(children_probs) > 0:
        total += max(children_probs)
        n += 1
    if len(parent_probs) > 0:
        total += min(parent_probs)
        n += 1
    
    return total / n

def correct_by_child_parent_avg(df: pd.DataFrame, goidsubset, go_graph: nx.MultiDiGraph):
    annotations = {}
    print(df.head())
    print(df.shape)
    
    children_dict = {goid: find_children(goid, goidsubset, go_graph)
                     for goid in tqdm(goidsubset)}
    parent_dict = {goid: find_parents(goid, goidsubset, go_graph)
                     for goid in tqdm(goidsubset)}
    goids_by_n_parents = sorted(goidsubset, key= lambda x: len(parent_dict[x]), reverse=True)
    for index, row in tqdm(df.iterrows()):
        proteinid = row['protein']+'\t'+row['taxid']
        local_probs = {goid: row[goid] for goid in goidsubset}
        for goid in goids_by_n_parents:
            children_probs = [local_probs[c] for c in children_dict[goid]]
            parent_probs = [local_probs[p] for p in parent_dict[goid]]
            
            new_prob = children_parent_avg(row[goid], parent_probs, children_probs)
            local_probs[goid] = new_prob
        annotations[proteinid] = [local_probs[x] for x in goidsubset]
    
    return annotations, children_dict, parent_dict

def correct_by_max_children(df: pd.DataFrame, goidsubset, go_graph: nx.MultiDiGraph):
    annotations = {}
    print(df.head())
    print(df.shape)
    
    children_dict = {goid: find_children(goid, goidsubset, go_graph)
                     for goid in tqdm(goidsubset)}
    goids_by_n_children = sorted(goidsubset, key= lambda x: len(children_dict[x]))
    for index, row in tqdm(df.iterrows()):
        proteinid = row['protein']+'\t'+row['taxid']
        local_probs = {goid: row[goid] for goid in goidsubset}
        for goid in goids_by_n_children:
            children_probs = [local_probs[c] for c in children_dict[goid]]
            new_prob = max([local_probs[goid]] + children_probs)
            local_probs[goid] = new_prob
        annotations[proteinid] = [local_probs[x] for x in goidsubset]
    
    return annotations, children_dict

#%%
annotations0 = uncorrected(validation_df, top_600_goids)
uncorrected = annotations_dict_to_df(annotations0, top_600_goids)
uncorrected_values = uncorrected[top_600_goids]
#%%
annotations1, children_dict, parent_dict = correct_by_child_parent_avg(
    validation_df, top_600_goids, go_graph)
child_parent = annotations_dict_to_df(annotations1, top_600_goids)
child_parent_values = child_parent[top_600_goids]
#%%
annotations2, children_dict = correct_by_max_children(
    validation_df, top_600_goids, go_graph)
child_max = annotations_dict_to_df(annotations2, top_600_goids)
child_max_values = child_max[top_600_goids]
        
#%%
print('Calculating metrics')
predictions = [('uncorrected', uncorrected_values), ('child_parent', child_parent_values),
               ('child_max_values', child_max_values)]
for name, values in predictions:
    print(name)
    with_th_pred = (values > 0.6).astype(int)
    keep = []
    for goid in top_600_goids:
        freq = with_th_pred[goid].sum()
        if freq >= 1:
            keep.append(goid)
    with_th_pred_f = with_th_pred[keep]
    true_probs_binary_f = true_probs_binary[keep]
    print('\thamming_loss', metrics.hamming_loss(true_probs_binary_f, with_th_pred_f))
    print('\tprecision_score', metrics.precision_score(true_probs_binary_f, with_th_pred_f, average='weighted'))
    print('\taccuracy_score', metrics.accuracy_score(true_probs_binary_f, with_th_pred_f))
    print('\trecall_score', metrics.recall_score(true_probs_binary_f, with_th_pred_f, average='weighted'))
    print('\troc_auc_ma_bin', metrics.roc_auc_score(true_probs_binary_f, with_th_pred_f, average='macro'))
    