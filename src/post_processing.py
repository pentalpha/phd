import json
from multiprocessing import Pool
import os
import sys
import pandas as pd
from tqdm import tqdm
import networkx as nx
import numpy as np
from gene_ontology import load_go_graph
from sklearn import metrics

from util import chunks, run_command

def annotations_dict_to_df(ann: dict, goids: list, proteinids: list):
    lines = []
    for proteinid in proteinids:
        goprobs = ann[proteinid]
        line = {goids[i]: goprobs[i] 
            for i in range(len(goids))}
        line['id'] = proteinid
        lines.append(line)
    return pd.DataFrame(lines)

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

def correct_proteins(args_dict):
    proteinids = args_dict['proteinids']
    children_dict = args_dict['children_dict']
    parent_dict = args_dict['parent_dict']
    goids_by_n_parents = args_dict['goids_by_n_parents']
    goidsubset = args_dict['goidsubset']
    df = args_dict['df']
    go_graph = args_dict['go_graph']

    annotations = {}
    bar = tqdm(total=len(proteinids))
    for index, row in tqdm(df.iterrows()):
        proteinid = row['protein']+'\t'+row['taxid']
        if proteinid in proteinids:
            local_probs = {goid: row[goid] for goid in goidsubset}
            for goid in goids_by_n_parents:
                children_probs = [local_probs[c] for c in children_dict[goid]]
                parent_probs = [local_probs[p] for p in parent_dict[goid]]
                
                new_prob = children_parent_avg(row[goid], parent_probs, children_probs)
                local_probs[goid] = new_prob
            annotations[proteinid] = [local_probs[x] for x in goidsubset]
            bar.update(1)
    bar.close()
    return annotations

def correct_by_child_parent_avg(df: pd.DataFrame, goidsubset, go_graph: nx.MultiDiGraph, proteinids):
    annotations = {}
    print(df.head())
    print(df.shape)
    
    children_dict = {goid: find_children(goid, goidsubset, go_graph)
                     for goid in tqdm(goidsubset)}
    parent_dict = {goid: find_parents(goid, goidsubset, go_graph)
                     for goid in tqdm(goidsubset)}
    goids_by_n_parents = sorted(goidsubset, key= lambda x: len(parent_dict[x]), reverse=True)

    protid_chunks = chunks(proteinids, 4000)
    arglist = []
    for protid_chunk in protid_chunks:
        arglist.append({
            'proteinids': protid_chunk,
            'children_dict': children_dict,
            'parent_dict': parent_dict,
            'goids_by_n_parents': goids_by_n_parents,
            'goidsubset': goidsubset,
            'df': df,
            'go_graph': go_graph
        })

    with Pool(4) as pool:
        annotation_dicts = pool.map(correct_proteins, arglist)

        annotations = {}
        for d in annotation_dicts:
            for prot, annots in d.items():
                annotations[prot] = annots

        return annotations, children_dict, parent_dict
    return None

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

def post_process_and_validate(experiment_json_path, validation_df_path, is_test):
    
    labels_validation_path = 'input/validation/labels.tsv'
    print('Loading results')
    experiment = json.load(open(experiment_json_path, 'r'))
    validation_df = pd.read_csv(validation_df_path, sep='\t', index_col=False)
    all_goids = [col for col in validation_df.columns if col.startswith('GO:')]
    go_graph = load_go_graph()
    validated_proteinids = []
    for index, row in validation_df.iterrows():
        validated_proteinids.append(row['protein']+'\t'+row['taxid'])
    validated_proteinids = validated_proteinids
    print(len(validated_proteinids))
    print('Loading true labels')
    annotations_true = {}
    proteinids = []
    not_predicted = 0
    for rawline in open(labels_validation_path, 'r'):
        cells = rawline.rstrip('\n').split('\t')
        proteinid = cells[0]+'\t'+cells[1]
        if proteinid in validated_proteinids:
            proteinids.append(proteinid)
            annotated = cells[2].split(',')
            probs = [1.0 if goid in annotated else 0.0 for goid in all_goids]
            annotations_true[proteinid] = probs
        else:
            not_predicted += 1
    
    print(len(proteinids), 'validation proteins predictions')
    print(not_predicted, 'validation proteins not in predictions table')

    true_probs = annotations_dict_to_df(annotations_true, all_goids, proteinids)
    goid_freqs = [(goid, true_probs[goid].sum()) for goid in all_goids]
    goid_freqs.sort(key=lambda tp: tp[1])
    goids = [goid for goid, freq in goid_freqs]
    true_probs_values = true_probs[goids]
    true_probs_binary = (true_probs_values == 1.0).astype(int)

    annotated_goids = []
    for col in goids:
        if sum(true_probs_binary[col]) > 0:
            annotated_goids.append(col)
    target_sets = [('all', annotated_goids)]
    for clustername, targetlist in experiment['go_clusters'].items():
        targets_with_ann = [t for t in targetlist if t in annotated_goids]
        target_sets.append((clustername, targets_with_ann))
    
    corrected_path = validation_df_path.replace('.tsv', '.corrected.tsv')
    if is_test:
        validation_corrected = validation_df
        run_command(['cp', validation_df_path, corrected_path])
    else:
        if not os.path.exists(corrected_path):
            print('Correcting probabilities by avg(node,min(parents),max(children)) method')
            annotations1, children_dict, parent_dict = correct_by_child_parent_avg(
                validation_df, annotated_goids, go_graph, proteinids)
            output = open(corrected_path, 'w')
            output.write('protein\ttaxid\t'+ '\t'.join(annotated_goids)+'\n')
            for protid in proteinids:
                predicted_probs_str = [str(x) for x in annotations1[protid]]
                output.write(protid+'\t' + '\t'.join(predicted_probs_str)+'\n')
            output.close()
            #validation_corrected = annotations_dict_to_df(annotations1, annotated_goids, proteinids)
        validation_corrected = pd.read_csv(corrected_path, sep='\t', index_col=False)
    
    validation_results = {}

    print('Calculating metrics')
    n_validated = 0
    individual_metrics = []
    for clustername, targetlist in tqdm(target_sets):
        print(clustername, len(targetlist), targetlist[0], targetlist[-1])
        if clustername == 'all':
            short_cluster_name = clustername
        else:
            short_cluster_name = clustername.split('_')[0] + '_' + clustername.split('_')[1]
        print('filtering y_true')
        indexes_with_annot = set()
        y_true_vec = []
        i = 0
        for index, row in true_probs_binary.iterrows():
            vec = [float(row[t]) for t in targetlist]
            #if sum(vec) > 0:
            y_true_vec.append(np.array(vec))
            indexes_with_annot.add(i)
            i += 1
        y_true = np.asarray(y_true_vec)
        #print(len(indexes_with_annot), 'proteins with annotation in cluster')
        print('filtering y_pred')
        i = 0
        y_pred = []
        for index, row in validation_corrected.iterrows():
            #if i in indexes_with_annot:
            vec = [1.0 if row[t] > 0.5 else 0.0 for t in targetlist]
            y_pred.append(np.array(vec))
            i += 1
        y_pred = np.asarray(y_pred)
        #print(with_th_pred)
        #print('\n')
        #print(true_probs_binary_f)

        roc_auc_ma_bin = metrics.roc_auc_score(y_true_vec, y_pred, 
            average='macro')
        hamming_loss = metrics.hamming_loss(y_true, y_pred)
        y_true_argmax = np.argmax(y_true, axis = 1)
        y_pred_argmax = np.argmax(y_pred, axis = 1)
        precision_score = metrics.precision_score(y_true_argmax, y_pred_argmax, 
            average='weighted')
        recall_score = metrics.recall_score(y_true_argmax, y_pred_argmax, 
            average='weighted')
        accuracy_score = metrics.accuracy_score(y_true_argmax, y_pred_argmax)

        validation_results[short_cluster_name] = {
            'hamming_loss': hamming_loss,
            'precision_score': precision_score,
            'recall_score': recall_score,
            'accuracy_score': accuracy_score,
            'roc_auc_ma_bin': roc_auc_ma_bin
        }

        if clustername != 'all':
            individual_metrics.append((validation_results[short_cluster_name], len(targetlist)))

        print(short_cluster_name, validation_results[short_cluster_name])
        
    print(n_validated, 'validated of', len(target_sets)-1)
    experiment['validation_all_goids'] = goids
    experiment['validation'] = validation_results
    w_sum = sum([w for m, w in individual_metrics])
    experiment['validation_mean_metrics'] = {
        'hamming_loss': sum([x['hamming_loss']*w for x, w in individual_metrics]) / w_sum,
        'precision_score': sum([x['precision_score']*w for x, w in individual_metrics]) / w_sum,
        'recall_score': sum([x['recall_score']*w for x, w in individual_metrics]) / w_sum,
        'accuracy_score': sum([x['accuracy_score']*w for x, w in individual_metrics]) / w_sum,
        'roc_auc_ma_bin': sum([x['roc_auc_ma_bin']*w for x, w in individual_metrics]) / w_sum
    }
    experiment['validation_std'] = {
        'hamming_loss': np.std([x['hamming_loss'] for x, w in individual_metrics]),
        'precision_score':  np.std([x['precision_score'] for x, w in individual_metrics]),
        'recall_score':  np.std([x['recall_score'] for x, w in individual_metrics]),
        'accuracy_score':  np.std([x['accuracy_score'] for x, w in individual_metrics]),
        'roc_auc_ma_bin':  np.std([x['roc_auc_ma_bin'] for x, w in individual_metrics])
    }
    validated_path = experiment_json_path.rstrip('.json')+'.validated.json'
    json.dump(experiment, open(validated_path, 'w'), indent=4)

    return validated_path

if __name__ == '__main__':
    experiment_json_path = sys.argv[1]
    validation_df_path = sys.argv[2]

    validated_json = post_process_and_validate(experiment_json_path, validation_df_path)