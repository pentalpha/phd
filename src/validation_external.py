import json
import numpy as np
import pandas as pd
import os
from sklearn import metrics
from tqdm import tqdm

from gene_ontology import load_go_graph
from post_processing import annotations_dict_to_df, correct_by_child_parent_avg
from util import open_file, write_file


probs_deepfri_validation_path = 'input/validation/deepfri_probs.tsv.gz'
results_deepfri_validation_path = 'input/validation/deepfri_validation.json'
all_probs_path = 'input/deepfri_probs.tsv.gz'
validation_ids_path = 'input/validation/ids.txt'
labels_validation_path = 'input/validation/labels.tsv'

def post_process_and_validate_external(validation_df_path):
    
    print('Loading results')
    validation_df = pd.read_csv(validation_df_path, sep='\t', index_col=False)
    all_goids = [col for col in validation_df.columns if col.startswith('GO:')]
    print(len(all_goids), 'go ids in dataset')
    goids_predicted = []
    for goid in all_goids:
        all_values = validation_df[goid].values.tolist()
        min_val = 0.01
        n_predicted = [x for x in all_values if x > min_val]
        if len(n_predicted) > 0:
            goids_predicted.append(goid)
    print(len(goids_predicted), 'go ids actually predicted')
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
            probs = [1.0 if goid in annotated else 0.0 for goid in goids_predicted]
            annotations_true[proteinid] = probs
        else:
            not_predicted += 1
    
    print(len(proteinids), 'validation proteins predictions')
    print(not_predicted, 'validation proteins not in predictions table')

    true_probs = annotations_dict_to_df(annotations_true, goids_predicted, proteinids)
    goid_freqs = [(goid, true_probs[goid].sum()) for goid in goids_predicted]
    goid_freqs.sort(key=lambda tp: tp[1])
    goids = [goid for goid, freq in goid_freqs]
    true_probs_values = true_probs[goids]
    true_probs_binary = (true_probs_values == 1.0).astype(int)

    annotated_goids = []
    for col in goids:
        if sum(true_probs_binary[col]) > 0:
            annotated_goids.append(col)
    print(len(annotated_goids), 'annotated goids')
    '''for clustername, targetlist in experiment['go_clusters'].items():
        targets_with_ann = [t for t in targetlist if t in annotated_goids]
        target_sets.append((clustername, targets_with_ann))'''
    
    corrected_path = validation_df_path.replace('.tsv', '.corrected.tsv').rstrip('.gz')
    if not os.path.exists(corrected_path):
        print('Correcting probabilities by avg(node,min(parents),max(children)) method')
        annotations1, children_dict, parent_dict = correct_by_child_parent_avg(
            validation_df, annotated_goids, go_graph, proteinids)
        output = write_file(corrected_path)
        output.write('protein\ttaxid\t'+ '\t'.join(annotated_goids)+'\n')
        for protid in proteinids:
            predicted_probs_str = [str(x) for x in annotations1[protid]]
            output.write(protid+'\t' + '\t'.join(predicted_probs_str)+'\n')
        output.close()
        #validation_corrected = annotations_dict_to_df(annotations1, annotated_goids, proteinids)
    
    
    validation_results = {}
    print('Calculating metrics')
    n_validated = 0
    target_sets = [('pitagoras', annotated_goids, 'experiments/2024-02-29_23-54-28_Max-40-epochs/validation.corrected.tsv'),
                   ('no_postprocessing', annotated_goids, validation_df_path),
                   ('postprocessed', annotated_goids, corrected_path)]
    for clustername, targetlist, probs_df_path in tqdm(target_sets):
        probs_df = pd.read_csv(probs_df_path, sep='\t', index_col=False)
        print(clustername, len(targetlist), targetlist[0], targetlist[-1])
        print('filtering y_true')
        indexes_with_annot = set()
        y_true_vec = []
        i = 0
        for index, row in true_probs_binary.iterrows():
            vec = [float(row[t]) for t in targetlist]
            y_true_vec.append(np.array(vec))
            indexes_with_annot.add(i)
            i += 1
        y_true = np.asarray(y_true_vec)
        #print(len(indexes_with_annot), 'proteins with annotation in cluster')
        print('filtering y_pred')

        i = 0
        y_pred_raw = []
        for index, row in probs_df.iterrows():
            vec_raw = [float(row[t]) if t in row else 0.0 for t in targetlist]
            y_pred_raw.append(np.array(vec_raw))
        y_pred_raw = np.asarray(y_pred_raw)
        
        roc_auc_ma_bin_raw = metrics.roc_auc_score(y_true_vec, y_pred_raw, 
            average='macro')

        validation_results[clustername] = {
            'roc_auc_ma_bin_raw': roc_auc_ma_bin_raw,
            'by_threshold': {}
        }

        print(clustername, validation_results[clustername])

        for th in [0.4, 0.45, 0.5, 0.55, 0.6]:
            i = 0
            y_pred = []
            for index, row in probs_df.iterrows():
                vec = [(1.0 if row[t] > th else 0.0) if t in row else 0.0 for t in targetlist]
                y_pred.append(np.array(vec))
                i += 1
            y_pred = np.asarray(y_pred)

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

            validation_results[clustername]['by_threshold'][th] = {
                'hamming_loss': hamming_loss,
                'precision_score': precision_score,
                'recall_score': recall_score,
                'accuracy_score': accuracy_score,
                'roc_auc_ma_bin': roc_auc_ma_bin
            }

            print('\t', clustername, th, validation_results[clustername]['by_threshold'][th])
        
    print(n_validated, 'validated of', len(target_sets)-1)

    val_experiment = {}

    val_experiment['validation_all_goids'] = goids
    val_experiment['validation'] = validation_results
    
    json.dump(val_experiment, open(results_deepfri_validation_path, 'w'), indent=4)

    return results_deepfri_validation_path

if __name__ == "__main__":
    
    validation_ids = open(validation_ids_path, 'r').read().split('\n')
    filtered_prob_lines = ['' for i in range(len(validation_ids))]
    validation_id_to_line = {validation_ids[i]: i for i in range(len(validation_ids))}

    in_stream = open_file(all_probs_path)
    header_raw = in_stream.readline().rstrip('\n')
    header = header_raw.split('\t')
    all_goids = [col for col in header if col.startswith('GO:')]
    
    for rawline in in_stream:
        cells = rawline.split('\t')
        protid = cells[0] + '\t' + cells[1]
        if protid in validation_id_to_line:
            line_n = validation_id_to_line[protid]
            filtered_prob_lines[line_n] = rawline.rstrip('\n')
    
    not_found = 0
    for i in range(len(filtered_prob_lines)):
        if filtered_prob_lines[i] == '':
            not_found += 1
            
            validation_id = validation_ids[i]
            newline = validation_id + '\t' + '\t'.join(['0.0' for _ in all_goids])
            filtered_prob_lines[i] = newline
    
    print(not_found, 'validation proteins had no deepfri probabilities')
    with write_file(probs_deepfri_validation_path) as out_stream:
        out_stream.write(header_raw.replace('proteinid\t', 'protein\t') + '\n')
        for line in filtered_prob_lines:
            out_stream.write(line+'\n')
    
    post_process_and_validate_external(probs_deepfri_validation_path)