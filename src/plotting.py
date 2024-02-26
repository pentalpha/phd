# -*- coding: utf-8 -*-
import sys
from os import path, mkdir
import json
from json import JSONDecodeError
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from glob import glob

def plot_experiment(input_json):
    artifacts_dir = input_json.rstrip('.json')
    if not path.exists(artifacts_dir):
        mkdir(artifacts_dir)
    print('Plotting', input_json)
    try:
        data = json.load(open(input_json, 'r'))
    except JSONDecodeError as err:
        print('Invalid json at', input_json)
        data = None
    
    if data:
        if 'classifiers' in data:
            results = [x['results'] for name, x in data['classifiers'].items()]
            if len(results) > 0:
                roc_auc = np.array([x['ROC AUC'] for x in results])
                accuracy = np.array([x['Accuracy'] for x in results])
                
                fig, axes = plt.subplots(1, 1, figsize=(10, 6))
                plot = sns.kdeplot(data={'ROC AUC': roc_auc, 'Accuracy': accuracy}, 
                    fill=True, alpha=0.6, legend=True, ax = axes)
                fig.savefig(path.join(artifacts_dir, 'testing.png'), dpi=400)

def plot_nodes(experiment_json_path):
    experiment = json.load(open(experiment_json_path, 'r'))
    validation = experiment['validation']

def letter_to_int(str_letter):
    b = bytes(str_letter, encoding='utf-8')
    char_int = b[0]
    x = char_int - 65
    return x

def convert_name_to_abc(node_names):
    levels = {}
    for node_name in node_names:
        level = node_name.split('_')[0].split('-')[1]
        if not level in levels:
            levels[level] = []
            
        levels[level].append(node_name)
    new_names = {}
    
    for level, node_names in levels.items():
        node_names.sort(key=lambda x: int(x.split('_')[1].split('-')[1]))
        i = 0
        for n in node_names:
            parts = n.split('_')
            letter = str(bytes([i+65]).decode())
            new_names[n] = '_'.join([parts[0], letter])
            i += 1
    
    return new_names
        
def sep_by_level(node_names):
    levels = {}
    level_n = set()
    largest_letter = 'A'
    for node_name in node_names:
        level = node_name.split('_')[0].split('-')[1]
        letter = node_name.split('_')[1]
        if letter > largest_letter:
            largest_letter = letter
        if not level in levels:
            levels[level].append(node_name)
    
    by_level = []
    n_by_level = letter_to_int(largest_letter)+1
    for level in range(1, 8):
        if str(level) in levels:
            nodes_vec = [None for _ in range(n_by_level)]
            node_names = levels[str(level)]
            for node in node_names:
                x = letter_to_int(node.split('_')[1])
                nodes_vec[x] = node
            node_names.sort()
            by_level.append(nodes_vec)
        else:
            by_level.append([None for _ in range(n_by_level)])
    return by_level

if __name__ == "__main__":
    experiment_json_path = '../experiments/2024-02-21_09-14-59_Full-training-2.validated.json'
    '''input_jsons = glob('../experiments/*.json')
    for x in input_jsons:
        plot_experiment(x)'''
    artifacts_dir = experiment_json_path.rstrip('.json')
    if not path.exists(artifacts_dir):
        mkdir(artifacts_dir)
    experiment = json.load(open(experiment_json_path, 'r'))
    data = []
    for group, metrics in experiment['validation'].items():
        if group != 'all':
            data.append([group, metrics])
    
    new_names = convert_name_to_abc([x for x,_ in data])
    for i in range(len(data)):
        data[i][0] = new_names[data[i][0]]
        
    x = []
    y = []
    labels = []
    hamming_loss = []
    precision_score = []
    recall_score = []
    accuracy_score = []
    roc_auc = []
    
    for node_name, metrics in data:
        level_str, x_pos_str = node_name.split('_')
        level = int(level_str.split('-')[1])
        x_pos = letter_to_int(x_pos_str)
        
        x.append(x_pos)
        y.append(level-1)
        labels.append(node_name)
        hamming_loss.append(metrics['hamming_loss'])
        precision_score.append(metrics['precision_score'])
        recall_score.append(metrics['recall_score'])
        accuracy_score.append(metrics['accuracy_score'])
        roc_auc.append(metrics['roc_auc_ma_bin'])
    
    
        
    