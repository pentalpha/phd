# -*- coding: utf-8 -*-
import sys
from os import path, mkdir
import json
from json import JSONDecodeError
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.patches as patches
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
        if 'validation' in data: 
            results = [x for name, x in data['validation'].items()]
            if len(results) > 0:
                roc_auc = np.array([x['roc_auc_ma_bin'] for x in results])
                accuracy = np.array([x['accuracy_score'] for x in results])
                
                fig, axes = plt.subplots(1, 1, figsize=(10, 6))
                plot = sns.kdeplot(data={'ROC AUC': roc_auc, 'Accuracy': accuracy}, 
                    fill=True, alpha=0.6, legend=True, ax = axes, bw_adjust=0.15)
                fig.savefig(path.join(artifacts_dir, 'testing.png'), dpi=400)

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

def plot_nodes_graph(experiment_json_path):
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
    global_min = 0.2
    for node_name, metrics in data:
        level_str, x_pos_str = node_name.split('_')
        level = int(level_str.split('-')[1])
        x_pos = letter_to_int(x_pos_str)
        
        x.append(x_pos)
        y.append(level)
        labels.append(node_name)
        hamming_loss.append(metrics['hamming_loss'])
        precision_score.append(max(global_min, metrics['precision_score']))
        recall_score.append(max(global_min, metrics['recall_score']))
        accuracy_score.append(max(global_min, metrics['accuracy_score']))
        roc_auc.append(max(global_min, metrics['roc_auc_ma_bin']))
    
    min_metric = min(precision_score + recall_score + accuracy_score + roc_auc)
    import matplotlib as mpl
    import matplotlib.cm as cm
    
    norm = mpl.colors.Normalize(vmin=global_min, vmax=1.0)
    cmap = cm.winter
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    
    fig, ax = plt.subplots(1,1, figsize=(13,9))
    ax.scatter(x, y, c='white')
    for index in range(len(x)):
        current_x = x[index]
        current_y = y[index]
        label = labels[index]
        current_hamming_loss = hamming_loss[index]
        current_precision_score = precision_score[index]
        current_recall_score = recall_score[index]
        current_accuracy_score = accuracy_score[index]
        current_roc_auc = roc_auc[index]
        
        minus_hl = 1.0 - current_hamming_loss
        minus_hl = minus_hl*minus_hl*minus_hl
        
        roc_auc_color = m.to_rgba(current_roc_auc)
        roc_auc_pos = (current_x, current_y-0.5)
        roc_auc_w = 0.47
        roc_auc_h = 0.9
        roc_auc_rect = patches.Rectangle(roc_auc_pos, roc_auc_w, roc_auc_h, 
            linewidth=0, facecolor=roc_auc_color)
        ax.text(roc_auc_pos[0]+roc_auc_w/2, roc_auc_pos[1]+0.45, 
            'ROC='+str(round(current_roc_auc, 2)),
            horizontalalignment='center',
            verticalalignment='center')
        
        prec_color = m.to_rgba(current_precision_score)
        prec_pos = (current_x+roc_auc_w, current_y-0.05)
        prec_w = roc_auc_w/2
        prec_h = roc_auc_h/2
        prec_rect = patches.Rectangle(prec_pos, prec_w, prec_h, 
            linewidth=0, facecolor=prec_color)
        ax.text(prec_pos[0]+prec_w/2, prec_pos[1]+prec_h/2, 
            'PREC', horizontalalignment='center',
            verticalalignment='center')
        
        acc_color = m.to_rgba(current_accuracy_score)
        acc_pos = (current_x+roc_auc_w+prec_w, current_y-0.05)
        acc_w = prec_w
        acc_h = prec_h
        acc_rect = patches.Rectangle(acc_pos, acc_w, acc_h, 
            linewidth=0, facecolor=acc_color)
        ax.text(acc_pos[0]+acc_w/2, acc_pos[1]+acc_h/2, 
            'ACC', horizontalalignment='center',
            verticalalignment='center')
        
        rec_color = m.to_rgba(current_recall_score)
        rec_pos = (current_x+roc_auc_w, current_y-prec_h-0.05)
        rec_w = prec_w
        rec_h = prec_h
        rec_rect = patches.Rectangle(rec_pos, rec_w, rec_h, 
            linewidth=0, facecolor=rec_color)
        ax.text(rec_pos[0]+rec_w/2, rec_pos[1]+rec_h/2, 
            'REC', horizontalalignment='center',
            verticalalignment='center')
        
        '''hl_color = m.to_rgba(minus_hl)
        hl_pos = (current_x+roc_auc_w+rec_w, current_y-prec_h-0.05)
        hl_w = prec_w
        hl_h = prec_h
        hl_rect = patches.Rectangle(hl_pos, hl_w, hl_h, 
            linewidth=0, facecolor=hl_color)
        ax.text(hl_pos[0]+rec_w/2, hl_pos[1]+rec_h/2, 
            '1-HL', horizontalalignment='center',
            verticalalignment='center')'''
        
        print(current_x, current_y+1, 0.5, 1)
        ax.add_patch(roc_auc_rect)
        ax.add_patch(prec_rect)
        ax.add_patch(acc_rect)
        ax.add_patch(rec_rect)
        
    ax.get_xaxis().set_visible(False)
    
    ax.set_ylabel("Gene Ontology Level")
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    fig.colorbar(m, ax=ax, label='Metric Value')
    fig.tight_layout()
    ax.set_title("Molecular Function Classifiers")
    plt.show()
    
    output_path = artifacts_dir+'/nodes_graph.png'
    fig.savefig(output_path, dpi=200)

if __name__ == "__main__":
    experiment_json_path = '../experiments/2024-02-21_09-14-59_Full-training-2.validated.json'
    #plot_nodes_graph(experiment_json_path)
    plot_experiment(experiment_json_path)