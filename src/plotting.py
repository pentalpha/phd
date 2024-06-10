# -*- coding: utf-8 -*-
from datetime import datetime
import sys
from os import path, mkdir
import json
from json import JSONDecodeError
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from glob import glob
from validation_external import results_deepfri_validation_path

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
    freq_min_max = {}
    
    for level, node_names in levels.items():
        node_names.sort(key=lambda x: int(x.split('_')[1].split('-')[1]))
        i = 0
        for n in node_names:
            parts = n.split('_')
            letter = str(bytes([i+65]).decode())
            new_names[n] = '_'.join([parts[0], letter])
            i += 1
            
            #print(parts)
            _, min_str, max_str = parts[1].split('-')
            freq_min_max[new_names[n]] = (int(min_str), int(max_str))
            print(n, min_str, max_str)
    
    return new_names, freq_min_max

def plot_nodes_graph(experiment_json_path):
    artifacts_dir = experiment_json_path.rstrip('.json')
    if not path.exists(artifacts_dir):
        mkdir(artifacts_dir)
    experiment = json.load(open(experiment_json_path, 'r'))
    data = []
    for group, metrics in experiment['validation'].items():
        if group != 'all':
            data.append([group, metrics])
    
    new_names, freq_min_max = convert_name_to_abc([x for x,_ in data])
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
        #x_pos = letter_to_int(x_pos_str)
        x_min, x_max = freq_min_max[node_name]
        
        x.append((x_min, x_max))
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
    
    fig, ax = plt.subplots(1,1, figsize=(20,9))
    ax.scatter([(x1+x2)/2 for x1,x2 in x], y, c='white')
    for index in range(len(x)):
        x_min, x_max = x[index]
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
        roc_auc_pos = (x_min, current_y-0.5)
        roc_auc_w = x_max-x_min
        roc_auc_h = 0.9
        roc_auc_rect = patches.Rectangle(roc_auc_pos, roc_auc_w, roc_auc_h, 
            linewidth=2, facecolor=roc_auc_color, edgecolor='white')
        ax.text(roc_auc_pos[0]+(roc_auc_w)/2, roc_auc_pos[1]+0.45, 
            'ROC='+str(round(current_roc_auc, 2)),
            horizontalalignment='center',
            verticalalignment='center')
        
        '''prec_color = m.to_rgba(current_precision_score)
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
            verticalalignment='center')'''
        
        '''hl_color = m.to_rgba(minus_hl)
        hl_pos = (current_x+roc_auc_w+rec_w, current_y-prec_h-0.05)
        hl_w = prec_w
        hl_h = prec_h
        hl_rect = patches.Rectangle(hl_pos, hl_w, hl_h, 
            linewidth=0, facecolor=hl_color)
        ax.text(hl_pos[0]+rec_w/2, hl_pos[1]+rec_h/2, 
            '1-HL', horizontalalignment='center',
            verticalalignment='center')'''
        
        print(x_min, x_max, current_y+1, 0.5, 1)
        ax.add_patch(roc_auc_rect)
        #ax.add_patch(prec_rect)
        #ax.add_patch(acc_rect)
        #ax.add_patch(rec_rect)
        
    #ax.get_xaxis().set_visible(False)
    ax.set_xscale('log', base=30)
    ax.set_ylabel("Gene Ontology Level")
    ax.set_xlabel("Number of Positive Samples (Annotations) per Class")
    ax.spines['right'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    fig.colorbar(m, ax=ax, label='Metric Value')
    fig.tight_layout()
    ax.set_title("Molecular Function Classifiers")
    plt.show()
    
    output_path = artifacts_dir+'/nodes_graph.png'
    fig.savefig(output_path, dpi=200)

def name_to_date(name):
    return datetime.strptime(name.split('_')[0]+'_'+name.split('_')[1], '%Y-%m-%d_%H-%M-%S')

def plot_experiments(plot_tests):
    current_dir = path.dirname(__file__)
    expsdir = path.dirname(current_dir) + '/experiments'
    experiments = glob(expsdir+'/*.json')
    if plot_tests:
        experiments = [e for e in experiments if '_TEST_' in e]
    else:
        experiments = [e for e in experiments if not '_TEST_' in e]
    mean_roc_auc = []
    nodes_roc_auc = {}
    for p in experiments:
        print(p)
        experiment = None
        try:
            experiment = json.load(open(p, 'r'))
        except JSONDecodeError as err:
            print(err)
            print('Cannot load', p)
        if experiment:  
            if 'validation_mean_metrics' in experiment and 'validation' in experiment:
                if len(experiment['validation']) > 1:
                    name = path.basename(p).rstrip('.json')
                    new_date = name_to_date(name)
                    new_mean = experiment['validation_mean_metrics']['roc_auc_ma_bin']
                    mean_roc_auc.append((new_date, new_mean))
                    for nodename, metrics in experiment['validation'].items():
                        if nodename.startswith('Level-'):
                            if not nodename in nodes_roc_auc:
                                nodes_roc_auc[nodename] = []
                            nodes_roc_auc[nodename].append((new_date, metrics['roc_auc_ma_bin']))
                else:
                    print('No node validation at', p)

    mean_roc_auc.sort(key=lambda x: x[0])
    for nodename, vals in nodes_roc_auc.items():
          vals.sort(key=lambda x: x[0])
    
    fig, ax = plt.subplots(1,1, figsize=(12,8))
    #fig.gca().invert_xaxis()

    mean_dates = [m[0] for m in mean_roc_auc]
    mean_vals = [m[1] for m in mean_roc_auc]
    
    for nodename, vals in nodes_roc_auc.items():
        node_dates = [d for d, roc in vals]
        node_vals = [roc for d, roc in vals]
        #print(node_dates)
        #print(node_vals)
        ax.plot(node_dates, node_vals, label='Node ROC AUC', marker='o', linewidth=2, alpha=0.5, color='lightblue')

    deepfri_val = json.load(open(results_deepfri_validation_path, 'r'))
    deepfri_postproc_roc_auc = deepfri_val['validation']['postprocessed']['roc_auc_ma_bin_raw']
    ax.hlines(deepfri_postproc_roc_auc, min(mean_dates), max(mean_dates), linewidth=8, 
        color='orange', label='DeepFRI ROC AUC', linestyles='--')
    #print(mean_dates)
    #print(mean_vals)
    ax.plot(mean_dates, mean_vals, label='Mean ROC AUC', linewidth=10, color='blue')

    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    ax.set_title("Protein GO Classification - ROC AUC")
    fig.tight_layout()
    #ax.set_yticks(x_indexes)
    #ax.set_yticklabels(names)
    #ax.set_xlim(ax.get_xlim()[::-1])
    #ax.set_ylim(ax.get_ylim()[::-1])

    #plt.show()
    
    output_path = expsdir+'/progress.full_model.png'
    if plot_tests:
        output_path = expsdir+'/progress.tests.png'
    fig.savefig(output_path, dpi=200)
    
def plot_progress():
    plot_experiments(True)
    plot_experiments(False)

if __name__ == "__main__":
    '''experiments = ['../experiments/2024-02-21_09-14-59_Full-training-2.validated.json',
       '../experiments/2024-02-27_16-04-20_Full-training-With-Early-Stopping.json',
       '../experiments/2024-02-29_23-54-28_Max-40-epochs.json']
    for exp in experiments:
        plot_nodes_graph(exp)
        #plot_experiment(exp)'''
    
    plot_progress()