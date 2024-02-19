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

if __name__ == "__main__":
    input_jsons = glob('../experiments/*.json')
    for x in input_jsons:
        plot_experiment(x)
    