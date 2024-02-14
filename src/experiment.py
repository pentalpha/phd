import os
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import keras
print(keras.__version__)

from sklearn import metrics
from skmultilearn.model_selection import iterative_train_test_split

from random import sample

from util import create_labels_matrix, load_features_from_dir, load_labels_from_dir, config

def go_frequencies(go_labels: dict):
    go_count = {}
    for golist in go_labels.values():
        for go in golist:
            if not go in go_count:
                go_count[go] = 0
            go_count[go] += 1

    tps = [(go, n) for go, n in go_count.items()]
    tps.sort(key= lambda tp: tp[1])

    return tps

def cluster_gos_by_freq_percentiles(go_labels: dict):
    go_freqs = go_frequencies(go_labels)
    go_freqs = [(go, freq) for go, freq in go_freqs if (freq / len(traintest_ids)) < 0.9 and freq >= 20]
    print(len(go_freqs), 'labels')
    freq_vec = [y for x,y in go_freqs]

    print('Counting percentiles')
    percentiles = [30, 60, 90, 96]
    perc_index = []
    last_index = -1
    for perc in percentiles:
        print(perc, np.percentile(freq_vec, perc))
        index = int(len(go_freqs)*(perc/100))
        perc_index.append((last_index+1, index))
        last_index = index 
    perc_index.append((last_index+1, len(go_freqs)-1))   
    print(perc_index)

    total_len = 0
    go_mega_clusters = []
    for start, end in perc_index:
        sub_gos = go_freqs[start:end+1]
        print(len(sub_gos))
        total_len += len(sub_gos)
        go_mega_clusters.append(sub_gos)
    print(total_len)

    go_sets = []
    for index in range(len(go_mega_clusters)):
        clust = go_mega_clusters[index]
        print('Go Cluster', index, 'with', len(clust), 'GOs')
        freq_vec = [y for x,y in clust]
        print('\tMean samples:', np.mean(freq_vec), 'Min:', np.min(freq_vec), 'Max:', np.max(freq_vec))
        go_sets.append([x for x,y in clust])

    return go_sets

def makeMultiClassifierModel(train_x, train_y):
    print('Defining')
    param_dict = {'batch_size': 1070, 'learning_rate': 0.001, 'epochs': 20, 
                      'hidden1': 0.71, 'hidden2': 0.82}
    size_factors = [param_dict['hidden1'], param_dict['hidden2']]
    optimizer = keras.optimizers.Adam(learning_rate=param_dict['learning_rate'])

    n_features = len(train_x[0])
    first = keras.layers.BatchNormalization(input_shape=[n_features])
    last = keras.layers.Dense(units=len(train_y[0]), activation='sigmoid')
    hidden_sizes = [n_features*size_factor for size_factor in size_factors]
    hidden_layers = [keras.layers.Dense(units=hidden_sizes[0], activation='relu'),
                    keras.layers.Dense(units=hidden_sizes[1], activation='relu')]
    model = keras.Sequential([first] + hidden_layers + [last])
    
    # Compile model
    '''self.model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=''
    )'''

    print('Compiling')
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy'
    )
    
    print('Fitting')
    history = model.fit(
        train_x, train_y,
        batch_size=param_dict['batch_size'],
        epochs=param_dict['epochs']
    )

    '''history_df = pd.DataFrame(history.history)
    history_df.loc[:, ['mean_absolute_percentage_error']].plot(title="mean_absolute_percentage_error")
    history_df.loc[:, ['val_mean_absolute_percentage_error']].plot(title="Val mean_absolute_percentage_error")'''
    
    return model

if __name__ == '__main__':
    print('Loading train/test ids')
    traintest_ids = open('input/traintest/ids.txt', 'r').read().split('\n')
    print(len(traintest_ids), 'proteins')
    print('Loading labels')
    go_labels, go_annotations = load_labels_from_dir('input/traintest', ids_allowed=traintest_ids)
    go_lists = cluster_gos_by_freq_percentiles(go_labels)

    test_go_set = go_lists[-1][-6:]

    print(test_go_set)
    all_proteins = set()
    for go in test_go_set:
        annots = go_annotations[go]
        print(go, len(annots))
        all_proteins.update(annots)
    
    print(len(all_proteins), 'proteins')
    protein_list = sorted(all_proteins)
    protein_list = sample(protein_list, 10000)
    print('Loading features')
    local_features, new_index = load_features_from_dir('input/traintest', ids_allowed=protein_list)
    print(local_features.shape)

    print('Creating go label one hot encoding')
    local_labels = create_labels_matrix(go_labels, protein_list, test_go_set)
    print(local_labels.shape)

    print('Splitting train and test')
    train_x, train_y, test_x, test_y = iterative_train_test_split(
        local_features, 
        local_labels, 
        test_size = config['testing_perc'])

    annot_model = makeMultiClassifierModel(train_x, train_y)

    print('Testing')
    y_pred = annot_model.predict(test_x)
    roc_auc_score = metrics.roc_auc_score(test_y, y_pred)
    print(roc_auc_score)