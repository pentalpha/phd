import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import keras
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, Callback
from keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU, Concatenate, Input, ReLU, MultiHeadAttention, Layer
from keras.losses import Loss
from keras.constraints import max_norm
from keras.optimizers import Adam
from keras.metrics import AUC
from keras.callbacks import CSVLogger
from keras import Model
from keras.models import load_model
from keras import backend as K
print(keras.__version__)

from sklearn import metrics
from skmultilearn.model_selection import iterative_train_test_split

from random import sample

from util import create_labels_matrix, get_items_at_indexes, load_dataset_from_dir, load_features_from_dir, load_labels_from_dir, config

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

def makeMultiClassifierModel(train_x, train_y, test_x, test_y):
    
    print('Converting features to np')
    for i in range(len(train_x)):
        feature_name, feature_vec = train_x[i]
        train_x[i] = (feature_name, np.asarray([np.array(vec) for vec in feature_vec]))
        print(feature_name, train_x[i][1].shape)
    for i in range(len(test_x)):
        feature_name, feature_vec = test_x[i]
        test_x[i] = (feature_name, np.asarray([np.array(vec) for vec in feature_vec]))
        print(feature_name, test_x[i][1].shape)
    
    print('go labels', train_y.shape)
    print('go labels', test_y.shape)

    print('Defining network')

    l1_dim = 600
    l2_dim = 300
    taxa_l1_dim = 128
    taxa_l2_dim = 64
    dropout_rate = 0.5

    final_dim = 256

    keras_inputs = []
    keras_input_networks = []

    for feature_name, feature_vec in train_x:
        if 'taxa' in feature_name:
            start_dim = taxa_l1_dim
            end_dim = taxa_l2_dim
        elif 'esm' in feature_name:
            start_dim = l1_dim
            end_dim = l2_dim
        
        input_start = Input(shape=(feature_vec.shape[1],))

        input_network = Dense(start_dim, name=feature_name+'_dense_1')(input_start)
        input_network = BatchNormalization(name=feature_name+'_batchnorm_1')(input_network)
        input_network = LeakyReLU(alpha=0.1, name=feature_name+'_leakyrelu_1')(input_network)
        input_network = Dropout(dropout_rate, name=feature_name+'_dropout_1')(input_network)
        input_network = Dense(end_dim, name=feature_name+'_dense_2')(input_network)

        keras_inputs.append(input_start)
        keras_input_networks.append(input_network)
    
    # Concatenate the networks
    combined = Concatenate()(keras_input_networks)
    #combined = LeakyReLU(alpha=0.1, name='combined_leakyrelu_1')(combined)
    combined = BatchNormalization(name = 'combined_batchnorm_1')(combined)
    combined = Dense(final_dim, name='combined_dense_1', activation='relu')(combined)
    output_1 = Dense(train_y.shape[1], activation='sigmoid', name='final_output_123')(combined)

    # Create the model
    print('Creating Modle')
    model = Model(inputs=keras_inputs,
        outputs=output_1)

    #from keras.utils import plot_model
    #plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
    
    #lr scheduling
    def lr_schedule(epoch, lr):
        if epoch > 0 and epoch % 10 == 0:
            lr = lr * 0.5
        return lr
    lr_callback = LearningRateScheduler(lr_schedule, verbose=1)

    print("Compiling")
    model.compile(optimizer=Adam(learning_rate=0.0003),
        loss = 'binary_crossentropy',
        metrics=['binary_accuracy', keras.metrics.AUC()])

    print('Fiting')
    history = model.fit([x for name, x in train_x], train_y,
        validation_data=([x for name, x in test_x], test_y),
        epochs=30, batch_size=256,
        callbacks=[lr_callback])

    '''param_dict = {'batch_size': 1070, 'learning_rate': 0.001, 'epochs': 20, 
                      'hidden1': 0.71, 'hidden2': 0.82}
    size_factors = [param_dict['hidden1'], param_dict['hidden2']]
    optimizer = keras.optimizers.Adam(learning_rate=param_dict['learning_rate'])

    n_features = len(train_x[0])
    first = keras.layers.BatchNormalization(input_shape=[n_features])
    last = keras.layers.Dense(units=len(train_y[0]), activation='sigmoid')
    hidden_sizes = [n_features*size_factor for size_factor in size_factors]
    hidden_layers = [keras.layers.Dense(units=hidden_sizes[0], activation='relu'),
                    keras.layers.Dense(units=hidden_sizes[1], activation='relu')]
    model = keras.Sequential([first] + hidden_layers + [last])'''
    
    # Compile model
    '''self.model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=''
    )'''

    '''print('Compiling')
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
    )'''

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
    features, labels, annotations = load_dataset_from_dir('input/traintest', protein_list)
    for feature_name, feature_vec in features:
        print('\t', feature_name, len(feature_vec), len(feature_vec[0]))

    print('Creating go label one hot encoding')
    local_labels = create_labels_matrix(labels, protein_list, test_go_set)
    print('\t', local_labels.shape)

    train_perc = 1.0 - config['validation_perc'] - config['testing_perc']

    print('Train/test is', len(protein_list))
    total_with_val = len(protein_list) / (1.0 - config['validation_perc'])
    print('Total proteins was', total_with_val)
    test_n = total_with_val*config['testing_perc']
    print('Test total should be', test_n)
    testing_perc_local = test_n / len(protein_list)
    print('So the local testing perc. is', testing_perc_local)
    print('Splitting train and test')
    protein_indexes = np.asarray([np.array([i]) for i in range(len(protein_list))])
    train_ids, train_y, test_ids, test_y = iterative_train_test_split(
        protein_indexes, 
        local_labels, 
        test_size = testing_perc_local)
    #protein_indices = {protein_list[i]: i for i in range(len(protein_list))}
    train_feature_indices = [i_vec[0] for i_vec in train_ids]
    test_feature_indices = [i_vec[0] for i_vec in test_ids]
    train_x = []
    test_x = []
    for name, feature_vec in features:
        sub_vec_train = get_items_at_indexes(feature_vec, train_feature_indices)
        sub_vec_test = get_items_at_indexes(feature_vec, test_feature_indices)
        train_x.append((name, sub_vec_train))
        test_x.append((name, sub_vec_test))

    annot_model = makeMultiClassifierModel(train_x, train_y, test_x, test_y)

    print('Loading validation')
    val_protein_list = open('input/validation/ids.txt', 'r').read().split('\n')
    val_features, val_labels, _ = load_dataset_from_dir('input/validation', val_protein_list)
    for feature_name, feature_vec in val_features:
        print('\t', feature_name, len(feature_vec), len(feature_vec[0]))
    print('Creating go label one hot encoding')
    val_y = create_labels_matrix(val_labels, val_protein_list, test_go_set)
    print('\t', val_y.shape)
    print('Converting features to np')
    for i in range(len(val_features)):
        feature_name, feature_vec = val_features[i]
        val_features[i] = (feature_name, np.asarray([np.array(vec) for vec in feature_vec]))
        print(feature_name, val_features[i][1].shape)
    val_x = [x for name, x in val_features]

    print('Validating')
    val_y_pred = annot_model.predict(val_x)
    roc_auc_score = metrics.roc_auc_score(val_y, val_y_pred)
    print(roc_auc_score)