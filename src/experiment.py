import datetime
import sys
import json
import os
from os import path
from multiprocessing import Pool
from tqdm import tqdm

from gene_ontology import load_go_graph
from post_processing import post_process_and_validate
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
from mealpy import FloatVar, PSO, GOA
from mealpy.bio_based import SMA
from mealpy.evolutionary_based import DE, GA
from random import sample
from pickle import dump, load

from util import (create_labels_matrix, get_items_at_indexes, load_dataset_from_dir, load_features_from_dir, 
    load_labels_from_dir, config, run_command)
from go_clustering import cluster_go_by_levels_and_freq
from plotting import plot_experiment, plot_nodes_graph, plot_progress
from params import RandomSearchMetaheuristic, default_params, PARAM_TRANSLATOR

def make_dataset(dirname, protein_list, go_set):
    print('Loading features')
    X, labels, annotations = load_dataset_from_dir(dirname, protein_list)
    for feature_name, feature_vec in X:
        print('\t', feature_name, len(feature_vec), len(feature_vec[0]))

    print('Creating go label one hot encoding')
    Y = create_labels_matrix(labels, protein_list, go_set)
    print('\t', Y.shape)

    return X, Y

def split_train_test(ids, X, Y):

    print('Train/test is', len(ids))
    total_with_val = len(ids) / (1.0 - config['validation_perc'])
    print('Total proteins was', total_with_val)
    test_n = total_with_val*config['testing_perc']
    print('Test total should be', test_n)
    testing_perc_local = test_n / len(ids)
    print('So the local testing perc. is', testing_perc_local)
    print('Splitting train and test')
    protein_indexes = np.asarray([np.array([i]) for i in range(len(ids))])
    train_ids, train_y, test_ids, test_y = iterative_train_test_split(
        protein_indexes, 
        Y, 
        test_size = testing_perc_local)
    
    train_feature_indices = [i_vec[0] for i_vec in train_ids]
    test_feature_indices = [i_vec[0] for i_vec in test_ids]
    train_x = []
    test_x = []
    for name, feature_vec in X:
        sub_vec_train = get_items_at_indexes(feature_vec, train_feature_indices)
        sub_vec_test = get_items_at_indexes(feature_vec, test_feature_indices)
        train_x.append((name, sub_vec_train))
        test_x.append((name, sub_vec_test))

    return train_ids, train_x, train_y, test_ids, test_x, test_y

def x_to_np(x):
    #print('Converting features to np')
    for i in range(len(x)):
        feature_name, feature_vec = x[i]
        x[i] = (feature_name, np.asarray([np.array(vec) for vec in feature_vec]))
        #print(feature_name, x[i][1].shape)
    return x

def makeMultiClassifierModel(train_x, train_y, test_x, test_y, params_dict = default_params):
    
    #print('Converting features to np')
    for i in range(len(train_x)):
        feature_name, feature_vec = train_x[i]
        train_x[i] = (feature_name, np.asarray([np.array(vec) for vec in feature_vec]))
        #print(feature_name, train_x[i][1].shape)
    for i in range(len(test_x)):
        feature_name, feature_vec = test_x[i]
        test_x[i] = (feature_name, np.asarray([np.array(vec) for vec in feature_vec]))
        #print(feature_name, test_x[i][1].shape)
    
    #print('go labels', train_y.shape)
    #print('go labels', test_y.shape)

    #print('Defining network')

    keras_inputs = []
    keras_input_networks = []

    for feature_name, feature_vec in train_x:
        feature_params = params_dict[feature_name]
        start_dim = feature_params['l1_dim']
        end_dim = feature_params['l2_dim']
        leakyrelu_1_alpha = feature_params['leakyrelu_1_alpha']
        dropout_rate = feature_params['dropout_rate']
        input_start = Input(shape=(feature_vec.shape[1],))

        input_network = Dense(start_dim, name=feature_name+'_dense_1')(input_start)
        input_network = BatchNormalization(name=feature_name+'_batchnorm_1')(input_network)
        input_network = LeakyReLU(alpha=leakyrelu_1_alpha, name=feature_name+'_leakyrelu_1')(input_network)
        input_network = Dropout(dropout_rate, name=feature_name+'_dropout_1')(input_network)
        input_network = Dense(end_dim, name=feature_name+'_dense_2')(input_network)

        keras_inputs.append(input_start)
        keras_input_networks.append(input_network)
    
    final_params = params_dict['final']
    final_dim = final_params['final_dim']
    dropout_rate = final_params['final_dim']
    patience = final_params['patience']
    epochs = final_params['epochs']
    learning_rate = final_params['learning_rate']
    batch_size = final_params['batch_size']

    # Concatenate the networks
    combined = Concatenate()(keras_input_networks)
    #combined = LeakyReLU(alpha=0.1, name='combined_leakyrelu_1')(combined)
    combined = BatchNormalization(name = 'combined_batchnorm_1')(combined)
    combined = Dense(final_dim, name='combined_dense_1', activation='relu')(combined)
    output_1 = Dense(train_y.shape[1], activation='sigmoid', name='final_output_123')(combined)

    # Create the model
    #print('Creating Modle')
    model = Model(inputs=keras_inputs,
        outputs=output_1)

    #from keras.utils import plot_model
    #plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
    
    #lr scheduling
    def lr_schedule(epoch, lr):
        if epoch > 0 and epoch % 10 == 0:
            lr = lr * 0.5
        return lr
    lr_callback = LearningRateScheduler(lr_schedule, verbose=0)
    es = EarlyStopping(monitor='val_loss', patience=patience)

    #print("Compiling")
    model.compile(optimizer=Adam(learning_rate=learning_rate),
        loss = 'binary_crossentropy',
        metrics=['binary_accuracy', keras.metrics.AUC()])

    #print('Fiting')
    x_test_vec = [x for name, x in test_x]
    history = model.fit([x for name, x in train_x], train_y,
        validation_data=(x_test_vec, test_y),
        epochs=epochs, batch_size=batch_size,
        callbacks=[lr_callback, es],
        verbose=0)
    
    y_pred = model.predict(x_test_vec, verbose=0)
    roc_auc_score = metrics.roc_auc_score(test_y, y_pred)
    acc = np.mean(keras.metrics.binary_accuracy(test_y, y_pred).numpy())

    return model, {'ROC AUC': float(roc_auc_score), 'Accuracy': float(acc)}

def prepare_data(params, max_proteins=60000):
    basename = 'tmp/'+params['cluster_name']
    train_x_name = basename + '.train_x.obj'
    train_y_name = basename + '.train_y.obj'
    test_x_name = basename + '.test_x.obj'
    test_y_name = basename + '.test_y.obj'

    if all([path.exists(p) for p in [train_x_name, train_y_name, test_x_name, test_y_name]]):
        return True
    else:
        print('Preparing', basename)
        test_go_set = params['test_go_set']
        go_annotations = params['go_annotations']
        all_proteins = set()
        for go in test_go_set:
            annots = go_annotations[go]
            print(go, len(annots))
            all_proteins.update(annots)
        
        print(len(all_proteins), 'proteins')
        protein_list = sorted(all_proteins)
        if len(protein_list) > max_proteins:
            protein_list = sample(protein_list, max_proteins)

        print('Loading features')
        features, local_labels = make_dataset('input/traintest', protein_list, test_go_set)
        train_ids, train_x, train_y, test_ids, test_x, test_y = split_train_test(
            protein_list, features, local_labels)

        to_save = [(train_x_name, train_x), (train_y_name, train_y),
                (test_x_name, test_x), (test_y_name, test_y),]
        if not path.exists('tmp'):
            run_command(['mkdir', 'tmp'])
        for p, obj in to_save:
            dump(obj, open(p, 'wb'))
        
        return True
    

def train_node(params, max_proteins=60000, params_dict = default_params):
    if 'params_dict' in params:
        params_dict = params['params_dict']
    
    basename = 'tmp/'+params['cluster_name']
    train_x_name = basename + '.train_x.obj'
    train_y_name = basename + '.train_y.obj'
    test_x_name = basename + '.test_x.obj'
    test_y_name = basename + '.test_y.obj'

    if all([path.exists(p) for p in [train_x_name, train_y_name, test_x_name, test_y_name]]):
        #print('Loading input from', basename+'.*')
        train_x = load(open(train_x_name, 'rb'))
        train_y = load(open(train_y_name, 'rb'))
        test_x = load(open(test_x_name, 'rb'))
        test_y = load(open(test_y_name, 'rb'))
    else:
        test_go_set = params['test_go_set']
        go_annotations = params['go_annotations']
        all_proteins = set()
        for go in test_go_set:
            annots = go_annotations[go]
            print(go, len(annots))
            all_proteins.update(annots)
        
        print(len(all_proteins), 'proteins')
        protein_list = sorted(all_proteins)
        if len(protein_list) > max_proteins:
            protein_list = sample(protein_list, max_proteins)

        print('Loading features')
        features, local_labels = make_dataset('input/traintest', protein_list, test_go_set)
        train_ids, train_x, train_y, test_ids, test_x, test_y = split_train_test(
            protein_list, features, local_labels)

    annot_model, stats = makeMultiClassifierModel(train_x, train_y, test_x, test_y, params_dict=params_dict)
    #print(params['cluster_name'], stats)

    return annot_model, stats

def predict_with_model(nodes, experiment_dir):
    print('Loading validation')
    val_protein_list = open('input/validation/ids.txt', 'r').read().split('\n')
    print('Loading features')
    val_features, labels, annotations = load_dataset_from_dir('input/validation', val_protein_list)
    for feature_name, feature_vec in val_features:
        print('\t', feature_name, len(feature_vec), len(feature_vec[0]))
    val_features = x_to_np(val_features)
    val_x = [x for name, x in val_features]

    roc_auc_scores = []
    all_targets = []
    all_probas = [[] for _ in range(len(val_protein_list))]
    for mod_name, data in nodes.items():
        val_tsv = experiment_dir+'/'+mod_name+'.val.tsv'
        output = open(val_tsv, 'w')
        annot_model, targets = data
        all_targets += targets
        output.write('protein\ttaxid\t'+ '\t'.join(targets)+'\n')
        #print('Creating go label one hot encoding')
        val_y = create_labels_matrix(labels, val_protein_list, targets)
        #print('\t', val_y.shape)
    
        print('Validating')
        val_y_pred = annot_model.predict(val_x)
        #roc_auc_score = metrics.roc_auc_score(val_y, val_y_pred)
        #print(roc_auc_score)
        #roc_auc_scores.append(roc_auc_score)

        for i in range(len(val_protein_list)):
            predicted_probs = [x for x in val_y_pred[i]]
            predicted_probs_str = [str(x) for x in predicted_probs]
            all_probas[i] += predicted_probs
            output.write(val_protein_list[i]+'\t' + '\t'.join(predicted_probs_str)+'\n')
        output.close()
    
    big_table_path = experiment_dir+'/validation.tsv'
    output = open(big_table_path, 'w')
    output.write('protein\ttaxid\t'+ '\t'.join(all_targets)+'\n')
    for i in range(len(val_protein_list)):
        predicted_probs_str = [str(x) for x in all_probas[i]]
        output.write(val_protein_list[i]+'\t' + '\t'.join(predicted_probs_str)+'\n')
    output.close()

    return big_table_path

class MetaheuristicTest():

    def __init__(self, params) -> None:
        self.nodes = params

        '''self.problem_constrained = {
            "obj_func": self.objective_func,
            "bounds": PARAM_TRANSLATOR.to_bounds(),
            "minmax": "max",
            "log_to": "file",
            "log_file": "result.log",         # Default value = "mealpy.log"
        }'''

        self.heuristic_model = RandomSearchMetaheuristic(60, 
            PARAM_TRANSLATOR.upper_bounds, PARAM_TRANSLATOR.lower_bounds,
            n_jobs=8)
    
    def objective_func(self, solution):
        new_params_dict = PARAM_TRANSLATOR.decode(solution)
        
        for node in self.nodes:
            node['params_dict'] = new_params_dict
        '''n_procs = config['training_processes']
        if n_procs > len(self.nodes):
            n_procs = len(self.nodes)'''

        roc_aucs = [train_node(node)[1]['ROC AUC'] for node in self.nodes]
        print(np.mean(roc_aucs), np.std(roc_aucs), min(roc_aucs))
        mean_training_roc = np.mean(roc_aucs)
        min_roc_auc = min(roc_aucs)
        std = np.std(roc_aucs)
        return mean_training_roc, min_roc_auc, std

    def test(self):
        best_solution, best_fitness, report = self.heuristic_model.run_tests(self.objective_func)
        solution_dict = PARAM_TRANSLATOR.decode(best_solution)
        print(json.dumps(solution_dict, indent=4))
        print(best_fitness)

        return solution_dict, best_fitness, report

def optimize_params_for_classification(percentiles, go_graph, go_annotations, n_ids,
        min_annotations, experiment_dir):
    
    go_lists = cluster_go_by_levels_and_freq(go_annotations, n_ids, 
        percentiles, go_graph, min_annotations, True)
    
    clusters = list(go_lists.items())
    params = []
    for cluster_name, cluster_gos in clusters:
        print('Creating params for', cluster_name, 'with', len(cluster_gos), 'targets')
        params.append({
            'cluster_name': cluster_name,
            'test_go_set': cluster_gos,
            'go_annotations': go_annotations
        })
    
    with Pool(config['parsing_processes']) as pool:
        _ = pool.map(prepare_data, params)
    
    meta = MetaheuristicTest(params)
    solution_dict, fitness, report = meta.test()
    meta_report_path = experiment_dir + '.optimization.txt'
    open(meta_report_path, 'w').write(report)
    return solution_dict

if __name__ == '__main__':
    print(sys.argv)
    assert sys.argv[1] in ['all', 'test']
    assert sys.argv[2] in ['default', 'optimize']
    is_test = sys.argv[1] != 'all'
    is_optimized = sys.argv[2] != 'default'
    experiment_comment = sys.argv[3]
    timestamp = '{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.datetime.now())
    if is_test:
        experiment_name = timestamp+'_TEST_'+experiment_comment.replace(" ","-")
    else:
        experiment_name = timestamp+'_'+experiment_comment.replace(" ","-")

    experiment_dir = 'experiments/'+experiment_name
    json_path = experiment_dir+'.json'
    print('Loading train/test ids')
    traintest_ids = open('input/traintest/ids.txt', 'r').read().split('\n')
    print(len(traintest_ids), 'proteins')
    print('Loading labels')
    go_labels, go_annotations = load_labels_from_dir('input/traintest', 
        ids_allowed=traintest_ids)
    go_graph = load_go_graph()
    percentiles = [40, 70, 90]

    if is_optimized:
        solution_dict = optimize_params_for_classification(percentiles, go_graph, 
            go_annotations, len(traintest_ids), config['min_annotations'], experiment_dir)
    else:
        solution_dict = default_params

    go_lists = cluster_go_by_levels_and_freq(go_annotations, len(traintest_ids), 
        percentiles, go_graph, config['min_annotations'], is_test)
    print(go_lists.keys())

    params = {
        'go_frequency_percentiles': percentiles,
        'n_traintest_proteins': len(traintest_ids)
    }
    for key, val in config.items():
        params[key] = val
    experiment_json = {
        'id': timestamp,
        'comment': experiment_comment,
        'params': params,
        'classifiers': {},
        'go_clusters': go_lists
    }

    clusters = list(go_lists.items())
    params = []
    for cluster_name, cluster_gos in clusters:
        print('Creating params for', cluster_name, 'with', len(cluster_gos), 'targets')
        params.append({
            'cluster_name': cluster_name,
            'test_go_set': cluster_gos,
            'go_annotations': go_annotations
        })
    
    with Pool(config['parsing_processes']) as pool:
        _ = pool.map(prepare_data, params)

    for node in params:
        node['params_dict'] = solution_dict
    
    with Pool(config['training_processes']) as pool:
        models_and_stats = pool.map(train_node, params)
    
    models = {

    }
    for i in range(len(clusters)):
        cluster_name, cluster_gos = clusters[i]
        annot_model, stats = models_and_stats[i]
        
        experiment_json['classifiers'][cluster_name] = {
            'results': stats,
            'labels': cluster_gos
        }
        models[cluster_name] = (annot_model, cluster_gos)
    
    if not os.path.exists('experiments'):
        os.mkdir('experiments')

    
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
    big_table_path = predict_with_model(models, experiment_dir)
    
    json.dump(experiment_json, open(json_path, 'w'), indent=4)

    json_path_val = post_process_and_validate(json_path, big_table_path, is_test)
    run_command(['mv', json_path_val, json_path])
    
    plot_experiment(json_path)
    plot_nodes_graph(json_path)
    plot_progress()