import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
#from skmultilearn.model_selection import iterative_train_test_split

from util import (input_features_ids_path, 
    input_labels_path, label_lists_to_onehot, load_features_from_dir, load_label_lists, load_labels_from_dir, open_file, config,
    input_features_ids_validation_path, input_features_ids_traintest_path, run_command)

if __name__ == '__main__':
    proteins_by_taxid = {}
    all_proteins = set()
    for rawline in open_file(input_features_ids_path):
        protid, taxid = rawline.rstrip('\n').split('\t')
        if not taxid in proteins_by_taxid:
            proteins_by_taxid[taxid] = []
        proteins_by_taxid[taxid].append(protid)
        

    taxon_sizes = np.array([len(v) for k, v in proteins_by_taxid.items()])
    print(min(taxon_sizes))
    print(max(taxon_sizes))
    print(np.percentile(taxon_sizes, 10))
    print(np.percentile(taxon_sizes, 25))
    print(np.mean(taxon_sizes))
    print(np.percentile(taxon_sizes, 75))
    print(np.percentile(taxon_sizes, 90))

    validation_set = []
    sorted_taxons = sorted(list(proteins_by_taxid.keys()), key=lambda taxid: len(proteins_by_taxid[taxid]))
    for taxonid in tqdm(sorted_taxons):
        protids = [x+'\t'+taxonid for x in proteins_by_taxid[taxonid]]
        all_proteins.update(protids)
        n_validation = int(len(protids)*config['validation_perc'])
        print('\n', taxonid, len(protids), n_validation, n_validation/len(protids))

        #label_lists = load_label_lists(protids)
        #print('encoding labels')
        #encoding = label_lists_to_onehot(label_lists)

        print('\t', 'spliting')
        ids_train, ids_validation = train_test_split(
            protids, test_size = config['validation_perc'])
        #ids_validation = ids_validation.tolist()
        #ids_validation = [x[0] for x in ids_validation]
        validation_set += ids_validation
        print('\t', len(ids_train), len(ids_validation))
        print('\t', len(ids_validation), len(ids_validation) / (len(ids_validation) + len(ids_train)))
        print('\t', len(validation_set), 'proteins for validation')
        print('\t', len(validation_set) / (len(all_proteins) + len(validation_set)))
    
    print('Sorting')
    #print(len(validation_set), 'proteins for validation')
    validation_list = sorted(validation_set)
    #open(input_features_ids_validation_path, 'w').write('\n'.join(validation_list))
    traintest_set = [x for x in all_proteins if not x in validation_set]
    #print(len(traintest_set), 'proteins for train/test')
    train_test_list = sorted(traintest_set)
    #open(input_features_ids_traintest_path, 'w').write('\n'.join(train_test_list))
    print(len(validation_set) / (len(traintest_set) + len(validation_set)))

    #all_ids, all_features, index = load_features_from_dir('input')

    protein_sets = [(validation_list, 'validation'), (train_test_list, 'traintest')]
    for protein_list, setname in protein_sets:
        print(len(protein_list), 'proteins for', setname)
        directory = 'input/'+setname
        run_command(['mkdir', directory])
        ids_file = directory+'/ids.txt'
        labels_file = directory+'/labels.tsv'
        features_file = directory+'/features.npy'
        
        open(ids_file, 'w').write('\n'.join(protein_list))
        local_features, local_index = load_features_from_dir('input', ids_allowed=protein_list)
        np.save(features_file, local_features)
        print('Writing labels')
        local_labels = load_labels_from_dir('input', ids_allowed=protein_list)
        label_output = open(labels_file, 'w')
        for protein in protein_list:
            labellist = local_labels[protein]
            label_output.write(protein+'\t'+','.join(labellist)+'\n')
        label_output.close()