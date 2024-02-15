from glob import glob
from os import path
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
#from skmultilearn.model_selection import iterative_train_test_split

from util import (load_features, load_labels_from_dir, open_file, 
    config, run_command, write_file)

from util import (features_taxon_path, input_features_ids_path, features_taxon_prefix,
                  features_esm_base_path)

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
    print(len(proteins_by_taxid.keys()), 'taxa')
    sorted_taxons = sorted(list(proteins_by_taxid.keys()), key=lambda taxid: len(proteins_by_taxid[taxid]), reverse=True)
    frequent_taxa = sorted_taxons[:config['max_taxons']]
    unfrequent_taxa = [key for key, prots in proteins_by_taxid.items() if not key in frequent_taxa]
    proteins_unfrequent = set()
    for unfrequent_taxon in unfrequent_taxa:
        proteins_unfrequent.update([prot+'\t'+unfrequent_taxon for prot in proteins_by_taxid[unfrequent_taxon]])
        #del proteins_by_taxid[unfrequent_taxon]
    #proteins_by_taxid['UNFREQUENT'] = sorted(proteins_unfrequent)
    #print(len(proteins_by_taxid.keys()), 'taxa')
    #sorted_taxons = sorted(list(proteins_by_taxid.keys()), key=lambda taxid: len(proteins_by_taxid[taxid]), reverse=False)

    protid_lists = [list(proteins_unfrequent)]
    for taxonid in tqdm(frequent_taxa):
        protids = [x+'\t'+taxonid for x in proteins_by_taxid[taxonid]]
        protid_lists.append(protids)
    
    for protids in tqdm(protid_lists):
        all_proteins.update(protids)
        n_validation = int(len(protids)*config['validation_perc'])
        print('\n', len(protids), n_validation, n_validation/len(protids))
        
        if n_validation > 0:
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
        taxon_file = directory+'/'+features_taxon_prefix
        
        open(ids_file, 'w').write('\n'.join(protein_list))
        
        print('Writing labels')
        local_labels, annotations = load_labels_from_dir('input', ids_allowed=protein_list)
        label_output = open(labels_file, 'w')
        for protein in protein_list:
            labellist = local_labels[protein]
            label_output.write(protein+'\t'+','.join(labellist)+'\n')
        label_output.close()

        print('Writing taxa')
        taxon_features = load_features(features_taxon_path, protein_list, int)
        taxon_output = write_file(taxon_file)
        for i in tqdm(range(len(taxon_features))):
            taxa_onehot = taxon_features[i]
            line = '\t'.join([str(x) for x in taxon_features])
            startline = '\n' if i > 0 else ''
            taxon_output.write(startline+line)
        taxon_output.close()

        print('Writing ESM')
        esm_feature_paths = glob(features_esm_base_path)
        for esm_feature_path in esm_feature_paths:
            basename = path.basename(esm_feature_path)
            local_esm_path = directory+'/'+basename
            features = load_features(esm_feature_path, protein_list, float)
            esm_output = write_file(local_esm_path)
            for i in tqdm(range(len(features))):
                f = features[i]
                line = '\t'.join([str(x) for x in f])
                startline = '\n' if i > 0 else ''
                esm_output.write(startline+line)
            esm_output.close()
