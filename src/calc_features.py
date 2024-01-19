from collections import Counter
from multiprocessing import Pool
import sys
from os import path
import torch
from glob import glob
import numpy as np
from tqdm import tqdm

from util import (load_parsed_goa, open_file, run_command,
                  concat_vecs, 
                  input_features_path, input_features_ids_path, 
                  input_annotation_path, config, 
                  taxon_features, taxon_features_ids,
                  fairesm_features)
from fasta import fasta_equal_split, ids_from_fasta

from esm import ESM_Embedder

def calc_fairesm(fasta_input, output_dir):
    embedder = ESM_Embedder()
    uniprot_ids, _ = ids_from_fasta(fasta_input)
    feature_dfs = []
    for emb_len in config['esm_models_to_use']:
        print('Calculating ESM', emb_len, 'embeddings')
        features_path = fairesm_features.replace('*', str(emb_len)+'.npy')
        embedder.calc_embeddings(fasta_input, emb_len)
        embedder.find_calculated()
        embedder.export_embeddings(emb_len, uniprot_ids, features_path)
        features_ids_path = features_path.replace('.npy', '_ids.txt')
        open(features_ids_path, 'w').write('\n'.join(uniprot_ids))
        feature_dfs.append((features_ids_path, features_path))
    
    return feature_dfs

def calc_taxon_features(output_dir):
    annotations = load_parsed_goa(input_annotation_path)
    print(len(annotations), 'annotations loaded')
    taxons = [cells[-1] for cells in annotations]
    taxon_counts = [(taxid, count) for taxid, count  in Counter(taxons).items()]
    taxon_counts.sort(key=lambda tp: tp[1], reverse=True)
    print(len(taxon_counts), 'taxons')
    '''for t, c in taxon_counts:
        print(t,c)'''
    taxon_counts = taxon_counts[:config['max_taxons']]
    print(taxon_counts[-1])
    print(len(taxon_counts), 'taxons after filtering by most frequent')
    taxids = [taxid for taxid, count in taxon_counts]
    one_hot = [np.array([1.0 if taxids[i] == current_tax else 0.0 
            for i in range(len(taxids))])
        for current_tax in taxids]
    one_hot = np.asarray(one_hot)
    print(one_hot.shape)
    print(one_hot)
    np.save(open(taxon_features, 'wb'), one_hot)
    open(taxon_features_ids, 'w').write('\n'.join(taxids))

    return taxon_features_ids, taxon_features
    
def make_features(esm_feature_dfs):
    print('Loading proteins')
    proteins_and_taxid = set()
    for rawline in open_file(input_annotation_path):
        cells = rawline.rstrip('\n').split('\t')
        proteins_and_taxid.add((cells[0], cells[3]))
    proteins_and_taxid = list(proteins_and_taxid)
    proteins_and_taxid.sort(key=lambda tp: tp[0])

    print('Loading taxon features')
    taxon_features_list = np.load(taxon_features)
    taxon_features_ids_list = open(taxon_features_ids, 'r').read().split('\n')
    assert len(taxon_features_list) == len(taxon_features_ids_list)
    taxons = {taxon_features_ids_list[i]: taxon_features_list[i] 
        for i in range(len(taxon_features_ids_list))}
    
    print('Loading esm features')
    esms = []
    for esm_ids_path, esm_path in esm_feature_dfs:
        features_list = np.load(esm_path)
        prot_ids_list = open(esm_ids_path, 'r').read().split('\n')
        assert len(features_list) == len(prot_ids_list)
        esm_features_dict = {prot_ids_list[i]: features_list[i] 
            for i in range(len(prot_ids_list))}
        esm_len = int(path.basename(esm_path).rstrip('.npy').split('_')[-1])
        esms.append((esm_len, esm_features_dict))
    
    print('Creating features')
    all_features = []
    no_taxid = 0
    no_esm = {esm_len: 0 for esm_len, esm_features_dict in esms}
    for protid, taxid in tqdm(proteins_and_taxid):
        feature_vec = []

        all_data = True

        if taxid in taxons:
            feature_vec.append(taxons[taxid])
        else:
            feature_vec.append(None)
            no_taxid += 1
            all_data = False
        
        for esm_len, esm_features_dict in esms:
            if protid in esm_features_dict:
                feature_vec.append(esm_features_dict[protid])
            else:
                feature_vec.append(None)
                no_esm[esm_len] += 1
                all_data = False
        
        if all_data:
            all_features.append(feature_vec)
        else:
            all_features.append([])
    
    n_prots = len(proteins_and_taxid)
    print(n_prots, 'proteins')
    print(no_taxid, 'removed because of unfrequent taxon')
    print(no_esm, 'removed because of no ESM embedding taxon')

    proteins_and_taxid = [proteins_and_taxid[i] for i in range(n_prots)
        if len(all_features[i]) > 0]
    all_features = [concat_vecs(all_features[i]) for i in range(n_prots)
        if len(all_features[i]) > 0]
    assert len(proteins_and_taxid) == len(all_features), (str(len(proteins_and_taxid)) 
        + " " + str(len(all_features)))
    
    all_features = np.asarray(all_features)
    np.save(input_features_path, all_features)
    open(input_features_ids_path, 'w').write('\n'.join(
        [seq+'\t'+t for seq, t in proteins_and_taxid]))

    return proteins_and_taxid, all_features


if __name__ == '__main__':
    fasta_input = sys.argv[1]
    output_dir = sys.argv[2]
    if not path.exists(output_dir):
        run_command(['mkdir', output_dir])
    calc_taxon_features(output_dir)
    esm_feature_dfs = calc_fairesm(fasta_input, output_dir)
    make_features(esm_feature_dfs)
    