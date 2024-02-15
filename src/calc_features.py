from collections import Counter
from multiprocessing import Pool
from pickle import load
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
                  fairesm_features, features_taxon_path,
                  features_esm_base_path,
                  write_file)
from fasta import fasta_equal_split, ids_from_fasta

from esm import ESM_Embedder

def calc_fairesm(fasta_input, output_dir):
    #embedder = ESM_Embedder()
    uniprot_ids, _ = ids_from_fasta(fasta_input)
    feature_dfs = []
    print('Q6UY62 is in uniprot_ids', 'Q6UY62' in uniprot_ids)
    for emb_len in config['esm_models_to_use']:
        print('Calculating ESM', emb_len, 'embeddings')
        features_path = fairesm_features.replace('*', str(emb_len)+'.tsv.gz')
        features_ids_path = features_path.replace('.tsv.gz', '_ids.txt')
        '''embedder.calc_embeddings(fasta_input, emb_len)
        embedder.find_calculated()
        embedder.export_embeddings(emb_len, 
            uniprot_ids, features_path)'''
        
        feature_dfs.append((features_ids_path, features_path))
    
    return feature_dfs

def taxon_to_onehot(taxid, all_taxids):
    return [1 if all_taxids[i] == taxid else 0
        for i in range(len(all_taxids))]

def calc_taxon_features(output_dir, min_taxon_freq = 50):
    annotations = load_parsed_goa(input_annotation_path)
    print(len(annotations), 'annotations loaded')
    taxons = [cells[-1] for cells in annotations]
    taxon_counts = [(taxid, count) for taxid, count  in Counter(taxons).items()]
    taxon_counts.sort(key=lambda tp: tp[1], reverse=True)
    frequent = [taxid for taxid, count in taxon_counts[:config['max_taxons']]]
    unfrequent = [taxid for taxid, count in taxon_counts if not taxid in frequent]
    
    taxids = []
    one_hot = []
    for taxid in frequent + unfrequent:
        taxids.append(taxid)
        one_hot.append(taxon_to_onehot(taxid, frequent))
    
    print(len(frequent), 'frequent taxons')
    print(len(unfrequent), 'unfrequent taxons')
    
    onehot_lines = ['\t'.join([str(x) for x in onehot_line]) for onehot_line in one_hot]
    write_file(taxon_features).write('\n'.join(onehot_lines))
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
    taxon_features_list = [[int(x_raw) for x_raw in rawline.split('\t')] 
        for rawline in open_file(taxon_features).read().split('\n')]
    taxon_features_ids_list = open(taxon_features_ids, 'r').read().split('\n')
    assert len(taxon_features_list) == len(taxon_features_ids_list)
    taxons = {taxon_features_ids_list[i]: taxon_features_list[i] 
        for i in range(len(taxon_features_ids_list))}
    
    print('Loading esm features')
    esms = []
    for esm_ids_path, esm_path in esm_feature_dfs:
        features_list = []
        print('Opening', esm_path)
        for rawline in open_file(esm_path):
            features_list.append(
                [float(x) for x in rawline.rstrip('\n').split('\t')])
        prot_ids_list = open(esm_ids_path, 'r').read().split('\n')
        assert len(features_list) == len(prot_ids_list)
        esm_features_dict = {prot_ids_list[i]: features_list[i] 
            for i in range(len(prot_ids_list)) if features_list[i]}
        esm_len = int(path.basename(esm_path).rstrip('.tsv.gz').split('_')[-1])
        esms.append((esm_len, esm_features_dict))
    
    print('Creating features')
    
    no_taxid = 0
    no_esm = {esm_len: 0 for esm_len, esm_features_dict in esms}
    taxon_features_file = write_file(features_taxon_path)
    esm_paths = {esm_len: features_esm_base_path.replace('*',str(esm_len)) 
        for esm_len, _ in esms}
    esm_files = {esm_len: write_file(esm_path)
        for esm_len, esm_path in esm_paths.items()}

    ids_file = open(input_features_ids_path, 'w')
    n_prots = 0
    for protid, taxid in tqdm(proteins_and_taxid):
        feature_vec = []

        all_esms = True
        for esm_len, esm_features_dict in esms:
            if not protid in esm_features_dict:
                no_esm[esm_len] += 1
                all_esms = False
        in_taxa = taxid in taxons
        if not in_taxa:
            no_taxid += 1
        
        if all_esms and in_taxa:
            startline = '\n' if n_prots > 0 else ''
            ids_file.write(startline+protid+'\t'+taxid)
            for esm_len, esm_features_dict in esms:
                esm_vec = [str(x) for x in esm_features_dict[protid]]
                esm_line = startline+'\t'.join(esm_vec)
                esm_files[esm_len].write(esm_line)
            taxa_line = startline+'\t'.join([str(x) for x in taxons[taxid]])
            taxon_features_file.write(taxa_line)
            n_prots += 1
    ids_file.close()
    for esm_len, esm_file in esm_files.items():
        esm_file.close()
    taxon_features_file.close()
    
    print(n_prots, 'proteins')
    print(no_taxid, 'removed because of unfrequent taxon')
    print(no_esm, 'removed because of no ESM embedding taxon')

    return input_features_ids_path

if __name__ == '__main__':
    fasta_input = sys.argv[1]
    output_dir = sys.argv[2]
    if not path.exists(output_dir):
        run_command(['mkdir', output_dir])
    calc_taxon_features(output_dir)
    esm_feature_dfs = calc_fairesm(fasta_input, output_dir)
    make_features(esm_feature_dfs)
    