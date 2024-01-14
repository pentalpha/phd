from collections import Counter
from multiprocessing import Pool
import sys
from os import path
import torch
from glob import glob
import numpy as np

from util import (load_parsed_goa, run_command, 
                  input_annotation_path, config, 
                  taxon_features, taxon_features_ids,
                  fairesm_features)
from fasta import fasta_equal_split, ids_from_fasta

from esm import ESM_Embedder

def calc_fairesm(fasta_input, output_dir):
    embedder = ESM_Embedder()
    uniprot_ids, _ = ids_from_fasta(fasta_input)
    for emb_len in config['esm_models_to_use']:
        print('Calculating ESM', emb_len, 'embeddings')
        features_path = fairesm_features.replace('*', str(emb_len)+'.npy')
        embedder.calc_embeddings(fasta_input, emb_len)
        embedder.export_embeddings(emb_len, uniprot_ids, features_path)
        open(features_path.replace('.npy', '_ids.txt'), 'w').write('\n'.join(uniprot_ids))

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
    print(len(taxon_counts), 'taxons after filtering by most frequent')
    taxids = [taxid for taxid, count in taxon_counts]
    one_hot = [np.array([1 if taxids[i] == current_tax else 0 
            for i in range(len(taxids))])
        for current_tax in taxids]
    one_hot = np.asarray(one_hot)
    print(one_hot.shape)
    print(one_hot)
    np.save(open(taxon_features, 'wb'), one_hot)
    open(taxon_features_ids, 'w').write('\n'.join(taxids))
    
if __name__ == '__main__':
    fasta_input = sys.argv[1]
    output_dir = sys.argv[2]
    if not path.exists(output_dir):
        run_command(['mkdir', output_dir])
    calc_taxon_features(output_dir)
    calc_fairesm(fasta_input, output_dir)
    