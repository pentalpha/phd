from collections import Counter
import sys
from os import path
from fasta import ids_from_fasta
import torch
from glob import glob
import numpy as np

from util import (load_parsed_goa, run_command, 
                  input_annotation_path, config, 
                  taxon_features, taxon_features_ids)


def calc_fairesm(fasta_input, output_dir):
    esmmodels = [(12, 'esm2_t12_35M_UR50D'),
        (30, 'esm2_t30_150M_UR50D')]

    for esm_model_n, esm_model_name in esmmodels:
        esm_output_dir = path.join(output_dir, 'fairesm_'+str(esm_model_n))
        if path.exists(esm_output_dir):
            run_command(['rm -rf', esm_output_dir])
        esm_cmd = ['python esm/scripts/extract.py', esm_model_name, 
            fasta_input, esm_output_dir, '--include mean --toks_per_batch 10000']
        run_command(esm_cmd)

        pt_files = glob(esm_output_dir+'/*.pt')
        print(len(pt_files), 'proteins with embedding')
        esm_features = []
        esm_protids = []
        for file_path in pt_files:
            x = torch.load(file_path)
            esm_protids.append(x['label'])
            esm_features.append(np.array(x['mean_representations'][esm_model_n].tolist()))
            #print(esm_features[-1])
            #print(esm_protids[-1])

        esm_features = np.asarray(esm_features)
        np.save(open(output_dir+'/esm_'+str(esm_model_n)+'.npy', 'wb'), esm_features)
        open(output_dir+'/protein_ids_'+str(esm_model_n)+'.txt', 'w').write('\n'.join(esm_protids))

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
    
    