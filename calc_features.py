import sys
from os import path
import torch
from glob import glob
import numpy as np

from util import run_command

if __name__ == '__main__':
    fasta_input = sys.argv[1]
    output_dir = sys.argv[2]
    if not path.exists(output_dir):
        run_command(['mkdir', output_dir])
    
    esmmodels = [(12, 'esm2_t12_35M_UR50D'),
        (30, 'esm2_t30_150M_UR50D')]

    for esm_model_n, esm_model_name in esmmodels:
        esm_output_dir = path.join(output_dir, 'fairesm_'+str(esm_model_n))
        '''if path.exists(esm_output_dir):
            run_command(['rm -rf', esm_output_dir])
        esm_cmd = ['python esm/scripts/extract.py', esm_model_name, 
            fasta_input, esm_output_dir, '--include mean --toks_per_batch 10000']
        run_command(esm_cmd)'''

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