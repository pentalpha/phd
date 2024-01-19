from os import path
from glob import glob
import torch
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

from fasta import fasta_equal_split, remove_from_fasta
from util import config, run_command

def run_esm_extract(args: dict):
    fasta_input = args['fasta_input']
    esm_model_name = args['esm_model_name']
    esm_output_dir = args['esm_output_dir']
    #tmp_output_dir = fasta_input.rstrip('.fasta')+'_output'
    #run_command(['mkdir', tmp_output_dir])
    esm_cmd = ['python esm/scripts/extract.py', esm_model_name, 
        fasta_input, esm_output_dir, '--include mean --toks_per_batch 10000']
    run_command(esm_cmd)

def download_esm_model(model_name):
    fake_fasta_content = '>seqnameA\nMATPGASSARDEFVYMAKLAEQAERYEEMVTHPIRLGLALNFSVFYYEI\n'
    fake_fasta_path = 'fakefasta.fasta'
    open(fake_fasta_path, 'w').write(fake_fasta_content)

    download_cmd = ['python esm/scripts/extract.py', model_name, 
        fake_fasta_path, 'output_tmp', '--include mean --toks_per_batch 10000']
    run_command(download_cmd)

    run_command(['rm -Rf', fake_fasta_path, 'output_tmp'])

class ESM_Embedder():

    model_names = {
        33: 'esm2_t33_650M_UR50D',
        30: 'esm2_t30_150M_UR50D',
        12: 'esm2_t12_35M_UR50D' 
    }

    def __init__(self) -> None:

        print('Loading fair-esm embedding utility')
        self.esm_dir = config['esm_libraries']
        self.processes = config['esm_processes']
        assert path.exists(self.esm_dir), 'self.esm_dir ('+self.esm_dir+') must exist'

        self.lib_paths = {x: path.join(self.esm_dir, 'fairesm_'+str(x)) 
                          for x in ESM_Embedder.model_names.keys()}
        
        for x, p in self.lib_paths.items():
            if not path.exists(p):
                run_command(['mkdir', p])
        
        print('Libraries:', self.lib_paths)
        self.find_calculated()
    
    def find_calculated(self):
        self.calculated_embs = {x: set() for x in ESM_Embedder.model_names.keys()}
        for emb_len, calculated in self.calculated_embs.items():
            if emb_len in self.lib_paths:
                embs_dir = self.lib_paths[emb_len]
                emb_files = glob(embs_dir+'/*.pt')
                uniprotids = [path.basename(f.rstrip('.pt')) for f in emb_files]
                calculated.update(uniprotids)

                print('ESM', emb_len, ':', len(calculated), 'proteins calculated')
            else:
                print('ESM', emb_len, ': not created yet')

    def load_embedding(self, emb_len: int, uniprotid: str):
        if uniprotid in self.calculated_embs[emb_len]:
            emb_path = path.join(self.lib_paths[emb_len], uniprotid+'.pt')
            x = torch.load(emb_path)
            emb = np.array(x['mean_representations'][emb_len].tolist())
            return emb
        else:
            return np.nan
    
    def get_embeddings(self, emb_len: int, uniprotids: list, as_np=False):
        print('Loading', emb_len, 'embeddings for', len(uniprotids), 'proteins')
        embs_list = [self.load_embedding(emb_len, ID) for ID in tqdm(uniprotids)]
        if as_np:
            embs_list = np.asarray(embs_list)
        return embs_list
    
    def export_embeddings(self, emb_len: int, uniprotids: list, output_path: str):
        embs_np_obj = self.get_embeddings(emb_len, uniprotids, as_np=True)
        print('saving embeddings for', len(uniprotids), 'proteins')
        np.save(open(output_path, 'wb'), embs_np_obj)

    def calc_embeddings(self, input_fasta: str, emb_len: int):
        calculated_proteins = self.calculated_embs[emb_len]
        if len(calculated_proteins) > 0:
            print('Some embeddings have already been calculated')
            print('Removing them from the input fasta')
            to_process_fasta = self.esm_dir+'/to_calc_'+str(emb_len)+'.fasta'
            kept = remove_from_fasta(input_fasta, calculated_proteins, to_process_fasta)
            if len(kept) == 0:
                run_command(['rm', to_process_fasta])
                print('All proteins calculated')
                return
            input_fasta = to_process_fasta
        
        fasta_parts = fasta_equal_split(input_fasta, self.processes)
        for f in fasta_parts:
            print(f)

        esm_model_name = self.model_names[emb_len]
        download_esm_model(esm_model_name)
        
        esm_output_dir = self.lib_paths[emb_len]
        '''if path.exists(esm_output_dir):
            run_command(['rm -rf', esm_output_dir])'''
        esm_run_params = [{'fasta_input': fasta_part, 'esm_model_name': esm_model_name,
                        'esm_output_dir': esm_output_dir}
            for fasta_part in fasta_parts]
        
        with Pool(self.processes) as pool:
            print("Parallel processing with", self.processes, 'processes')
            pool.map(run_esm_extract, esm_run_params)

        pt_files = glob(esm_output_dir+'/*.pt')
        print(len(pt_files), 'proteins with embedding')
        uniprotids = [path.basename(f.rstrip('.pt')) for f in pt_files]
        self.calculated_embs[emb_len].update(uniprotids)

        for f in fasta_parts:
            run_command(['rm', f])