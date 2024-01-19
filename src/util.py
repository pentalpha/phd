import gzip
import subprocess
import yaml
import numpy as np

config = yaml.safe_load(open("config.yml", "r"))
goa_parsed = 'databases/goa_parsed.tsv.gz'
goa_parsed_expanded = 'databases/goa_parsed_expanded.tsv.gz'
goa_parsed_frequent = 'databases/goa_parsed_frequent.tsv.gz'
go_not_use_path = 'databases/gocheck_do_not_annotate.json'
go_basic_path = "databases/go-basic.obo"

#quickgo_expanded_path = "input/quickgo_expanded.tsv.gz"
input_annotation_path = 'input/annotation.tsv'
input_features_path = 'input/features.npy'
input_features_ids_path = 'input/ids.txt'
input_labels_path = 'input/labels.tsv'
taxon_features = 'input/features/taxon_one_hot.npy'
taxon_features_ids = 'input/features/taxon_one_hot_ids.txt'
fairesm_features = 'input/features/fairesm_*'

def run_command(cmd_vec, stdin="", no_output=True):
    '''Executa um comando no shell e retorna a saída (stdout) dele.'''
    cmd_vec = " ".join(cmd_vec)
    #logging.info(cmd_vec)
    if no_output:
        #print(cmd_vec)
        result = subprocess.run(cmd_vec, shell=True)
        return result.returncode
    else:
        result = subprocess.run(cmd_vec, capture_output=True, 
            text=True, input=stdin, shell=True)
        return result.stdout

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def open_file(input_path: str):
    if input_path.endswith('.gz'):
        return gzip.open(input_path, 'rt')
    else:
        return open(input_path, 'r')
    
def count_lines(input_path: str):
    stream = open_file(input_path)
    n = 0
    for line in stream:
        n += 1
    return n
    
def write_file(input_path: str):
    if input_path.endswith('.gz'):
        return gzip.open(input_path, 'wt')
    else:
        return open(input_path, 'w')
    
def load_parsed_goa(file_path=goa_parsed):
    anns = []
    for rawline in open_file(file_path):
        protid, goid, evi, taxid = rawline.rstrip('\n').split('\t')
        if goid.startswith('GO') and taxid.startswith('tax'):
            anns.append([protid, goid, evi, taxid])

    return anns

def write_parsed_goa(annotations, file_path):
    output = write_file(file_path)
    for cells in annotations:
        line = '\t'.join(cells)
        output.write(line+'\n')

def concat_vecs(vecs: list):
    vec = []
    for v in vecs:
        vec += v.tolist()
    return np.array(vec)