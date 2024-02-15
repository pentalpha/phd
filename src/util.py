import glob
import gzip
from os import path
import subprocess
from tqdm import tqdm
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
input_features_ids_traintest_path = 'input/ids_traintest.txt'
input_features_ids_validation_path = 'input/ids_validation.txt'
input_labels_path = 'input/labels.tsv'
taxon_features = 'input/features/taxon_one_hot.tsv.gz'
taxon_features_ids = 'input/features/taxon_one_hot_ids.txt'
fairesm_features = 'input/features/fairesm_*'

features_esm_prefix = 'feature_esm_*.tsv.gz'
features_taxon_prefix = 'feature_taxa.tsv.gz'
features_taxon_path = 'input/'+features_taxon_prefix
features_esm_base_path = 'input/'+features_esm_prefix

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
        vec += v
    return np.array(vec)

def concat_vecs_np(vecs: list):
    vec = []
    for v in vecs:
        vec += v.tolist()
    return np.array(vec)

def load_label_lists(uniprotids: list):
    label_lists = {}
    for rawline in open_file(input_labels_path):
        uniprotid, taxid, gos = rawline.rstrip('\n').split('\t')
        if uniprotid in uniprotids:
            label_lists[uniprotid] = gos.split(',')
    label_lists2 = [label_lists[uniprotid] for uniprotid in uniprotids]
    return label_lists2

def label_lists_to_onehot(label_lists: list):
    all_labels = set()
    for label_list in label_lists:
        all_labels.update(label_list)
    all_labels = sorted(list(all_labels))
    label_pos = {all_labels[pos]: pos for pos in range(len(all_labels))}

    n_labels = len(all_labels)
    print(n_labels, 'GO ids')
    one_hot = []
    for label_list in label_lists:
        vec = [0]*n_labels
        for go in label_list:
            vec[label_pos[go]] = 1
        one_hot.append(np.array(vec))

    return np.asarray(one_hot)

def load_features_from_dir(dirname: str, ids_allowed: list = []):
    features_path = dirname+'/features.npy'
    ids_path = dirname+'/ids.txt'

    features = np.load(features_path)
    ids = open(ids_path, 'r').read().split('\n')

    protein_indexes = {ids[i]: i for i in range(len(ids))}

    new_index = {}
    local_features = []
    i = 0
    print('Selecting features of proteins')
    for protein in tqdm(ids_allowed):
        feature_index = protein_indexes[protein]
        feature_vec = features[feature_index]
        local_features.append(feature_vec)
        new_index[protein] = i
        i+= 1
    print('Converting')
    local_features = np.asarray(local_features)

    return local_features, new_index

def load_features(feature_file_path: str, subset: list, converter):
    ids_path = path.dirname(feature_file_path)+'/ids.txt'

    ids = open(ids_path, 'r').read().split('\n')
    feature_file = open_file(feature_file_path)
    features = []
    id_to_line = {}
    line_i = 0
    for protid in subset:
        features.append(None)
        id_to_line[protid] = line_i
        line_i += 1

    line_i = 0
    while line_i < len(ids):
        current_line = feature_file.readline()
        current_id = ids[line_i]
        if current_id in subset:
            cols = current_line.rstrip('\n').split('\t')
            vals = [converter(x) for x in cols]
            correct_line = id_to_line[current_id]
            features[correct_line] = vals
        line_i += 1
    assert not (None in features)
    return features

def load_labels_from_dir(dirname: str, ids_allowed: list = []):
    labels_path = dirname+'/labels.tsv'

    labels = {}
    anns = {}
    for rawline in open(labels_path, 'r'):
        uniprotid, taxonid, gos = rawline.rstrip('\n').split('\t')
        protid = uniprotid+'\t'+taxonid
        if protid in ids_allowed:
            go_list = gos.split(',')
            labels[protid] = go_list
            for go in go_list:
                if not go in anns:
                    anns[go] = set()
                anns[go].add(protid)


    return labels, anns

def create_labels_matrix(labels: dict, ids_allowed: list, gos_allowed: list):
    label_vecs = []
    for protid in ids_allowed:
        gos_in_prot = labels[protid]
        one_hot_labels = [1 if go in gos_in_prot else 0 for go in gos_allowed]
        label_vecs.append(np.array(one_hot_labels))
    
    return np.asarray(label_vecs)


def load_dataset_from_dir(dirname: str, subset: list = []):
    if len(subset) == 0:
        ids_path = dirname+'/ids.txt'
        subset = open(ids_path, 'r').read().split('\n')
    labels, annotations = load_labels_from_dir(dirname, ids_allowed=subset)

    taxa_path = dirname+'/'+features_taxon_prefix
    taxa_features = load_features(taxa_path, subset, int)

    esm_paths = glob(dirname+'/'+features_esm_prefix)
    esms = []
    for esm_path in esm_paths:
        esm_len_str = esm_path.split('.')[-3].split('_')[-1]
        esm_len = int(esm_len_str)
        esm_features = load_features(esm_path, subset, float)
        esms.append(esm_len, esm_features)
    
    return taxa_features, esms, labels, annotations