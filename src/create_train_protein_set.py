from collections import Counter
import gzip
from fasta import filter_fasta

from util import (load_parsed_goa, run_command, config, write_parsed_goa,
    input_annotation_path, goa_parsed_frequent)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def open_file(input_path: str):
    if input_path.endswith('.gz'):
        return gzip.open(input_path, 'rt')
    else:
        return open(input_path, 'r')

def create_filtered_annotation_and_fasta(allowed_uniprot):
    
    protein_ann = {}
    ann_dropped = 0
    anns = 0
    all_annotations = load_parsed_goa(file_path=goa_parsed_frequent)
    all_proteins = set([protid for protid, goid, evi, taxid in all_annotations])
    print(len(all_proteins), 'proteins in', goa_parsed_frequent)
    all_goids = set([goid for protid, goid, evi, taxid in all_annotations])
    print(len(all_goids), 'GO IDs in', goa_parsed_frequent)

    all_annotations = [x for x in all_annotations if x[0] in allowed_uniprot]
    all_proteins = set([protid for protid, goid, evi, taxid in all_annotations])
    print(len(all_proteins), 'uniprot proteins in', goa_parsed_frequent)

    goid_list = [goid for protid, goid, evi, taxid in all_annotations]
    print(len(set(goid_list)), 'GO IDs from uniprot proteins in', goa_parsed_frequent)
    go_counts = Counter(goid_list)
    frequent_go_ids = set()
    for goid, freq in go_counts.items():
        if freq >= config['min_annotations']:
            frequent_go_ids.add(goid)
    print(len(frequent_go_ids), 'frequent GO IDs from uniprot proteins in', goa_parsed_frequent)

    all_annotations = [x for x in all_annotations if x[1] in frequent_go_ids]
    frequent_prots = set([protid for protid, goid, evi, taxid in all_annotations])
    print(len(frequent_prots), 'uniprot proteins with frequent GOs in', goa_parsed_frequent)

    run_command(['mkdir', 'input'])

    proteins_for_learning_fasta = "input/proteins.fasta"
    filter_fasta(swiss_prot_fasta, frequent_prots, proteins_for_learning_fasta,
        id_pos = 1)

    print(len(all_annotations), 'annotations kept for training')
    write_parsed_goa(all_annotations, input_annotation_path)

if __name__ == '__main__':
    swiss_prot_fasta = "databases/uniprot_sprot.fasta.gz"
    swiss_prot_ids = []
    for rawline in open_file(swiss_prot_fasta):
        if rawline.startswith('>'):
            title_parts = rawline.rstrip('\n').split('|')
            swiss_prot_ids.append(title_parts[1])
    print(len(set(swiss_prot_ids)), len(swiss_prot_ids), 'IDs in swiss prot')
    allowed_ids = set(swiss_prot_ids)
    
    create_filtered_annotation_and_fasta(allowed_ids)

    '''frequent_prots = list(frequent_prots)
    frequent_prots.sort()
    print(len(frequent_prots), 'frequent proteins')
    output_path = 'databases/uniprot_ids'
    output = open(output_path+'.txt', 'w')
    for protid in frequent_prots:
            output.write(protid+'\n')'''