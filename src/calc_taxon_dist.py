from collections import Counter
from os import path
import sys
import taxoniq
from tqdm import tqdm

from util import (load_parsed_goa, input_annotation_path, config, open_file, 
                taxon_profiles, taxon_profiles_ids, write_file,
                run_command, input_features_ids_path, features_taxon_profile_path)

def get_lineage(taxid):
    ncbi_taxon = taxoniq.Taxon(int(taxid.lstrip('taxon:')))
    lineage = []
    for tax in ncbi_taxon.lineage:
        str_id = 'taxon:'+str(tax.tax_id)
        lineage.append(str_id)
    return lineage

def lineage_closeness(lineage1: list, lineage2: list) -> float:
    last_common_ancestor = len(lineage1)-1
    for lineage1_i in range(len(lineage1)):
        taxid_i = lineage1[lineage1_i]
        if taxid_i in lineage2:
            last_common_ancestor = lineage1_i
            break
    
    normalized_distance = last_common_ancestor / len(lineage1)
    return 1.0 - normalized_distance

def taxon_closeness(taxid: str, other_taxid: str) -> float:
    lineage1 = get_lineage(taxid)
    lineage2 = get_lineage(other_taxid)

    return lineage_closeness(lineage1, lineage2)

def calc_taxon_dist_features(min_taxon_freq = 50):
    annotations = load_parsed_goa(input_annotation_path)
    print(len(annotations), 'annotations loaded')
    taxons = []
    for cells in annotations:
        for taxonid_new in cells[-1].split('|'):
            taxons.append(taxonid_new)
    taxon_counts = [(taxid, count) for taxid, count  in Counter(taxons).items()]
    taxon_counts.sort(key=lambda tp: tp[1], reverse=True)
    
    frequent = [taxid for taxid, count in taxon_counts[:config['max_taxons']]]
    unfrequent = [taxid for taxid, count in taxon_counts if not taxid in frequent]
    
    print(len(frequent), 'frequent taxons')
    print(len(unfrequent), 'unfrequent taxons')

    taxids = frequent + unfrequent
    closeness_vecs = []
    taxon_lineages = {taxid: get_lineage(taxid) for taxid in frequent + unfrequent}
    sapiens_representation = []
    for taxid in tqdm(taxids):
        closeness = [lineage_closeness(taxon_lineages[taxid], taxon_lineages[other_taxid])
                for other_taxid in frequent]
        closeness_vecs.append(closeness)
        if taxid == taxids[0]:
            for other_taxid in frequent:
                try:
                    name = taxoniq.Taxon(int(other_taxid.lstrip('taxon:'))).common_name
                except Exception as err:
                    name = taxoniq.Taxon(int(other_taxid.lstrip('taxon:'))).scientific_name
                sapiens_representation.append((name, lineage_closeness(taxon_lineages[taxid], taxon_lineages[other_taxid])))
    sapiens_representation.sort(key=lambda tp: tp[1], reverse=True)
    for n, c in sapiens_representation:
        print(n+':', c)
    closeness_dict = {taxids[i]: closeness_vecs[i] for i in range(len(taxids))}

    onehot_lines = ['\t'.join([str(x) for x in closeness_vec]) for closeness_vec in closeness_vecs]
    write_file(taxon_profiles).write('\n'.join(onehot_lines))
    open(taxon_profiles_ids, 'w').write('\n'.join(taxids))

    complete_taxonid_list = [line.split('\t')[1] 
        for line in open_file(input_features_ids_path).read().split('\n')]
    
    closeness_output = write_file(features_taxon_profile_path)
    for taxid in tqdm(complete_taxonid_list):
        line = [str(x) for x in closeness_dict[taxid]]
        line = '\t'.join(line)
        closeness_output.write(line+'\n')
    closeness_output.close()

    return closeness_dict

if __name__ == '__main__':
    calc_taxon_dist_features()

    print('Loading taxon features')
    taxon_features_list = [[float(x_raw) for x_raw in rawline.split('\t')] 
        for rawline in open_file(taxon_profiles).read().split('\n')]
    taxon_features_ids_list = open(taxon_profiles_ids, 'r').read().split('\n')
    assert len(taxon_features_list) == len(taxon_features_ids_list)
    taxons = {taxon_features_ids_list[i]: taxon_features_list[i] 
        for i in range(len(taxon_features_ids_list))}