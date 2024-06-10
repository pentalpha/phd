
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from math import ceil, floor
from util import load_labels_from_dir
from gene_ontology import load_go_graph
#%%

def go_frequencies(go_labels: dict):
    go_count = {}
    for golist in go_labels.values():
        for go in golist:
            if not go in go_count:
                go_count[go] = 0
            go_count[go] += 1

    tps = [(go, n) for go, n in go_count.items()]
    tps.sort(key= lambda tp: tp[1])

    return tps

def cluster_gos_by_freq_percentiles(go_labels: dict, n_proteins: int):
    go_freqs = go_frequencies(go_labels)
    go_freqs = [(go, freq) for go, freq in go_freqs if (freq / n_proteins) < 0.9 and freq >= 20]
    print(len(go_freqs), 'labels')
    freq_vec = [y for x,y in go_freqs]

    print('Counting percentiles')
    percentiles = [30, 60, 90, 96]
    perc_index = []
    last_index = -1
    for perc in percentiles:
        print(perc, np.percentile(freq_vec, perc))
        index = int(len(go_freqs)*(perc/100))
        perc_index.append((last_index+1, index))
        last_index = index 
    perc_index.append((last_index+1, len(go_freqs)-1))   
    print(perc_index)

    total_len = 0
    go_mega_clusters = []
    for start, end in perc_index:
        sub_gos = go_freqs[start:end+1]
        print(len(sub_gos))
        total_len += len(sub_gos)
        go_mega_clusters.append(sub_gos)
    print(total_len)

    go_sets = {}
    for index in range(len(go_mega_clusters)):
        clust = go_mega_clusters[index]
        print('Go Cluster', index, 'with', len(clust), 'GOs')
        freq_vec = [y for x,y in clust]
        print('\tMean samples:', np.mean(freq_vec), 'Min:', np.min(freq_vec), 'Max:', np.max(freq_vec))
        go_sets['Cluster_'+str(index)] = [x for x,y in clust]

    return go_sets

def cluster_go_by_levels_and_freq(go_annotations, n_proteins, percentiles, go_graph, min_annots,
        only_test_nodes):
    go_graph = load_go_graph()
    root = 'GO:0003674'
    go_levels_2 = {}
    go_n_annotations = {}
    all_goids = list(go_annotations.keys())
    valid_goids = [x for x in all_goids if x in go_graph]
    for goid in tqdm(valid_goids):
        n_annots = len(go_annotations[goid])
        if n_annots >= min_annots and goid != root:
            simple_paths = nx.all_simple_paths(go_graph, source=goid, target=root)
            simple_path_lens = [len(p) for p in simple_paths]
            try:
                mean_dist = floor(np.mean(simple_path_lens)-1)
                go_levels_2[goid] = min(7, mean_dist)
                go_n_annotations[goid] = n_annots
            except ValueError as err:
                print(simple_path_lens)
                print('No path from', goid, 'to', root)
                print(err)
                raise(err)
    
    levels = {l: [] for l in set(go_levels_2.values())}
    for goid, level in go_levels_2.items():
        levels[level].append(goid)
    
    clusters = {}

    #test_nodes = {3: [0, 2], 5: [1,2], 7: [1, 3]}
    #test_nodes = {5: [0], 6: [0, 1], 7: [0, 1]}
    test_nodes = {4: [0], 5: [0], 6: [0, 1], 7: [0, 1, 2]}
    
    clusters_to_keep = []
    for level, goids in levels.items():
        go_freqs = [(go, go_n_annotations[go]) for go in goids 
            if (go_n_annotations[go] / n_proteins) < 0.9]
        go_freqs.sort(key=lambda tp: tp[1])
        
        #print('Counting percentiles')
        perc_index = []
        last_index = -1
        for perc in percentiles:
            index = int(len(go_freqs)*(perc/100))
            perc_index.append((last_index+1, index))
            last_index = index 
        perc_index.append((last_index+1, len(go_freqs)-1))   
        #print(perc_index)
        
        total_len = 0
        last_cluster_name = None
        to_use = test_nodes[level] if level in test_nodes else []
        
        current_percentile = 0
        for start, end in perc_index:
            sub_gos = go_freqs[start:end+1]
            min_freq = sub_gos[0][1]
            max_freq = sub_gos[-1][1]
            #print(len(sub_gos))
            total_len += len(sub_gos)
            cluster_goids = [x for x, y in sub_gos]
            
            if len(sub_gos) > 2:
                cluster_name = ('Level-'+str(level)+'_Freq-'+str(min_freq)+'-'
                    +str(max_freq)+'_N-'+str(len(sub_gos)))
                if current_percentile in to_use or not only_test_nodes:
                    clusters_to_keep.append(cluster_name)
                clusters[cluster_name] = cluster_goids
                #print(cluster_name.split('_'))
            else:
                last_cluster = clusters[last_cluster_name]
                level_str, freq_str, n_str = last_cluster_name.split('_')
                _, last_min_str, _ = freq_str.split('-')
                last_min_freq = int(last_min_str)
                new_cluster = last_cluster + cluster_goids
                cluster_name = ('Level-'+str(level)+'_Freq-'+str(last_min_freq)+'-'
                    +str(max_freq)+'_N-'+str(len(new_cluster)))
                if current_percentile in to_use or not only_test_nodes:
                    clusters_to_keep.append(cluster_name)
                clusters[cluster_name] = new_cluster
                del clusters[last_cluster_name]
                #print(cluster_name.split('_'))
            last_cluster_name = cluster_name
            current_percentile += 1
        
    all_cluster_names = list(clusters.keys())
    for name in all_cluster_names:
        if not name in clusters_to_keep:
            print('Ignoring protein cluster', name)
            del clusters[name]
        else:
            print('Using protein cluster', name)
        #print(total_len)
    return clusters

if __name__ == "__main__":
    deeppred_clusters_csv = "../databases/deeppred_clusters.tsv"
    go_basic_obo = "../databases/go-basic.obo"
    deeppred_clusters = pd.read_csv(deeppred_clusters_csv, sep='\t')
    traintest_ids = open('../input/traintest/ids.txt', 'r').read().split('\n')
    go_labels, go_annotations = load_labels_from_dir('../input/traintest', ids_allowed=traintest_ids)
    go_graph = load_go_graph()
    percentiles = [50, 80]
    go_clusters = cluster_go_by_levels_and_freq(go_labels, go_annotations, len(go_labels), 
       percentiles, go_graph)
    