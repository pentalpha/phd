import gzip

from util import filter_fasta, run_command


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def open_file(input_path: str):
    if input_path.endswith('.gz'):
        return gzip.open(input_path, 'rt')
    else:
        return open(input_path, 'r')

if __name__ == '__main__':
    swiss_prot_fasta = "databases/uniprot_sprot.fasta.gz"
    swiss_prot_ids = []
    for rawline in open_file(swiss_prot_fasta):
        if rawline.startswith('>'):
            title_parts = rawline.rstrip('\n').split('|')
            swiss_prot_ids.append(title_parts[1])
    print(len(set(swiss_prot_ids)), len(swiss_prot_ids), 'IDs in swiss prot')
    allowed_ids = set(swiss_prot_ids)

    goa_parsed = 'databases/goa_parsed.tsv.gz'
    protein_ann = {}
    ann_dropped = 0
    anns = 0
    for rawline in open_file(goa_parsed):
        protid, goid, evi, taxid = rawline.rstrip('\n').split('\t')
        if protid in allowed_ids:
            if not goid in protein_ann:
                protein_ann[goid] = set()
            protein_ann[goid].add(protid)
        else:
            #print('Not swiss prot')
            ann_dropped += 1
        anns += 1

    print(ann_dropped, 'of', anns, 'dropped')
        
    print(len(protein_ann), 'GOs with annotation')
    frequent = [(goid,protids) for goid, protids in protein_ann.items() if len(protids) >= 10]
    print(len(frequent), 'frequent GOs')
    frequent_prots = set()
    for goid, protids in frequent:
        frequent_prots.update(protids)
    print(len(frequent_prots), 'frequent proteins')
    run_command(['mkdir', 'input'])
    proteins_for_learning_fasta = "input/proteins.fasta"
    filter_fasta(swiss_prot_fasta, frequent_prots, proteins_for_learning_fasta,
        id_pos = 1)

    '''frequent_prots = list(frequent_prots)
    frequent_prots.sort()
    print(len(frequent_prots), 'frequent proteins')
    output_path = 'databases/uniprot_ids'
    output = open(output_path+'.txt', 'w')
    for protid in frequent_prots:
            output.write(protid+'\n')'''