from util import open_file, write_file


def filter_fasta(fasta_path, allowed_ids, output_path, id_pos = 0):
    keep = []

    last_title = None
    current_seq = ''
    for rawline in open_file(fasta_path):
        if rawline.startswith('>'):
            if last_title:
                keep.append((last_title, current_seq))
                current_seq = ""
            title_parts = rawline.rstrip('\n').split('|')
            current_id = title_parts[id_pos]
            if current_id in allowed_ids:
                last_title = current_id
            else:
                last_title = None
        else:
            if last_title:
                current_seq += rawline.rstrip('\n')
    
    output = write_file(output_path)
    for name, seq in keep:
        output.write('>'+name+'\n')
        output.write(seq+'\n')

def ids_from_fasta(fasta_path):
    ids = []
    taxons = []
    for rawline in open_file(fasta_path):
        if rawline.startswith('>'):
            title_parts = rawline.rstrip('\n').split('|')
            ids.append(title_parts[0])
            taxids = [x for x in title_parts if 'taxon:' in x]
            taxid = taxons[0] if len(taxids) > 0 else None
            taxons.append(taxid)
    return ids, taxons