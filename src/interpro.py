from os import path
from tqdm import tqdm

from util import open_file, write_file, config, input_features_ids_path, run_command

aprox_interprodb_n_lines = 1355591115

if __name__ == "__main__":
    interpro_raw = config['interpro_dir'] + '/protein2ipr.dat.gz'
    interpro_go_raw = config['interpro_dir'] + '/interpro2go'
    if not path.exists(interpro_go_raw):
        download_cmd_1 = ['cd', config['interpro_dir'], '&&', 'wget https://ftp.ebi.ac.uk/pub/databases/interpro/current_release/interpro2go']
        run_command(download_cmd_1)
    if not path.exists(interpro_raw):
        download_cmd_2 = ['cd', config['interpro_dir'], '&&', 'wget https://ftp.ebi.ac.uk/pub/databases/interpro/current_release/protein2ipr.dat.gz']
        run_command(download_cmd_2)
    
    interpro_filtered = config['interpro_dir'] + '/protein2ipr.pita.tsv.gz'
    interpro_filtered2 = config['interpro_dir'] + '/interpro_by_protein.tsv.gz'

    ids = set([line.split('\t')[0] for line in open_file(input_features_ids_path)])
    print('Looking for', len(ids), 'at interpro db')

    ids_found = set()
    protein_categories = {x: [] for x in ids}
    prot_categories_input = open_file(interpro_raw)
    prot_categories_output = write_file(interpro_filtered)

    annots = 0
    progress_bar = tqdm(total=aprox_interprodb_n_lines)
    for rawline in prot_categories_input:
        cells = rawline.split('\t')
        uniprot_id = cells[0]
        if uniprot_id in ids:
            interpro_id = cells[1]
            protein_categories[uniprot_id].append(interpro_id)
            prot_categories_output.write(uniprot_id+'\t'+interpro_id+'\n')
            ids_found.add(uniprot_id)
            annots += 1
        
        progress_bar.update(1)
    progress_bar.close()
    prot_categories_output.close()
    print('Found', annots, 'annotations')
    print('Found', len(ids_found), 'uniprot ids')

    prot_categories_output2 = write_file(interpro_filtered2)
    for x in ids:
        if len(protein_categories[x]) > 0:
            line = x+'\t'+','.join(protein_categories[x])+'\n'
            prot_categories_output2.write(line)
    prot_categories_output2.close()
    

