import gzip
from os import path, mkdir
import subprocess
import pandas as pd

from util import open_file, write_file
#manual urls:
#quickgo:
#https://www.ebi.ac.uk/QuickGO/annotations?aspect=molecular_function&evidenceCode=ECO:0000352,ECO:0000269,ECO:0000314,ECO:0000315,ECO:0000316,ECO:0000353,ECO:0000270,ECO:0007005,ECO:0007001,ECO:0007003,ECO:0007007,ECO:0006056,ECO:0000318,ECO:0000320,ECO:0000321,ECO:0000304,ECO:0000305&evidenceCodeUsage=descendants&withFrom=UniProtKB
#geneontology.org
#http://current.geneontology.org/annotations/filtered_goa_uniprot_all_noiea.gaf.gz


    
if __name__ == '__main__':
    dbs_dir = 'databases'
    '''goa_url = "https://ftp.ebi.ac.uk/pub/databases/GO/goa/UNIPROT/goa_uniprot_all.gaf.gz"
    download_cmd = ['wget', goa_url]
    goa_path = path.join(dbs_dir, goa_url.split('/')[-1])
    if not path.exists(goa_path):
        run_command(['mkdir', dbs_dir])
        run_command(download_cmd)'''

    goa_gaf = open_file('databases/QuickGO-annotations-20240111.gaf')
    evi_df = pd.read_csv('evi_not_to_use.txt',sep=',')
    evi_not_use = set(evi_df['code'].tolist())

    parsed = ['prot_id\tgoid\tevi\ttaxonid']
    droped = 0
    try:
        for line in goa_gaf:
            if line.startswith('UniProtKB'):
                cells = line.rstrip('\n').split('\t')
                if len(cells) < 13:
                    break
                if cells[8] == 'F':
                    protid = cells[1]
                    goid = cells[4]
                    evi = cells[6]
                    taxid = cells[12]
                    if len(evi) < 2 or len(evi) > 3:
                        print('strange evidence:', evi)
                    else:
                        if not evi in evi_not_use:
                            parsed.append('\t'.join([protid,goid,evi,taxid]))
                        else:
                            droped += 1
    except EOFError as err:
        print('GAF file download incomplete')
        print(err)
    print(droped, 'with evi codes we cant use')
    
    output_path = path.join(dbs_dir, 'goa_parsed.tsv.gz')
    write_file(output_path).write('\n'.join(parsed))

