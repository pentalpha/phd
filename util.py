import gzip
import subprocess


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
    
def write_file(input_path: str):
    if input_path.endswith('.gz'):
        return gzip.open(input_path, 'wt')
    else:
        return open(input_path, 'w')
    
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
