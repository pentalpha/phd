from glob import glob
import sys
from tqdm import tqdm
import json
from util import write_file

if __name__ == '__main__':
    all_ids_path = 'input/ids.txt'
    taxids = {}
    for line in open(all_ids_path, 'r'):
        proteinid, taxid = line.rstrip('\n').split('\t')
        taxids[proteinid] = taxid

    deepfri_path = sys.argv[1]
    pred_jsons = glob(deepfri_path+'/n*_MF_pred_scores.json')
    lines = []
    taxid_not_found = 0
    for pred_json in tqdm(pred_jsons):
        data = json.load(open(pred_json, 'r'))
        print(pred_json, len(data['pdb_chains']), len(data['Y_hat']))
        
        for i in range(len(data["pdb_chains"])):
            protein_id = data["pdb_chains"][i]
            if protein_id in taxids:
                probs = data['Y_hat'][i]
                lines.append((protein_id+'\t'+taxids[protein_id], probs))
            else:
                taxid_not_found += 1
        print('taxid not found:', taxid_not_found)

    model_params_path = deepfri_path+"/trained_models/DeepCNN-MERGED_molecular_function_model_params.json"
    model_params = json.load(open(model_params_path, 'r'))
    goterms_list = model_params['goterms']

    output_pred_path = 'input/deepfri_probs.tsv.gz'
    output_ids_path = 'input/deepfri_ids.txt'

    print('Saving DeepFRI probs in one table')
    output_pred = write_file(output_pred_path)
    output_pred.write('proteinid\t' + 'taxid\t' + '\t'.join(goterms_list) + '\n')
    for proteinid, probs in tqdm(lines):
        output_pred.write(proteinid + '\t' + '\t'.join([str(x) for x in probs])+'\n')
    output_pred.close()
    open(output_ids_path, 'w').write('\n'.join([proteinid for proteinid, probs in lines]))