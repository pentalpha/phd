from util import (input_features_ids_path, input_annotation_path, 
    input_labels_path, open_file)

if __name__ == '__main__':
    annotations = {}
    print('Loading annotation')
    for rawline in open_file(input_annotation_path):
        cells = rawline.rstrip('\n').split('\t')
        if len(cells) == 4:
            ID = cells[0]+'_'+cells[3]
            if not ID in annotations:
                annotations[ID] = set()
            annotations[ID].add(cells[1])

    print('Loading labels by uniprotid')
    label_lists = []
    for rawline in open_file(input_features_ids_path):
        cells = rawline.rstrip('\n').split('\t')
        ID = cells[0]+'_'+cells[1]
        labels = sorted(annotations[ID])
        label_lists.append((cells[0], cells[1], ','.join(labels)))

    print('Saving')
    output = open(input_labels_path, 'w')
    for protid, taxid, labels in label_lists:
        output.write(protid+'\t'+taxid+'\t'+labels+'\n')