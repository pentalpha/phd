configfile: "config.yml"

extract_esm_script = "esm/scripts/extract.py"

quickgo_gaf = "databases/QuickGO-annotations-1697773507369-20231020.gaf"
quickgo_parsed = "databases/goa_parsed.tsv.gz"
uniprot_fasta = "databases/uniprot_sprot.fasta.gz"
go_basic = "databases/go-basic.obo"

quickgo_expanded = "input/quickgo_expanded.tsv.gz"
proteins_for_learning = "input/proteins.fasta"
input_annotation_path = 'input/annotation.tsv'
taxon_features = 'input/features/taxon_one_hot.npy'
esm_features = 'input/features/esm.npy'
input_features_path = 'input/features.npy'
input_labels_path = 'input/labels.tsv'

rule download_go:
    output:
        go_basic
    shell:
        "cd databases && wget https://current.geneontology.org/ontology/subsets/gocheck_do_not_annotate.json"
        " && wget https://purl.obolibrary.org/obo/go/go-basic.obo"

rule download_esm:
    output:
        extract_esm_script
    shell:
        "rm -rf esm && git clone git@github.com:facebookresearch/esm.git"

rule parse_quickgo:
    input:
        config['go_annotation_raw'],
        'evi_not_to_use.txt',
        go_basic
    output:
        quickgo_parsed,
        quickgo_expanded
    shell:
        "conda run --live-stream -n plm python src/download_annotation.py"

rule download_uniprot:
    output:
        uniprot_fasta
    shell:
        "cd databases && wget https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/complete/uniprot_sprot.fasta.gz"

rule annotated_protein_list:
    input:
        quickgo_expanded,
        uniprot_fasta
    output:
        proteins_for_learning,
        input_annotation_path
    shell:
        "conda run --live-stream -n plm python src/create_train_protein_set.py"

rule create_features:
    input:
        proteins_for_learning,
        extract_esm_script
    output:
        input_features_path
    shell:
        "conda run --live-stream -n plm python src/calc_features.py "+proteins_for_learning+" input/features"

rule list_labels:
    input:
        input_features_path
    output:
        input_labels_path
    shell:
        "conda run --live-stream -n plm python src/label_lists.py"