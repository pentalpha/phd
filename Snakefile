configfile: "config.yml"

extract_esm_script = "esm/scripts/extract.py"

quickgo_gaf = "databases/QuickGO-annotations-1697773507369-20231020.gaf"
goa_parsed = "databases/goa_parsed.tsv.gz"

goa_parsed = 'databases/goa_parsed.tsv.gz'
goa_parsed_expanded = 'databases/goa_parsed_expanded.tsv.gz'
goa_parsed_frequent = 'databases/goa_parsed_frequent.tsv.gz'

uniprot_fasta = "databases/uniprot_sprot.fasta.gz"
go_basic = "databases/go-basic.obo"

#quickgo_expanded = "input/quickgo_expanded.tsv.gz"
proteins_for_learning = "input/proteins.fasta"
input_annotation_path = 'input/annotation.tsv'
taxon_features = 'input/features/taxon_one_hot.npy'
esm_features = 'input/features/esm.npy'
input_features_path = 'input/features.npy'
input_labels_path = 'input/labels.tsv'
input_features_ids_path = 'input/ids.txt'
input_features_ids_traintest_path = 'input/ids_traintest.txt'
input_features_ids_validation_path = 'input/ids_validation.txt'

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

rule download_goa:
    input:
        'evi_not_to_use.txt',
        go_basic
    output:
        goa_parsed_frequent
    shell:
        "conda run --live-stream -n plm python src/download_annotation.py"

rule download_uniprot:
    output:
        uniprot_fasta
    shell:
        "cd databases && wget https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/complete/uniprot_sprot.fasta.gz"

rule annotated_protein_list:
    input:
        goa_parsed_frequent,
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
        input_features_path,
        input_features_ids_path
    shell:
        "conda run --live-stream -n plm python src/calc_features.py "+proteins_for_learning+" input/features"

rule list_labels:
    input:
        input_features_path
    output:
        input_labels_path
    shell:
        "conda run --live-stream -n plm python src/label_lists.py"

rule sep_validation:
    input:
        input_features_ids_path
    output:
        input_features_ids_traintest_path,
        input_features_ids_validation_path
    shell:
        "conda run --live-stream -n plm python src/validation.py"
