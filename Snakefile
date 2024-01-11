quickgo_gaf = "databases/QuickGO-annotations-20240111.gaf"
quickgo_parsed = "databases/goa_parsed.tsv.gz"
uniprot_fasta = "databases/uniprot_sprot.fasta.gz"
proteins_for_learning = "input/proteins.fasta"
extract_esm_script = "esm/scripts/extract.py"

rule download_esm:
    output:
        extract_esm_script
    shell:
        "rm -rf esm & git clone git@github.com:facebookresearch/esm.git"

rule parse_quickgo:
    input:
        quickgo_gaf,
        'evi_not_to_use.txt'
    output:
        quickgo_parsed
    shell:
        "python download_annotation.py"

rule download_uniprot:
    output:
        uniprot_fasta
    shell:
        "cd databases && wget https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/complete/uniprot_sprot.fasta.gz"

rule annotated_protein_list:
    input:
        quickgo_parsed,
        uniprot_fasta
    output:
        proteins_for_learning
    shell:
        "python create_train_protein_set.py"

rule create_features:
    input:
        proteins_for_learning,
        extract_esm_script
    output:
        'input/features/esm.npy'
    shell:
        "conda run --live-stream -n plm python calc_features.py "+proteins_for_learning+" input/features"