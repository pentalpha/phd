quickgo_gaf = "databases/QuickGO-annotations-20240111.gaf"
quickgo_parsed = "databases/goa_parsed.tsv.gz"
uniprot_fasta = "databases/uniprot_sprot.fasta.gz"
proteins_for_learning = "input/proteins.fasta"

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