quickgo_gaf = "databases/QuickGO-annotations-20240111.gaf"
quickgo_parsed = "databases/goa_parsed.tsv.gz"
uniprot_fasta = "databases/uniprot_sprot.fasta.gz"

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
        databases/uniprot_ids.txt
    shell:
        "python annotated_protein_list.py"