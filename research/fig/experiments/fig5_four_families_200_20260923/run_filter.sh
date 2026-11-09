#!/bin/sh
set -eu
cd /data1/zshuo23/GraBSpArk/fig/experiments/fig5_four_families_200_20260923
PY=/data1/zshuo23/anaconda3/envs/hfds/bin/python
MM=/data1/zshuo23/anaconda3/envs/hfds/bin/mmseqs
"$PY" prepare_candidates.py
"$MM" easy-search candidates.fasta reference_with_pilot.fasta old_similarity.tsv /tmp/crosslatent_four_family_search --threads 8 -s 7.5 --max-seqs 5000 --alignment-mode 3 --min-seq-id 0.2 -c 0.5 --cov-mode 0 --format-output query,target,fident,qcov,tcov,evalue,bits > mmseqs_search.log 2>&1
"$PY" select_cohort.py --prepare-clustering
"$MM" easy-cluster eligible.fasta new_clusters /tmp/crosslatent_four_family_cluster --min-seq-id 0.8 -c 0.8 --cov-mode 0 --threads 8 > mmseqs_cluster.log 2>&1
"$PY" select_cohort.py
