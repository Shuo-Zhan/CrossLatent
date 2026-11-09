#!/usr/bin/env python3
"""Freeze the diverse RNA40 composition while archiving superseded RBP tasks."""
from __future__ import annotations
import csv, json, shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];SPLITS=ROOT/"splits/rna"
KEEP_RBP={"rbp_uchl5","rbp_ybx3","rbp_znf622","rbp_aqr","rbp_ddx24","rbp_upf1","rbp_lin28b",
          "rbp_ddx3x","rbp_pabpc4","rbp_igf2bp1","rbp_srsf1","rbp_fmr1","rbp_pum2","rbp_igf2bp2","rbp_fxr2"}
STRUCTURE={"paired_stem","unpaired_region","hairpin_loop","bulge_loop","internal_loop","multibranch_loop",
           "external_region","pseudoknot","canonical_pair","gu_wobble_pair"}
TRANSCRIPT={"five_prime_utr","cds","three_prime_utr","start_codon","stop_codon"}
MIRNA={"mirna_mature_5p","mirna_mature_3p","mirna_seed_region","mirna_terminal_loop"}
PROCESS={"splice_donor","splice_acceptor","polya_cleavage_site","intronic_polya_site","polya_signal"}
MODIFICATION={"m6a_site"}

def main():
    archive=ROOT/"archive/rbp_not_in_diverse40";archive.mkdir(parents=True,exist_ok=True)
    for path in sorted(SPLITS.glob("rbp_*")):
        if path.name not in KEEP_RBP:
            destination=archive/path.name
            if destination.exists():shutil.rmtree(destination)
            shutil.move(str(path),str(destination));print("ARCHIVED",path.name)
    rows=[]
    for path in sorted(p for p in SPLITS.iterdir() if p.is_dir()):
        if path.name in KEEP_RBP:category,source="RBP binding","ENCODE K562 eCLIP"
        elif path.name in STRUCTURE:category,source="RNA structure","Rfam seed"
        elif path.name in TRANSCRIPT:category,source="Transcript region","GENCODE v49"
        elif path.name in MIRNA:category,source="miRNA biogenesis","miRBase release 23"
        elif path.name in PROCESS:category,source="RNA processing","GENCODE v49 / PolyASite 2.0"
        elif path.name in MODIFICATION:category,source="RNA modification","m6A-Atlas 2.0"
        else:raise RuntimeError(f"Unclassified task {path.name}")
        definition=path/"definition.json";label=path.name
        if definition.exists():
            d=json.loads(definition.read_text());label=d.get("label_column",d.get("task",label))
        rows.append({"task":path.name,"label_column":label,"category":category,"source":source,
                     "train":3000,"valid":500,"test":500})
    if len(rows)!=40:raise RuntimeError(f"Expected 40 tasks, found {len(rows)}")
    out=ROOT/"manifests/rna40_diverse_tasks.csv"
    with out.open("w",newline="") as h:
        w=csv.DictWriter(h,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    print("FROZEN diverse RNA40",len(rows))

if __name__=="__main__":main()
