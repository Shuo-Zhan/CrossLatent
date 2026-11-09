#!/usr/bin/env python3
"""Extract flattened ERNIE-RNA layer-12 token embeddings for one RNA40 CSV."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(REPO/"grabspark/extract_embedding"))
from extract_ERNIERNA_embedding import extract_embedding_twod_token  # noqa:E402

def main():
    p=argparse.ArgumentParser();p.add_argument("--input-csv",type=Path,required=True);p.add_argument("--output-csv",type=Path,required=True)
    p.add_argument("--output-npy",type=Path,required=True);p.add_argument("--device",default="cuda:0");a=p.parse_args()
    df=pd.read_csv(a.input_csv,dtype=str,keep_default_na=False)
    if "sequence" not in df:raise ValueError("CSV lacks sequence column")
    normalized=df["sequence"].str.upper().str.replace("T","U",regex=False)
    lengths=normalized.str.len()
    keep=lengths.between(1,1022,inclusive="both")
    skipped_empty=int((lengths==0).sum())
    skipped_long=int((lengths>1022).sum())
    if skipped_empty or skipped_long:
        print(
            f"Skipped {skipped_empty + skipped_long} invalid row(s): "
            f"empty={skipped_empty}, longer_than_1022={skipped_long}",
            flush=True,
        )
    df=df.loc[keep].reset_index(drop=True)
    raw_seqs=normalized.loc[keep].tolist()
    if not raw_seqs:
        raise ValueError("No RNA sequences remain after filtering to 1..1022 nt")
    iupac_ambiguity=set("RYSWKMBDHV")
    unsupported=sorted(set().union(*(set(s)-set("ACGUN")-iupac_ambiguity for s in raw_seqs)))
    if unsupported:raise ValueError(f"Unsupported sequence characters: {unsupported}")
    ambiguous_bases=sum(c in iupac_ambiguity for s in raw_seqs for c in s)
    seqs=["".join("N" if c in iupac_ambiguity else c for c in s) for s in raw_seqs]
    if ambiguous_bases:
        print(f"Normalized {ambiguous_bases} IUPAC ambiguous base(s) to N",flush=True)
    a.output_csv.parent.mkdir(parents=True,exist_ok=True);a.output_npy.parent.mkdir(parents=True,exist_ok=True)
    df["sequence"]=seqs
    df.to_csv(a.output_csv,index=False)
    embeddings=extract_embedding_twod_token(seqs,if_cls=False,device=a.device)
    expected=sum(map(len,seqs))
    if embeddings.shape!=(expected,768):raise RuntimeError(f"Unexpected embedding shape {embeddings.shape}, expected {(expected,768)}")
    np.save(a.output_npy,embeddings.astype(np.float32,copy=False))
    a.output_npy.with_suffix(".metadata.json").write_text(json.dumps({
        "input_csv":str(a.input_csv.resolve()),"rows":len(seqs),"tokens":expected,
        "shape":list(embeddings.shape),"dtype":"float32","model":"ERNIE-RNA","layer":12,
        "iupac_ambiguous_bases_mapped_to_N":ambiguous_bases,
        "input_rows":int(len(keep)),"skipped_empty_rows":skipped_empty,
        "skipped_rows_longer_than_1022":skipped_long,
    },indent=2)+"\n")
    print(f"SAVED {a.output_npy} shape={embeddings.shape}",flush=True)

if __name__=="__main__":main()
