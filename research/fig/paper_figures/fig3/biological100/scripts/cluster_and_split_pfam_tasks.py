#!/usr/bin/env python3
"""Cluster 60 Pfam tasks and create exact homology-isolated splits."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPO = next(path for path in ROOT.parents if (path / "InterPLM").is_dir())
MMSEQS = REPO / "data/Protein/fig5_pfam00005_pfam00004_labels/tools/mmseqs/bin/mmseqs"
TARGETS = {"train": 3000, "validation": 500, "test": 500}


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def accession(domain_id: str) -> str:
    return domain_id.split("|", 1)[0]


def stable_key(seed: str, *values: str) -> str:
    return hashlib.sha256("|".join((seed, *values)).encode()).hexdigest()


def cluster_family(task_dir: Path, threads: int) -> Path:
    fasta = task_dir / "processed/target_domains.fasta"
    cluster_dir = task_dir / "clustering"
    cluster_dir.mkdir(exist_ok=True)
    prefix = cluster_dir / "domain30"
    tsv = cluster_dir / "domain30_cluster.tsv"
    if tsv.exists() and tsv.stat().st_size:
        return tsv
    subprocess.run(
        [
            str(MMSEQS), "easy-linclust", str(fasta), str(prefix),
            str(cluster_dir / "tmp"), "--min-seq-id", "0.30", "-c", "0.80",
            "--cov-mode", "0", "--threads", str(threads),
        ],
        check=True,
    )
    return tsv


def component_mapping(tsv: Path) -> dict[str, str]:
    uf = UnionFind()
    domains_by_protein: dict[str, list[str]] = defaultdict(list)
    with tsv.open() as handle:
        for line in handle:
            representative, member = line.rstrip("\n").split("\t")
            uf.union(representative, member)
            domains_by_protein[accession(member)].append(member)
    for domains in domains_by_protein.values():
        for domain in domains[1:]:
            uf.union(domains[0], domain)
    mapping = {}
    for protein, domains in domains_by_protein.items():
        roots = sorted({uf.find(domain) for domain in domains})
        mapping[protein] = hashlib.sha256("|".join(roots).encode()).hexdigest()[:16]
    return mapping


def select_splits(frame: pd.DataFrame, seed: str, family: str) -> pd.DataFrame:
    groups = {
        component: indices.tolist()
        for component, indices in frame.groupby("homology_component_30pct").groups.items()
    }
    available = set(groups)
    selected: list[pd.DataFrame] = []
    # Allocate held-out sets first; train receives the remaining diversity.
    for split in ("test", "validation", "train"):
        remaining = TARGETS[split]
        order = sorted(available, key=lambda c: stable_key(seed, family, split, c))
        chosen: list[str] = []
        for component in order:
            size = len(groups[component])
            if size <= remaining:
                chosen.append(component)
                remaining -= size
                if remaining == 0:
                    break
        for component in chosen:
            available.remove(component)
            part = frame.loc[groups[component]].copy()
            part["split"] = split
            selected.append(part)
        if remaining:
            oversized = [c for c in available if len(groups[c]) > remaining]
            if not oversized:
                raise RuntimeError(f"{family}: cannot fill {split}; missing {remaining} rows")
            component = min(
                oversized,
                key=lambda c: (len(groups[c]), stable_key(seed, family, split, c)),
            )
            available.remove(component)
            indices = sorted(
                groups[component],
                key=lambda i: stable_key(seed, family, split, str(frame.at[i, "uniprot_id"])),
            )[:remaining]
            part = frame.loc[indices].copy()
            part["split"] = split
            selected.append(part)
    result = pd.concat(selected, ignore_index=True)
    counts = result.groupby("split").size().to_dict()
    if counts != TARGETS:
        raise RuntimeError(f"{family}: unexpected split counts {counts}")
    if (result.groupby("homology_component_30pct")["split"].nunique() > 1).any():
        raise RuntimeError(f"{family}: homology component crosses splits")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", type=Path, default=ROOT / "raw/pfam/full_alignment_tasks"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "splits/protein")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", default="CrossLatent-biological100-pfam-split-v1")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    task_dirs = sorted(p.parent for p in args.input_dir.glob("PF*/COMPLETE.json"))
    for number, task_dir in enumerate(task_dirs, 1):
        family = task_dir.name
        print(f"START {number}/{len(task_dirs)} {family}", flush=True)
        tsv = cluster_family(task_dir, args.threads)
        mapping = component_mapping(tsv)
        labels = task_dir / "processed/labels.csv.gz"
        frame = pd.read_csv(labels, compression="gzip", dtype={"mask_target_domain": str})
        raw_rows = len(frame)
        # Different UniProt accessions can encode an identical full sequence.
        # Keep one deterministic row so exact duplicates cannot cross splits or
        # inflate a task while retaining the original raw files for auditing.
        frame = frame.sort_values("uniprot_id").drop_duplicates("sequence_md5").copy()
        frame["homology_component_30pct"] = frame["uniprot_id"].map(mapping)
        if frame["homology_component_30pct"].isna().any():
            raise RuntimeError(f"{family}: labels missing clustering assignments")
        split_frame = select_splits(frame, args.seed, family)
        family_dir = args.output_dir / family
        family_dir.mkdir(exist_ok=True)
        for split in TARGETS:
            subset = split_frame[split_frame["split"] == split].copy()
            subset.to_csv(family_dir / f"{split}.csv", index=False, quoting=csv.QUOTE_MINIMAL)
        summary = {
            "accession": family,
            "raw_source_rows": raw_rows,
            "source_rows": len(frame),
            "source_components": frame["homology_component_30pct"].nunique(),
            "selected_rows": len(split_frame),
            "selected_components": split_frame["homology_component_30pct"].nunique(),
            "train": 3000,
            "validation": 500,
            "test": 500,
            "components_crossing_splits": 0,
        }
        (family_dir / "split_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
        summaries.append(summary)
        print(
            f"COMPLETE {family}: components={summary['source_components']} splits=3000/500/500",
            flush=True,
        )
    pd.DataFrame(summaries).to_csv(
        ROOT / "reports/protein_homology_split_summary.csv", index=False
    )
    print(f"SUMMARY complete={len(summaries)}", flush=True)


if __name__ == "__main__":
    main()
