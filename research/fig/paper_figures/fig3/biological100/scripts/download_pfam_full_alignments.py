#!/usr/bin/env python3
"""Build the frozen protein tasks from Pfam full alignments.

This replaces pagination over all InterPro protein matches. Each family is an
independent resumable unit: download one official Pfam full alignment,
deterministically sample candidate domain instances, batch-fetch their UniProt
sequences, validate coordinates, and save domain-only labels.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = next(path for path in ROOT.parents if (path / "InterPLM").is_dir())
FIG5_CODE = REPO / "data/Protein/fig5_pfam00005_pfam00004_labels"
sys.path.insert(0, str(FIG5_CODE))

import download_fig5_labels as labels_module  # noqa: E402
from download_fig5_labels import build_labels, fetch_uniprot_sequences  # noqa: E402
from sample_pfam_full_alignment import (  # noqa: E402
    build_hits_and_alignment_audit,
    collect_alignment_sequences,
    download_alignment,
    sample_alignment_ids,
    versionless_accession,
)


LABEL_FIELDS = [
    "uniprot_id",
    "sequence",
    "length",
    "target_family",
    "family_name",
    "reviewed",
    "organism_id",
    "sequence_md5",
    "n_domain_instances",
    "mask_target_domain",
]


def write_csv_gz(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    with gzip.open(temporary, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        writer.writerows({key: row[key] for key in LABEL_FIELDS} for row in rows)
    temporary.replace(path)


def write_domain_fasta(path: Path, audit: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w") as handle:
        for row in audit:
            if row["status"] != "keep":
                continue
            sequence = str(row["aligned_sequence"]).replace(".", "").replace("-", "").upper()
            handle.write(
                f">{row['accession']}|{row['pfam']}|coords={row['start']}-{row['end']}\n"
                f"{sequence}\n"
            )
    temporary.replace(path)


def process_family(row: dict, args: argparse.Namespace) -> dict:
    pfam, family_name = row["accession"], row["id"]
    task_dir = args.output_dir / pfam
    raw_dir = task_dir / "raw"
    processed_dir = task_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    complete = task_dir / "COMPLETE.json"
    labels_path = processed_dir / "labels.csv.gz"
    fasta_path = processed_dir / "target_domains.fasta"
    if complete.exists() and labels_path.exists() and fasta_path.exists():
        return json.loads(complete.read_text())

    # build_labels consults this mapping for family_name. Restrict it to the
    # current task so no kinase/P-loop-specific labels are mixed across tasks.
    labels_module.FAMILIES.clear()
    labels_module.FAMILIES[pfam] = family_name

    alignment = raw_dir / f"{pfam}_full.stockholm.gz"
    import requests
    session = requests.Session()
    # The server may export a loopback proxy (for example 127.0.0.1:17890)
    # that is unavailable to nohup jobs. Direct access to EBI/UniProt is
    # supported, so do not inherit HTTP(S)_PROXY from the shell.
    session.trust_env = False
    session.headers.update({"User-Agent": "CrossLatent-biological100-full-alignment/1.0"})
    download_alignment(
        session, pfam, alignment, timeout=args.timeout, retries=args.retries, refresh=False
    )
    candidate_count = args.sample_per_family * args.oversample_factor
    selected, population, release = sample_alignment_ids(
        alignment, pfam, candidate_count, args.seed
    )
    aligned = collect_alignment_sequences(alignment, set(selected))
    accessions = sorted({versionless_accession(value) for value in selected.values()})
    sequences = fetch_uniprot_sequences(
        session,
        accessions,
        raw_dir=raw_dir / "uniprot",
        timeout=args.timeout,
        retries=args.retries,
        sleep=args.sleep,
        refresh=False,
    )
    hits, audit = build_hits_and_alignment_audit(
        pfam,
        selected,
        aligned,
        sequences,
        args.max_length,
        args.seed,
        args.sample_per_family,
    )
    label_rows, _, _ = build_labels(hits, sequences, c_window=21)
    if len(label_rows) < args.minimum_rows:
        raise RuntimeError(
            f"{pfam}: only {len(label_rows)} unique coordinate-valid proteins; "
            f"minimum is {args.minimum_rows}"
        )
    write_csv_gz(labels_path, label_rows)
    write_domain_fasta(fasta_path, audit)
    summary = {
        "status": "complete",
        "accession": pfam,
        "family_name": family_name,
        "description": row["description"],
        "pfam_release": release,
        "population_domain_instances": population,
        "candidate_domain_instances": len(selected),
        "downloaded_uniprot_sequences": len(sequences),
        "coordinate_valid_domain_instances": len(hits),
        "unique_label_rows": len(label_rows),
        "training_label": "mask_target_domain only",
        "sampling_seed": args.seed,
        "max_length": args.max_length,
    }
    complete.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "manifests/protein_tasks_frozen.csv"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "raw/pfam/full_alignment_tasks"
    )
    parser.add_argument("--sample-per-family", type=int, default=5000)
    parser.add_argument("--oversample-factor", type=int, default=2)
    parser.add_argument("--minimum-rows", type=int, default=4000)
    parser.add_argument("--max-length", type=int, default=1022)
    parser.add_argument("--seed", default="CrossLatent-biological100-pfam-full-v1")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--families", nargs="*", default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.manifest.open() as handle:
        tasks = list(csv.DictReader(handle))
    if args.families:
        requested = set(args.families)
        tasks = [row for row in tasks if row["accession"] in requested]
        missing = requested - {row["accession"] for row in tasks}
        if missing:
            raise ValueError(f"Families absent from manifest: {sorted(missing)}")

    summaries, failures = [], []
    for index, row in enumerate(tasks, 1):
        pfam = row["accession"]
        print(f"START {index}/{len(tasks)} {pfam} {row['id']}", flush=True)
        try:
            summary = process_family(row, args)
            summaries.append(summary)
            print(
                f"COMPLETE {index}/{len(tasks)} {pfam}: "
                f"rows={summary['unique_label_rows']}",
                flush=True,
            )
        except Exception as exc:
            failures.append({"accession": pfam, "error": str(exc)})
            print(f"FAILED {index}/{len(tasks)} {pfam}: {exc}", flush=True)
            traceback.print_exc()

    report = ROOT / "manifests/protein_full_alignment_counts.csv"
    fields = [
        "accession", "family_name", "pfam_release", "population_domain_instances",
        "candidate_domain_instances", "downloaded_uniprot_sequences",
        "coordinate_valid_domain_instances", "unique_label_rows",
    ]
    with report.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: item.get(key, "") for key in fields} for item in summaries)
    failure_path = ROOT / "reports/pfam_full_alignment_failures.csv"
    with failure_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["accession", "error"])
        writer.writeheader()
        writer.writerows(failures)
    print(f"SUMMARY complete={len(summaries)} failed={len(failures)}", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} full-alignment tasks failed; rerun resumes")


if __name__ == "__main__":
    main()
