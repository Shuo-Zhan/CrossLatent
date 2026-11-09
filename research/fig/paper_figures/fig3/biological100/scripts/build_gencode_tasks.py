#!/usr/bin/env python3
"""Build five gene-disjoint transcript-coordinate tasks from GENCODE v49."""
from __future__ import annotations

import csv
import gzip
import hashlib
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("five_prime_utr", "cds", "three_prime_utr", "start_codon", "stop_codon")


def attrs(text: str) -> dict[str, str]:
    return {k: v for k, v in re.findall(r'(\w+) "([^"]+)"', text)}


def load_selected(gtf: Path):
    selected = {}
    with gzip.open(gtf, "rt") as handle:
        for line in handle:
            if line.startswith("#"): continue
            fields = line.rstrip().split("\t")
            if len(fields) != 9 or fields[2] != "transcript": continue
            a = attrs(fields[8])
            if a.get("transcript_type") != "protein_coding": continue
            if 'tag "MANE_Select"' not in fields[8] and 'tag "Ensembl_canonical"' not in fields[8]: continue
            gene = a["gene_id"]
            # Prefer MANE when both annotations exist for a gene.
            priority = 0 if 'tag "MANE_Select"' in fields[8] else 1
            item = (priority, a["transcript_id"], gene, fields[0], fields[6])
            if gene not in selected or item < selected[gene]: selected[gene] = item
    transcripts = {item[1]: {"gene": item[2], "chrom": item[3], "strand": item[4],
                              "exons": [], "CDS": [], "start_codon": [], "stop_codon": []}
                   for item in selected.values()}
    with gzip.open(gtf, "rt") as handle:
        for line in handle:
            if line.startswith("#"): continue
            fields = line.rstrip().split("\t")
            if len(fields) != 9 or fields[2] not in {"exon", "CDS", "start_codon", "stop_codon"}: continue
            a = attrs(fields[8]); tid = a.get("transcript_id")
            if tid in transcripts:
                key = "exons" if fields[2] == "exon" else fields[2]
                transcripts[tid][key].append((int(fields[3]) - 1, int(fields[4])))
    return transcripts


def load_fasta(path: Path, wanted: set[str]) -> dict[str, str]:
    result, name, parts = {}, None, []
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line.startswith(">"):
                if name in wanted: result[name] = "".join(parts).upper().replace("T", "U")
                name = line[1:].split("|", 1)[0]; parts = []
            else: parts.append(line.strip())
        if name in wanted: result[name] = "".join(parts).upper().replace("T", "U")
    return result


def project(record: dict, intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    exons = sorted(record["exons"], reverse=record["strand"] == "-")
    output, offset = [], 0
    for es, ee in exons:
        for fs, fe in intervals:
            start, end = max(es, fs), min(ee, fe)
            if start >= end: continue
            if record["strand"] == "+": output.append((offset + start - es, offset + end - es))
            else: output.append((offset + ee - end, offset + ee - start))
        offset += ee - es
    return output


def crop(sequence: str, spans: list[tuple[int, int]], max_len: int = 1022):
    if not spans: return None
    lo, hi = min(x[0] for x in spans), max(x[1] for x in spans)
    center = (lo + hi) // 2
    start = max(0, min(len(sequence) - max_len, center - max_len // 2)) if len(sequence) > max_len else 0
    end = min(len(sequence), start + max_len)
    mask = bytearray(b"0" * (end - start))
    for a, b in spans:
        a, b = max(a, start), min(b, end)
        if a < b: mask[a-start:b-start] = b"1" * (b-a)
    if b"1" not in mask or end - start < 20: return None
    return sequence[start:end], mask.decode(), start


def main() -> None:
    raw = ROOT / "raw/rna"
    records = load_selected(raw / "gencode.v49.annotation.gtf.gz")
    sequences = load_fasta(raw / "gencode.v49.transcripts.fa.gz", set(records))
    examples = defaultdict(list)
    for tid, record in records.items():
        sequence = sequences.get(tid)
        if not sequence or set(sequence) - set("ACGUN"): continue
        cds = project(record, record["CDS"])
        if not cds: continue
        cds_start, cds_end = min(a for a, _ in cds), max(b for _, b in cds)
        spans_by_task = {
            "five_prime_utr": [(0, cds_start)] if cds_start else [],
            "cds": [(cds_start, cds_end)],
            "three_prime_utr": [(cds_end, len(sequence))] if cds_end < len(sequence) else [],
            "start_codon": project(record, record["start_codon"]),
            "stop_codon": project(record, record["stop_codon"]),
        }
        for task, spans in spans_by_task.items():
            item = crop(sequence, spans)
            if item:
                seq, mask, crop_start = item
                examples[task].append({"name": tid, "gene": record["gene"], "sequence": seq,
                                       task: mask, "crop_start": crop_start})
    rng = random.Random(20260824)
    out = ROOT / "splits/rna"
    summary = []
    for task in TASKS:
        rows = examples[task]; rng.shuffle(rows)
        # Different genes can contain identical cropped sequence. Retain only one
        # copy before splitting so sequence identity cannot leak across splits.
        unique, seen = [], set()
        for row in rows:
            digest = hashlib.md5(row["sequence"].encode()).hexdigest()
            if digest in seen: continue
            seen.add(digest); row["sequence_md5"] = digest; unique.append(row)
        rows = unique
        if len(rows) < 4000: raise RuntimeError(f"{task}: only {len(rows)} eligible genes")
        task_dir = out / task; task_dir.mkdir(parents=True, exist_ok=True)
        cursor = 0
        for split, n in (("train", 3000), ("valid", 500), ("test", 500)):
            subset, cursor = rows[cursor:cursor+n], cursor+n
            with (task_dir / f"{split}.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["name", "gene", "sequence", task, "crop_start", "sequence_md5"])
                writer.writeheader(); writer.writerows(subset)
            summary.append((task, split, n, sum(r[task].count("1") for r in subset)))
    with (ROOT / "reports/gencode_task_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["task", "split", "rows", "positive_tokens"]); writer.writerows(summary)
    print({task: len(rows) for task, rows in examples.items()})


if __name__ == "__main__":
    main()
