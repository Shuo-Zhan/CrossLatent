#!/usr/bin/env python3
"""Build 25 gene-disjoint RBP binding tasks from reproducible K562 ENCODE eCLIP peaks."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from build_gencode_tasks import attrs, load_fasta, project


ROOT = Path(__file__).resolve().parents[1]
ENCODE = "https://www.encodeproject.org"
SPLIT_SIZES = {"train": 3000, "valid": 500, "test": 500}


def session() -> requests.Session:
    value = requests.Session(); value.trust_env = False
    value.headers["User-Agent"] = "CrossLatent-RNA40/1.0"
    return value


def get_json(url: str, attempts: int = 6) -> dict:
    error = None
    for attempt in range(attempts):
        try:
            response = session().get(url, timeout=120); response.raise_for_status(); return response.json()
        except Exception as exc:
            error = exc; time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"GET failed {url}: {error}")


def download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size: return
    path.parent.mkdir(parents=True, exist_ok=True); part = path.with_suffix(path.suffix + ".part")
    error = None
    for attempt in range(6):
        try:
            headers = {"Range": f"bytes={part.stat().st_size}-"} if part.exists() else {}
            with session().get(url, headers=headers, stream=True, timeout=180) as response:
                if response.status_code == 416: part.rename(path); return
                response.raise_for_status()
                mode = "ab" if response.status_code == 206 else "wb"
                with part.open(mode) as handle:
                    for chunk in response.iter_content(1 << 20):
                        if chunk: handle.write(chunk)
            part.rename(path); return
        except Exception as exc:
            error = exc; time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"download failed {url}: {error}")


def merged_peak_file(experiment: dict) -> dict | None:
    eligible = []
    for item in experiment.get("files", []):
        if item.get("status") != "released" or item.get("file_format") != "bed": continue
        if item.get("file_format_type") != "narrowPeak" or item.get("assembly") != "GRCh38": continue
        reps = item.get("biological_replicates") or []
        if len(set(reps)) < 2 or "peak" not in item.get("output_type", "").lower(): continue
        eligible.append(item)
    if not eligible: return None
    return max(eligible, key=lambda x: (bool(x.get("preferred_default")), x.get("file_size", 0)))


def fetch_experiment(item: dict, cache: Path) -> dict | None:
    accession = item["accession"]; path = cache / "metadata" / f"{accession}.json"
    if path.exists(): detail = json.loads(path.read_text())
    else:
        detail = get_json(f"{ENCODE}/experiments/{accession}/?format=json")
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(detail))
    peak = merged_peak_file(detail)
    if not peak: return None
    target = detail.get("target", {}).get("label") or detail.get("target", {}).get("genes", [{}])[0].get("symbol")
    if not target: return None
    href = peak["href"]; bed = cache / "peaks" / f"{peak['accession']}.bed.gz"
    download(ENCODE + href, bed)
    return {"experiment": accession, "target": re.sub(r"[^A-Za-z0-9_.-]", "_", target),
            "file": peak["accession"], "bed": str(bed), "href": href}


def build_exon_index(records: dict, bin_size: int = 100000):
    bins = defaultdict(set)
    for tid, record in records.items():
        for start, end in record["exons"]:
            for bucket in range(start // bin_size, (end - 1) // bin_size + 1):
                bins[(record["chrom"], bucket)].add(tid)
    return bins


def load_basic_transcriptome(gtf: Path) -> dict:
    """Load GENCODE basic transcripts across coding and non-coding gene classes."""
    records = {}
    with gzip.open(gtf, "rt") as handle:
        for line in handle:
            if line.startswith("#"): continue
            fields = line.rstrip().split("\t")
            if len(fields) != 9 or fields[2] != "transcript" or 'tag "basic"' not in fields[8]: continue
            a = attrs(fields[8]); tid = a.get("transcript_id"); gene = a.get("gene_id")
            if tid and gene:
                records[tid] = {"gene": gene, "chrom": fields[0], "strand": fields[6], "exons": []}
    with gzip.open(gtf, "rt") as handle:
        for line in handle:
            if line.startswith("#"): continue
            fields = line.rstrip().split("\t")
            if len(fields) != 9 or fields[2] != "exon": continue
            tid = attrs(fields[8]).get("transcript_id")
            if tid in records: records[tid]["exons"].append((int(fields[3]) - 1, int(fields[4])))
    return {tid: record for tid, record in records.items() if record["exons"]}


def read_peaks(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        for line in handle:
            if line.startswith(("#", "track", "browser")): continue
            fields = line.rstrip().split("\t")
            if len(fields) < 3: continue
            try: start, end = int(fields[1]), int(fields[2])
            except ValueError: continue
            strand = fields[5] if len(fields) > 5 and fields[5] in "+-" else None
            if end > start: yield fields[0], start, end, strand


def task_rows(info: dict, records: dict, sequences: dict, bins: dict, window: int = 401):
    spans_by_tid = defaultdict(list)
    for chrom, start, end, strand in read_peaks(Path(info["bed"])):
        candidates = set()
        for bucket in range(start // 100000, (end - 1) // 100000 + 1): candidates.update(bins.get((chrom, bucket), ()))
        for tid in candidates:
            record = records[tid]
            if strand and record["strand"] != strand: continue
            spans = project(record, [(start, end)])
            if spans: spans_by_tid[tid].extend(spans)
    rows = []
    for tid, all_spans in spans_by_tid.items():
        sequence = sequences.get(tid); record = records[tid]
        if not sequence or set(sequence) - set("ACGUN"): continue
        # One example per distinct reproducible-peak center. Nearby sites may have
        # overlapping biological context, but remain in the same gene-level split.
        centers = sorted({(a + b) // 2 for a, b in all_spans})
        for center in centers:
            start = max(0, min(len(sequence) - window, center - window // 2)) if len(sequence) > window else 0
            end = min(len(sequence), start + window)
            mask = bytearray(b"0" * (end - start))
            for a, b in all_spans:
                lo, hi = max(a, start), min(b, end)
                if lo < hi: mask[lo-start:hi-start] = b"1" * (hi-lo)
            if b"1" not in mask or end - start < 20: continue
            seq = sequence[start:end]
            rows.append({"name": f"{tid}:{start}-{end}", "transcript": tid,
                         "gene": record["gene"], "sequence": seq, info["target"]: mask.decode(),
                         "crop_start": start, "sequence_md5": hashlib.md5(seq.encode()).hexdigest()})
    return rows


def gene_disjoint_splits(rows: list[dict], seed: int):
    by_gene = defaultdict(list)
    for row in rows: by_gene[row["gene"]].append(row)
    rng = random.Random(seed); genes = list(by_gene); rng.shuffle(genes)
    genes.sort(key=lambda g: len(by_gene[g]))
    assigned = {s: [] for s in SPLIT_SIZES}; cursor = 0; reserved = set()
    for split in ("test", "valid"):
        hashes = set()
        while len(hashes) < 520 and cursor < len(genes):
            gene = genes[cursor]; cursor += 1; assigned[split].append(gene)
            hashes.update(r["sequence_md5"] for r in by_gene[gene] if r["sequence_md5"] not in reserved)
        reserved.update(hashes)
    assigned["train"] = genes[cursor:]
    output = {}; used = set()
    for split, target in SPLIT_SIZES.items():
        pool = [r for gene in assigned[split] for r in by_gene[gene]]; rng.shuffle(pool); selected = []
        for row in pool:
            if row["sequence_md5"] in used: continue
            used.add(row["sequence_md5"]); selected.append(row)
            if len(selected) == target: break
        if len(selected) < target: return None
        output[split] = selected
    return output, assigned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--num-tasks", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--window", type=int, default=401)
    args = parser.parse_args()
    raw = ROOT / "raw/rna"; cache = raw / "encode_eclip"
    inventory = json.loads((raw / "encode_k562_eclip_experiments.json").read_text())["@graph"]
    infos = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(fetch_experiment, item, cache): item["accession"] for item in inventory}
        for future in as_completed(futures):
            try:
                info = future.result()
                if info: infos.append(info); print(f"FETCHED {info['target']} {info['experiment']}", flush=True)
            except Exception as exc: print(f"FETCH_FAILED {futures[future]} {exc}", flush=True)
    records = load_basic_transcriptome(raw / "gencode.v49.annotation.gtf.gz")
    sequences = load_fasta(raw / "gencode.v49.transcripts.fa.gz", set(records))
    bins = build_exon_index(records)
    candidates = []
    for i, info in enumerate(sorted(infos, key=lambda x: (x["target"], x["experiment"]))):
        rows = task_rows(info, records, sequences, bins, args.window)
        split_result = gene_disjoint_splits(rows, args.seed + i)
        print(f"AUDIT {info['target']} rows={len(rows)} eligible={bool(split_result)}", flush=True)
        if split_result: candidates.append((len(rows), info, split_result))
    # A target with multiple experiments contributes only its highest-coverage eligible dataset.
    best = {}
    for item in candidates:
        target = item[1]["target"]
        if target not in best or item[0] > best[target][0]: best[target] = item
    selected = sorted(best.values(), key=lambda x: (-x[0], x[1]["target"]))[:args.num_tasks]
    if len(selected) < args.num_tasks:
        raise RuntimeError(f"Only {len(selected)} RBP targets satisfy exact 3000/500/500 gene-disjoint splits")
    manifest = []
    for rank, (available, info, (splits, assigned)) in enumerate(selected, 1):
        target = info["target"]; task = f"rbp_{target.lower()}"; task_dir = ROOT / "splits/rna" / task
        task_dir.mkdir(parents=True, exist_ok=True)
        for split, rows in splits.items():
            fields = ["name", "transcript", "gene", "sequence", target, "crop_start", "sequence_md5"]
            with (task_dir / f"{split}.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        definition = {"task": task, "label_column": target, "biological_target": target,
                      "source": "ENCODE K562 eCLIP reproducible merged GRCh38 peaks",
                      "experiment": info["experiment"], "file": info["file"],
                      "available_examples": available, "window": args.window,
                      "gene_disjoint": True, "genes_by_split": {k: len(v) for k, v in assigned.items()}}
        (task_dir / "definition.json").write_text(json.dumps(definition, indent=2))
        manifest.append({"rank": rank, **definition}); print(f"COMPLETE {rank}/25 {task}", flush=True)
    path = ROOT / "manifests/rna_eclip_tasks_frozen.csv"
    with path.open("w", newline="") as handle:
        fields = ["rank", "task", "label_column", "biological_target", "experiment", "file", "available_examples", "window"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(manifest)


if __name__ == "__main__": main()
