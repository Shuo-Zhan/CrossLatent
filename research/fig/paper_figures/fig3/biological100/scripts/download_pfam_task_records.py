#!/usr/bin/env python3
"""Download full sequences and exact domain intervals for 60 frozen Pfam tasks."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def fetch_json(url: str) -> dict:
    last = None
    for attempt in range(10):
        try:
            req = Request(url, headers={"User-Agent": "CrossLatent-biological100/1.0"})
            with urlopen(req, timeout=180) as response:
                return json.load(response)
        except Exception as exc:
            last = exc; time.sleep(min(300, 2 ** attempt))
    raise RuntimeError(f"Failed URL after retries: {url}: {last}")


def process_task(row: dict, target: int) -> tuple[str, int]:
    accession = row["accession"]
    out_dir = ROOT / "raw/pfam/tasks" / accession
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "records.jsonl.gz"
    done = out_dir / "COMPLETE.json"
    if done.exists() and output.exists():
        return accession, json.loads(done.read_text())["records"]
    url = (f"https://www.ebi.ac.uk/interpro/api/protein/UniProt/entry/pfam/{accession}/"
           "?page_size=200&extra_fields=sequence")
    page_dir = out_dir / "pages"
    page_dir.mkdir(exist_ok=True)
    records, seen, page = [], set(), 0
    while url and len(records) < target:
        page += 1
        page_path = page_dir / f"page_{page:05d}.json.gz"
        if page_path.exists():
            with gzip.open(page_path, "rt") as handle:
                payload = json.load(handle)
        else:
            payload = fetch_json(url)
            temporary = page_path.with_suffix(".json.gz.part")
            with gzip.open(temporary, "wt") as handle:
                json.dump(payload, handle, separators=(",", ":"))
            temporary.replace(page_path)
        for result in payload.get("results", []):
            metadata = result.get("metadata", {})
            protein = metadata.get("accession")
            sequence = (result.get("extra_fields") or {}).get("sequence", "")
            if not protein or protein in seen or not (20 <= len(sequence) <= 1022):
                continue
            intervals = []
            for entry in result.get("entries", []):
                if entry.get("accession") != accession:
                    continue
                for loc in entry.get("entry_protein_locations", []):
                    for fragment in loc.get("fragments", []):
                        start, end = int(fragment["start"]) - 1, int(fragment["end"])
                        if 0 <= start < end <= len(sequence): intervals.append([start, end])
            if intervals:
                seen.add(protein)
                records.append({"accession": protein, "sequence": sequence, "intervals": intervals,
                                "protein_name": metadata.get("name", ""),
                                "tax_id": (metadata.get("source_organism") or {}).get("taxId")})
        url = payload.get("next")
        if page % 10 == 0:
            print(f"{accession}: pages={page}, eligible={len(records)}", flush=True)
    if len(records) < 4000:
        raise RuntimeError(f"{accession}: only {len(records)} length-eligible proteins")
    with gzip.open(output, "wt") as handle:
        for record in records: handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    done.write_text(json.dumps({"accession": accession, "records": len(records), "pages": page}, indent=2) + "\n")
    return accession, len(records)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=12000)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    with (ROOT / "manifests/protein_tasks_frozen.csv").open() as handle:
        tasks = list(csv.DictReader(handle))
    results, failures = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(process_task, row, args.target): row["accession"] for row in tasks}
        for i, job in enumerate(as_completed(jobs), 1):
            accession = jobs[job]
            try:
                accession, count = job.result(); results.append((accession, count))
                print(f"COMPLETE {i}/60: {accession}, records={count}", flush=True)
            except Exception as exc:
                failures.append((accession, str(exc)))
                print(f"FAILED {accession}: {exc}", flush=True)
    with (ROOT / "manifests/protein_download_counts.csv").open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["accession", "records"]); writer.writerows(sorted(results))
    if failures:
        with (ROOT / "reports/pfam_download_failures.csv").open("w", newline="") as handle:
            writer = csv.writer(handle); writer.writerow(["accession", "error"]); writer.writerows(failures)
        raise RuntimeError(f"{len(failures)} tasks failed; rerun is safe and resumes cached pages")


if __name__ == "__main__":
    main()
# 3937703