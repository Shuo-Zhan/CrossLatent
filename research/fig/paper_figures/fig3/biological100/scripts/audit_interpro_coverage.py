#!/usr/bin/env python3
"""Query InterPro coverage for diverse Pfam seed candidates with resumable cache."""
from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def query(accession: str, cache: Path) -> dict:
    path = cache / f"{accession}.json"
    if path.exists():
        return json.loads(path.read_text())
    url = f"https://www.ebi.ac.uk/interpro/api/protein/UniProt/entry/pfam/{accession}/?page_size=1"
    last = None
    for attempt in range(6):
        try:
            req = Request(url, headers={"User-Agent": "CrossLatent-biological100/1.0"})
            with urlopen(req, timeout=90) as response:
                payload = json.load(response)
            result = {"accession": accession, "uniprot_count": int(payload["count"])}
            path.write_text(json.dumps(result) + "\n")
            return result
        except Exception as exc:
            last = exc; time.sleep(2 ** attempt)
    raise RuntimeError(f"InterPro query failed for {accession}: {last}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    with (ROOT / "manifests/pfam_seed_candidates.csv").open() as handle:
        rows = [r for r in csv.DictReader(handle) if r["type"].lower() == "domain"][:args.candidates]
    cache = ROOT / "raw/pfam/interpro_count_cache"; cache.mkdir(parents=True, exist_ok=True)
    by_accession = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(query, r["accession"], cache): r["accession"] for r in rows}
        for i, job in enumerate(as_completed(jobs), 1):
            result = job.result(); by_accession[result["accession"]] = result["uniprot_count"]
            if i % 25 == 0: print(f"Queried {i}/{len(jobs)}", flush=True)
    for row in rows:
        row["uniprot_count"] = by_accession[row["accession"]]
    # Once a diverse seed-derived candidate pool is fixed, rank by actual UniProt
    # coverage so each task can survive length filtering and 30%-identity clustering.
    rows.sort(key=lambda r: (-int(r["uniprot_count"]), -int(r["seed_sequences"]), r["accession"]))
    eligible = [r for r in rows if int(r["uniprot_count"]) >= 20000
                and "unknown function" not in r["description"].lower()
                and not r["id"].startswith(("DUF", "UPF"))]
    selected = eligible[:60]
    if len(selected) < 60:
        raise RuntimeError(f"Only {len(selected)} candidates have >=8000 UniProt entries")
    fields = list(rows[0])
    for name, values in (("pfam_interpro_coverage.csv", rows), ("protein_tasks_frozen.csv", selected)):
        with (ROOT / "manifests" / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(values)
    print(f"Frozen 60 named Pfam domain tasks; minimum raw coverage={min(int(r['uniprot_count']) for r in selected)}")


if __name__ == "__main__":
    main()
