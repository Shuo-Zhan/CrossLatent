#!/usr/bin/env python3
"""Rank named Pfam domains by seed diversity before InterPro coverage queries."""
from __future__ import annotations

import csv
import gzip
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def metadata_types(path: Path) -> dict[str, str]:
    result, accession, typ = {}, None, None
    with gzip.open(path, "rt", errors="replace") as handle:
        for line in handle:
            if line.startswith("#=GF AC"):
                accession = line.split()[2].split(".")[0]
            elif line.startswith("#=GF TP"):
                typ = line.split(maxsplit=2)[2].strip()
            elif line.rstrip() == "//":
                if accession:
                    result[accession] = typ or "unknown"
                accession, typ = None, None
    return result


def main() -> None:
    types = metadata_types(ROOT / "raw/pfam/Pfam-A.hmm.dat.gz")
    rows, record = [], {}
    with gzip.open(ROOT / "raw/pfam/Pfam-A.seed.gz", "rt", errors="replace") as handle:
        for line in handle:
            if line.startswith("#=GF AC"):
                record["accession"] = line.split()[2].split(".")[0]
            elif line.startswith("#=GF ID"):
                record["id"] = line.split(maxsplit=2)[2].strip()
            elif line.startswith("#=GF DE"):
                record["description"] = line.split(maxsplit=2)[2].strip()
            elif line.startswith("#=GF SQ"):
                record["seed_sequences"] = int(line.split()[2])
            elif line.rstrip() == "//":
                if record.get("accession"):
                    record["type"] = types.get(record["accession"], "unknown")
                    rows.append(record)
                record = {}
    rows.sort(key=lambda x: (-x.get("seed_sequences", 0), x["accession"]))
    out = ROOT / "manifests/pfam_seed_candidates.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["accession", "id", "description", "type", "seed_sequences"])
        writer.writeheader(); writer.writerows(rows)
    domain = [r for r in rows if r["type"].lower() == "domain"]
    print(f"Parsed {len(rows)} Pfam entries; {len(domain)} are typed as domains; wrote {out}")


if __name__ == "__main__":
    main()
