#!/usr/bin/env python3
"""Create a unique-target inventory of released K562 ENCODE eCLIP experiments."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    data = json.loads((ROOT / "raw/rna/encode_k562_eclip_experiments.json").read_text())
    rows = []
    for experiment in data.get("@graph", []):
        target = experiment.get("target") or {}
        rows.append({
            "experiment": experiment.get("accession", ""),
            "target": target.get("label", ""),
            "target_id": target.get("@id", ""),
            "lab": (experiment.get("lab") or {}).get("title", ""),
            "file_records": len(experiment.get("files") or []),
        })
    counts = Counter(row["target"] for row in rows)
    for row in rows:
        row["experiments_for_target"] = counts[row["target"]]
    rows.sort(key=lambda r: (r["target"], r["experiment"]))
    out = ROOT / "manifests/encode_k562_eclip_experiments.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader(); writer.writerows(rows)
    print(f"Found {len(rows)} experiments for {len(counts)} unique RBP targets; wrote {out}")


if __name__ == "__main__":
    main()
