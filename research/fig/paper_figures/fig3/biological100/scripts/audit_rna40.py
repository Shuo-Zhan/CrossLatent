#!/usr/bin/env python3
"""Strict integrity audit for the 40-task RNA benchmark."""
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {"train": 3000, "valid": 500, "test": 500}


def label_for(task_dir: Path) -> str:
    definition = task_dir / "definition.json"
    if definition.exists():
        data = json.loads(definition.read_text()); return data.get("label_column", data["task"])
    return task_dir.name


def main() -> None:
    task_dirs = sorted(p for p in (ROOT / "splits/rna").iterdir() if p.is_dir())
    errors, summary = [], []
    for task_dir in task_dirs:
        label = label_for(task_dir); seen_names = set(); seen_hashes = set(); groups = {}
        for split, expected in EXPECTED.items():
            path = task_dir / f"{split}.csv"
            if not path.exists(): errors.append(f"{task_dir.name}/{split}: missing"); continue
            with path.open() as handle: rows = list(csv.DictReader(handle))
            positives = 0
            if len(rows) != expected: errors.append(f"{task_dir.name}/{split}: rows={len(rows)} expected={expected}")
            for number, row in enumerate(rows, 2):
                seq, mask = row.get("sequence", ""), row.get(label, "")
                if len(seq) != len(mask): errors.append(f"{task_dir.name}/{split}:{number}: length mismatch")
                if set(mask) - {"0", "1"}: errors.append(f"{task_dir.name}/{split}:{number}: nonbinary mask")
                positives += "1" in mask
                name = row.get("name", ""); digest = row.get("sequence_md5") or hashlib.md5(seq.encode()).hexdigest()
                if name in seen_names: errors.append(f"{task_dir.name}: duplicate name across rows {name}")
                if digest in seen_hashes: errors.append(f"{task_dir.name}: duplicate sequence across splits {digest}")
                seen_names.add(name); seen_hashes.add(digest)
                group = row.get("gene") or row.get("rfam_accession") or row.get("cluster")
                if group:
                    if group in groups and groups[group] != split: errors.append(f"{task_dir.name}: group leakage {group}")
                    groups[group] = split
            if positives < int(0.8 * expected): errors.append(f"{task_dir.name}/{split}: positive rows={positives}")
            summary.append({"task": task_dir.name, "label_column": label, "split": split,
                            "rows": len(rows), "positive_rows": positives,
                            "positive_tokens": sum(r.get(label, "").count("1") for r in rows)})
    if len(task_dirs) != 40: errors.append(f"task count={len(task_dirs)} expected=40")
    report = ROOT / "reports/rna40_qc_summary.csv"
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0] if summary else ["task"]); writer.writeheader(); writer.writerows(summary)
    status = {"tasks": len(task_dirs), "errors": len(errors), "error_messages": errors[:500]}
    (ROOT / "reports/rna40_qc_status.json").write_text(json.dumps(status, indent=2))
    if errors: raise RuntimeError("RNA40 QC failed:\n" + "\n".join(errors[:30]))
    print("RNA40 COMPLETE: 40 tasks, 160000 rows, strict QC passed")


if __name__ == "__main__": main()
