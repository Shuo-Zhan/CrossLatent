"""Verify released data hashes, sequence partitions, metrics and fitted predictors."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from verify_evidence import verify as verify_evidence
from verify_readouts import verify as verify_readouts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.data_root.resolve()
    report = verify_evidence(root)
    report.update(verify_readouts(root / "readouts", root))
    entries = json.loads((root / "source_data/index.json").read_text())["files"]
    for entry in entries:
        assert hashlib.sha256((root / entry["file"]).read_bytes()).hexdigest() == entry["sha256"]
    report["numerical_source_files_verified"] = len(entries)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
