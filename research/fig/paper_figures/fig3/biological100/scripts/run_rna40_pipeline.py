#!/usr/bin/env python3
"""Resumable orchestration entry point for the complete RNA40 build."""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def run(name: str, arguments: list[str] | None = None) -> None:
    command = [sys.executable, str(SCRIPTS / name), *(arguments or [])]
    print("RUN", " ".join(command), flush=True); subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--skip-gencode", action="store_true")
    parser.add_argument("--skip-rfam", action="store_true")
    parser.add_argument("--skip-eclip", action="store_true")
    args = parser.parse_args()
    if not args.skip_gencode: run("build_gencode_tasks.py")
    if not args.skip_rfam: run("build_rfam_structure_tasks.py")
    if not args.skip_eclip: run("build_encode_eclip_tasks.py", ["--jobs", str(args.jobs)])
    run("build_mirbase_tasks.py")
    run("build_rna_processing_tasks.py")
    run("freeze_diverse_rna40.py")
    run("audit_rna40.py")


if __name__ == "__main__": main()
