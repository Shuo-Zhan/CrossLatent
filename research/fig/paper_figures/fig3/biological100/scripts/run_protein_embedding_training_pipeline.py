#!/usr/bin/env python3
"""Serial, resumable ESM2 embedding -> CrossLatent training pipeline for 60 Pfams."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO = next(path for path in ROOT.parents if (path / "InterPLM").is_dir())
EXTRACTOR = REPO / "data/Protein/fig5_pfam00005_pfam00004_labels/extract_fig5_train_ESM2_embedding.py"
TRAINER = REPO / "grabspark/boost_pipeline/ESM2_train_neuron_boosting.py"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def validate_result(path: Path, expected_latents: int) -> dict:
    payload = json.loads(path.read_text())
    feature = payload["features"]["mask_target_domain"]
    neuron_ids = [int(value) for value in feature["combination"]["neuron_ids"]]
    steps = feature["boosting_steps"]
    if len(neuron_ids) != expected_latents:
        raise ValueError(
            f"{path}: expected {expected_latents} selected latents, found {len(neuron_ids)}"
        )
    if len(set(neuron_ids)) != expected_latents:
        raise ValueError(f"{path}: selected latent IDs are not unique")
    if len(steps) != expected_latents + 1 or int(steps[0]["neuron_id"]) != -1:
        raise ValueError(f"{path}: incomplete boosting steps")
    return {
        "selected_latents": neuron_ids,
        "composite_r2": feature["boosting_final_composite_r2"],
        "threshold": feature["combination"]["best_threshold"],
        "macro_f1": feature["combination"]["best_macro_f1"],
    }


def run_logged(command: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as handle:
        handle.write(f"\nRUN {now()}\nCOMMAND: " + " ".join(command) + "\n")
        handle.flush()
        subprocess.run(
            command,
            cwd=REPO,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_embedding(npy: Path, csv_path: Path, family: str) -> dict:
    """Validate an existing embedding without loading its multi-GB payload into RAM."""
    metadata_path = npy.with_suffix(".metadata.json")
    if not npy.is_file() or not metadata_path.is_file() or not csv_path.is_file():
        raise FileNotFoundError("Embedding, metadata, or paired CSV is missing")
    metadata = json.loads(metadata_path.read_text())
    required = {
        "status": "complete", "family": family, "split": "train", "layer": 33,
        "embedding_dim": 1280, "dtype": "float32",
        "source_csv_sha256": sha256_file(csv_path),
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in required.items() if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Embedding metadata mismatch: {mismatches}")
    array = np.load(npy, mmap_mode="r")
    expected_shape = tuple(int(value) for value in metadata["shape"])
    if array.shape != expected_shape or array.dtype != np.float32:
        raise ValueError(
            f"Embedding array mismatch: shape={array.shape}, dtype={array.dtype}, "
            f"expected={expected_shape}/float32"
        )
    expected_size = int(array.offset) + int(array.nbytes)
    if npy.stat().st_size != expected_size:
        raise ValueError(
            f"Truncated/oversized embedding: {npy.stat().st_size} bytes, expected {expected_size}"
        )
    if expected_shape != (int(metadata["num_tokens"]), 1280):
        raise ValueError("Embedding shape and metadata token count disagree")
    return {"shape": list(array.shape), "bytes": npy.stat().st_size}


def remove_embedding(npy: Path) -> list[str]:
    removed = []
    for path in (npy, npy.with_suffix(".metadata.json")):
        if path.exists():
            path.unlink()
            removed.append(str(path))
    return removed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "manifests/protein_tasks_frozen.csv"
    )
    parser.add_argument("--split-dir", type=Path, default=ROOT / "splits/protein")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "work/protein_latent10")
    parser.add_argument(
        "--result-dir", type=Path, default=ROOT / "results/protein_domain_latent10"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batch-tokens", type=int, default=64000)
    parser.add_argument("--num-latents", type=int, default=10)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--families", nargs="*", default=None)
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.result_dir / "pipeline_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
    else:
        state = {"created_at": now(), "num_latents": args.num_latents, "families": {}}
    if int(state.get("num_latents", args.num_latents)) != args.num_latents:
        raise ValueError("Existing pipeline state was created with a different --num-latents")

    with args.manifest.open() as handle:
        tasks = list(csv.DictReader(handle))
    if args.families:
        requested = set(args.families)
        tasks = [row for row in tasks if row["accession"] in requested]
        missing = requested - {row["accession"] for row in tasks}
        if missing:
            raise ValueError(f"Families absent from manifest: {sorted(missing)}")

    env = os.environ.copy()
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env.pop(key, None)

    for index, task in enumerate(tasks, 1):
        family = task["accession"]
        family_work = args.work_dir / family
        family_result = args.result_dir / family
        family_work.mkdir(parents=True, exist_ok=True)
        family_result.mkdir(parents=True, exist_ok=True)
        result_json = family_result / f"{family}_mask_target_domain_latent{args.num_latents}.json"
        npy = family_work / f"{family}_train_ESM2_650M_layer33.npy"
        record = state["families"].setdefault(family, {})

        print(f"[{index}/{len(tasks)}] {family} {task['id']}", flush=True)
        try:
            if result_json.exists():
                metrics = validate_result(result_json, args.num_latents)
                removed = remove_embedding(npy)
                record.update(
                    status="complete", verified_at=now(), result_json=str(result_json.resolve()),
                    metrics=metrics, removed_embedding_files=removed,
                )
                atomic_json(state_path, state)
                print(f"SKIP verified complete: {family}", flush=True)
                continue

            csv_path = family_work / f"{family}_train.csv"
            try:
                embedding_info = validate_embedding(npy, csv_path, family)
                record.update(
                    status="training", resumed_at=now(), reused_embedding=True,
                    embedding_bytes=embedding_info["bytes"], embedding_shape=embedding_info["shape"],
                )
                print(f"REUSE verified embedding: {npy}", flush=True)
            except FileNotFoundError:
                record.update(status="extracting", started_at=record.get("started_at", now()))
                atomic_json(state_path, state)
                extract_started = time.monotonic()
                run_logged(
                    [
                        sys.executable, "-u", str(EXTRACTOR),
                        "--input-csv", str((args.split_dir / family / "train.csv").resolve()),
                        "--output-dir", str(family_work.resolve()),
                        "--families", family, "--splits", "train",
                        "--device", args.device, "--layer", "33",
                        "--batch-size", str(args.batch_size),
                    ],
                    family_result / "extract_embedding.log",
                    env,
                )
                embedding_info = validate_embedding(npy, csv_path, family)
                record.update(
                    status="training", reused_embedding=False,
                    embedding_seconds=round(time.monotonic() - extract_started, 3),
                    embedding_bytes=embedding_info["bytes"], embedding_shape=embedding_info["shape"],
                )
            atomic_json(state_path, state)

            train_started = time.monotonic()
            run_logged(
                [
                    sys.executable, "-u", str(TRAINER),
                    "--csv_path", str(csv_path.resolve()),
                    "--embedding_npy", str(npy.resolve()),
                    "--label_column", "mask_target_domain",
                    "--plm_model", "esm2-650m", "--plm_layer", "33",
                    "--num_iterations", str(args.num_latents),
                    "--max_batch_tokens", str(args.max_batch_tokens),
                    "--device", args.device,
                    "--output_json", str(result_json.resolve()),
                ],
                family_result / "train.log",
                env,
            )
            metrics = validate_result(result_json, args.num_latents)
            removed = remove_embedding(npy)
            record.update(
                status="complete", completed_at=now(),
                training_seconds=round(time.monotonic() - train_started, 3),
                result_json=str(result_json.resolve()), metrics=metrics,
                removed_embedding_files=removed,
            )
            atomic_json(state_path, state)
            print(f"COMPLETE {family}; embedding deleted", flush=True)
        except Exception as exc:
            record.update(status="failed", failed_at=now(), error=repr(exc))
            atomic_json(state_path, state)
            print(f"FAILED {family}: {exc!r}; embedding retained", flush=True)
            if not args.continue_on_error:
                raise

    state["finished_at"] = now()
    atomic_json(state_path, state)
    print(f"PIPELINE COMPLETE: {len(tasks)} families", flush=True)


if __name__ == "__main__":
    main()
