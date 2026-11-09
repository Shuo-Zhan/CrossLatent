#!/usr/bin/env python3
"""Validation-select K/threshold and evaluate all RNA40 tasks on held-out test."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[3]
sys.path.insert(0, str(REPO / "grabspark/sae_train"))
sys.path.insert(0, str(REPO / "grabspark/boost_pipeline"))
from sae_models import load_sae_from_checkpoint  # noqa: E402
from neuron_boosting import build_sequence_batches, encode_features  # noqa: E402

EXTRACTOR = ROOT / "scripts/extract_rna40_ernierna_embedding.py"
THRESHOLDS = np.round(np.arange(0.0, 1.0001, 0.05), 2)


def parse_mask(value: str, length: int) -> np.ndarray:
    text = str(value)
    if len(text) != length or any(c not in "01" for c in text):
        raise ValueError(f"Invalid binary mask: length={len(text)}, expected={length}")
    return np.fromiter((c == "1" for c in text), dtype=bool, count=length)


def macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    tp = int(np.sum(y & pred)); tn = int(np.sum(~y & ~pred))
    fp = int(np.sum(~y & pred)); fn = int(np.sum(y & ~pred))
    pd = 2 * tp + fp + fn; nd = 2 * tn + fp + fn
    return 0.5 * ((2 * tp / pd if pd else 0.0) + (2 * tn / nd if nd else 0.0))


def load_models(task_dir: Path, task: str, label: str) -> tuple[list[dict], dict, str, str]:
    base = task_dir / f"{task}_{label}_latent10"
    configs: list[dict] = []
    single = None; checkpoint = None; sae_type = None
    for k in range(1, 11):
        path = Path(f"{base}_{k}.json")
        payload = json.loads(path.read_text())
        item = payload["features"][label]
        combo = item["combination"]
        if len(combo["neuron_ids"]) != k:
            raise ValueError(f"{path}: expected {k} selected latents")
        configs.append({
            "K": k,
            "ids": np.asarray(combo["neuron_ids"], dtype=np.int64),
            "coef": np.asarray(combo["coefficients"], dtype=np.float64),
            "intercept": float(combo["intercept_b"]),
            "source_json": str(path.resolve()),
        })
        current = item["best_single_neuron"]
        if single is None:
            single = {"id": int(current["neuron_id"]), "source_json": str(path.resolve())}
        elif single["id"] != int(current["neuron_id"]):
            raise ValueError(f"{task}: best-single ID changes across iteration files")
        checkpoint = payload["config"]["sae_checkpoint"]
        sae_type = payload["config"].get("sae_type", "relu")
    return configs, single, str(checkpoint), str(sae_type)


def extract(task: str, split_dir: Path, work: Path, extract_python: Path, device: str) -> tuple[Path, Path]:
    frames = []
    for split in ("valid", "test"):
        frame = pd.read_csv(split_dir / task / f"{split}.csv", dtype=str, keep_default_na=False)
        frame["_eval_split"] = split
        frame["_eval_source_row"] = np.arange(len(frame))
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    input_csv = work / "combined_input.csv"
    output_csv = work / "combined_filtered.csv"
    output_npy = work / "combined_ERNIERNA_layer12.npy"
    work.mkdir(parents=True, exist_ok=True)
    combined.to_csv(input_csv, index=False)
    command = [str(extract_python.resolve()), "-u", str(EXTRACTOR.resolve()),
               "--input-csv", str(input_csv.resolve()), "--output-csv", str(output_csv.resolve()),
               "--output-npy", str(output_npy.resolve()), "--device", device]
    with (work / "extract.log").open("w") as handle:
        handle.write("COMMAND: " + " ".join(command) + "\n"); handle.flush()
        subprocess.run(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT, check=True)
    return output_csv, output_npy


@torch.no_grad()
def evaluate_task(task: str, label: str, category: str, args, model_cache: dict) -> dict:
    task_result = args.result_dir / task
    summary_path = task_result / "summary.json"
    if summary_path.exists() and not args.force:
        payload = json.loads(summary_path.read_text())
        if payload.get("status") == "complete":
            print(f"SKIP {task}: verified summary exists", flush=True)
            return payload
    task_result.mkdir(parents=True, exist_ok=True)
    work = args.work_dir / task
    configs, single, checkpoint, sae_type = load_models(args.train_result_dir / task, task, label)
    csv_path, npy_path = extract(task, args.split_dir, work, args.extract_python, args.device)
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    lengths = df.sequence.str.len().astype(int).tolist()
    embedding = np.load(npy_path, mmap_mode="r")
    if embedding.shape[0] != sum(lengths): raise ValueError(f"{task}: token count mismatch")

    cache_key = (checkpoint, sae_type, args.device)
    if cache_key not in model_cache:
        model_cache.clear()
        model_cache[cache_key] = load_sae_from_checkpoint(checkpoint, sae_type=sae_type, device=torch.device(args.device)).eval()
    model = model_cache[cache_key]
    offsets = np.cumsum([0] + lengths)
    accum = {"combo": {k: {s: np.zeros(len(THRESHOLDS)) for s in ("valid", "test")} for k in range(1, 11)},
             "single": {s: np.zeros(len(THRESHOLDS)) for s in ("valid", "test")}}
    counts = df._eval_split.value_counts().to_dict()
    per_sequence: list[dict] = []
    cached_scores: dict[int, tuple[np.ndarray, list[np.ndarray], np.ndarray]] = {}
    batches = build_sequence_batches(lengths, args.max_batch_tokens)
    for batch_no, rows in enumerate(batches, 1):
        start, end = int(offsets[rows[0]]), int(offsets[rows[-1] + 1])
        features = encode_features(model, np.asarray(embedding[start:end], dtype=np.float32), device=args.device)
        local = 0
        for row in rows:
            length = lengths[row]; f_seq = features[local:local + length]; local += length
            y = parse_mask(df.iloc[row][label], length); split = df.iloc[row]["_eval_split"]
            combo_scores = []
            for cfg in configs:
                score = cfg["intercept"] + f_seq[:, cfg["ids"]].astype(np.float64) @ cfg["coef"]
                combo_scores.append(score)
                accum["combo"][cfg["K"]][split] += [macro_f1(y, score >= t) for t in THRESHOLDS]
            single_score = f_seq[:, single["id"]].astype(np.float64)
            accum["single"][split] += [macro_f1(y, single_score >= t) for t in THRESHOLDS]
            cached_scores[row] = (y, combo_scores, single_score)
        print(f"{task}: SAE batch {batch_no}/{len(batches)}", flush=True)

    curve_rows = []; candidates = []
    for k in range(1, 11):
        valid = accum["combo"][k]["valid"] / counts["valid"]
        test = accum["combo"][k]["test"] / counts["test"]
        for threshold, vf1, tf1 in zip(THRESHOLDS, valid, test):
            curve_rows.append({"method": "CrossLatent", "K": k, "threshold": threshold,
                               "valid_macro_f1": vf1, "test_macro_f1": tf1})
            candidates.append((float(vf1), k, float(threshold), float(tf1)))
    chosen = sorted(candidates, key=lambda x: (-x[0], x[1], x[2]))[0]
    single_valid = accum["single"]["valid"] / counts["valid"]
    single_test = accum["single"]["test"] / counts["test"]
    si = sorted(range(len(THRESHOLDS)), key=lambda i: (-single_valid[i], THRESHOLDS[i]))[0]
    for threshold, vf1, tf1 in zip(THRESHOLDS, single_valid, single_test):
        curve_rows.append({"method": "Best single latent", "K": 1, "threshold": threshold,
                           "valid_macro_f1": vf1, "test_macro_f1": tf1})
    for row, (y, combos, single_score) in cached_scores.items():
        if df.iloc[row]["_eval_split"] != "test": continue
        per_sequence.append({"task": task, "label": label, "category": category,
                             "source_row": int(df.iloc[row]["_eval_source_row"]),
                             "sequence_md5": df.iloc[row].get("sequence_md5", ""),
                             "sequence_length": len(y), "positive_fraction": float(y.mean()),
                             "grabspark_macro_f1": macro_f1(y, combos[chosen[1]-1] >= chosen[2]),
                             "best_single_macro_f1": macro_f1(y, single_score >= THRESHOLDS[si])})
    pd.DataFrame(curve_rows).to_csv(task_result / "selection_curve.csv", index=False)
    pd.DataFrame(per_sequence).to_csv(task_result / "test_per_sequence.csv", index=False)
    summary = {"status": "complete", "task": task, "label": label, "category": category,
               "metric": "mean per-sequence class-macro F1", "valid_sequences": int(counts["valid"]),
               "test_sequences": int(counts["test"]), "selected_k": chosen[1],
               "selected_threshold": chosen[2], "valid_macro_f1": chosen[0], "test_macro_f1": chosen[3],
               "best_single_neuron": single["id"], "best_single_valid_threshold": float(THRESHOLDS[si]),
               "best_single_valid_macro_f1": float(single_valid[si]),
               "best_single_test_macro_f1": float(single_test[si]),
               "test_delta_macro_f1": float(chosen[3] - single_test[si]),
               "selection_rule": "max valid macro F1; ties: smaller K, then lower threshold",
               "threshold_grid": THRESHOLDS.tolist(), "sae_checkpoint": checkpoint,
               "filtered_rows": int(1000 - len(df)), "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    for path in (npy_path, npy_path.with_suffix(".metadata.json"), work / "combined_input.csv"):
        if path.exists(): path.unlink()
    print(f"COMPLETE {task}: K={chosen[1]} t={chosen[2]:.2f} test={chosen[3]:.4f} delta={summary['test_delta_macro_f1']:+.4f}", flush=True)
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=ROOT / "manifests/rna40_diverse_tasks.csv")
    p.add_argument("--split-dir", type=Path, default=ROOT / "splits/rna")
    p.add_argument("--train-result-dir", type=Path, default=ROOT / "results/rna40_latent10")
    p.add_argument("--result-dir", type=Path, default=ROOT / "evaluation/rna40_valid_test")
    p.add_argument("--work-dir", type=Path, default=ROOT / "work/rna40_evaluation")
    p.add_argument("--extract-python", type=Path, default=Path("/data1/zshuo23/anaconda3/envs/sparse_rna/bin/python"))
    p.add_argument("--device", default="cuda:0"); p.add_argument("--max-batch-tokens", type=int, default=32000)
    p.add_argument("--tasks", nargs="*"); p.add_argument("--force", action="store_true")
    args = p.parse_args(); args.result_dir.mkdir(parents=True, exist_ok=True); args.work_dir.mkdir(parents=True, exist_ok=True)
    with args.manifest.open() as handle: tasks = list(csv.DictReader(handle))
    if args.tasks: tasks = [r for r in tasks if r["task"] in set(args.tasks)]
    summaries = []; model_cache = {}
    for i, row in enumerate(tasks, 1):
        print(f"[{i}/{len(tasks)}] {row['task']} label={row['label_column']}", flush=True)
        summaries.append(evaluate_task(row["task"], row["label_column"], row["category"], args, model_cache))
        pd.DataFrame(summaries).to_csv(args.result_dir / "summary.csv", index=False)
    frames = [pd.read_csv(args.result_dir / r["task"] / "test_per_sequence.csv") for r in tasks]
    pd.concat(frames, ignore_index=True).to_csv(args.result_dir / "test_per_sequence_all.csv", index=False)
    print(f"RNA40 EVALUATION COMPLETE: {len(summaries)} tasks", flush=True)


if __name__ == "__main__": main()
