#!/usr/bin/env python3
"""Stream ESM2/SAE once per sequence and evaluate Protein60 valid/test splits."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[3]
INTERPLM = REPO / "InterPLM"
sys.path.insert(0, str(INTERPLM))
sys.path.insert(0, str(REPO / "grabspark/boost_pipeline"))
from interplm.embedders import get_embedder  # noqa: E402
from interplm.sae.inference import load_sae_from_hf  # noqa: E402
from neuron_boosting import encode_features  # noqa: E402

THRESHOLDS = np.round(np.arange(0.0, 1.0001, 0.05), 2)
MODEL_DIR = INTERPLM / "esm2_model/esm2_650M_UR50D"
LABEL = "mask_target_domain"


def parse_mask(value: str, length: int) -> np.ndarray:
    text = str(value)
    if len(text) != length or any(c not in "01" for c in text):
        raise ValueError(f"Invalid mask length/content: {len(text)} versus {length}")
    return np.fromiter((c == "1" for c in text), dtype=bool, count=length)


def macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    tp = int(np.sum(y & pred)); tn = int(np.sum(~y & ~pred))
    fp = int(np.sum(~y & pred)); fn = int(np.sum(y & ~pred))
    pd = 2 * tp + fp + fn; nd = 2 * tn + fp + fn
    return 0.5 * ((2 * tp / pd if pd else 0.0) + (2 * tn / nd if nd else 0.0))


def load_configs(folder: Path, family: str) -> tuple[list[dict], dict]:
    base = folder / f"{family}_{LABEL}_latent10"
    configs = []; single = None
    for k in range(1, 11):
        path = Path(f"{base}_{k}.json")
        payload = json.loads(path.read_text())
        item = payload["features"][LABEL]; combo = item["combination"]
        ids = np.asarray(combo["neuron_ids"], dtype=np.int64)
        coef = np.asarray(combo["coefficients"], dtype=np.float64)
        if len(ids) != k or len(coef) != k:
            raise ValueError(f"{path}: expected {k} coefficients/latents")
        configs.append({"K": k, "ids": ids, "coef": coef,
                        "intercept": float(combo["intercept_b"]), "source_json": str(path.resolve())})
        current = item["best_single_neuron"]
        if single is None:
            single = {"id": int(current["neuron_id"]), "source_json": str(path.resolve())}
        elif single["id"] != int(current["neuron_id"]):
            raise ValueError(f"{family}: best-single ID changes across iteration checkpoints")
    return configs, single


def check_inputs(tasks: list[dict], args) -> None:
    total_tokens = 0
    for task in tasks:
        family = task["accession"]
        load_configs(args.train_result_dir / family, family)
        for split in ("validation", "test"):
            path = args.split_dir / family / f"{split}.csv"
            df = pd.read_csv(path, dtype=str, keep_default_na=False)
            required = {"uniprot_id", "sequence", LABEL, "sequence_md5", "homology_component_30pct"}
            missing = required - set(df.columns)
            if missing: raise ValueError(f"{path}: missing {sorted(missing)}")
            lengths = df.sequence.str.len()
            if len(df) != 500: raise ValueError(f"{path}: expected 500 rows, found {len(df)}")
            if not lengths.between(1, 1022).all(): raise ValueError(f"{path}: sequence outside 1..1022 aa")
            if not (lengths == df[LABEL].str.len()).all(): raise ValueError(f"{path}: mask length mismatch")
            total_tokens += int(lengths.sum())
    print(f"CHECK OK: {len(tasks)} families, {2*len(tasks)*500} sequences, {total_tokens:,} tokens", flush=True)


@torch.no_grad()
def score_split(df: pd.DataFrame, configs: list[dict], single: dict, embedder, sae,
                device: str, batch_size: int, split: str, family: str):
    accum = {k: np.zeros(len(THRESHOLDS), dtype=np.float64) for k in range(1, 11)}
    single_accum = np.zeros(len(THRESHOLDS), dtype=np.float64)
    raw_test = []
    n_batches = (len(df) + batch_size - 1) // batch_size
    for batch_no, start in enumerate(range(0, len(df), batch_size), 1):
        frame = df.iloc[start:start + batch_size]
        seqs = frame.sequence.astype(str).tolist(); lengths = [len(s) for s in seqs]
        embedding = embedder.extract_embeddings(sequences=seqs, layer=33, batch_size=batch_size)
        if isinstance(embedding, torch.Tensor): embedding = embedding.detach().cpu().numpy()
        features = encode_features(sae, np.asarray(embedding, dtype=np.float32), device=device)
        offset = 0
        for local_row, (_, row) in enumerate(frame.iterrows()):
            length = lengths[local_row]; f_seq = features[offset:offset + length]; offset += length
            y = parse_mask(row[LABEL], length); combo_scores = []
            for cfg in configs:
                score = cfg["intercept"] + f_seq[:, cfg["ids"]].astype(np.float64) @ cfg["coef"]
                combo_scores.append(score)
                accum[cfg["K"]] += [macro_f1(y, score >= t) for t in THRESHOLDS]
            single_score = f_seq[:, single["id"]].astype(np.float64)
            single_accum += [macro_f1(y, single_score >= t) for t in THRESHOLDS]
            if split == "test":
                raw_test.append({"source_row": int(start + local_row), "row": row,
                                 "y": y, "combo_scores": combo_scores, "single_score": single_score})
        if offset != sum(lengths): raise RuntimeError(f"{family}/{split}: embedding split mismatch")
        print(f"{family}/{split}: ESM2+SAE batch {batch_no}/{n_batches}", flush=True)
    return accum, single_accum, raw_test


def evaluate_family(task: dict, args, embedder, sae) -> dict:
    family = task["accession"]; output = args.result_dir / family; output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    if summary_path.exists() and not args.force:
        summary = json.loads(summary_path.read_text())
        if summary.get("status") == "complete":
            print(f"SKIP {family}: complete", flush=True); return summary
    configs, single = load_configs(args.train_result_dir / family, family)
    valid_df = pd.read_csv(args.split_dir / family / "validation.csv", dtype=str, keep_default_na=False)
    test_df = pd.read_csv(args.split_dir / family / "test.csv", dtype=str, keep_default_na=False)
    va, sva, _ = score_split(valid_df, configs, single, embedder, sae, args.device, args.batch_size, "validation", family)
    te, ste, raw_test = score_split(test_df, configs, single, embedder, sae, args.device, args.batch_size, "test", family)

    curves = []; candidates = []
    for k in range(1, 11):
        vf = va[k] / len(valid_df); tf = te[k] / len(test_df)
        for threshold, valid_f1, test_f1 in zip(THRESHOLDS, vf, tf):
            curves.append({"method": "CrossLatent", "K": k, "threshold": threshold,
                           "valid_macro_f1": valid_f1, "test_macro_f1": test_f1})
            candidates.append((float(valid_f1), k, float(threshold), float(test_f1)))
    chosen = sorted(candidates, key=lambda x: (-x[0], x[1], x[2]))[0]
    sv = sva / len(valid_df); st = ste / len(test_df)
    si = sorted(range(len(THRESHOLDS)), key=lambda i: (-sv[i], THRESHOLDS[i]))[0]
    for threshold, valid_f1, test_f1 in zip(THRESHOLDS, sv, st):
        curves.append({"method": "Best single latent", "K": 1, "threshold": threshold,
                       "valid_macro_f1": valid_f1, "test_macro_f1": test_f1})

    per_sequence = []
    for item in raw_test:
        row = item["row"]; y = item["y"]
        per_sequence.append({"family": family, "pfam_id": task["id"], "description": task["description"],
            "source_row": item["source_row"], "uniprot_id": row["uniprot_id"],
            "sequence_md5": row["sequence_md5"], "homology_component_30pct": row["homology_component_30pct"],
            "sequence_length": len(y), "positive_fraction": float(y.mean()),
            "grabspark_macro_f1": macro_f1(y, item["combo_scores"][chosen[1]-1] >= chosen[2]),
            "best_single_macro_f1": macro_f1(y, item["single_score"] >= THRESHOLDS[si])})
    pd.DataFrame(curves).to_csv(output / "selection_curve.csv", index=False)
    pd.DataFrame(per_sequence).to_csv(output / "test_per_sequence.csv", index=False)
    summary = {"status": "complete", "family": family, "pfam_id": task["id"],
        "description": task["description"], "metric": "mean per-sequence class-macro F1",
        "valid_sequences": len(valid_df), "test_sequences": len(test_df), "selected_k": chosen[1],
        "selected_threshold": chosen[2], "valid_macro_f1": chosen[0], "test_macro_f1": chosen[3],
        "best_single_neuron": single["id"], "best_single_valid_threshold": float(THRESHOLDS[si]),
        "best_single_valid_macro_f1": float(sv[si]), "best_single_test_macro_f1": float(st[si]),
        "test_delta_macro_f1": float(chosen[3] - st[si]),
        "selection_rule": "max valid macro F1; ties: smaller K, then lower threshold",
        "threshold_grid": THRESHOLDS.tolist(), "plm_model": "ESM2-650M", "plm_layer": 33,
        "sae_source": "InterPLM load_sae_from_hf(plm_model=esm2-650m, plm_layer=33)",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"COMPLETE {family}: K={chosen[1]} t={chosen[2]:.2f} test={chosen[3]:.4f} delta={summary['test_delta_macro_f1']:+.4f}", flush=True)
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=ROOT / "manifests/protein_tasks_frozen.csv")
    p.add_argument("--split-dir", type=Path, default=ROOT / "splits/protein")
    p.add_argument("--train-result-dir", type=Path, default=ROOT / "results/protein_domain_latent10")
    p.add_argument("--result-dir", type=Path, default=ROOT / "evaluation/protein60_valid_test")
    p.add_argument("--device", default="cuda:0"); p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--families", nargs="*"); p.add_argument("--force", action="store_true")
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args(); args.result_dir.mkdir(parents=True, exist_ok=True)
    with args.manifest.open() as handle: tasks = list(csv.DictReader(handle))
    if args.families:
        wanted = set(args.families); tasks = [x for x in tasks if x["accession"] in wanted]
        missing = wanted - {x["accession"] for x in tasks}
        if missing: raise ValueError(f"Families absent from manifest: {sorted(missing)}")
    check_inputs(tasks, args)
    if args.check_only: return
    print(f"Loading ESM2-650M layer 33 model from {MODEL_DIR}", flush=True)
    embedder = get_embedder("esm", model_name=str(MODEL_DIR.resolve()), device=args.device)
    print("Loading ESM2-650M layer-33 SAE", flush=True)
    sae = load_sae_from_hf(plm_model="esm2-650m", plm_layer=33).to(torch.device(args.device)).eval()
    summaries = []
    for i, task in enumerate(tasks, 1):
        print(f"[{i}/{len(tasks)}] {task['accession']} {task['id']}", flush=True)
        summaries.append(evaluate_family(task, args, embedder, sae))
        pd.DataFrame(summaries).to_csv(args.result_dir / "summary.csv", index=False)
    frames = [pd.read_csv(args.result_dir / x["accession"] / "test_per_sequence.csv") for x in tasks]
    pd.concat(frames, ignore_index=True).to_csv(args.result_dir / "test_per_sequence_all.csv", index=False)
    print(f"PROTEIN60 EVALUATION COMPLETE: {len(summaries)} families", flush=True)


if __name__ == "__main__": main()
