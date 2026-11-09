#!/usr/bin/env python3
"""Strict refit leave-one-latent-out analysis for Fig. 4D.

The selected latent set is frozen.  The full model and every leave-one-out
model are fitted by ordinary least squares on training tokens.  Binary-task
thresholds are selected independently on validation sequences, then frozen
for the paired test evaluation.  DNA is evaluated when train/valid/test
selected-latent caches exported from the Evo2 server are available.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/data1/zshuo23/GraBSpArk")
FIG2 = ROOT / "fig/paper_figures/fig2"
FIG3 = ROOT / "fig/paper_figures/fig3"
FIG4 = ROOT / "fig/paper_figures/fig4"
sys.path[:0] = [str(ROOT / "InterPLM"), str(ROOT / "grabspark/sae_train")]

THRESHOLDS = np.round(np.arange(21) * 0.05, 2)


def binary_mask(value: object) -> np.ndarray:
    text = str(value)
    return np.fromiter((char == "1" for char in text), dtype=np.float64, count=len(text))


def flatten_labels(df: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray]:
    arrays = [binary_mask(value) for value in df[column]]
    return np.concatenate(arrays), np.cumsum([0] + [len(x) for x in arrays])


def fit_ols(x: np.ndarray, y: np.ndarray, keep: np.ndarray) -> tuple[float, np.ndarray]:
    design = np.column_stack((np.ones(len(y), dtype=np.float64), x[:, keep].astype(np.float64)))
    beta = np.linalg.lstsq(design, y.astype(np.float64), rcond=None)[0]
    return float(beta[0]), beta[1:]


def predict(x: np.ndarray, keep: np.ndarray, intercept: float, coef: np.ndarray) -> np.ndarray:
    return intercept + x[:, keep] @ coef


def class_macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    y = y.astype(bool); pred = pred.astype(bool)
    tp = np.sum(y & pred); tn = np.sum(~y & ~pred)
    fp = np.sum(~y & pred); fn = np.sum(y & ~pred)
    pos_den = 2 * tp + fp + fn; neg_den = 2 * tn + fp + fn
    pos = 2 * tp / pos_den if pos_den else 0.0
    neg = 2 * tn / neg_den if neg_den else 0.0
    return float((pos + neg) / 2)


def mean_sequence_f1(y: np.ndarray, score: np.ndarray, offsets: np.ndarray, threshold: float) -> tuple[float, np.ndarray]:
    values = np.asarray([
        class_macro_f1(y[offsets[i]:offsets[i + 1]], score[offsets[i]:offsets[i + 1]] >= threshold)
        for i in range(len(offsets) - 1)
    ])
    return float(values.mean()), values


def choose_threshold(y: np.ndarray, score: np.ndarray, offsets: np.ndarray) -> tuple[float, float]:
    candidates = [(mean_sequence_f1(y, score, offsets, t)[0], -abs(t - .5), -t, t) for t in THRESHOLDS]
    best = max(candidates)
    return float(best[3]), float(best[0])


def encode_selected(embedding_path: Path, model: object, latent_ids: list[int], device: str,
                    output_path: Path, batch_tokens: int) -> np.ndarray:
    import torch
    if output_path.exists():
        cached = np.load(output_path)
        if cached.shape[1] != len(latent_ids):
            raise ValueError(f"Stale cache shape: {output_path} {cached.shape}")
        return cached
    emb = np.load(embedding_path, mmap_mode="r")
    out = np.empty((len(emb), len(latent_ids)), dtype=np.float32)
    for start in range(0, len(emb), batch_tokens):
        end = min(start + batch_tokens, len(emb))
        tensor = torch.as_tensor(np.asarray(emb[start:end], dtype=np.float32), device=device)
        with torch.no_grad():
            out[start:end] = model.encode_feat_subset(tensor, latent_ids).detach().cpu().numpy()
        print(f"encode {output_path.name}: {end}/{len(emb)}", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, out)
    return out


def evaluate_binary(modality: str, latent_ids: list[int], train_x: np.ndarray, train_y: np.ndarray,
                    valid_x: np.ndarray, valid_y: np.ndarray, valid_offsets: np.ndarray,
                    test_x: np.ndarray, test_y: np.ndarray, test_offsets: np.ndarray) -> pd.DataFrame:
    rows = []
    all_keep = np.arange(len(latent_ids))
    variants: list[tuple[str, int | None, np.ndarray]] = [("full", None, all_keep)]
    variants += [("leave_one_out", j, np.delete(all_keep, j)) for j in range(len(latent_ids))]
    test_values: dict[int | None, np.ndarray] = {}
    fitted: dict[int | None, dict] = {}
    for kind, removed, keep in variants:
        intercept, coef = fit_ols(train_x, train_y, keep)
        valid_score = predict(valid_x, keep, intercept, coef)
        threshold, valid_f1 = choose_threshold(valid_y, valid_score, valid_offsets)
        test_score = predict(test_x, keep, intercept, coef)
        test_f1, per_sequence = mean_sequence_f1(test_y, test_score, test_offsets, threshold)
        test_values[removed] = per_sequence
        fitted[removed] = {
            "kind": kind, "removed_latent_id": None if removed is None else latent_ids[removed],
            "retained_latent_ids": [latent_ids[i] for i in keep], "intercept": intercept,
            "coefficients": coef.tolist(), "selected_threshold": threshold,
            "validation_macro_f1": valid_f1, "test_macro_f1": test_f1,
        }
    full = fitted[None]; full_values = test_values[None]
    for j, latent_id in enumerate(latent_ids):
        reduced = fitted[j]
        diffs = full_values - test_values[j]
        rows.append({
            "modality": modality, "step": j + 1, "latent_id": latent_id,
            "full_test_per_sequence_macro_f1": full["test_macro_f1"],
            "loo_refit_test_per_sequence_macro_f1": reduced["test_macro_f1"],
            "delta_refit": full["test_macro_f1"] - reduced["test_macro_f1"],
            "mean_paired_sequence_delta": float(diffs.mean()),
            "positive_sequence_fraction": float(np.mean(diffs > 0)),
            "full_valid_threshold": full["selected_threshold"],
            "loo_valid_threshold": reduced["selected_threshold"],
        })
    out = FIG4 / modality / "results"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "latent_loo_refit.csv", index=False)
    (out / "latent_loo_refit_models.json").write_text(json.dumps(list(fitted.values()), indent=2) + "\n")
    return pd.DataFrame(rows)


def protein(device: str, batch_tokens: int) -> pd.DataFrame:
    settings = pd.read_csv(FIG2 / "same_clan_evaluation/results/selected_valid_settings.csv")
    row = settings[(settings.experiment == "group3") & (settings.method == "CrossLatent")].iloc[0]
    latent_ids = [int(x) for x in json.loads(row.neuron_ids)]
    train_df = pd.read_csv(ROOT / "data/Protein/clan_CL0023/CL0023_group3.csv", dtype=str, keep_default_na=False)
    train_y, _ = flatten_labels(train_df, "mask_protein_domains")
    # Fig. 2's screened-baseline cache already contains every selected Group-3
    # latent, so reuse it instead of repeating SAE encoding.
    baseline_dir = FIG2 / "baselines/training/group3"
    baseline = json.loads((baseline_dir / "baseline_training_results.json").read_text())
    ranking = pd.read_csv(baseline_dir / "top256_univariate_ranking.csv").neuron_id.astype(int).tolist()
    missing = [latent for latent in latent_ids if latent not in ranking]
    if missing:
        raise ValueError(f"Selected protein latents absent from top-256 cache: {missing}")
    matrix = np.memmap(baseline["cache_file"], mode="r", dtype=np.float16,
                       shape=(len(train_y), int(baseline["screen_size"])))
    train_x = np.asarray(matrix[:, [ranking.index(x) for x in latent_ids]], dtype=np.float32)
    parts = {}
    for split in ("valid", "test"):
        z = np.load(FIG2 / f"same_clan_evaluation/activation_cache/{split}_needed_latents.npz", allow_pickle=True)
        ids = z["latent_ids"].astype(int).tolist()
        parts[split] = (z["activations"][:, [ids.index(x) for x in latent_ids]].astype(np.float32),
                        z["labels"].astype(np.float64), z["offsets"].astype(int))
    return evaluate_binary("protein", latent_ids, train_x, train_y, *parts["valid"], *parts["test"])


def rna(device: str, batch_tokens: int) -> pd.DataFrame:
    from sae_models import load_sae_from_checkpoint
    result = json.loads((ROOT / "ckpt/json/RNA/Rfam/mask_angle/ERNIERNA_mask_angle_results_9.json").read_text())
    cfg = result["features"]["mask_angle"]["combination"]
    latent_ids = [int(x) for x in cfg["neuron_ids"]]
    checkpoint = result["config"]["sae_checkpoint"]
    model = load_sae_from_checkpoint(checkpoint, sae_type=result["config"].get("sae_type", "relu"), device=device).eval()
    base = ROOT / "data/RNA/Rfam/ERNIERNA_embeddings"
    train_df = pd.read_csv(base / "Rfam_train.csv", dtype=str, keep_default_na=False)
    train_y, _ = flatten_labels(train_df, "mask_angle")
    train_x = encode_selected(base / "layer_12/Rfam_train.npy", model, latent_ids, device,
                              FIG4 / "rna/data/rfam_mask_angle_train_selected_latents.npy", batch_tokens)
    eval_df = pd.read_csv(base / "Rfam_test.csv", dtype=str, keep_default_na=False)
    all_x = encode_selected(base / "layer_12/Rfam_test.npy", model, latent_ids, device,
                            FIG4 / "rna/data/rfam_mask_angle_all_selected_latents.npy", batch_tokens)
    del model
    _, source_offsets = flatten_labels(eval_df, "mask_angle")
    manifest = pd.read_csv(FIG3 / "data_splits/rna/Rfam_manifest.csv")
    parts = {}
    for split in ("valid", "test"):
        rows = manifest.loc[manifest.split == split, "source_row"].astype(int).to_numpy()
        xs = []; ys = []; offsets = [0]
        for row in rows:
            a, b = source_offsets[row:row + 2]
            xs.append(all_x[a:b]); local_y = binary_mask(eval_df.iloc[row].mask_angle); ys.append(local_y)
            offsets.append(offsets[-1] + len(local_y))
        parts[split] = (np.concatenate(xs), np.concatenate(ys), np.asarray(offsets))
    return evaluate_binary("rna", latent_ids, train_x, train_y, *parts["valid"], *parts["test"])


def r2(y: np.ndarray, score: np.ndarray) -> float:
    denominator = float(np.sum((y - y.mean()) ** 2))
    return float(1 - np.sum((y - score) ** 2) / denominator)


def dna() -> pd.DataFrame:
    cache_dir = FIG4 / "dna/data"
    required = {split: cache_dir / f"k562_{split}_selected_latents.npz" for split in ("train", "valid", "test")}
    # Prefer explicit train caches.  On this machine the Evo2 train embedding is
    # unavailable, but the exact nested OLS coefficients/intercepts/R2 values
    # contain sufficient statistics to reconstruct the 3x3 centered Gram
    # matrix and X'y.  This is algebraically equivalent to refitting on train.
    if all(path.exists() for path in required.values()):
        data = {split: np.load(path) for split, path in required.items()}
        latent_ids = data["train"]["neuron_ids"].astype(int).tolist()
        train_x = data["train"]["activations"].reshape(-1, len(latent_ids)); train_y = data["train"]["labels"].reshape(-1)
        gram = np.cov(train_x, rowvar=False, bias=True)
        rhs = ((train_x - train_x.mean(0)).T @ (train_y - train_y.mean())) / len(train_y)
        means = train_x.mean(0); y_mean = float(train_y.mean())
        test = data["test"]
        provenance = "direct refit from cached training activations"
    else:
        parameter_dir = ROOT / "ckpt/json/DNA/data/test_sample_3000_a/results"
        configs = [json.loads((parameter_dir / f"Evo2_7b_262k_layer26_atac_K562_neuron_parameters_{k}.json").read_text())
                   ["features"]["atac_K562"]["combination"] for k in (1, 2, 3)]
        latent_ids = [int(x) for x in configs[-1]["neuron_ids"]]
        train_y = np.load(ROOT / "ckpt/json/DNA/data/test_sample_3000_a/atac_K562.npy", mmap_mode="r").reshape(-1).astype(np.float64)
        y_mean = float(train_y.mean()); y_var = float(np.mean((train_y - y_mean) ** 2))
        # Unknown vector: g11,g12,g13,g22,g23,g33,h1,h2,h3.
        def g_index(i: int, j: int) -> int:
            return {(0, 0): 0, (0, 1): 1, (0, 2): 2, (1, 1): 3, (1, 2): 4, (2, 2): 5}[tuple(sorted((i, j)))]
        equations = []; targets = []
        for k, cfg in enumerate(configs, start=1):
            beta = np.asarray(cfg["coefficients"], dtype=np.float64)
            for i in range(k):
                row = np.zeros(9)
                for j in range(k): row[g_index(i, j)] += beta[j]
                row[6 + i] -= 1
                equations.append(row); targets.append(0.0)
            row = np.zeros(9); row[6:6 + k] = beta
            equations.append(row); targets.append(float(cfg["composite_r2"]) * y_var)
        solution, residuals, rank, _ = np.linalg.lstsq(np.asarray(equations), np.asarray(targets), rcond=None)
        if rank < 9:
            raise ValueError(f"DNA sufficient-statistic system is rank deficient: {rank}/9")
        gram = np.asarray([[solution[g_index(i, j)] for j in range(3)] for i in range(3)])
        rhs = solution[6:]
        means = np.zeros(3)
        for k, cfg in enumerate(configs, start=1):
            beta = np.asarray(cfg["coefficients"], dtype=np.float64)
            means[k - 1] = (y_mean - float(cfg["intercept_b"]) - beta[:-1] @ means[:k - 1]) / beta[-1]
        test = np.load(cache_dir / "k562_test_selected_latents.npz")
        provenance = "exact train sufficient statistics reconstructed from nested OLS coefficients/intercepts/R2"
    test_x = test["activations"].reshape(-1, len(latent_ids)); test_y = test["labels"].reshape(-1)
    variants = [(None, np.arange(len(latent_ids)))] + [(j, np.delete(np.arange(len(latent_ids)), j)) for j in range(len(latent_ids))]
    scores = {}
    models = []
    for removed, keep in variants:
        coef = np.linalg.lstsq(gram[np.ix_(keep, keep)], rhs[keep], rcond=None)[0]
        intercept = float(y_mean - means[keep] @ coef)
        score = predict(test_x, keep, intercept, coef); scores[removed] = r2(test_y, score)
        models.append({"removed_latent_id": None if removed is None else latent_ids[removed], "retained_latent_ids": [latent_ids[i] for i in keep],
                       "intercept": intercept, "coefficients": coef.tolist(), "test_r2": scores[removed]})
    rows = [{"modality": "dna", "step": j + 1, "latent_id": latent_id, "full_test_r2": scores[None],
             "loo_refit_test_r2": scores[j], "delta_refit": scores[None] - scores[j]}
            for j, latent_id in enumerate(latent_ids)]
    out = FIG4 / "dna/results"; pd.DataFrame(rows).to_csv(out / "latent_loo_refit.csv", index=False)
    (out / "latent_loo_refit_models.json").write_text(json.dumps({"provenance": provenance, "models": models}, indent=2) + "\n")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modalities", default="protein,rna,dna")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-tokens", type=int, default=32000)
    args = parser.parse_args()
    outputs = []
    for name in [x.strip() for x in args.modalities.split(",") if x.strip()]:
        print(f"[{name}] strict LOO refit", flush=True)
        outputs.append({"protein": protein, "rna": rna, "dna": lambda *_: dna()}[name](args.device, args.batch_tokens))
    print(pd.concat(outputs, ignore_index=True).to_string(index=False))


if __name__ == "__main__":
    main()
