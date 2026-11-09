"""Auditable raw embedding probe experiment using frozen repository splits.

Run inventory, then Fig2, then the biological100 workers, then aggregate.
Every task writes independently; no existing inputs or results are modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from grabspark.raw_probes import accumulate, confusion, ols_path, predict, ridge_path, select_validation, threshold_grid

BIO = REPO / "fig/paper_figures/fig3/biological100"
FIG2 = REPO / "fig/paper_figures/fig2"
ESM = REPO / "InterPLM/esm2_model/esm2_t33_650M_UR50D"
RNA_WEIGHT = REPO / "BERT_new/model/two_mlm_checkpoint/checkpoint_best.pt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.12g")


def specs() -> list[dict]:
    config = json.loads((REPO / "ckpt/json/Protein/protein_domains/group3_protein_domains/ESM2_650M_33_mask_protein_domains_neuron_parameters_group3.json").read_text())["config"]
    tasks = [{"suite": "fig2", "task": "Group3", "modality": "protein", "dim": 1280,
              "label": "mask_protein_domains", "train_embedding": config["embedding_npy"],
              "splits": {"train": config["csv_path"],
                         "valid": str(FIG2 / "data/CL0023_filtered_valid.csv"),
                         "test": str(FIG2 / "data/CL0023_filtered_test.csv"),
                         "PF00069": str(FIG2 / "cross_clan/data/PF00069_filtered.csv"),
                         "PF04055": str(FIG2 / "cross_clan/data/PF04055_filtered.csv")}}]
    for row in pd.read_csv(BIO / "manifests/protein_tasks_frozen.csv").to_dict("records"):
        task = row["accession"]
        config_path = BIO / f"results/protein_domain_latent10/{task}/{task}_mask_target_domain_latent10_1.json"
        config = json.loads(config_path.read_text())["config"]
        tasks.append({"suite": "protein60", "task": task, "modality": "protein", "dim": 1280,
                      "label": "mask_target_domain", "train_embedding": config["embedding_npy"],
                      "historical_train_csv": config["csv_path"],
                      "splits": {s: str(BIO / f"splits/protein/{task}/{name}.csv")
                                 for s, name in (("train", "train"), ("valid", "validation"), ("test", "test"))}})
    for row in pd.read_csv(BIO / "manifests/rna40_diverse_tasks.csv").to_dict("records"):
        task, label = row["task"], row["label_column"]
        config = json.loads((BIO / f"results/rna40_latent10/{task}/{task}_{label}_latent10_1.json").read_text())["config"]
        tasks.append({"suite": "rna40", "task": task, "modality": "rna", "dim": 768,
                      "label": label, "train_embedding": config["embedding_npy"],
                      "historical_train_csv": config["csv_path"],
                      "splits": {s: str(BIO / f"splits/rna/{task}/{s}.csv") for s in ("train", "valid", "test")}})
    return tasks


def load_frame(path: Path, spec: dict, verify_original_md5: bool = True) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame["source_row"] = np.arange(len(frame))
    frame["original_sequence"] = frame.sequence
    if spec["modality"] == "rna":
        sequence = frame.sequence.str.upper().str.replace("T", "U", regex=False)
        keep = sequence.str.len().between(1, 1022)
        frame = frame.loc[keep].copy()
        frame["sequence"] = sequence.loc[keep].map(lambda s: "".join("N" if c in "RYSWKMBDHV" else c for c in s))
        if any(set(s) - set("ACGUN") for s in frame.sequence):
            raise ValueError(f"{path}: unsupported RNA bases")
    lengths = frame.sequence.str.len().to_numpy(dtype=int)
    if not len(lengths) or not np.all((lengths >= 1) & (lengths <= 1022)):
        raise ValueError(f"{path}: invalid lengths")
    masks = frame[spec["label"]].tolist()
    if any(len(m) != n or set(m) - set("01") for m, n in zip(masks, lengths)):
        raise ValueError(f"{path}: label boundary mismatch")
    if "length" in frame and not np.array_equal(frame.length.astype(int), lengths):
        raise ValueError(f"{path}: stored length differs from sequence")
    if "sequence_md5" in frame and verify_original_md5:
        md5 = frame.original_sequence.map(lambda s: hashlib.md5(s.encode()).hexdigest())
        if not np.array_equal(md5, frame.sequence_md5):
            raise ValueError(f"{path}: sequence MD5 mismatch")
    frame["sequence_sha256"] = frame.sequence.map(lambda s: hashlib.sha256(s.encode()).hexdigest())
    ids = next((c for c in ("uniprot_id", "name", "sequence_id", "id") if c in frame), None)
    frame["sequence_id"] = frame[ids] if ids else frame.source_row.astype(str)
    frame["group"] = frame.get("homology_component_30pct", pd.Series("", index=frame.index))
    offsets = np.r_[0, np.cumsum(lengths)]
    labels = np.frombuffer("".join(masks).encode(), dtype=np.uint8).copy() - ord("0")
    return frame.reset_index(drop=True), labels, offsets


def ordered_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256("\n".join(frame.sequence_sha256).encode()).hexdigest()


def inventory(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"seed": 20261006, "scope": "Fig2 Group3; Fig3 protein60, RNA40, K562 DNA",
                "k_grid": list(range(1, 11)), "lambda_grid": [10.0 ** i for i in range(-8, 5)],
                "threshold_grid": (np.arange(21) / 20).tolist(),
                "fit": "token-equal squared error; unpenalized intercept; raw linear scores",
                "OLS_gain": "(g_j.T @ r)^2 / (g_j.T @ g_j)",
                "OLS_numerical_zero": "residual norm <= 1e-12 * max(original norm, 1)",
                "OLS_ties": "lowest coordinate ID on exact gain tie",
                "ridge_objective": "SSE/(2*N) + lambda*||w_standardized||^2/2; alpha=N*lambda",
                "metric": "mean per-sequence (positive F1 + negative F1)/2; zero denominators = 0",
                "fig2_selection": "max valid F1, lower K / larger lambda, threshold closer .5, lower threshold",
                "biological100_selection": "max valid F1, lower K / larger lambda, lower threshold",
                "dna_selection": "max validation pooled raw R2, lower K / larger lambda",
                "dna_smoothing": {"sigma_nt": 64, "mode": "nearest", "truncate": 4.0, "within_window": 1024},
                "test_policy": "save frozen JSON/hash before reading test labels for evaluation",
                "training_statistics": "float64 stable batch-centered accumulation; train-only normalization",
                "extraction": "frozen local weights; eval/no_grad; float32; no autocast; BOS/EOS/pad removed",
                "historical_batch_note": "SAE train/eval batch composition differed; raw probes bypass SAE"}
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Existing frozen protocol differs")
    save_json(protocol_path, protocol)
    weights = {"ESM2-650M": str(ESM / "pytorch_model.bin"), "ERNIE-RNA": str(RNA_WEIGHT)}
    weight_info = {name: {"path": path, "exists": Path(path).exists(),
                          "sha256": sha256(Path(path)) if Path(path).exists() else None} for name, path in weights.items()}
    rows, checks = [], []
    for spec in specs():
        frames = {}
        for split, value in spec["splits"].items():
            path = Path(value)
            frame, labels, offsets = load_frame(path, spec)
            frames[split] = frame
            historical = Path(spec["train_embedding"]) if split == "train" else None
            row = {"suite": spec["suite"], "task": spec["task"], "split": split,
                   "csv_path": str(path.resolve()), "csv_sha256": sha256(path),
                   "rows": len(frame), "source_rows": len(pd.read_csv(path, usecols=["sequence"])),
                   "tokens": len(labels), "label_column": spec["label"],
                   "ordered_sequence_sha256": ordered_hash(frame), "dim": spec["dim"],
                   "model": "ESM2-650M" if spec["modality"] == "protein" else "ERNIE-RNA",
                   "layer": 33 if spec["modality"] == "protein" else 12,
                   "coordinates": "zero-based hidden columns; concatenated sequence positions; no special/pad tokens",
                   "embedding_path": str(historical) if historical and historical.exists() else str(output / f"cache/{spec['suite']}/{spec['task']}/{split}.npy"),
                   "embedding_exists": bool(historical and historical.exists()), "dtype": "pending extraction"}
            if historical and historical.exists():
                array = np.load(historical, mmap_mode="r")
                row.update(embedding_sha256=sha256(historical), dtype=str(array.dtype), embedding_shape=str(array.shape))
                if array.shape != (len(labels), spec["dim"]):
                    raise ValueError(f"{historical}: incorrect shape")
            rows.append(row)
        overlaps = {}
        group_overlaps = {}
        names = list(frames)
        for i, left in enumerate(names):
            for right in names[i+1:]:
                overlaps[f"{left}/{right}"] = len(set(frames[left].sequence_sha256) & set(frames[right].sequence_sha256))
                groups_left = set(frames[left].group) - {""}
                groups_right = set(frames[right].group) - {""}
                group_overlaps[f"{left}/{right}"] = len(groups_left & groups_right) if groups_left and groups_right else None
        train_match = None
        if spec.get("historical_train_csv") and Path(spec["historical_train_csv"]).exists():
            # Historical extraction replaces T/IUPAC in the sequence column but
            # retains the original MD5; verify against the frozen source below.
            old, old_y, _ = load_frame(Path(spec["historical_train_csv"]), spec, verify_original_md5=False)
            train_match = (ordered_hash(old) == ordered_hash(frames["train"]))
            if "sequence_md5" in old:
                train_match = train_match and np.array_equal(old.sequence_md5, frames["train"].sequence_md5)
            _, source_y, _ = load_frame(Path(spec["splits"]["train"]), spec)
            train_match = train_match and np.array_equal(old_y, source_y)
            if not train_match:
                raise ValueError(f"{spec['task']}: train row mapping differs from historical extraction")
        checks.append({"suite": spec["suite"], "task": spec["task"],
                       "exact_sequence_overlap_counts": overlaps, "component_overlap_counts": group_overlaps,
                       "historical_train_order_matches": train_match,
                       "split_claim": "only protein60 has frozen 30pct homology components; RNA/Fig2 exact overlap check does not establish homology separation"})
    # DNA's historical embedding paths refer to a different user's absent project.
    dna_json = REPO / "ckpt/json/DNA/data/test_sample_3000_a/results/Evo2_7b_262k_layer26_atac_K562_neuron_parameters_1.json"
    dna_config = json.loads(dna_json.read_text())["config"]
    dna = {"task": "K562", "suite": "dna", "status": "blocked_missing_inputs",
           "historical_config": str(dna_json), "historical_config_sha256": sha256(dna_json),
           "historical_embedding": dna_config["embedding_npy"],
           "embedding_exists": Path(dna_config["embedding_npy"]).exists(),
           "required_model": "Evo2 7B 262k block26, exact frozen weights and extraction node",
           "dimension": "4096 reported; cannot verify without model configuration/raw cache",
           "reason": "Local raw Evo2 embedding and frozen Evo2 base weights/config were not found; predictions/labels are not embeddings"}
    dna_inputs = [REPO / "fig/paper_figures/fig3/data_splits/dna/test_sample_3000_b_manifest.csv"]
    for folder in ("test_sample_3000_a", "test_sample_3000_b", "chr2_k562_peak150"):
        dna_inputs += [REPO / f"ckpt/json/DNA/data/{folder}/{file}" for file in ("sequences.fasta", "atac_K562.npy", "windows.tsv")]
    dna["available_inputs"] = [{"path": str(p), "exists": p.exists(), "sha256": sha256(p) if p.exists() else None} for p in dna_inputs]
    save_json(output / "dna_dependency_audit.json", dna)
    write_csv(output / "input_manifest.csv", rows)
    save_json(output / "qc_report.json", {"tasks": checks, "weights": weight_info, "DNA": dna,
                                          "status": "CSV/labels/IDs/splits checked; cache boundaries verified by task before fitting"})
    save_json(output / "environment.json", {"python": sys.executable, "python_version": sys.version,
                                            "numpy": np.__version__, "pandas": pd.__version__, "platform": platform.platform()})
    print(f"Inventory: {len(checks)} classification tasks, {len(rows)} splits; DNA dependencies recorded", flush=True)


class Extractor:
    def __init__(self, modality: str, device: str, batch_size: int) -> None:
        import torch
        torch.set_num_threads(4)
        torch.manual_seed(20261006)
        torch.backends.cuda.matmul.allow_tf32 = False
        self.torch, self.modality, self.device, self.batch_size = torch, modality, device, batch_size
        if modality == "protein":
            from transformers import AutoTokenizer, EsmModel
            self.tokenizer = AutoTokenizer.from_pretrained(str(ESM), local_files_only=True)
            self.model = EsmModel.from_pretrained(str(ESM), add_pooling_layer=False, local_files_only=True).to(device).eval()
            if self.model.config.hidden_size != 1280 or self.model.config.num_hidden_layers != 33:
                raise ValueError("Wrong ESM2 frozen architecture")
        else:
            sys.path.insert(0, str(REPO / "grabspark/extract_embedding"))
            from extract_ERNIERNA_embedding import Twod_mlm_onestage, load_model_pretrained, oned_to_twod_bert_input, seq_to_rna_index
            pretrained = load_model_pretrained(str(RNA_WEIGHT), {"data": str(REPO / "BERT_new/pretrain_dict")})
            self.model = Twod_mlm_onestage(pretrained.encoder).to(device).eval()
            self.rna_input, self.rna_index = oned_to_twod_bert_input, seq_to_rna_index

    def batches(self, sequences: list[str]) -> Iterator[tuple[int, list[np.ndarray]]]:
        torch = self.torch
        batch_size = self.batch_size if self.modality == "protein" else 1
        for start in range(0, len(sequences), batch_size):
            seqs = sequences[start:start+batch_size]
            with torch.no_grad():
                if self.modality == "protein":
                    inputs = self.tokenizer(seqs, return_tensors="pt", padding=True, truncation=False)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    hidden = self.model(**inputs, output_hidden_states=True).hidden_states[33]
                    arrays = [hidden[i, 1:len(s)+1].detach().cpu().numpy() for i, s in enumerate(seqs)]
                else:
                    indices, lengths = self.rna_index(seqs)
                    one, two = self.rna_input(indices[0], lengths[0])
                    hidden = self.model(one.to(self.device), two.to(self.device))
                    arrays = [hidden[0, 1:-1].detach().cpu().numpy()]
            yield start, arrays


def embedding(frame: pd.DataFrame, offsets: np.ndarray, spec: dict, split: str,
              args, extractor: Extractor) -> tuple[np.ndarray, dict]:
    folder = args.output / f"cache/{spec['suite']}/{spec['task']}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{split}.npy"
    historical = Path(spec["train_embedding"]) if split == "train" else None
    old_cache = bool(historical and historical.exists())
    if old_cache:
        path = historical
    meta_path = folder / f"{split}.metadata.json"
    expected = (int(offsets[-1]), spec["dim"])
    order = ordered_hash(frame)
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["ordered_sequence_sha256"] != order or meta["csv_sha256"] != sha256(Path(spec["splits"][split])):
            raise ValueError("Cached sequence mapping changed")
        model_name = "ESM2-650M" if spec["modality"] == "protein" else "ERNIE-RNA"
        expected_weight = json.loads((args.output / "qc_report.json").read_text())["weights"][model_name]["sha256"]
        if meta["weight_sha256"] != expected_weight:
            raise ValueError("Cached representation checkpoint changed")
        array = np.load(path, mmap_mode="r")
        if array.shape != expected or sha256(path) != meta["embedding_sha256"]:
            raise ValueError("Cache shape/hash changed")
        return array, meta
    sample_checks = []
    if old_cache:
        array = np.load(path, mmap_mode="r")
        if array.shape != expected:
            raise ValueError(f"{path}: shape {array.shape} != {expected}")
        # All token positions of first/middle/last sequences, both sequence boundaries.
        for row in sorted(set([0, len(frame)//2, len(frame)-1])):
            fresh = next(extractor.batches([frame.iloc[row].sequence]))[1][0]
            stored = np.asarray(array[offsets[row]:offsets[row+1]])
            difference = float(np.max(np.abs(fresh.astype(float)-stored.astype(float))))
            aligned = np.allclose(stored, fresh, atol=2e-4, rtol=2e-4)
            sample_checks.append({"row": row, "sequence_id": frame.iloc[row].sequence_id,
                                  "sequence_sha256": frame.iloc[row].sequence_sha256,
                                  "max_absolute_difference": difference, "allclose": bool(aligned)})
            if not aligned:
                raise ValueError(f"Historical cache sample mismatch at row {row}: max error {difference}")
    else:
        temporary = path.with_suffix(".partial.npy")
        array = np.lib.format.open_memmap(temporary, mode="w+", dtype="float32", shape=expected)
        for start, arrays in extractor.batches(frame.sequence.tolist()):
            for local, values in enumerate(arrays):
                row = start + local
                if values.shape != (offsets[row+1]-offsets[row], spec["dim"]):
                    raise ValueError("Extracted token boundaries differ")
                array[offsets[row]:offsets[row+1]] = values
            if start % 200 == 0:
                print(f"EXTRACT {spec['suite']}/{spec['task']}/{split} {start}/{len(frame)}", flush=True)
        array.flush()
        del array
        temporary.replace(path)
        array = np.load(path, mmap_mode="r")
    meta = {"csv_path": spec["splits"][split], "csv_sha256": sha256(Path(spec["splits"][split])),
            "embedding_path": str(path), "embedding_sha256": sha256(path),
            "rows": len(frame), "tokens": int(offsets[-1]), "shape": list(array.shape), "dtype": str(array.dtype),
            "ordered_sequence_sha256": order, "sample_boundary_checks": sample_checks,
            "historical_cache": old_cache, "model": "ESM2-650M" if spec["modality"] == "protein" else "ERNIE-RNA",
            "layer": 33 if spec["modality"] == "protein" else 12, "seed": 20261006,
            "device": args.device, "batch_size": args.batch_size if spec["modality"] == "protein" else 1,
            "precision": "float32 inference; no autocast; TF32 disabled", "mode": "eval/no_grad",
            "weight_sha256": json.loads((args.output / "qc_report.json").read_text())["weights"]["ESM2-650M" if spec["modality"] == "protein" else "ERNIE-RNA"]["sha256"]}
    save_json(meta_path, meta)
    write_csv(folder / f"{split}.sequence_mapping.csv", frame[["source_row", "sequence_id", "sequence_sha256", "group"]].assign(token_start=offsets[:-1], token_end=offsets[1:]).to_dict("records"))
    return array, meta


def score_models(array: np.ndarray, models: list[dict], chunk: int = 16384) -> np.ndarray:
    coefficients = np.zeros((array.shape[1], len(models)))
    for i, model in enumerate(models):
        coefficients[model["coordinates"], i] = model["coefficients"]
    intercepts = np.array([m["intercept"] for m in models])
    scores = np.empty((len(array), len(models)), dtype=np.float64)
    for start in range(0, len(array), chunk):
        scores[start:start+chunk] = np.asarray(array[start:start+chunk], dtype=np.float64) @ coefficients + intercepts
        if not np.isfinite(scores[start:start+chunk]).all():
            raise ValueError("Nonfinite validation/test prediction scores")
    return scores


def run_task(spec: dict, args, extractor: Extractor) -> None:
    start_time = time.time()
    folder = args.output / f"models/{spec['suite']}/{spec['task']}"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "complete.json").exists():
        print(f"SKIP complete {spec['suite']}/{spec['task']}", flush=True)
        return
    save_json(folder / "status.json", {"status": "running", "pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat()})
    frame, y, offsets = load_frame(Path(spec["splits"]["train"]), spec)
    array, train_meta = embedding(frame, offsets, spec, "train", args, extractor)
    statistics = accumulate((array[i:i+args.chunk_tokens], y[i:i+args.chunk_tokens]) for i in range(0, len(y), args.chunk_tokens))
    ols = ols_path(statistics)
    if len(ols) != 10:
        raise ValueError(f"Only {len(ols)} independent coordinates; cannot deliver K1..10")
    ridge, standardization = ridge_path(statistics)
    models = ols + ridge
    save_json(folder / "all_candidates.json", {"models": models, "OLS_train_mean": statistics.mean_x.tolist(),
                                               "train_label_mean": statistics.mean_y, "train_input": train_meta})
    save_json(folder / "standardization.json", standardization)
    del array, statistics
    frame, y, offsets = load_frame(Path(spec["splits"]["valid"]), spec)
    array, valid_meta = embedding(frame, offsets, spec, "valid", args, extractor)
    scores = score_models(array, models)
    candidates = []
    for index, model in enumerate(models):
        grid = threshold_grid(y, scores[:, index], offsets)
        for threshold, value in zip(np.arange(21)/20, grid):
            candidates.append({"suite": spec["suite"], "task": spec["task"], "model_index": index,
                               "method": model["method"], "K": model["K"], "lambda": model["lambda"],
                               "threshold": float(threshold), "valid_metric": float(value)})
    write_csv(folder / "validation_grid.csv", candidates)
    selected = [select_validation([r for r in candidates if r["method"] == method], spec["suite"] == "fig2") for method in ("Raw-OLS", "Raw-Ridge")]
    for row in selected:
        row["frozen_at"] = datetime.now(timezone.utc).isoformat()
        row["protocol_sha256"] = sha256(args.output / "protocol.json")
    # This durable freeze precedes loading any test label for scoring.
    save_json(folder / "frozen_settings.json", {"selected": selected, "valid_input": valid_meta})
    freeze_hash = sha256(folder / "frozen_settings.json")
    for row in selected:
        row["freeze_sha256"] = freeze_hash
    write_csv(folder / "frozen_settings.csv", selected)
    del array, scores
    summaries, per_sequence, curves = [], [], []
    for split in [s for s in spec["splits"] if s not in ("train", "valid")]:
        frame, y, offsets = load_frame(Path(spec["splits"][split]), spec)
        array, meta = embedding(frame, offsets, spec, split, args, extractor)
        scores = score_models(array, models)
        pred_folder = args.output / f"predictions/{spec['suite']}/{spec['task']}"
        pred_folder.mkdir(parents=True, exist_ok=True)
        np.savez(pred_folder / f"{split}.npz", scores=scores, labels=y, offsets=offsets,
                 sequence_ids=frame.sequence_id.to_numpy(dtype=str), sequence_sha256=frame.sequence_sha256.to_numpy(dtype=str))
        for setting in selected:
            index = setting["model_index"]
            sequence_values = []
            for i, row in frame.iterrows():
                counts = confusion(y[offsets[i]:offsets[i+1]], scores[offsets[i]:offsets[i+1], index], setting["threshold"])
                sequence_values.append(counts["macro_f1"])
                per_sequence.append({"suite": spec["suite"], "task": spec["task"], "split": split,
                                     "method": setting["method"], "source_row": int(row.source_row),
                                     "sequence_id": row.sequence_id, "sequence_sha256": row.sequence_sha256,
                                     "group": row.group, "length": int(offsets[i+1]-offsets[i]),
                                     "positive_prevalence": float(y[offsets[i]:offsets[i+1]].mean()), **counts})
            summaries.append({"suite": spec["suite"], "task": spec["task"], "split": split,
                              "method": setting["method"], "K": setting["K"], "lambda": setting["lambda"],
                              "threshold": setting["threshold"], "macro_f1": float(np.mean(sequence_values)),
                              "n_sequences": len(frame), "n_tokens": len(y), "freeze_sha256": freeze_hash})
        for index in range(10):
            krows = [r for r in candidates if r["model_index"] == index]
            setting = select_validation(krows, spec["suite"] == "fig2")
            metric = threshold_grid(y, scores[:, index], offsets)[round(setting["threshold"] * 20)]
            curves.append({"suite": spec["suite"], "task": spec["task"], "split": split,
                           "method": "Raw-OLS", "K": index+1, "threshold": setting["threshold"],
                           "valid_macro_f1": setting["valid_metric"], "test_macro_f1": float(metric),
                           "is_frozen_k": index == selected[0]["model_index"], "test_role": "descriptive, not for selection"})
        del array, scores
    write_csv(folder / "test_summary.csv", summaries)
    write_csv(folder / "test_per_sequence.csv", per_sequence)
    write_csv(folder / "budget_curves.csv", curves)
    completion = {"status": "complete", "suite": spec["suite"], "task": spec["task"],
                  "elapsed_seconds": time.time()-start_time, "completed_at": datetime.now(timezone.utc).isoformat(),
                  "worker_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  "device": args.device, "freeze_sha256": freeze_hash}
    save_json(folder / "complete.json", completion)
    save_json(folder / "status.json", completion)
    print(f"COMPLETE {spec['suite']}/{spec['task']} {summaries}", flush=True)


def aggregate(output: Path) -> None:
    folders = sorted((output / "models").glob("*/*/complete.json"))
    for filename in ("validation_grid.csv", "frozen_settings.csv", "test_summary.csv", "test_per_sequence.csv", "budget_curves.csv"):
        dtypes = {"sequence_id": str, "sequence_sha256": str, "group": str} if filename == "test_per_sequence.csv" else None
        frames = [pd.read_csv(p.parent / filename, dtype=dtypes, keep_default_na=filename != "test_per_sequence.csv") for p in folders]
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(output / filename, index=False, float_format="%.12g")
    summaries, pairs, sequence_pairs = [], [], []
    if not folders:
        return
    raw = pd.read_csv(output / "test_summary.csv")
    sequence_raw = pd.read_csv(output / "test_per_sequence.csv", dtype={"sequence_id": str, "sequence_sha256": str, "group": str}, keep_default_na=False)
    historical_checks = []
    for suite in ("fig2", "protein60", "rna40"):
        if suite == "fig2":
            same = pd.read_csv(FIG2 / "same_clan_evaluation/results/frozen_test_summary.csv")
            same = same[same.experiment == "group3"]
            external = pd.read_csv(FIG2 / "cross_clan/evaluation/results/cross_clan_frozen_summary.csv")
            external = external[external.experiment == "group3"]
            for split, expected in (("test", .5801), ("PF00069", .4141), ("PF04055", .5975)):
                table = same if split == "test" else external[external.external_dataset == split]
                column = "test_sequence_macro_class_f1" if split == "test" else "sequence_macro_class_f1"
                cl = float(table.loc[table.method == "CrossLatent", column].iloc[0])
                historical_checks.append({"split": split, "expected_rounded": expected, "observed": cl, "matches": round(cl, 4) == expected})
                for method in ("CrossLatent", "best_single"):
                    value = float(table.loc[table.method == method, column].iloc[0])
                    summaries.append({"suite": suite, "task": "Group3", "split": split, "method": method, "macro_f1": value, "source": "frozen historical table"})
        else:
            historical = pd.read_csv(BIO / f"evaluation/{'protein60' if suite == 'protein60' else 'rna40'}_valid_test/summary.csv")
            task_column = "family" if suite == "protein60" else "task"
            for _, row in historical.iterrows():
                for method, column in (("CrossLatent", "test_macro_f1"), ("best_single", "best_single_test_macro_f1")):
                    summaries.append({"suite": suite, "task": row[task_column], "split": "test", "method": method, "macro_f1": row[column], "source": "frozen historical table"})
    history = pd.DataFrame(summaries)
    baseline_path = FIG2 / "baselines/evaluation/results/frozen_test_summary.csv"
    baseline_table = pd.read_csv(baseline_path)
    for _, row in baseline_table[baseline_table.experiment == "group3"].iterrows():
        split = "test" if row.evaluation_dataset == "CL0023 test" else row.evaluation_dataset
        summaries.append({"suite": "fig2", "task": "Group3", "split": split,
                          "method": row.method, "macro_f1": row.sequence_macro_class_f1,
                          "source": str(baseline_path)})
    history = pd.DataFrame(summaries)
    write_csv(output / "historical_reference.csv", summaries)
    comparisons = pd.concat([raw[["suite", "task", "split", "method", "macro_f1"]],
                             history[["suite", "task", "split", "method", "macro_f1"]]], ignore_index=True)
    comparisons = comparisons.merge(raw[["suite", "task", "split"]].drop_duplicates(),
                                    on=["suite", "task", "split"], validate="many_to_one")
    comparisons.pivot(index=["suite", "task", "split"], columns="method", values="macro_f1").reset_index().to_csv(
        output / "all_method_comparison.csv", index=False, float_format="%.12g")
    modality_methods = comparisons[comparisons.suite != "fig2"].groupby(["suite", "method"]).macro_f1.agg(
        completed_tasks="count", task_mean="mean", task_median="median").reset_index()
    modality_methods.to_csv(output / "modality_method_summary.csv", index=False, float_format="%.12g")
    save_json(output / "historical_consistency.json", {"fig2": historical_checks})
    task_specs = {(s["suite"], s["task"]): s for s in specs()}
    for _, row in raw.iterrows():
        reference = history[(history.suite == row.suite) & (history.task == row.task) & (history.split == row.split) & (history.method == "CrossLatent")]
        if len(reference) != 1:
            raise ValueError("Historical task pairing is not unique")
        cl = float(reference.macro_f1.iloc[0])
        pairs.append({"suite": row.suite, "task": row.task, "split": row.split, "baseline": row.method,
                      "crosslatent_f1": cl, "baseline_f1": row.macro_f1, "delta_crosslatent_minus_baseline": cl-row.macro_f1})
        if row.suite == "fig2":
            old_path = (FIG2 / "same_clan_evaluation/results/frozen_test_per_sequence.csv" if row.split == "test"
                        else FIG2 / "cross_clan/evaluation/results/cross_clan_frozen_per_sequence.csv")
            old = pd.read_csv(old_path)
            old = old[(old.experiment == "group3") & (old.method == "CrossLatent")]
            if row.split != "test":
                old = old[old.external_dataset == row.split]
            part = sequence_raw[(sequence_raw.suite == "fig2") & (sequence_raw.method == row.method) & (sequence_raw.split == row.split)]
            merged = part.merge(old, left_on="sequence_id", right_on="uniprot_id", validate="one_to_one")
            if len(merged) != len(part) or not np.array_equal(merged.length, merged.seq_length):
                raise ValueError("Fig2 historical sequence-ID/length pairing mismatch")
            if not np.array_equal(merged.tp_x + merged.fn_x, merged.positive_count):
                raise ValueError("Fig2 historical positive-label count mismatch")
            if abs(float(merged.class_macro_f1.mean())-cl) > 1e-10:
                raise ValueError("Fig2 historical per-sequence mean differs from frozen table")
            for _, item in merged.iterrows():
                sequence_pairs.append({"suite": "fig2", "task": "Group3", "split": row.split,
                                       "baseline": row.method, "source_row": item.source_row,
                                       "sequence_sha256": item.sequence_sha256,
                                       "crosslatent_f1": item.class_macro_f1, "baseline_f1": item.macro_f1,
                                       "delta": item.class_macro_f1-item.macro_f1})
        if row.suite in ("protein60", "rna40"):
            hp = BIO / f"evaluation/{row.suite}_valid_test/{row.task}/test_per_sequence.csv"
            old = pd.read_csv(hp, dtype=str, keep_default_na=False)
            part = sequence_raw[(sequence_raw.suite == row.suite) & (sequence_raw.task == row.task) & (sequence_raw.method == row.method)]
            # Historical rows use source_row; independently verify MD5 via frozen split.
            spec = task_specs[(row.suite, row.task)]
            frame, _, _ = load_frame(Path(spec["splits"]["test"]), spec)
            mapping = frame.set_index("source_row")
            indices = old.source_row.astype(int).to_numpy()
            if len(set(indices)) != len(indices) or len(part) != len(old) or set(part.source_row) != set(indices):
                raise ValueError("Historical per-sequence source rows are not one-to-one")
            matched = part.set_index("source_row").loc[indices]
            expected = mapping.loc[indices]
            if not np.array_equal(matched.sequence_sha256, expected.sequence_sha256):
                raise ValueError("Historical per-sequence row pairing mismatch")
            if "sequence_md5" in old:
                original_md5 = expected.original_sequence.map(lambda s: hashlib.md5(s.encode()).hexdigest())
                if not np.array_equal(original_md5, old.sequence_md5):
                    raise ValueError("Historical paired MD5 mismatch")
            if "sequence_length" in old and not np.array_equal(expected.sequence.str.len(), old.sequence_length.astype(int)):
                raise ValueError("Historical paired sequence length mismatch")
            old_f1 = old.grabspark_macro_f1.astype(float).to_numpy()
            new_f1 = matched.macro_f1.to_numpy(dtype=float)
            sequence_pairs.extend(pd.DataFrame({"suite": row.suite, "task": row.task, "split": "test", "baseline": row.method,
                                                 "source_row": indices, "sequence_sha256": matched.sequence_sha256.to_numpy(),
                                                 "crosslatent_f1": old_f1, "baseline_f1": new_f1,
                                                 "delta": old_f1-new_f1}).to_dict("records"))
    sequence_pair_frame = pd.DataFrame(sequence_pairs)
    for pair in pairs:
        part = sequence_pair_frame[(sequence_pair_frame.suite == pair["suite"]) &
                                   (sequence_pair_frame.task == pair["task"]) &
                                   (sequence_pair_frame.baseline == pair["baseline"]) &
                                   (sequence_pair_frame.split == pair["split"])]
        pair["paired_sequences"] = len(part)
        pair["fraction_crosslatent_better"] = float((part.delta > 1e-12).mean())
        pair["fraction_tied"] = float((part.delta.abs() <= 1e-12).mean())
        pair["fraction_crosslatent_worse"] = float((part.delta < -1e-12).mean())
        pair["mean_paired_delta"] = float(part.delta.mean())
        if abs(pair["mean_paired_delta"] - pair["delta_crosslatent_minus_baseline"]) > 1e-9:
            raise ValueError("Paired per-sequence mean differs from task-level delta")
    write_csv(output / "paired_comparisons.csv", pairs)
    write_csv(output / "paired_per_sequence.csv", sequence_pairs)
    pair_frame = pd.DataFrame(pairs)
    task_summary = []
    for (suite, method), part in pair_frame.groupby(["suite", "baseline"]):
        delta = part.delta_crosslatent_minus_baseline
        task_summary.append({"suite": suite, "baseline": method, "completed_tasks": len(part),
                             "crosslatent_task_mean": part.crosslatent_f1.mean(), "baseline_task_mean": part.baseline_f1.mean(),
                             "crosslatent_task_median": part.crosslatent_f1.median(),
                             "baseline_task_median": part.baseline_f1.median(), "mean_delta": delta.mean(), "median_delta": delta.median(),
                             "crosslatent_better": int((delta > 1e-12).sum()), "tied": int((delta.abs() <= 1e-12).sum()),
                             "crosslatent_worse": int((delta < -1e-12).sum()),
                             "crosslatent_worse_tasks": ";".join(
                                 f"{r.task}:{r.split}" if suite == "fig2" else str(r.task)
                                 for _, r in part.loc[delta < -1e-12].iterrows())})
    write_csv(output / "task_summary.csv", task_summary)
    historical_curves = []
    for suite, task in sorted({(r["suite"], r["task"]) for r in pairs if r["suite"] != "fig2"}):
        table = pd.read_csv(BIO / f"evaluation/{suite}_valid_test/{task}/selection_curve.csv")
        table = table[table.method == "CrossLatent"]
        old_settings = json.loads((BIO / f"evaluation/{suite}_valid_test/{task}/summary.json").read_text())
        for k in range(1, 11):
            selected_curve = table[table.K == k].sort_values(["valid_macro_f1", "threshold"], ascending=[False, True]).iloc[0]
            historical_curves.append({"suite": suite, "task": task, "split": "test", "method": "CrossLatent",
                                      "K": k, "threshold": selected_curve.threshold,
                                      "valid_macro_f1": selected_curve.valid_macro_f1,
                                      "test_macro_f1": selected_curve.test_macro_f1,
                                      "is_frozen_k": k == old_settings["selected_k"],
                                      "test_role": "frozen historical descriptive trajectory; K threshold selected on validation"})
    write_csv(output / "historical_budget_curves.csv", historical_curves)
    if historical_curves:
        budget = pd.read_csv(output / "budget_curves.csv")
        pd.concat([budget, pd.DataFrame(historical_curves)], ignore_index=True).to_csv(output / "budget_curves.csv", index=False, float_format="%.12g")
    complete = {(json.loads(p.read_text())["suite"], json.loads(p.read_text())["task"]) for p in folders}
    missing = [{"suite": s["suite"], "task": s["task"]} for s in specs() if (s["suite"], s["task"]) not in complete]
    scope_path = output / "scope_amendment.json"
    dna_status = json.loads(scope_path.read_text()).get("dna_status", "deferred_by_user") if scope_path.exists() else "blocked_missing_raw_embedding_and_frozen_model"
    save_json(output / "completion.json", {"classification_completed": len(complete), "classification_required": 101,
                                           "classification_missing": missing,
                                           "dna": dna_status})
    # Merge extraction records into the inventory after results are complete.
    manifest = pd.read_csv(output / "input_manifest.csv").to_dict("records")
    for row in manifest:
        meta_path = output / f"cache/{row['suite']}/{row['task']}/{row['split']}.metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            row.update({key: meta[key] for key in ("embedding_path", "embedding_sha256", "dtype")})
            row["embedding_exists"] = True
            row["embedding_shape"] = str(meta["shape"])
    write_csv(output / "input_manifest.csv", manifest)
    qc = json.loads((output / "qc_report.json").read_text())
    overlap_notes = [f"{r['suite']}/{r['task']}: {r['exact_sequence_overlap_counts']}" for r in qc['tasks'] if any(r['exact_sequence_overlap_counts'].values())]
    display = pd.DataFrame(task_summary).drop(columns=["crosslatent_worse_tasks"])
    display = display[display.suite != "fig2"]
    effect_text = []
    for row in task_summary:
        if row["suite"] == "fig2":
            continue
        modality = "蛋白" if row["suite"] == "protein60" else "RNA"
        effect_text.append(
            f"{modality}已完成{row['completed_tasks']}个任务：CrossLatent对{row['baseline']}的任务等权平均Δ={row['mean_delta']:+.6f}，"
            f"中位Δ={row['median_delta']:+.6f}；CrossLatent较高/持平/较低={row['crosslatent_better']}/{row['tied']}/{row['crosslatent_worse']}。")
    fig2_display = pd.concat([raw[raw.suite == "fig2"][["split", "method", "macro_f1"]],
                              history[history.suite == "fig2"][["split", "method", "macro_f1"]]])
    fig2_display = fig2_display.pivot(index="split", columns="method", values="macro_f1")
    text = ["# 原始表示稀疏探针补充实验", "",
            f"已完成 {len(complete)}/101 个分类任务（Fig2 1组、蛋白60、RNA40）；DNA范围状态为{dna_status}，见scope_amendment.json。",
            "所有模型按验证集冻结后评价测试集；Raw-OLS 上限10坐标，Raw-Ridge使用全表示且参数量不同。", "",
            "Raw-OLS K=1为含自由系数与截距的单坐标平方损失读出，与历史best-single直接阈值化latent不同。原始hidden坐标不称作MLP neuron，也未赋予生物语义。", "",
            "## Fig2 Group3", "", fig2_display.to_markdown(), "",
            "Raw-OLS验证选择K=10、阈值0.50；Raw-Ridge选择λ=0.001、阈值0.45。三个评价集中Raw-OLS点估计均高于CrossLatent，当前Fig2结果不支持SAE坐标的低预算预测优势。PF04055差异较小；未报告区间，不作显著性断言。",
            "## 蛋白/RNA配对点估计", "", modality_methods.to_markdown(index=False), "",
            display.to_markdown(index=False), "",
            *effect_text, "",
            "完整下降任务列表在task_summary.csv，逐任务及逐序列差值在paired_comparisons.csv和paired_per_sequence.csv。",
            "每任务CrossLatent、Raw-OLS、Raw-Ridge与既有best-single并列表见all_method_comparison.csv；历史方法未重新训练。",
            "若尚未完成全范围，这些是进度性结果，未完成任务可能具有不同长度/计算难度，不能把当前子集汇总当作完整100任务结论。" if missing else "60蛋白与40RNA任务均已完成，汇总没有按新方法结果挑选任务。",
            "Δ定义为CrossLatent−原始表示基线，正值表示CrossLatent较高。蛋白/RNA汇总按任务等权。Fig2为三个评价集，不能称为三个独立训练任务。",
            "未报告置信区间或P值；预测性能比较不直接测量解释质量或专家收益。",
            "冻结历史结果来自既有表；逐序列配对仅在ID/源行号和序列hash核验后进行。",
            "RNA沿用各标签、T→U/IUPAC→N及1..1022过滤；保留source_row恢复原划分。", "",
            "## 划分与数据限制", "", *overlap_notes,
            "Fig2重复序列的标签另经逐位核验：train/valid及train/test重复的mask一致；valid/test的10条重复序列中1条mask不一致。保留原标签与划分，详见fig2_label_overlap_audit.json。",
            "Fig2沿用既有划分，上述重复未被删除，因此同家族结果不宜视为完全独立的泛化证据。蛋白60未检出跨划分30pct component重叠；RNA没有宣称同源隔离。", "",
            ("DNA按用户最终指示不再执行，已移出本轮补充实验范围，不列为待补任务。未产生DNA Raw-OLS/Ridge结果。" if dna_status == "excluded_by_user" else
             "DNA按用户后续指示暂缓，待提供模型/表示路径后另行补做。未产生Raw-OLS/Ridge R²。" if dna_status == "deferred_by_user" else
             "DNA缺少原始表示/冻结权重，未产生Raw-OLS/Ridge R²；4096维及提取节点尚未核验。"),
            "正式正文、原图和历史数值均未改动。"]
    (output / "report_zh.md").write_text("\n".join(text) + "\n")
    print(f"Aggregated {len(complete)}/101 classification tasks", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("inventory", "run", "aggregate"), required=True)
    parser.add_argument("--suite", choices=("fig2", "protein60", "rna40"))
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--chunk-tokens", type=int, default=8192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output = args.output.resolve()
    if args.stage == "inventory":
        inventory(args.output)
    elif args.stage == "aggregate":
        aggregate(args.output)
    else:
        if not args.suite:
            raise ValueError("--suite is required for run")
        if not (args.output / "protocol.json").exists():
            raise ValueError("Run inventory first")
        os.environ.setdefault("TRITON_CACHE_DIR", str(args.output / "runtime/triton"))
        tasks = [s for s in specs() if s["suite"] == args.suite]
        if args.tasks:
            tasks = [s for s in tasks if s["task"] in args.tasks]
        tasks = tasks[args.worker_index::args.workers]
        extractor = Extractor("rna" if args.suite == "rna40" else "protein", args.device, args.batch_size)
        failures = []
        for spec in tasks:
            try:
                run_task(spec, args, extractor)
            except Exception as error:
                traceback.print_exc()
                failure = {"suite": spec["suite"], "task": spec["task"], "status": "failed", "error": str(error), "traceback": traceback.format_exc()}
                save_json(args.output / f"models/{spec['suite']}/{spec['task']}/failure.json", failure)
                failures.append(failure)
        if failures:
            raise SystemExit(f"{len(failures)} tasks failed; see task failure.json")


if __name__ == "__main__":
    main()
