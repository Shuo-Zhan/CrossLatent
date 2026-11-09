#!/usr/bin/env python3
"""
Evaluate a range of boosting neuron JSONs on one eval CSV (streaming, one ESM2+SAE pass).

Given a JSON name stem and an inclusive index range, loads
``{json_dir}/{json_name}_{i}.json`` for each i, runs ESM2 embed + SAE encode **once**
on the evaluation CSV, then:

- for each JSON: mean per-sequence combo macro F1
- once for the whole run: mean per-sequence single-neuron macro F1
  (same training series shares ``best_single_neuron`` across checkpoints)

Example:
    python ESM2_eval_neuron_boosting_seq_multi_json.py \\
        --csv_path data/Protein/top20_pfam_families/Protein_kinase_domain_2151.csv \\
        --json_dir data/Protein/clan_CL0023/results \\
        --json_name ESM2_650M_33_mask_protein_domains_neuron_parameters_group1 \\
        --index_start 0 \\
        --index_end 15 \\
        --feature mask_protein_domains

Optional per-sequence CSV (off by default):
    ... --save_per_seq_macro_f1 [--per_seq_macro_f1_csv path.csv]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
INTERPLM_ROOT = SCRIPT_DIR.parents[1] / "InterPLM"
DATA_DIR = SCRIPT_DIR.parents[1] / "data"

sys.path.insert(0, str(INTERPLM_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from ESM2_eval_neuron_boosting import (
    build_combination_prediction_for_sequence,
    load_results_json,
    parse_binary_label,
    resolve_single_neuron_config,
    sequence_macro_f1_from_predictions,
)
from ESM2_eval_neuron_boosting_seq import (
    filter_dataframe_by_max_length,
    load_models,
)
from neuron_boosting import load_evaluation_dataframe, validate_row_labels
from neuron_boosting_seq import (  # pyright: ignore[reportMissingImports]
    StreamingContext,
    _encode_batch,
    build_streaming_context,
    compute_all_labels_from_df,
)

DEFAULT_CSV = (
    DATA_DIR / "Protein" / "top20_pfam_families" / "Protein_kinase_domain_2151.csv"
)
# DEFAULT_CSV = (
#     DATA_DIR / "Protein" / "top20_pfam_families" / "Radical_SAM_superfamily_2135.csv"
# )
# DEFAULT_CSV = (
#     DATA_DIR / "Protein" / "top20_pfam_families" / "7_transmembrane_receptor_rhodopsin_family_1870.csv"
# )
# DEFAULT_CSV = (
#     DATA_DIR / "Protein" / "top20_pfam_families" / "Anticodon_binding_domain_2290.csv"
# )
# DEFAULT_CSV = (
#     DATA_DIR / "Protein" / "top20_pfam_families" / "Cytidylyltransferase-like_1424.csv"
# )

# DEFAULT_CSV = (
#     DATA_DIR / "Protein" / "clan_CL0023" / "CL0023_test_drop_all0_mask_protein_domains.csv"
# )
# DEFAULT_CSV = (
#     DATA_DIR / "Protein" / "clan_CL0036" / "CL0036_test.csv"
# )
# DEFAULT_CSV = (
#     DATA_DIR / "Protein" / "clan_CL0063" / "CL0063_test.csv"
# )

# DEFAULT_JSON_DIR = DATA_DIR / "Protein" / "clan_CL0023" / "results"
# DEFAULT_JSON_NAME = "ESM2_650M_33_mask_protein_domains_neuron_parameters_pfam01926"
# DEFAULT_JSON_NAME = "ESM2_650M_boosting_neuron_feature_results"
# DEFAULT_JSON_NAME = "ESM2_650M_33_mask_protein_domains_neuron_parameters_group5"

# DEFAULT_JSON_DIR = DATA_DIR / "Protein" / "top20_pfam_families" / "results"
# DEFAULT_JSON_NAME = "ESM2_650M_boosting_ABC_transporter_feature_results"

# DEFAULT_JSON_DIR = DATA_DIR / "Protein" / "clan_CL0036" / "results"
# DEFAULT_JSON_NAME = "ESM2_650M_33_mask_disulfide_bonds_neuron_parameters_pfam04055"
# DEFAULT_JSON_NAME = "ESM2_650M_33_mask_disulfide_bonds_neuron_parameters_0.9clan0036"

# DEFAULT_JSON_DIR = DATA_DIR / "Protein" / "clan_CL0063" / "results"
# DEFAULT_JSON_NAME = "ESM2_650M_33_mask_regions_of_interest_neuron_parameters_pfam00185"
# DEFAULT_JSON_NAME = "ESM2_650M_33_mask_regions_of_interest_neuron_parameters_0.9clan0063"

DEFAULT_JSON_DIR = DATA_DIR / "Protein" / "High-quality-dataset" / "results"
# DEFAULT_JSON_NAME = "ESM2_650M_33_mask_regions_of_interest_neuron_parameters_pfam00185"
DEFAULT_JSON_NAME = "ESM2_650M_33_mask_protein_domains_CL0036_0.6000_neuron_parameters"

DEFAULT_ESM2_MODEL_DIR = INTERPLM_ROOT / "esm2_model" / "esm2_650M_UR50D"
MAX_SEQ_LEN = 1022


@dataclass
class JsonPredictor:
    """One results JSON combo predictor (single-neuron is shared across the series)."""

    index: int
    path: Path
    combo_neuron_ids: list[int]
    combo_coefficients: list[float]
    combo_intercept: float
    combo_threshold: float
    combo_train_macro_f1: float | None
    combo_f1_sum: float = 0.0
    num_sequences: int = 0


@dataclass
class SharedSingleNeuron:
    """Single-neuron predictor shared by all JSONs in one training series."""

    neuron_id: int
    threshold: float
    train_macro_f1: float | None
    source_json: Path
    f1_sum: float = 0.0
    num_sequences: int = 0


def resolve_json_paths(
    json_dir: Path,
    json_name: str,
    index_start: int,
    index_end: int,
) -> list[tuple[int, Path]]:
    if index_end < index_start:
        raise ValueError(f"index_end ({index_end}) < index_start ({index_start})")

    pairs: list[tuple[int, Path]] = []
    missing: list[str] = []
    for idx in range(index_start, index_end + 1):
        path = json_dir / f"{json_name}_{idx}.json"
        if path.exists():
            pairs.append((idx, path))
        else:
            missing.append(str(path))

    if not pairs:
        raise FileNotFoundError(
            f"No JSON found for {json_name}_{{{index_start}..{index_end}}}.json "
            f"under {json_dir}"
        )
    if missing:
        print(f"Warning: {len(missing)} JSON file(s) missing (skipped):")
        for path in missing[:10]:
            print(f"  - {path}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
    return pairs


def load_predictors_and_shared_single(
    json_pairs: list[tuple[int, Path]],
    feature: str,
) -> tuple[list[JsonPredictor], SharedSingleNeuron | None]:
    """
    Load combo configs for every JSON; take single-neuron once from the first JSON,
    and verify later JSONs match (same training series should be identical).
    """
    predictors: list[JsonPredictor] = []
    shared_single: SharedSingleNeuron | None = None

    for idx, path in json_pairs:
        payload = load_results_json(str(path), feature)
        feature_result = payload["features"][feature]
        combo_cfg = feature_result["combination"]
        single_cfg = resolve_single_neuron_config(feature_result, payload.get("config", {}))

        predictors.append(
            JsonPredictor(
                index=idx,
                path=path,
                combo_neuron_ids=[int(x) for x in combo_cfg["neuron_ids"]],
                combo_coefficients=[float(x) for x in combo_cfg["coefficients"]],
                combo_intercept=float(combo_cfg["intercept_b"]),
                combo_threshold=float(combo_cfg["best_threshold"]),
                combo_train_macro_f1=(
                    float(combo_cfg["best_macro_f1"])
                    if "best_macro_f1" in combo_cfg
                    else None
                ),
            )
        )

        if single_cfg is None:
            continue

        neuron_id = int(single_cfg["neuron_id"])
        threshold = float(single_cfg["best_threshold"])
        train_f1 = (
            float(single_cfg["best_macro_f1"]) if "best_macro_f1" in single_cfg else None
        )
        if shared_single is None:
            shared_single = SharedSingleNeuron(
                neuron_id=neuron_id,
                threshold=threshold,
                train_macro_f1=train_f1,
                source_json=path,
            )
        elif (
            shared_single.neuron_id != neuron_id
            or not np.isclose(shared_single.threshold, threshold)
        ):
            raise ValueError(
                f"best_single_neuron mismatch across JSON series: "
                f"{shared_single.source_json.name} has "
                f"(id={shared_single.neuron_id}, thr={shared_single.threshold}), but "
                f"{path.name} has (id={neuron_id}, thr={threshold}). "
                "Refusing to treat them as one shared single-neuron predictor."
            )

    return predictors, shared_single


def resolve_model_defaults_from_json(
    first_json: Path,
    feature: str,
    args: argparse.Namespace,
) -> tuple[str, str, int, int, int]:
    """Fill plm/esm/batch defaults from the first JSON config when CLI omits them."""
    with open(first_json, encoding="utf-8") as handle:
        payload = json.load(handle)
    config = payload.get("config", {})

    plm_model = args.plm_model or config.get("plm_model", "esm2-650m")
    plm_layer = int(args.plm_layer or config.get("plm_layer", 33))
    esm_model_dir = args.esm_model_dir or config.get(
        "esm_model_dir", str(DEFAULT_ESM2_MODEL_DIR)
    )
    max_batch_tokens = int(
        args.max_batch_tokens or config.get("max_batch_tokens", 128000)
    )
    embed_batch_size = int(
        args.embed_batch_size or config.get("embed_batch_size", 24)
    )
    if feature not in payload.get("features", {}):
        available = list(payload.get("features", {}).keys())
        raise ValueError(f"Feature {feature!r} not in {first_json}. Available: {available}")
    return plm_model, esm_model_dir, plm_layer, max_batch_tokens, embed_batch_size


@torch.no_grad()
def encode_once_evaluate_all_jsons(
    ctx: StreamingContext,
    df: pd.DataFrame,
    seq_lengths: list[int],
    feature: str,
    predictors: list[JsonPredictor],
    shared_single: SharedSingleNeuron | None,
    *,
    collect_per_seq: bool = False,
) -> list[dict[str, Any]] | None:
    """
    One streaming ESM2+SAE encode pass.

    For each sequence:
      - apply every JSON's combo predictor (per-JSON mean macro F1)
      - apply shared single-neuron predictor once (series-level mean macro F1)

    If ``collect_per_seq`` is True, return one dict per (json, sequence) with
    combo_macro_f1 (and single_macro_f1 when available); else return None.
    """
    print(
        f"Batched {len(seq_lengths)} sequences into "
        f"{len(ctx.row_batches)} streaming encode pass(es); "
        f"evaluating {len(predictors)} combo JSON(s)"
        + (
            f" + shared single neuron {shared_single.neuron_id}"
            if shared_single is not None
            else " (no single-neuron config)"
        )
    )

    per_seq_rows: list[dict[str, Any]] | None = [] if collect_per_seq else None

    for batch_row_indices in tqdm(
        ctx.row_batches,
        desc="Streaming encode + multi-JSON eval",
    ):
        batch_sequences = [ctx.sequences[i] for i in batch_row_indices]
        batch_features = _encode_batch(ctx, batch_sequences)

        offset = 0
        for local_row_idx in batch_row_indices:
            seq_len = seq_lengths[local_row_idx]
            f_seq = batch_features[offset : offset + seq_len]
            offset += seq_len

            if f_seq.shape[0] != seq_len:
                raise ValueError(
                    f"Row {local_row_idx}: SAE activations={f_seq.shape[0]} != "
                    f"sequence length={seq_len}"
                )

            y_seq = parse_binary_label(df.iloc[local_row_idx][feature])
            if y_seq.shape[0] != seq_len:
                raise ValueError(
                    f"Row {local_row_idx}: label len={y_seq.shape[0]} != seq_len={seq_len}"
                )

            uniprot_id = str(df.iloc[local_row_idx].get("uniprot_id", local_row_idx))

            single_macro_f1: float | None = None
            if shared_single is not None:
                single_binary = (
                    f_seq[:, shared_single.neuron_id] >= shared_single.threshold
                )
                single_item = sequence_macro_f1_from_predictions(
                    y_seq,
                    single_binary,
                    row_index=local_row_idx,
                    uniprot_id=uniprot_id,
                    seq_length=seq_len,
                )
                single_macro_f1 = float(single_item.macro_f1)
                shared_single.f1_sum += single_macro_f1
                shared_single.num_sequences += 1

            for pred in predictors:
                combo_continuous = build_combination_prediction_for_sequence(
                    f_seq=f_seq,
                    neuron_ids=pred.combo_neuron_ids,
                    coefficients=pred.combo_coefficients,
                    intercept_b=pred.combo_intercept,
                )
                combo_binary = combo_continuous >= pred.combo_threshold
                combo_item = sequence_macro_f1_from_predictions(
                    y_seq,
                    combo_binary,
                    row_index=local_row_idx,
                    uniprot_id=uniprot_id,
                    seq_length=seq_len,
                )
                combo_f1 = float(combo_item.macro_f1)
                pred.combo_f1_sum += combo_f1
                pred.num_sequences += 1

                if per_seq_rows is not None:
                    per_seq_rows.append(
                        {
                            "json_index": pred.index,
                            "results_json": str(pred.path),
                            "uniprot_id": uniprot_id,
                            "row_index": int(local_row_idx),
                            "seq_length": int(seq_len),
                            "combo_macro_f1": combo_f1,
                            "single_macro_f1": single_macro_f1,
                        }
                    )

    for pred in predictors:
        if pred.num_sequences != len(seq_lengths):
            raise ValueError(
                f"{pred.path.name}: expected {len(seq_lengths)} sequences, "
                f"got {pred.num_sequences}"
            )
    if shared_single is not None and shared_single.num_sequences != len(seq_lengths):
        raise ValueError(
            f"shared single: expected {len(seq_lengths)} sequences, "
            f"got {shared_single.num_sequences}"
        )
    return per_seq_rows


def default_per_seq_macro_f1_csv(output_json: str | Path | None) -> Path:
    if output_json:
        out = Path(output_json)
        return out.with_name(out.stem + "_per_seq_macro_f1.csv")
    return Path("per_seq_macro_f1.csv")


def default_output_json(
    json_name: str,
    index_start: int,
    index_end: int,
    csv_path: Path,
) -> Path:
    csv_stem = csv_path.stem
    return (
        DEFAULT_JSON_DIR
        / (
            f"ESM2_eval_multi_{json_name}_idx{index_start}-{index_end}"
            f"_on_{csv_stem}.json"
        )
    )


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    json_dir = Path(args.json_dir)
    json_pairs = resolve_json_paths(
        json_dir=json_dir,
        json_name=args.json_name,
        index_start=args.index_start,
        index_end=args.index_end,
    )
    print(
        f"Resolved {len(json_pairs)} JSON(s): "
        f"{args.json_name}_{{{json_pairs[0][0]}..{json_pairs[-1][0]}}}.json"
    )

    plm_model, esm_model_dir, plm_layer, max_batch_tokens, embed_batch_size = (
        resolve_model_defaults_from_json(json_pairs[0][1], args.feature, args)
    )

    predictors, shared_single = load_predictors_and_shared_single(
        json_pairs, args.feature
    )
    print(f"Loaded {len(predictors)} combo predictors for feature={args.feature!r}")
    if shared_single is not None:
        print(
            f"Shared single-neuron: id={shared_single.neuron_id}, "
            f"threshold={shared_single.threshold:.1f} "
            f"(from {shared_single.source_json.name})"
        )
    else:
        print("No single-neuron config found; skipping single-neuron evaluation.")

    df = load_evaluation_dataframe(args.csv_path)
    num_sequences_total = len(df)
    df, num_skipped_long = filter_dataframe_by_max_length(df, max_len=args.max_seq_len)
    if df.empty:
        raise ValueError(
            f"No sequences left after filtering len(sequence) <= {args.max_seq_len}"
        )
    validate_row_labels(df, args.feature)

    seq_lengths = df["sequence"].astype(str).str.len().tolist()
    print(
        f"Evaluating sequences: {len(seq_lengths)} "
        f"(skipped {num_skipped_long} with len>{args.max_seq_len}), "
        f"total tokens: {sum(seq_lengths)}"
    )

    device = torch.device(args.device)
    embedder, sae = load_models(
        esm_model_dir=esm_model_dir,
        plm_model=plm_model,
        plm_layer=plm_layer,
        device=device,
    )
    print(f"Max batch tokens: {max_batch_tokens}, embed batch size: {embed_batch_size}")

    labels = compute_all_labels_from_df(df, [args.feature])
    ctx = build_streaming_context(
        df=df,
        y=labels[args.feature],
        embedder=embedder,
        sae=sae,
        device=device,
        plm_layer=plm_layer,
        max_batch_tokens=max_batch_tokens,
        embed_batch_size=embed_batch_size,
    )

    per_seq_rows = encode_once_evaluate_all_jsons(
        ctx=ctx,
        df=df,
        seq_lengths=seq_lengths,
        feature=args.feature,
        predictors=predictors,
        shared_single=shared_single,
        collect_per_seq=bool(args.save_per_seq_macro_f1),
    )
    del embedder, sae, ctx
    if device.type == "cuda":
        torch.cuda.empty_cache()

    per_seq_csv_path: str | None = None
    if args.save_per_seq_macro_f1:
        if not per_seq_rows:
            raise RuntimeError(
                "--save_per_seq_macro_f1 enabled but no per-sequence rows were collected"
            )
        out_csv = (
            Path(args.per_seq_macro_f1_csv)
            if args.per_seq_macro_f1_csv
            else default_per_seq_macro_f1_csv(args.output_json)
        )
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(per_seq_rows).to_csv(out_csv, index=False)
        per_seq_csv_path = str(out_csv)
        print(
            f"Saved per-sequence macro F1 CSV ({len(per_seq_rows)} rows): {out_csv}"
        )

    per_json: list[dict[str, Any]] = []
    print(f"\n{'=' * 60}")
    print(f"Feature: {args.feature}")
    print(f"Sequences evaluated: {len(seq_lengths)}")
    for pred in predictors:
        combo_mean = pred.combo_f1_sum / max(pred.num_sequences, 1)
        entry = {
            "index": pred.index,
            "results_json": str(pred.path),
            "num_sequences": pred.num_sequences,
            "combination": {
                "neuron_ids": pred.combo_neuron_ids,
                "coefficients": pred.combo_coefficients,
                "intercept_b": pred.combo_intercept,
                "threshold": pred.combo_threshold,
                "train_macro_f1": pred.combo_train_macro_f1,
                "mean_macro_f1": combo_mean,
            },
        }
        per_json.append(entry)
        print(
            f"[{pred.index:>3d}] {pred.path.name}: "
            f"combo mean macro F1={combo_mean:.6f}"
        )

    single_neuron_out: dict[str, Any] | None = None
    if shared_single is not None:
        single_mean = shared_single.f1_sum / max(shared_single.num_sequences, 1)
        single_neuron_out = {
            "neuron_id": shared_single.neuron_id,
            "threshold": shared_single.threshold,
            "train_macro_f1": shared_single.train_macro_f1,
            "mean_macro_f1": single_mean,
            "source_json": str(shared_single.source_json),
            "num_sequences": shared_single.num_sequences,
        }
        print(
            f"Shared single neuron (id={shared_single.neuron_id}, "
            f"threshold={shared_single.threshold:.1f}): "
            f"mean macro F1={single_mean:.6f}"
        )

    output: dict[str, Any] = {
        "csv_path": str(args.csv_path),
        "json_dir": str(json_dir),
        "json_name": args.json_name,
        "index_start": args.index_start,
        "index_end": args.index_end,
        "resolved_indices": [p.index for p in predictors],
        "streaming": True,
        "encode_once": True,
        "esm_model_dir": esm_model_dir,
        "plm_model": plm_model,
        "plm_layer": plm_layer,
        "max_batch_tokens": max_batch_tokens,
        "embed_batch_size": embed_batch_size,
        "max_seq_len": args.max_seq_len,
        "num_sequences_total": num_sequences_total,
        "num_sequences_skipped_long": num_skipped_long,
        "num_sequences": len(df),
        "feature": args.feature,
        "single_neuron": single_neuron_out,
        "per_json": per_json,
        "save_per_seq_macro_f1": bool(args.save_per_seq_macro_f1),
        "per_seq_macro_f1_csv": per_seq_csv_path,
    }

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2, ensure_ascii=False)
        print(f"\nSaved multi-JSON evaluation: {out_path}")

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a range of boosting JSONs on one CSV with a single "
            "streaming ESM2+SAE encode pass."
        )
    )
    parser.add_argument("--csv_path", type=str, default=str(DEFAULT_CSV))
    parser.add_argument("--json_dir", type=str, default=str(DEFAULT_JSON_DIR))
    parser.add_argument(
        "--json_name",
        type=str,
        default=DEFAULT_JSON_NAME,
        help=(
            "JSON stem without trailing _{index}.json, e.g. "
            "ESM2_650M_33_mask_protein_domains_neuron_parameters_group1"
        ),
    )
    parser.add_argument("--index_start", type=int, default=0)
    parser.add_argument(
        "--index_end",
        type=int,
        default=100,
        help="Inclusive end index.",
    )
    parser.add_argument(
        "--feature",
        type=str,
        default="mask_protein_domains",
        help="Label column / JSON feature key.",
    )
    parser.add_argument("--esm_model_dir", type=str, default=None)
    parser.add_argument("--plm_model", type=str, default=None)
    parser.add_argument("--plm_layer", type=int, default=None)
    parser.add_argument("--max_batch_tokens", type=int, default=None)
    parser.add_argument("--embed_batch_size", type=int, default=None)
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cuda_visible_devices", type=str, default=None)
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Output path; default derived from json_name / indices / csv stem.",
    )
    parser.add_argument(
        "--save_per_seq_macro_f1",
        action="store_true",
        default=False,
        help=(
            "If set, also write a CSV of per-sequence combo macro F1 for every "
            "JSON (default: off)."
        ),
    )
    parser.add_argument(
        "--per_seq_macro_f1_csv",
        type=str,
        default=None,
        help=(
            "Path for the per-sequence macro F1 CSV when --save_per_seq_macro_f1 "
            "is set; default = <output_json stem>_per_seq_macro_f1.csv."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    if args.output_json is None:
        args.output_json = str(
            default_output_json(
                args.json_name,
                args.index_start,
                args.index_end,
                Path(args.csv_path),
            )
        )
    run_evaluation(args)


if __name__ == "__main__":
    main()
