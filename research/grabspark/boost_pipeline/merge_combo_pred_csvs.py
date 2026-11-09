#!/usr/bin/env python3
"""
Merge a ground-truth mask CSV with multiple combo/single prediction CSVs,
compute per-sequence macro F1 for the highest-k combo vs single, and sort by
their delta (combo_max - single).

Each pred CSV is expected to contain:
  id, sequence, length, pred_combo_<feature>, pred_single_<feature>

Combo columns are renamed to pred_combo_k{N}_<feature> using the index N
parsed from the filename (e.g. ..._results_3_pred.csv → k=3), or from an
explicit --pred_ks list aligned with --pred_csvs.

Example:
    python merge_combo_pred_csvs.py \\
        --base_csv data/Protein/clan_CL0023/CL0023_test_drop_all0_mask_protein_domains.csv \\
        --pred_csvs \\
            fig/ESM2_650M_boosting_pfam0005_transporter_feature_results_1_pred.csv \\
            fig/ESM2_650M_boosting_pfam0005_transporter_feature_results_2_pred.csv \\
            fig/ESM2_650M_boosting_pfam0005_transporter_feature_results_3_pred.csv \\
        --feature mask_protein_domains \\
        --output_csv fig/ESM2_650M_boosting_pfam0005_transporter_merged_sorted.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SAVE_DIR = REPO_ROOT / "fig"
DATA_DIR = REPO_ROOT / "data"

DEFAULT_BASE_CSV = (
    DATA_DIR
    / "Protein"
    / "clan_CL0023"
    / "CL0023_test_drop_all0_mask_protein_domains.csv"
)
DEFAULT_PRED_CSVS = [
    SAVE_DIR / "ESM2_650M_boosting_pfam0005_transporter_feature_results_1_pred.csv",
    SAVE_DIR / "ESM2_650M_boosting_pfam0005_transporter_feature_results_2_pred.csv",
    SAVE_DIR / "ESM2_650M_boosting_pfam0005_transporter_feature_results_3_pred.csv",
]
DEFAULT_FEATURE = "mask_protein_domains"
DEFAULT_OUTPUT_CSV = (
    SAVE_DIR / "ESM2_650M_boosting_pfam0005_transporter_merged_sorted.csv"
)

_K_FROM_NAME = re.compile(r"(?:^|[_\-])(\d+)_pred(?:\.csv)?$", re.IGNORECASE)


def normalize_feature(feature: str) -> str:
    return feature if feature.startswith("mask_") else f"mask_{feature}"


def pred_combo_col(feature: str) -> str:
    return f"pred_combo_{normalize_feature(feature)}"


def pred_single_col(feature: str) -> str:
    return f"pred_single_{normalize_feature(feature)}"


def pred_combo_k_col(feature: str, k: int) -> str:
    feat = normalize_feature(feature)
    return f"pred_combo_k{k}_{feat}"


def infer_k_from_path(path: Path) -> int | None:
    match = _K_FROM_NAME.search(path.name)
    if match is None:
        return None
    return int(match.group(1))


def macro_f1_for_sequence(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Same definition as ESM2_eval_neuron_boosting.macro_f1_for_sequence."""
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")

    tp = int(np.sum(y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))

    f1_pos_den = 2 * tp + fp + fn
    f1_pos = (2 * tp / f1_pos_den) if f1_pos_den > 0 else 0.0
    f1_neg_den = 2 * tn + fp + fn
    f1_neg = (2 * tn / f1_neg_den) if f1_neg_den > 0 else 0.0
    return 0.5 * (f1_pos + f1_neg)


def parse_mask_string(mask: object, *, length: int, label: str) -> np.ndarray:
    if mask is None or (isinstance(mask, float) and np.isnan(mask)):
        raise ValueError(f"{label}: missing mask")
    text = str(mask).strip()
    if len(text) != length:
        raise ValueError(
            f"{label}: mask length {len(text)} != sequence length {length}"
        )
    if any(ch not in "01" for ch in text):
        raise ValueError(f"{label}: mask contains non-01 characters")
    return np.fromiter((ch == "1" for ch in text), dtype=bool, count=length)


def resolve_ks(pred_paths: list[Path], pred_ks: list[int] | None) -> list[int]:
    if pred_ks is not None:
        if len(pred_ks) != len(pred_paths):
            raise ValueError(
                f"--pred_ks length ({len(pred_ks)}) must match "
                f"--pred_csvs length ({len(pred_paths)})"
            )
        return pred_ks

    ks: list[int] = []
    for path in pred_paths:
        k = infer_k_from_path(path)
        if k is None:
            raise ValueError(
                f"Cannot infer combo index k from {path.name}; "
                "pass --pred_ks explicitly."
            )
        ks.append(k)
    if len(set(ks)) != len(ks):
        raise ValueError(f"Duplicate combo indices inferred: {ks}")
    return ks


def load_pred_frame(path: Path, feature: str, k: int, id_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    combo_src = pred_combo_col(feature)
    single_src = pred_single_col(feature)
    missing = [c for c in (id_col, combo_src, single_src) if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}; have {list(df.columns)}")

    combo_dst = pred_combo_k_col(feature, k)
    out = df[[id_col, combo_src, single_src]].rename(
        columns={
            combo_src: combo_dst,
            single_src: "_pred_single_tmp",
        }
    )
    if out[id_col].duplicated().any():
        raise ValueError(f"{path}: duplicate {id_col} values")
    return out


def merge_frames(
    base: pd.DataFrame,
    pred_frames: list[tuple[int, pd.DataFrame]],
    *,
    feature: str,
    id_col: str,
) -> pd.DataFrame:
    feat = normalize_feature(feature)
    if feat not in base.columns:
        raise ValueError(
            f"Base CSV missing feature column {feat!r}; have {list(base.columns)}"
        )
    if id_col not in base.columns:
        raise ValueError(f"Base CSV missing id column {id_col!r}")

    keep_cols = [id_col]
    for optional in ("sequence", "length"):
        if optional in base.columns:
            keep_cols.append(optional)
    keep_cols.append(feat)

    merged = base[keep_cols].copy()
    if merged[id_col].duplicated().any():
        raise ValueError(f"Base CSV has duplicate {id_col} values")

    single_dst = pred_single_col(feature)
    for k, pred in sorted(pred_frames, key=lambda item: item[0]):
        combo_col = pred_combo_k_col(feature, k)
        piece = pred[[id_col, combo_col, "_pred_single_tmp"]]
        before = len(merged)
        merged = merged.merge(piece, on=id_col, how="inner")
        if len(merged) < before:
            print(
                f"Warning: k={k} dropped {before - len(merged)} ids "
                f"not present in pred CSV"
            )
        if single_dst not in merged.columns:
            merged = merged.rename(columns={"_pred_single_tmp": single_dst})
            continue
        mismatch = (
            merged["_pred_single_tmp"].astype(str) != merged[single_dst].astype(str)
        ).sum()
        if mismatch:
            print(
                f"Warning: k={k} pred_single differs on {int(mismatch)} rows; "
                "keeping single from lowest-k pred CSV"
            )
        merged = merged.drop(columns=["_pred_single_tmp"])

    return merged


def add_metrics_and_sort(
    df: pd.DataFrame,
    *,
    feature: str,
    id_col: str,
    max_k: int,
) -> pd.DataFrame:
    feat = normalize_feature(feature)
    combo_col = pred_combo_k_col(feature, max_k)
    single_col = pred_single_col(feature)
    length_col = "length" if "length" in df.columns else None

    combo_f1s: list[float] = []
    single_f1s: list[float] = []
    deltas: list[float] = []

    for row in df.itertuples(index=False):
        row_dict = row._asdict()
        uid = row_dict[id_col]
        true_mask = str(row_dict[feat])
        length = int(row_dict[length_col]) if length_col else len(true_mask)
        y_true = parse_mask_string(true_mask, length=length, label=f"{uid}/{feat}")
        y_combo = parse_mask_string(
            row_dict[combo_col], length=length, label=f"{uid}/{combo_col}"
        )
        y_single = parse_mask_string(
            row_dict[single_col], length=length, label=f"{uid}/{single_col}"
        )
        combo_f1 = macro_f1_for_sequence(y_true, y_combo)
        single_f1 = macro_f1_for_sequence(y_true, y_single)
        combo_f1s.append(combo_f1)
        single_f1s.append(single_f1)
        deltas.append(combo_f1 - single_f1)

    out = df.copy()
    out["combo_max_macro_f1"] = combo_f1s
    out["single_macro_f1"] = single_f1s
    out["delta_macro_f1"] = deltas
    out["combo_max_k"] = max_k
    out = out.sort_values(
        by=["delta_macro_f1", "combo_max_macro_f1", id_col],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    front = [c for c in (id_col, "sequence", "length", feat) if c in out.columns]
    combo_cols = sorted(
        [c for c in out.columns if c.startswith("pred_combo_k")],
        key=lambda name: int(re.search(r"pred_combo_k(\d+)_", name).group(1)),
    )
    single = pred_single_col(feature)
    metrics = [
        c
        for c in (
            "combo_max_macro_f1",
            "single_macro_f1",
            "delta_macro_f1",
            "combo_max_k",
        )
        if c in out.columns
    ]
    ordered = front + combo_cols + ([single] if single in out.columns else []) + metrics
    remaining = [c for c in out.columns if c not in ordered]
    return out[ordered + remaining]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge GT + multi-k combo/single pred CSVs and sort by "
            "max-k combo vs single macro-F1 delta."
        )
    )
    parser.add_argument("--base_csv", type=Path, default=DEFAULT_BASE_CSV)
    parser.add_argument(
        "--pred_csvs",
        type=Path,
        nargs="+",
        default=DEFAULT_PRED_CSVS,
        help="Prediction CSVs (k inferred from *_N_pred.csv unless --pred_ks).",
    )
    parser.add_argument(
        "--pred_ks",
        type=int,
        nargs="+",
        default=None,
        help="Optional explicit combo indices aligned with --pred_csvs.",
    )
    parser.add_argument("--feature", type=str, default=DEFAULT_FEATURE)
    parser.add_argument("--id_col", type=str, default="uniprot_id")
    parser.add_argument("--output_csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature = normalize_feature(args.feature)
    pred_paths = [Path(p) for p in args.pred_csvs]
    ks = resolve_ks(pred_paths, args.pred_ks)
    max_k = max(ks)

    print(f"base_csv={args.base_csv}")
    print(f"feature={feature} id_col={args.id_col}")
    for k, path in zip(ks, pred_paths):
        print(f"  k={k}: {path}")
    print(f"sort_by=delta(combo_k{max_k} - single)")

    base = pd.read_csv(args.base_csv)
    pred_frames = [
        (k, load_pred_frame(path, feature, k, args.id_col))
        for k, path in zip(ks, pred_paths)
    ]
    merged = merge_frames(
        base, pred_frames, feature=feature, id_col=args.id_col
    )
    ranked = add_metrics_and_sort(
        merged, feature=feature, id_col=args.id_col, max_k=max_k
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(args.output_csv, index=False)

    print(f"rows={len(ranked)} → {args.output_csv}")
    print(
        "delta_macro_f1: "
        f"mean={ranked['delta_macro_f1'].mean():.6f} "
        f"median={ranked['delta_macro_f1'].median():.6f} "
        f"max={ranked['delta_macro_f1'].max():.6f} "
        f"min={ranked['delta_macro_f1'].min():.6f}"
    )
    top = ranked.head(3)[
        [args.id_col, "combo_max_macro_f1", "single_macro_f1", "delta_macro_f1"]
    ]
    print("top3 by delta:")
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()

# uniprot_id,sequence,length,
# mask_protein_domains,pred_combo_k1_mask_protein_domains,pred_combo_k2_mask_protein_domains,pred_combo_k3_mask_protein_domains,
# pred_single_mask_protein_domains,combo_max_macro_f1,single_macro_f1,delta_macro_f1,combo_max_k

# B2SFC9,
# MAKEKFERSKPHVNVGTIGHVDHGKTTLTAAITKVMAEKNGGMARKFDEIDSAPEEKARGITINTSHVEYESPNRHYAHVDCPGHADYVKNMITGAAQMDGAILVCSAADGPMPQTREHILLSRQVGVPKIVVFLNKCDMVDDEELLELVEMEVRELLDQYEFPGDDTPVIMGSALRAIEGDEAYVEKIVELVQAMDDYIPAPERDTEKPFILPIEDVFSISGRGTVVTGRIERGVVNVGDEVEVVGIRPTQKTTVTGVEMFRKLLDRGEAGDNVGILVRGLKRDDVERGQVLCKPGSIKPHTKFEAEVYVLSKEEGGRHTPFFKGYRPQFYFRTTDITGAVELPEGVEMVMPGDNVKMTITLINPIAMDEGLRFAIREGGRTVGAGVVAKIIE,394,
# 0000000001111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111110000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000,
# 1111101111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111,
# 1011101111111111111111111111111111111111111111111111111111111111111011111111111111101111000110111111111111110111111111111111111111111111111111111111111111111111111111111111100101111111111111111110010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000,
# 1011101111111111111111111111111111111111111101111111111111111111111011111111111111101111000110111111111111110111111111111111111111111111111111111111111111111111111111111111100101111111111111111110010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000,
# 1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111,
# 0.9339011485352948,0.3310696095076401,0.6028315390276547,3

# Q03QN5,
# MAEKEHYERTKPHVNIGTIGHIDHGKTTLTAAITKVLADKGLAKAEDYADIDAAPEERERGITINTAHVEYETEKRHYAHIDAPGHADYIKNMITGAAQMDGAILVVAATDGPMPQTREHILLARQVGVDYIVVFLNKTDLVDDDELVDLVEMEVRELLSEYDYPGDDIPVIRGSALKALEGDEEQEKVILHLMDVVDDYIPTPERENDKPFLMPVEDVFTITGRGTVASGRIDRGTVKVGDEVEVVGLHEDVLKTTVTGLEMFRKTLDLGEAGDNVGALLRGVNREQVVRGQVLAQPGSIQTHEKFKGEVYILSKEEGGRHTPFFSNYRPQFYFHTTDITGVIELPDGVEMVMPGDNVTFTVELIQPAAIEKGTKFTVREGGHTVGAGTVTEIDD,396,
# 000000000011111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000,
# 111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111,
# 101111111111111111111111111111111111111111111111111111111111111111110111111111111111011100001101111111111111111111111111101111111111111111111111111111111111111101111111111111001011111111111111111100100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000,
# 101110101111111111111111111111111111111111101111111101111111111111110111111111111111111100001101111111111111111111111111101111111111111111111111111111111111111101111111111111101011111111111111111100100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000,
# 111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111,
# 0.9316608609504331,0.3299492385786802,0.6017116223717529,3