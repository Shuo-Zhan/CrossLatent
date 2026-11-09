import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
INTERPLM_ROOT = SCRIPT_DIR.parents[1] / "InterPLM"

sys.path.insert(0, str(INTERPLM_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from interplm.embedders import get_embedder  # pyright: ignore[reportMissingImports]

# ===== 在这里指定 =====
# 可以是单个 CSV，也可以是包含多个 CSV 的文件夹
INPUT_PATH = Path("/data1/zshuo23/GraBSpArk/data/Protein/High-quality-dataset")
OUTPUT_DIR = None  # None: 与各 CSV 同目录；否则写到该文件夹
DEVICE = "cuda:0"
LAYER = 33
BATCH_SIZE = 25
ESM2_MODEL_DIR = str(INTERPLM_ROOT / "esm2_model" / "esm2_650M_UR50D")
# =====================


def resolve_csv_paths(input_path: Path) -> list[Path]:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    if input_path.is_file():
        if input_path.suffix.lower() != ".csv":
            raise ValueError(f"输入文件不是 CSV: {input_path}")
        return [input_path.resolve()]

    if input_path.is_dir():
        csv_paths = sorted(input_path.glob("*.csv"))
        if not csv_paths:
            raise FileNotFoundError(f"文件夹下没有 CSV: {input_path}")
        return [p.resolve() for p in csv_paths]

    raise ValueError(f"既不是文件也不是文件夹: {input_path}")


def default_output_npy(csv_path: Path, output_dir: Path | None) -> Path:
    name = f"{csv_path.stem}_ESM2_650M_embedding.npy"
    if output_dir is None:
        return csv_path.with_name(name)
    return Path(output_dir) / name


def extract_embeddings_for_csv(
    csv_path: Path,
    *,
    embedder,
    layer: int,
    batch_size: int,
    output_npy: Path,
) -> Path:
    df = pd.read_csv(csv_path)
    if "sequence" not in df.columns:
        raise ValueError(f"{csv_path}: 缺少 sequence 列")
    if "length" not in df.columns:
        raise ValueError(f"{csv_path}: 缺少 length 列")

    sequences = df["sequence"].astype(str).tolist()
    length_list = df["length"].astype(int).tolist()
    if not sequences:
        raise ValueError(f"{csv_path}: 空 CSV")

    print(f"\n{'=' * 60}")
    print(f"csv: {csv_path}")
    print(f"sequences: {len(sequences)}, tokens: {sum(length_list):,}")
    print(f"output: {output_npy}")

    embeddings_list = []
    num_batches = (len(sequences) - 1) // batch_size + 1
    for i in range(num_batches):
        print(f"  Processing batch {i + 1}/{num_batches}")
        batch = sequences[i * batch_size : (i + 1) * batch_size]
        embeddings = embedder.extract_embeddings(
            sequences=batch,
            layer=layer,
            batch_size=batch_size,
        )
        embeddings_list.append(embeddings)

    concatenated = torch.cat(embeddings_list, dim=0)
    print(f"  拼接后形状: {tuple(concatenated.shape)}")
    print(f"  总长度(length 列之和): {sum(length_list):,}")
    if concatenated.shape[0] != sum(length_list):
        print(
            f"  WARNING: embedding rows ({concatenated.shape[0]}) "
            f"!= sum(length) ({sum(length_list)})"
        )

    output_npy = Path(output_npy)
    output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_npy, concatenated.cpu().numpy())
    print(f"  saved: {output_npy.resolve()}")
    return output_npy


print(ESM2_MODEL_DIR)
csv_paths = resolve_csv_paths(INPUT_PATH)
print(f"input: {Path(INPUT_PATH).resolve()}")
print(f"found {len(csv_paths)} CSV file(s)")

embedder = get_embedder(
    "esm",
    model_name=str(ESM2_MODEL_DIR),
    device=DEVICE,
)

for csv_path in csv_paths:
    out_npy = default_output_npy(csv_path, OUTPUT_DIR)
    extract_embeddings_for_csv(
        csv_path,
        embedder=embedder,
        layer=LAYER,
        batch_size=BATCH_SIZE,
        output_npy=out_npy,
    )

print(f"\nDone. Processed {len(csv_paths)} CSV file(s).")
