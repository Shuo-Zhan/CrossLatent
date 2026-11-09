#!/usr/bin/env python3
"""
Extract Evo2-40B embedding + logits via NVIDIA hosted /forward API.

Default layers (StripedHyena / Vortex naming on hosted 40B):
  - embedding_layer  → token embeddings
  - unembed          → final logits

Example:
    python extract_evo2_layer26_embedding.py --sequence ACGTACGT
    python extract_evo2_layer26_embedding.py --sequence ACGTACGT --save_dir ./outs
"""

from __future__ import annotations

import argparse
import base64
import io
import os
from pathlib import Path

import numpy as np
import requests

DEFAULT_FORWARD_URL = (
    "https://health.api.nvidia.com/v1/biology/arc/evo2-40b/forward"
)
# 托管 40B：embedding + logits（不要用 decoder.layers.*）
DEFAULT_EMBED_LAYER = "embedding_layer"
DEFAULT_LOGITS_LAYER = "unembed"
DEFAULT_OUTPUT_LAYERS = [DEFAULT_EMBED_LAYER, DEFAULT_LOGITS_LAYER]

# 直接粘贴 Run Key（优先于环境变量）。不要把含真实 key 的文件提交到 git。
HARDCODED_RUN_KEY = "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"


def get_api_key() -> str:
    if HARDCODED_RUN_KEY and not HARDCODED_RUN_KEY.startswith("nvapi-在这里"):
        return HARDCODED_RUN_KEY.strip()
    key = os.getenv("NVCF_RUN_KEY") or os.getenv("NGC_API_KEY") or os.getenv(
        "NVIDIA_API_KEY"
    )
    if key:
        return key.strip()
    return input("Paste the Run Key (nvapi-...): ").strip()


def forward_npz(
    sequence: str,
    *,
    output_layers: list[str],
    url: str = DEFAULT_FORWARD_URL,
    api_key: str | None = None,
    timeout_s: float = 600.0,
) -> np.lib.npyio.NpzFile:
    key = api_key or get_api_key()
    response = requests.post(
        url=url,
        headers={"Authorization": f"Bearer {key}"},
        json={"sequence": sequence, "output_layers": output_layers},
        timeout=timeout_s,
    )
    if not response.ok:
        raise RuntimeError(
            f"HTTP {response.status_code} {response.reason}\n"
            f"url={url}\n"
            f"payload={{'sequence': <len={len(sequence)}>, "
            f"'output_layers': {output_layers!r}}}\n"
            f"response={response.text[:2000]}"
        )
    content_type = response.headers.get("Content-Type", "")

    if "application/json" in content_type:
        payload = response.json()
        if "data" not in payload:
            raise RuntimeError(f"Unexpected JSON keys: {list(payload.keys())}")
        return np.load(io.BytesIO(base64.b64decode(payload["data"])))

    if "application/zip" in content_type or "octet-stream" in content_type:
        return np.load(io.BytesIO(response.content))

    raise RuntimeError(
        f"Unsupported Content-Type={content_type!r}; "
        f"body[:200]={response.content[:200]!r}"
    )


def _pick_array(npz: np.lib.npyio.NpzFile, layer_name: str) -> np.ndarray:
    keys = list(npz.files)
    candidates = [
        layer_name,
        f"{layer_name}.output",
        layer_name.replace(".", "_"),
        f"{layer_name.replace('.', '_')}.output",
    ]
    for key in candidates:
        if key in npz.files:
            return np.asarray(npz[key])
    if len(keys) == 1:
        return np.asarray(npz[keys[0]])
    tail = layer_name.split(".")[-1]
    matched = [k for k in keys if layer_name in k or tail in k]
    if len(matched) == 1:
        return np.asarray(npz[matched[0]])
    raise KeyError(
        f"Could not find layer {layer_name!r} in NPZ keys={keys}. Tried {candidates}."
    )


def _to_token_major(acts: np.ndarray) -> np.ndarray:
    arr = np.asarray(acts)
    if arr.ndim == 3:
        if arr.shape[1] == 1:
            arr = arr[:, 0, :]
        elif arr.shape[0] == 1:
            arr = arr[0]
        else:
            arr = arr[:, 0, :] if arr.shape[1] < arr.shape[0] else arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D activations after squeeze, got {arr.shape}")
    return arr.astype(np.float32, copy=False)


def extract_embedding_and_logits(
    sequence: str,
    *,
    embed_layer: str = DEFAULT_EMBED_LAYER,
    logits_layer: str = DEFAULT_LOGITS_LAYER,
    url: str = DEFAULT_FORWARD_URL,
    api_key: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One /forward call → (embedding, logits), both float32 [T, D].
    """
    layers = [embed_layer, logits_layer]
    npz = forward_npz(
        sequence, output_layers=layers, url=url, api_key=api_key
    )
    print(f"NPZ keys: {list(npz.files)}")
    embedding = _to_token_major(_pick_array(npz, embed_layer))
    logits = _to_token_major(_pick_array(npz, logits_layer))
    return embedding, logits


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract Evo2-40B embedding + logits via hosted /forward."
    )
    p.add_argument("--sequence", type=str, default="ACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTG")
    p.add_argument(
        "--url",
        type=str,
        default=os.getenv("URL", DEFAULT_FORWARD_URL),
    )
    p.add_argument(
        "--embed_layer",
        type=str,
        default=DEFAULT_EMBED_LAYER,
        help="Embedding layer name (default: embedding_layer).",
    )
    p.add_argument(
        "--logits_layer",
        type=str,
        default=DEFAULT_LOGITS_LAYER,
        help="Logits layer name (default: unembed).",
    )
    p.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="If set, save embedding.npy and logits.npy under this dir.",
    )
    p.add_argument(
        "--list_only",
        action="store_true",
        help="Only print NPZ keys/shapes.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    layers = [args.embed_layer, args.logits_layer]
    print(f"url={args.url}")
    print(f"layers={layers}")
    print(f"sequence length={len(args.sequence)}")

    if args.list_only:
        npz = forward_npz(
            args.sequence, output_layers=layers, url=args.url
        )
        for key in npz.files:
            print(f"  {key}: shape={npz[key].shape} dtype={npz[key].dtype}")
        return

    embedding, logits = extract_embedding_and_logits(
        args.sequence,
        embed_layer=args.embed_layer,
        logits_layer=args.logits_layer,
        url=args.url,
    )
    print(f"embedding shape={embedding.shape}")
    print(f"logits shape={logits.shape}")

    if args.save_dir:
        out_dir = Path(args.save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        emb_path = out_dir / "embedding.npy"
        log_path = out_dir / "logits.npy"
        np.save(emb_path, embedding)
        np.save(log_path, logits)
        print(f"saved: {emb_path.resolve()}")
        print(f"saved: {log_path.resolve()}")


if __name__ == "__main__":
    main()
