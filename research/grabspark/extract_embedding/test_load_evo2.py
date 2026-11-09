"""Load evo2_7b_262k, extract layer-26 embeddings, run SAE encode."""
import os

# 必须在 import torch / evo2 之前设置，Vortex 会按可见 GPU 数自动切层
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")  # 改成你想用的物理卡号

import torch
import torch.nn as nn
from evo2 import Evo2

EVO2_CKPT = "/data1/zshuo_zzy/evo2sae/ckpt/evo2_7b_262k.pt"
SAE_CKPT = "/data1/zshuo_zzy/evo2sae/ckpt/sae-layer26-mixed-expansion_8-k_64.pt"
LAYER = "blocks.26"  # notebook 里是 blocks-26，Evo2 API 用点号
D_HIDDEN = 4096
EXPANSION = 8
K = 64


class BatchTopKTiedSAE(nn.Module):
    """与 sparse_autoencoder.ipynb 一致的 TopK SAE。"""

    def __init__(self, d_in, d_hidden, k, device, dtype, tiebreaker_epsilon: float = 1e-6):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.k = k
        W_mat = torch.randn((d_in, d_hidden))
        W_mat = 0.1 * W_mat / torch.linalg.norm(W_mat, dim=0, ord=2, keepdim=True)
        self.W = nn.Parameter(W_mat)
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.device = device
        self.dtype = dtype
        self.tiebreaker = torch.linspace(0, tiebreaker_epsilon, d_hidden)
        self.to(device, dtype)

    def encode(self, x, tiebreak=False):
        f = torch.nn.functional.relu(x @ self.W + self.b_enc)
        if tiebreak:
            f = f + self.tiebreaker.broadcast_to(f)
        *input_shape, _ = f.shape
        from math import prod

        numel = self.k * prod(input_shape)
        f_topk = torch.topk(f.flatten(), numel, dim=-1)
        return (
            torch.zeros_like(f.flatten())
            .scatter(-1, f_topk.indices, f_topk.values)
            .reshape(f.shape)
        )


def load_topk_sae(sae_path, device, dtype):
    state = torch.load(sae_path, weights_only=True, map_location="cpu")
    state = {
        k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in state.items()
    }
    sae = BatchTopKTiedSAE(D_HIDDEN, D_HIDDEN * EXPANSION, K, device, dtype)
    sae.load_state_dict(state)
    sae.eval()
    return sae


print(
    "cuda:",
    torch.cuda.is_available(),
    f"n_visible={torch.cuda.device_count()}",
    torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
)
assert torch.cuda.device_count() == 1, "可见多卡时 Vortex 会自动切层；请只暴露 1 张卡"

evo2_model = Evo2("evo2_7b_262k", local_path=EVO2_CKPT)

seq = "ACGTACGTACGT"
input_ids = torch.tensor(
    evo2_model.tokenizer.tokenize(seq),
    dtype=torch.int,
).unsqueeze(0).to("cuda:0")

with torch.no_grad():
    logits, embeddings = evo2_model(
        input_ids,
        return_embeddings=True,
        layer_names=[LAYER],
    )
    layer26 = embeddings[LAYER]  # (batch, seq, 4096)
    print("logits shape:", tuple(logits[0].shape))
    print(f"{LAYER} embedding shape:", tuple(layer26.shape))
    print(f"{LAYER} dtype/device:", layer26.dtype, layer26.device)

    sae = load_topk_sae(SAE_CKPT, device=layer26.device, dtype=torch.bfloat16)
    feats = sae.encode(layer26[0].to(torch.bfloat16))  # (seq, 4096*8)
    print("SAE features shape:", tuple(feats.shape))  # (seq, 32768)
    print("nonzero features / token (mean):", (feats != 0).sum(-1).float().mean().item())
    print("OK: layer26 embedding + SAE encode succeeded")
