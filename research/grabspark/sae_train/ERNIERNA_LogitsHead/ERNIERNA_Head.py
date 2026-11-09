import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class ERNIERNALogitsHead(nn.Module):
    def __init__(
        self,
        layer_norm=None,
        activation_fn=None,
        lm_head_transform_weight=None,
        embed_tokens_weight=None,
        embed_out=None,
        lm_output_learned_bias=None,
        share_input_output_embed=True,
    ):
        super().__init__()

        # 直接引用原模型的模块与参数，不复制，不改变
        if layer_norm is not None:
            self.layer_norm = layer_norm
        else:
            self.layer_norm = nn.LayerNorm(768)

        if lm_head_transform_weight is not None:
            self.lm_head_transform_weight = lm_head_transform_weight
        else:
            self.lm_head_transform_weight = nn.Linear(768, 768)

        if embed_tokens_weight is not None:
            self.embed_tokens_weight = embed_tokens_weight
        else:
            self.embed_tokens_weight = nn.Parameter(torch.empty(25, 768))
        
        if lm_output_learned_bias is not None:
            self.lm_output_learned_bias = lm_output_learned_bias
        else:
            self.lm_output_learned_bias = nn.Parameter(torch.zeros(25))

        if activation_fn is not None:
            self.activation_fn = activation_fn
        else:
            self.activation_fn = F.gelu
        self.embed_out = embed_out
        self.share_input_output_embed = share_input_output_embed


    def forward(self, x, masked_tokens=None):
        """
        x: (B, T, C)
        """

        if masked_tokens is not None:
            x = x[masked_tokens, :]

        x = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(x)))

        if self.share_input_output_embed and self.embed_tokens_weight is not None:
            x = F.linear(x, self.embed_tokens_weight)
        elif self.embed_out is not None:
            x = self.embed_out(x)

        if self.lm_output_learned_bias is not None:
            x = x + self.lm_output_learned_bias

        return x
    
if __name__ == "__main__":
    _ckpt = Path(__file__).resolve().parents[3] / "ckpt" / "ERNIERNA_LogitsHead" / "model_weights.pt"
    output_head = ERNIERNALogitsHead()
    output_head.load_state_dict(torch.load(_ckpt, map_location="cpu"))
    output_head.eval()

    all_embedding = np.random.rand(3, 8+2, 768)  # all_embedding shape like [3, 10, 768]
    recover_logits = output_head(torch.tensor(all_embedding, dtype=torch.float32)).cpu().detach().numpy()
    print(recover_logits.shape) # recover_logits shape like [3, 10, 25]
            # if sequences[i][j] in set("Aa"):
            #     rna_index[i][j+1] = 5
            # elif sequences[i][j] in set("Cc"):
            #     rna_index[i][j+1] = 7
            # elif sequences[i][j] in set("Gg"):
            #     rna_index[i][j+1] = 4
            # elif sequences[i][j] in set('TUtu'):
            #     rna_index[i][j+1] = 6

