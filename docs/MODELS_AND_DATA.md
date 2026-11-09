# Exact model sources and data access

| Model | Configuration | Download |
|---|---|---|
| ESM-2-650M | layer 33; 1,280-dimensional representations | [esm2_t33_650M_UR50D.pt](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt) |
| InterPLM protein SAE | 10,240 features; normalized feature activations | [layer_33/ae_normalized.pt](https://huggingface.co/Elana/InterPLM-esm2-650m/blob/main/layer_33/ae_normalized.pt) |
| ERNIE-RNA | layer 12; 768-dimensional representations | [authors' pretrained-model downloads](https://github.com/Bruce-ywj/ERNIE-RNA#access-pre-trained-models) |
| Authors' RNA SAE | ReLU; 9,216 features; retained epoch-3 checkpoint | [CrossLatent-rna-sae.zip](https://drive.google.com/file/d/1b2zQmqIVEXz0Nag2CQyxXXQTWU1RsiXu/view?usp=drive_link) |
| Evo 2 7B | 262k context; `blocks.26`; width 4,096 | [evo2_7b_262k.pt](https://huggingface.co/arcinstitute/evo2_7b_262k/blob/main/evo2_7b_262k.pt) |
| Goodfire DNA SAE | layer-26 mixed; expansion 8; width 32,768; BatchTopK budget 64 | [sae-layer26-mixed-expansion_8-k_64.pt](https://huggingface.co/Goodfire/Evo-2-Layer-26-Mixed/blob/main/sae-layer26-mixed-expansion_8-k_64.pt) |

For protein prediction weights use the **normalized** InterPLM file in **layer 33**. The different file `ae_unnormalized.pt` is not the checkpoint specified for this study. For DNA, the base-model variant is specifically `evo2_7b_262k`. Keep the recorded batch composition when reproducing BatchTopK outputs; sparsity is allocated across a batch.

`model_catalog.json` records local SAE hashes and source links. Obtain public models from their authors; neither the code nor study-data archive bundles those external weights. The RNA SAE archive retains exactly the checkpoint used by the saved RNA predictors, along with loading metadata and file hashes. No new normalization or training is performed during packaging.

## Study archive downloads

- Study data: [CrossLatent-study-data.zip](https://drive.google.com/file/d/1vw_L7JU67ECAWKoeZmfX9KRCN_U0xn1q/view?usp=drive_link).
- Authors' RNA SAE: [CrossLatent-rna-sae.zip](https://drive.google.com/file/d/1b2zQmqIVEXz0Nag2CQyxXXQTWU1RsiXu/view?usp=drive_link).
- Code: [Shuo-Zhan/CrossLatent](https://github.com/Shuo-Zhan/CrossLatent).

Both archive links were verified by complete anonymous downloads, ZIP CRC checks and exact SHA256 comparison with the prepared release. See [download instructions and checksums](DOWNLOADS.md).

The study archive's `SOURCES.md` and original metadata supply biological database identifiers and attribution. The software MIT licence does not relicense databases or model weights.
