# Reproducing the study

## Verify the saved evidence

Install CrossLatent, download the [study-data archive](https://drive.google.com/file/d/1vw_L7JU67ECAWKoeZmfX9KRCN_U0xn1q/view?usp=drive_link), and extract `CrossLatent-study-data` alongside the code directory. [Download instructions](DOWNLOADS.md) provide checksums and the separate RNA SAE parameters:


```sh
python tools/verify_data.py --data-root ../CrossLatent-study-data
python tools/verify_readouts.py --data-root ../CrossLatent-study-data
```

Verification checks file hashes, eleven source-specific sequence partitions, confusion-count F1 calculations, 100 raw-coordinate comparison tasks, 800 additional proteins and their selected-feature activation arrays, numerical source tables, and 100 saved SAE predictors with all 1,000 fitted budgets. It checks supplied records without repeating base-model inference.

| Evidence | Study-data location |
|---|---|
| Eleven source-specific benchmarks | `fig2/`; `supplementary_data/` |
| Protein60 and RNA40 benchmarks | `inputs/fig/paper_figures/fig3/biological100/`; `readouts/` |
| Raw-OLS and Raw-Ridge controls | `raw_probes/`; group3 six-method results |
| Continuous DNA prediction and feature-removal analyses | `inputs/ckpt/json/DNA/`; `inputs/fig/paper_figures/fig4/`; `source_data/fig4/` |
| Family and common-motif analyses | `fig5/`; `supplementary_data/` |
| Additional 800 proteins | `fig5_new800/`; `supplementary_data/S19/` |
| Protein-language-model output interventions | `fig6/`; `source_data/fig6/` |
| Numerical data underlying the article figures | `source_data/figN/tables/` and `source_data/figN/inputs/` |

## Run new predictions

Load the 100 final models with `LinearReadout.load`, using the matching SAE checkpoint, feature-column ordering and activation scale. Indices are zero-based. Feature count and binary threshold are frozen from validation. The K=1–10 files contain separately refitted predictors. `predict-steps` instead accumulates contributions under the final model's fixed coefficients.

## Full extraction, training and evaluation

`research/` contains original analysis implementations, including historical workstation defaults. Obtain the external models listed in `MODELS_AND_DATA.md`, install their modality-specific dependencies, provide explicit input/model/output/cache paths, and regenerate large embeddings as needed. Full-dictionary embedding and activation caches are not included.

The separate RNA SAE archive supplies the exact parameter file used by the RNA predictors, with loading metadata. Its SAE training and development FASTAs are not distributed in this release. Downstream evaluation sequences and annotations remain in the study archive. Source database information is in `SOURCES.md`.

Plotting implementations, rendered figure files and figure-formatting audits are omitted. Numerical results, source tables and analysis code are retained. Validation during packaging does not constitute a full GPU inference or model-training rerun.
