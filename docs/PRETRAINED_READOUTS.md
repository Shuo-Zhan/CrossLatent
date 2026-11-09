# Saved predictors for 60 protein and 40 RNA tasks

Location: `crosslatent/pretrained_readouts/` in the source repository and installed wheel; the study archive includes an identical `readouts/` directory.

| File | Contents |
|---|---|
| `index.csv`, `index.json` | All task IDs, selected K, threshold, feature indices, model paths and hashes |
| `protein60/TASK.json` | Validation-selected protein predictor, directly loadable by `LinearReadout.load` |
| `rna40/TASK.json` | Validation-selected RNA predictor, directly loadable by `LinearReadout.load` |
| `trajectories/SUITE/TASK.json` | Ten independently fitted models, K=1–10, with validation-selected thresholds for each K |
| `export_validation.json` | Counts, task means and export checks |

All feature indices are zero-based. Protein width is 10,240; RNA width is 9,216. A model computes `intercept + activations[:, latent_ids] @ coefficients`; positive annotations use `score >= threshold`, without a sigmoid. Signed coefficients are preserved.

The final K and threshold were selected from validation curves using the original rule: maximize mean per-sequence class-macro F1; ties favor smaller K then lower threshold. Historical training JSONs have a training `best_threshold`; it is not the published operating point. The exporter uses the final evaluation summary and validates it against every validation candidate.

Each model's metadata records its SAE checkpoint hash, source fitting JSON/hash, evaluation summary/hash, validation-curve hash and best-single-feature baseline. Model metadata retain test results for traceability; test results are not used in model selection. Export does not refit coefficients or modify the source results.

The activation columns and scales must correspond to the same checkpoint and study encoding procedure. The protein checkpoint is specifically `Elana/InterPLM-esm2-650m/layer_33/ae_normalized.pt`. The RNA checkpoint bytes are supplied separately; no additional rescaling was performed during release preparation. Historical rescaling of the RNA parameters remains unresolved, so the release identifies the exact checkpoint by hash rather than asserting a normalization history.

`dictionary_id` identifies the study suite, checkpoint hash and study protocol. Do not substitute a different SAE or change per-feature scales while retaining these prediction weights. Full-dictionary activations are not bundled in GitHub; compute them from matching embeddings and checkpoint or supply existing matched activations.

```sh
python tools/verify_readouts.py
python tools/verify_readouts.py --data-root ../CrossLatent-study-data
```

The second command additionally checks all 1,000 fitted models against source coefficients, all final validation operating points, and the prediction formula using deterministic synthetic responses. It is not a rerun of biological model inference.
