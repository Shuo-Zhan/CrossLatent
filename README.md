# CrossLatent

CrossLatent discovers shared constituents of biological concepts through sparse combinations of fixed SAE features. Orthogonal residual selection builds a concept readout; independently fitted readouts can then be compared to identify shared latents, examine position responses and test decoder-direction effects on a supplied model output head.

## Scope

This repository provides the core algorithms and portable interfaces: residual feature selection, validation-based budget/threshold selection, frozen prediction, leave-one-latent-out refitting, shared-set analysis, position profiles and readout interventions. It also includes fixed-SAE encoding adapters and independent Top-K/best-single comparisons. It does not contain paper figures, frozen paper-result snapshots, raw biological data or model weights.

InterPLM, ERNIE-RNA and Evo 2 are external projects assumed to be available in the user's environment when required. CrossLatent does not install or download them. For the synthetic example and activation-based learning, no language-model checkpoint is required.

## Installation

Python 3.10 or newer. Use an isolated environment with your required model dependencies available:

```sh
python -m pip install .
crosslatent --help
```

For development, use `python -m pip install -e '.[test]'`. If all declared dependencies are already installed, `python -m pip install --no-deps .` preserves them. Direct dependencies are NumPy, pandas, PyTorch, tqdm and einops.

## Minimal working example

```sh
crosslatent demo --output ./demo
crosslatent run --config ./demo/run.json
```

This seeded CPU example generates synthetic activations, fits a readout, selects its budget and threshold on validation data and evaluates a held-out test set. It also compares a single feature. Outputs include three NPZ files, a portable JSON configuration, model JSON and metrics. It is an interface demonstration, not biological evidence. Configuration paths resolve relative to the configuration file, so the entire demo directory can be moved.

## Use your own activations

Each train/validation/test NPZ file contains:

| Field | Shape | Meaning |
|---|---|---|
| `activations` | N × M | SAE activations with the same dictionary column order |
| `labels` | N | Binary 0/1 labels or continuous values |
| `lengths` | Number of sequences | Positive token counts summing to N |
| `dictionary_id` | Scalar string | Fixed checkpoint and encoding-protocol identifier |
| `sequence_ids` | Optional string vector | Unique sequence identifiers for split-overlap checks |

Remove padding and special tokens before aligning labels. Dictionary identifiers must match across splits; sequence IDs do not replace sequence-level deduplication or homology separation.

```python
from crosslatent import ActivationDataset, fit, evaluate

train = ActivationDataset.load("train.npz")
valid = ActivationDataset.load("valid.npz")
test = ActivationDataset.load("test.npz")
model = fit(train, valid, max_latents=10, task="binary")
model.save("model.json")
print(evaluate(model, test))
```

Equivalent commands:

```sh
crosslatent fit --train train.npz --valid valid.npz --max-latents 10 --output model.json
crosslatent evaluate --model model.json --test test.npz --output metrics.json
crosslatent refit --model model.json --train train.npz --valid valid.npz --test test.npz --output refit.json
```

Use `--task continuous` for pooled R². Binary scores are raw linear scores, with validation thresholds in 0–1 at steps of 0.05; the metric averages class-macro F1 within each sequence and then across sequences. `--method topk` ranks independent training gains before joint fitting; `--method best-single` selects one training feature. No screened-Lasso solver is included.

The default fit uses validation to choose K. For a fixed-budget constituent analysis, pass `--fixed-budget`; validation then chooses only the binary threshold. Numerical degeneracy can yield fewer unique features than the requested budget. Greedy selection does not guarantee the globally optimal K-feature subset.

## Shared features and interventions

Fit each family or concept independently using the same SAE dictionary, then compare:

```sh
crosslatent shared --models family_a.json family_b.json --output shared.json
```

`crosslatent.interpretation` provides `shared_latents`, `position_profile`, `ablate_latents` and `readout_effects`. The intervention subtracts selected activation-weighted decoder directions from the original hidden state and uses the caller's frozen output head. It preserves the SAE reconstruction residual. Position labels and pretrained models are supplied by the caller. See [core API details](docs/CORE_API.md).

## Encode existing embeddings

```sh
crosslatent encode --embeddings embeddings.npy --kind rna --checkpoint checkpoints/rna_sae.pt --batch-tokens 128000 --device cpu --output activations.npy
```

Supported adapters are `protein`, `rna` and `dna`. Explicit checkpoint, precision and batch settings are recorded alongside the output. BatchTopK operates across the entire encoding batch; batch composition can change activations. The adapter uses contiguous token blocks and does not recreate a historical sequence-batching protocol automatically.

## Memory and reproducibility

The NPZ interface is an in-memory workflow and creates float64 working arrays. The retained streaming engine requires caller-provided embeddings/SAE objects and explicit numerical settings; see the API documentation. A saved readout uses the name-independent `latent-readout/v1` schema. The command `import-legacy` can import an explicitly chosen prior combination JSON without following embedded machine paths.

## Tests

```sh
python -m pytest tests -q
```

Tests cover selection against an independent OLS reference, split validation, relocated configurations, fixed evaluation, redundant versus unique information under refitting, BatchTopK semantics and decoder interventions. They require no downloaded model or GPU.

## Data and pretrained SAE

- [CrossLatent study data (~149 MB)]([STUDY_DATA_DOWNLOAD_URL](https://drive.google.com/file/d/1UXEZiUoYh7w9C8AQaHnPrH1HixI7DoA3/view?usp=drive_link)): Processed datasets, evaluation splits, fitted readouts, and analysis results supporting the CrossLatent study, together with source metadata and file verification utilities.
- [CrossLatent RNA SAE (~99 MB)]([RNA_SAE_DOWNLOAD_URL](https://drive.google.com/file/d/1YTzhJXR1afEieFXQGArMO3jg7-1Dw1a-/view?usp=drive_link)): The pretrained ERNIE-RNA sparse autoencoder checkpoint used in this study, together with its training and validation sequences, configuration, and model documentation.

These archives complement the core code in this repository. Pretrained backbone models are not included and should be obtained from their respective providers.

## License

This source code is licensed under the MIT license found in the LICENSE file in the root directory of this source tree.
