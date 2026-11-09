# CrossLatent

CrossLatent progressively combines sparse autoencoder (SAE) features that provide additional predictive information and traces feature contributions to predictions at individual sequence positions. This repository provides the implementation and saved task predictors for *CrossLatent traces the assembly and reuse of sparse features in biological sequence models*. Software version: **0.1.4**.

## Install

Use Python 3.10 or newer and an isolated environment:

```sh
git clone https://github.com/Shuo-Zhan/CrossLatent.git
cd CrossLatent
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
crosslatent demo --output ./demo
python -m pytest tests -q
```

The seeded CPU demo uses synthetic activations and requires no downloaded model. It checks the software interface; its scores are not biological results.

The [tested environment](docs/ENVIRONMENT.md) records the Python and package versions used for release validation.

## The 100 saved task predictors

**All 60 protein-domain and 40 RNA annotation predictors are included in this repository**, under `crosslatent/pretrained_readouts/`. Each task has a directly loadable final model with zero-based SAE feature indices, signed coefficients, intercept, validation-selected feature count and threshold. The accompanying trajectories preserve the independently fitted models for all feature counts K=1–10. These small JSON files are also included in the installed wheel.

```python
from importlib.resources import files
from crosslatent import LinearReadout

root = files('crosslatent').joinpath('pretrained_readouts')
model = LinearReadout.load(root.joinpath('protein60/PF13545.json'))
# activations: [sequence positions, 10240], from the matching fixed protein SAE
scores = model.predict(activations)
positive = scores >= model.threshold
```

`index.csv` and `index.json` enumerate every task. RNA dictionaries contain 9,216 features. Feature identities and activation scales must match the checkpoint and encoding procedure recorded in each model's metadata. Saved test scores are descriptive results, not inputs for selecting a model.

See [saved predictor guide](docs/PRETRAINED_READOUTS.md). The final operating threshold comes from held-out validation, not the `best_threshold` field in historical training JSONs.

## Fit your own data

```python
from crosslatent import ActivationDataset, fit, evaluate

train = ActivationDataset.load('train.npz')
valid = ActivationDataset.load('valid.npz')
test = ActivationDataset.load('test.npz')
model = fit(train, valid, max_latents=10, task='binary')
model.save('model.json')
print(evaluate(model, test))
```

Each NPZ contains `activations`, `labels`, `lengths`, and `dictionary_id`. Optional `sequence_ids` enable ID-overlap checks. Align sequence positions and labels, exclude special/padding tokens, and establish sequence separation before fitting. Selection uses training data; feature count and binary threshold use validation; test evaluation uses fixed settings. Continuous tasks use `task='continuous'` and the linear score directly.

## Data and model downloads

| Archive | Contents | Download |
|---|---|---|
| `CrossLatent-study-data.zip` (222.4 MB) | Study inputs, partitions, numerical results, source tables and saved predictors | [Google Drive](https://drive.google.com/file/d/1vw_L7JU67ECAWKoeZmfX9KRCN_U0xn1q/view?usp=drive_link) |
| `CrossLatent-rna-sae.zip` (52.6 MB) | Authors' RNA SAE parameter file and loading metadata | [Google Drive](https://drive.google.com/file/d/1b2zQmqIVEXz0Nag2CQyxXXQTWU1RsiXu/view?usp=drive_link) |

See [checksums and extraction instructions](docs/DOWNLOADS.md). Both archives were downloaded without signing into a Google account during release validation and matched the prepared release files byte for byte. Google may display a large-file scan notice; select **Download anyway** to retrieve the ZIP.

## Reproduce the paper

Extract `CrossLatent-study-data` alongside this code directory:

```sh
python tools/verify_data.py --data-root ../CrossLatent-study-data
python tools/verify_readouts.py --data-root ../CrossLatent-study-data
```

The data archive provides final partitions, result tables, frozen settings, compact selected-feature responses for 800 additional proteins, and numerical source data underlying the article figures. Verification checks the supplied files and saved numerical evidence without base-model inference.

[Reproduction guide](docs/REPRODUCING.md) distinguishes saved-result verification from full model inference/training. [Model downloads](docs/MODELS_AND_DATA.md) identify the exact public files. The authors' RNA SAE parameters and loading metadata are in the separate `CrossLatent-rna-sae` archive. Public base-model and external-SAE weights are obtained from their authors' links.

## Predictions and interpretation

`crosslatent predict-steps` exports contributions and intermediate predictions with fixed final coefficients. This differs from the separately refitted K=1–10 models supplied for each benchmark task. [Stepwise prediction guide](docs/STEPWISE_PREDICTION.md) explains the distinction.

`crosslatent shared` compares selected features within one fixed dictionary. `crosslatent.interpretation` supports positional profiles, feature removal/refitting and subtraction of SAE reconstruction contributions. `crosslatent.raw_probes` supplies the Raw-OLS and Raw-Ridge controls. See [API](docs/CORE_API.md).

## Code, data and attribution

- `crosslatent/`: installable algorithm/API and the 100 saved task predictors.
- `research/`: original extraction, training and evaluation implementations. Historical workspace defaults are provenance and need explicit model/input/output paths before rerunning.
- `tools/`: portable export and validation commands.
- `docs/`: reproduction, model sources and release notes.

Author-owned code uses [MIT](LICENSE); [third-party notices](THIRD_PARTY_NOTICES.md) retain upstream attribution. Database sequences and external weights keep their own conditions. Repository: [Shuo-Zhan/CrossLatent](https://github.com/Shuo-Zhan/CrossLatent).

## Citation

Software citation metadata are provided in [CITATION.cff](CITATION.cff). The associated study is *CrossLatent traces the assembly and reuse of sparse features in biological sequence models*. A preprint or publication link can be added when available.
