# Tested environment

The installed release wheel and the 32 software tests were checked with Python 3.10.16 and these package versions:

| Package | Version |
|---|---|
| CrossLatent | 0.1.4 |
| NumPy | 2.0.1 |
| pandas | 2.3.3 |
| PyTorch | 2.5.1 |
| SciPy | 1.15.1 |
| tqdm | 4.67.1 |
| einops | 0.8.1 |
| pytest | 9.1.0 |
| setuptools | 65.5.0 |
| wheel | 0.45.1 |

This is a record of the tested versions, not a complete dependency lock or a claim that every supported dependency version was tested. The isolated validation environment reused existing scientific dependencies. The software tests and synthetic demo run on CPU without downloaded biological models.

The original model-extraction workflows in `research/` require the corresponding external model packages, checkpoints and modality-specific environments. See [model sources](MODELS_AND_DATA.md) and [reproduction instructions](REPRODUCING.md).
