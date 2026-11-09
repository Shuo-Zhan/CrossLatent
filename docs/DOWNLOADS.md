# Study data and RNA SAE downloads

| Archive | Contents | Download |
|---|---|---|
| `CrossLatent-study-data.zip` (222.4 MB) | Study inputs, partitions, numerical results, source tables and saved predictors | [Google Drive](https://drive.google.com/file/d/1vw_L7JU67ECAWKoeZmfX9KRCN_U0xn1q/view?usp=drive_link) |
| `CrossLatent-rna-sae.zip` (52.6 MB) | Authors' RNA SAE parameter file and loading metadata | [Google Drive](https://drive.google.com/file/d/1b2zQmqIVEXz0Nag2CQyxXXQTWU1RsiXu/view?usp=drive_link) |

The study-data archive includes downstream benchmark sequences, annotations, data partitions, saved numerical results, all 100 final readouts and 1,000 fitted-budget models, selected-feature responses for 800 additional proteins, and numerical source data. It does not bundle rendered figures or plotting scripts.

The RNA SAE archive contains the exact `rna_sae.pt` used by the saved RNA predictors, a model card, model-source metadata and a manifest. It does not contain SAE training/development FASTAs or the ERNIE-RNA base model. Public base models and external protein/DNA SAEs are listed in [model sources](MODELS_AND_DATA.md).

## Download and extract

Open the Google Drive links and use the Download action. A large-file scan notice can appear; select **Download anyway**. During release validation, both complete ZIPs were downloaded anonymously, passed ZIP CRC checks, and matched the release SHA256 values.

Place the two ZIP files in the parent directory containing `CrossLatent/`. From that parent directory run:

```sh
sha256sum -c CrossLatent/docs/DATA_SHA256SUMS.txt
unzip CrossLatent-study-data.zip
unzip CrossLatent-rna-sae.zip
```

The extracted folders are `CrossLatent-study-data/` and `CrossLatent-rna-sae/`. Install CrossLatent as described in the repository README, then verify the data from the code directory:

```sh
cd CrossLatent
python tools/verify_data.py --data-root ../CrossLatent-study-data
```

The RNA parameter file is required for new RNA SAE encoding; it is not required to verify the saved numerical results. Preserve the study checkpoint, feature ordering and activation scales when using the saved predictors.

## SHA256

```text
41d384678cf8dd038dd6a4144e2bfa9837a35795c5dd0f3fef3afd066ab07cab  CrossLatent-study-data.zip
bc121277fd9f8f9ca98884901cb3f150fdf0dff9dec12fdbe488239ec8eaa84b  CrossLatent-rna-sae.zip
```

The same checksums are in [DATA_SHA256SUMS.txt](DATA_SHA256SUMS.txt); machine-readable access records are in [downloads.json](downloads.json).
