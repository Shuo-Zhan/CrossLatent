# Software 0.1.4

- Supply 100 validation-selected prediction models: 60 protein and 40 RNA annotation tasks.
- Include all 1,000 separately fitted K=1–10 models, with source hashes and validation-selected thresholds.
- Preserve original feature indices, signed coefficients and intercepts. Final operating points come from validation.
- Identify exact external model files, including the normalized layer-33 protein SAE and the Evo 2 262k model variant.
- Include the saved model JSONs in the installed wheel.
- Supply portable model/data verification commands and numerical source tables.
- Distribute the authors' RNA SAE parameters separately, without its training/development FASTAs.
- Omit plotting implementations, rendered figures and internal manuscript-version labels from publication-facing documentation.

Scientific parameters and numerical results are unchanged. The existing NumPy compatibility fix and research-script compatibility imports are retained. No feature selection, biological model inference or model training was rerun to prepare these archives.
