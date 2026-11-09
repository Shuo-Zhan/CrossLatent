# Validation

Software version: 0.1.4.

- All 32 core, raw-probe and fixed-coefficient trajectory tests passed.
- The release wheel was built without changing scientific algorithms. It includes all 100 final models and 100 trajectory JSONs covering 1,000 separately fitted budgets.
- All 1,000 sets of feature indices, signed coefficients and intercepts match their source fitting JSONs exactly. Final feature counts and thresholds, and each budget's threshold, match validation curves.
- The data verifier checks every payload hash, eleven source-specific sequence partitions, 100 raw-coordinate comparison tasks, 800 additional proteins and their selected-feature activation arrays, and the retained numerical source files.
- The retained original scientific payload files (5,818 files) and the RNA SAE checkpoint are byte-identical to the source release. RNA SAE training/development FASTAs and all standalone plotting assets are excluded from these public packages.
- The publication-facing documentation and commands use release-neutral names. No remote upload is performed by local packaging.
- Both public archive links passed complete anonymous downloads, ZIP CRC checks and exact size/SHA256 comparison with the prepared study-data and RNA SAE archives.

The verification of synthetic matrix products uses float64 tolerance 1e-12 for BLAS rounding; saved coefficients are compared exactly. Tests use an isolated Python environment with existing scientific dependencies, rather than a fresh operating-system installation. Full biological model inference and SAE training were not rerun. Archive checksums are provided alongside the ZIP files.
