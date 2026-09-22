# Validation — 2026-09-12

A non-editable wheel was built and installed in a Python 3.10 environment with existing dependencies. Tests and the CLI ran outside the source directory.

- 13 core tests passed in 2.43 seconds. The previous paper-snapshot test was intentionally excluded with its data and module.
- The CPU synthetic demo selected K=2 and obtained test macro-F1=1.0; the single-feature reference obtained 0.7832166735. These are synthetic interface checks.
- CLI exposes demo, run, fit, evaluate, refit, predict, import-legacy, shared and encode. There is no paper-summary command.
- The wheel contains only crosslatent and crosslatent._core, with no paper data, figures or model weights.
- Direct dependencies were reused; this was not a clean operating-system dependency-installation test. No external-model encoding or GPU inference was performed.

```sh
python -m pip install --no-deps crosslatent-0.1.1-py3-none-any.whl
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest /path/to/CrossLatent/tests -q --import-mode=importlib
crosslatent demo --output ./demo
crosslatent --help
```
