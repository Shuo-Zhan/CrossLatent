"""Seeded raw-probe API example; synthetic data, no biological claims."""
import numpy as np
from crosslatent.raw_probes import accumulate, ols_path, ridge_path, predict, threshold_grid, select_validation

rng = np.random.default_rng(20261011)
train_x = rng.normal(size=(400, 12))
valid_x = rng.normal(size=(100, 12))
train_y = (train_x[:, 0] + train_x[:, 3] > 0).astype(float)
valid_y = (valid_x[:, 0] + valid_x[:, 3] > 0).astype(float)
stats = accumulate([(train_x, train_y)])
ols = ols_path(stats, max_k=10)
ridge, normalization = ridge_path(stats)
for models in [ols, ridge]:
    candidates = []
    for model in models:
        scores = predict(valid_x, model)
        grid = threshold_grid(valid_y, scores, np.arange(0, 101, 10))
        for i, metric in enumerate(grid):
            candidates.append({**model, 'threshold': i/20, 'valid_metric': float(metric)})
    selected = select_validation(candidates)
    print(selected['method'], 'validation F1:', selected['valid_metric'])
# Serialize each validation-selected model before evaluating an independent test set.
