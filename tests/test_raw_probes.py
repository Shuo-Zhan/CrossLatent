"""Numerical and leakage checks for the raw-coordinate probe experiment."""
import sys
from pathlib import Path
# Two retained input-QC checks use the archived extraction implementation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research'))

import numpy as np
import pytest

from crosslatent.raw_probes import accumulate, confusion, ols_path, predict, ridge_path, select_validation, threshold_grid


def sample():
    rng = np.random.default_rng(20261006)
    x = rng.normal(size=(137, 7))
    x[:, 1] = .8*x[:, 0] + .2*x[:, 1]
    x[:, 5] = x[:, 3]
    x[:, 6] = 3
    y = 2*x[:, 0] - x[:, 2] + .3*x[:, 4] + 1.4 + rng.normal(size=len(x))*.1
    return x, y


def test_conditional_gain_matches_joint_ols():
    x, y = sample()
    stats = accumulate([(x, y)])
    path = ols_path(stats, 5)
    selected = []
    residual = y-y.mean()
    for model in path:
        gains = []
        for j in range(x.shape[1]):
            if j in selected:
                gains.append(-np.inf)
                continue
            design = np.c_[np.ones(len(x)), x[:, selected+[j]]]
            fit = design @ np.linalg.lstsq(design, y, rcond=None)[0]
            gains.append(float(residual@residual - (y-fit)@(y-fit)))
        j = model['coordinates'][-1]
        assert gains[j] == pytest.approx(max(gains), abs=1e-9)
        assert model['conditional_gain'] == pytest.approx(gains[j], abs=1e-9)
        selected.append(j)
        residual = y-predict(x, model)
        assert model['K'] == len(model['coordinates'])
        assert 6 not in selected


def test_streamed_stats_and_original_coordinates():
    x, y = sample()
    a = accumulate([(x, y)])
    b = accumulate((x[i:i+17], y[i:i+17]) for i in range(0, len(x), 17))
    np.testing.assert_allclose(a.gram, b.gram, atol=1e-12)
    pa, pb = ols_path(a, 5), ols_path(b, 5)
    for ma, mb in zip(pa, pb):
        assert ma['coordinates'] == mb['coordinates']
        np.testing.assert_allclose(predict(x, ma), predict(x, mb), atol=1e-10)
        design = np.c_[np.ones(len(x)), x[:, ma['coordinates']]]
        explicit = design @ np.linalg.lstsq(design, y, rcond=None)[0]
        np.testing.assert_allclose(explicit, predict(x, ma), atol=1e-10)


def test_ridge_train_standardization_and_alpha():
    x, y = sample()
    stats = accumulate([(x, y)])
    models, norm = ridge_path(stats)
    assert norm['removed_coordinates'] == [6]
    np.testing.assert_allclose(norm['mean'], x.mean(0))
    ids = norm['retained_coordinates']
    z = (x[:, ids]-x.mean(0)[ids])/np.array(norm['scale'])
    for model in models:
        assert model['alpha'] == pytest.approx(len(x)*model['lambda'])
        coef = np.linalg.solve(z.T@z + model['alpha']*np.eye(len(ids)), z.T@(y-y.mean()))
        np.testing.assert_allclose(predict(x, model), y.mean()+z@coef, atol=1e-7)


def test_macro_f1_and_sequence_equal_weight():
    assert confusion(np.ones(3), np.ones(3), .5)['macro_f1'] == .5
    assert confusion(np.zeros(3), np.zeros(3), .5)['macro_f1'] == .5
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0])
    scores = np.array([1, 0, 0, 1, 1, 1, 1, 1])
    offsets = np.array([0, 2, 8])
    expected = (confusion(y[:2], scores[:2], .5)['macro_f1'] + confusion(y[2:], scores[2:], .5)['macro_f1'])/2
    assert threshold_grid(y, scores, offsets)[10] == pytest.approx(expected)
    assert expected != confusion(y, scores, .5)['macro_f1']


def test_alignment_errors_detected():
    with pytest.raises(ValueError):
        accumulate([(np.zeros((3, 2)), np.zeros(2))])
    with pytest.raises(ValueError):
        threshold_grid(np.zeros(3), np.zeros(3), np.array([0, 2]))
    with pytest.raises(ValueError):
        threshold_grid(np.zeros(3), np.zeros(3), np.array([0, 0, 3]))


def test_test_labels_cannot_change_frozen_selection():
    candidates = [dict(method='Raw-OLS', K=k, **{'lambda': None}, threshold=t, valid_metric=.7)
                  for k in (1, 2) for t in (.25, .5, .75)]
    before = select_validation(candidates, fig2=True)
    assert before['K'] == 1 and before['threshold'] == .5
    for test_labels in (np.zeros(4), np.ones(4)):
        confusion(test_labels, np.array([.1, .3, .6, .9]), before['threshold'])
        assert select_validation(candidates, fig2=True) == before
    assert select_validation(candidates)['threshold'] == .25
    with pytest.raises(ValueError):
        select_validation([dict(candidates[0], test_metric=1)])
    ridge = [dict(method='Raw-Ridge', K=None, **{'lambda': p}, threshold=.5, valid_metric=.7) for p in (.1, 10)]
    assert select_validation(ridge)['lambda'] == 10


def test_rna_filter_preserves_source_rows_and_detects_label_errors(tmp_path):
    import hashlib
    import pandas as pd
    from grabspark.run_raw_probe_experiment import load_frame
    sequences = ['ACT', 'A'*1023, 'AGRY']
    rows = [{'name': str(i), 'sequence': s, 'mask': '0'*len(s),
             'sequence_md5': hashlib.md5(s.encode()).hexdigest()} for i, s in enumerate(sequences)]
    path = tmp_path/'input.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    frame, labels, offsets = load_frame(path, {'modality': 'rna', 'label': 'mask'})
    assert frame.source_row.tolist() == [0, 2]
    assert frame.sequence.tolist() == ['ACU', 'AGNN']
    assert offsets.tolist() == [0, 3, 7]
    assert len(labels) == 7
    rows[0]['mask'] = '00'
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(ValueError, match='label boundary'):
        load_frame(path, {'modality': 'rna', 'label': 'mask'})


def test_nonfinite_prediction_scores_rejected():
    from grabspark.run_raw_probe_experiment import score_models
    model = {'coordinates': [0], 'coefficients': [1.0], 'intercept': 0.0}
    with pytest.raises(ValueError, match='Nonfinite'):
        score_models(np.array([[1., 0.], [np.nan, 1.]]), [model])
