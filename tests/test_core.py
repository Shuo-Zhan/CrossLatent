"""Core scientific invariants; no language-model downloads or GPU required."""
import json

import numpy as np
import pytest
import torch

from crosslatent import ActivationDataset, LinearReadout, evaluate, fit, refit_leave_one_out
from crosslatent._core.neuron_boosting import select_neurons_iteratively
from crosslatent.adapters import BatchTopKTiedSAE
from crosslatent.cli import demo, run_config
from crosslatent.interpretation import ablate_latents, readout_effects, shared_latents


def dataset(x, y, prefix="s"):
    return ActivationDataset(x, y, np.full(len(y) // 10, 10), "test-dictionary",
                             np.array([f"{prefix}-{i}" for i in range(len(y) // 10)]))


def test_each_selection_matches_exhaustive_one_step_ols():
    rng = np.random.default_rng(133)
    x = rng.normal(size=(150, 8))
    x[:, 5] += 0.8 * x[:, 0]
    y = 2 * x[:, 5] - x[:, 3] + 0.2 * rng.normal(size=150)
    original = x.copy()
    result = select_neurons_iteratively(x, y, 4, verbose=False, compute_step_f1=False)
    chosen = [step.neuron_id for step in result.steps if step.neuron_id >= 0]
    prefix = []
    for actual in chosen:
        losses = {}
        for candidate in set(range(x.shape[1])) - set(prefix):
            design = np.column_stack([np.ones(len(y)), x[:, prefix + [candidate]]])
            residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
            losses[candidate] = residual @ residual
        assert actual == min(losses, key=losses.get)
        prefix.append(actual)
    np.testing.assert_array_equal(x, original)


def test_sequence_equal_weight_not_pooled():
    model = LinearReadout([0], [1.0], 0, 1, "d")
    data = ActivationDataset(np.array([[0], [1], [1], [0], [1], [0]]),
                             np.array([0, 1, 0, 1, 0, 1]), np.array([2, 4]), "d")
    assert evaluate(model, data)["value"] == 0.5


@pytest.mark.parametrize("bad_lengths", [np.array([0, 10]), np.array([9]), np.array([10.0])])
def test_invalid_lengths_rejected(bad_lengths):
    with pytest.raises(ValueError):
        ActivationDataset(np.ones((10, 2)), np.ones(10), bad_lengths, "d")


def test_split_overlap_and_dictionary_mismatch_rejected():
    x = np.arange(20).reshape(10, 2)
    a = dataset(x, np.arange(10) % 2)
    with pytest.raises(ValueError, match="overlap"):
        fit(a, a, max_latents=1)
    b = dataset(x, np.arange(10) % 2, "v")
    b.dictionary_id = "other"
    with pytest.raises(ValueError, match="dictionary"):
        fit(a, b, max_latents=1)


def test_nan_and_nonbinary_labels_rejected():
    with pytest.raises(ValueError, match="finite"):
        dataset(np.full((10, 2), np.nan), np.zeros(10))
    a = dataset(np.ones((10, 2)), np.full(10, 0.3), "a")
    b = dataset(np.ones((10, 2)), np.full(10, 0.3), "b")
    with pytest.raises(ValueError, match="binary labels"):
        fit(a, b, max_latents=1)


def test_demo_roundtrip_relocated_config_and_frozen_evaluation(tmp_path):
    result = demo(tmp_path / "original")
    assert result["test"]["value"] > result["single_test"]["value"]
    assert result["test"]["k"] == 2
    (tmp_path / "original").rename(tmp_path / "relocated")
    report = run_config(tmp_path / "relocated/run.json")
    assert report["test"]["value"] == 1.0
    model = LinearReadout.load(tmp_path / "relocated/result/model.json")
    before = model.to_dict()
    test = ActivationDataset.load(tmp_path / "relocated/test.npz")
    test.labels = 1 - test.labels
    assert evaluate(model, test)["value"] < 0.1
    assert model.to_dict() == before


def test_refit_distinguishes_redundant_and_unique_information():
    rng = np.random.default_rng(9)

    def make(prefix):
        z = rng.normal(size=(100, 2))
        return dataset(np.column_stack([z[:, 0], z[:, 0], z[:, 1]]), z.sum(axis=1), prefix)

    train, valid, test = make("t"), make("v"), make("e")
    model = LinearReadout([0, 1, 2], [1.0, 0.0, 1.0], 0, 3, "test-dictionary", "continuous", None)
    rows = refit_leave_one_out(model, train, valid, test)["leave_one_out"]
    assert abs(rows[0]["delta_refit"]) < 1e-12
    assert abs(rows[1]["delta_refit"]) < 1e-12
    assert rows[2]["delta_refit"] > 0.2
    assert rows[0]["model"]["latent_ids"] == [1, 2]


def test_fixed_budget_retains_last_prefix():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(100, 4))
    train = dataset(x, x[:, 0] + 0.1 * rng.normal(size=100), "t")
    z = rng.normal(size=(100, 4))
    valid = dataset(z, z[:, 0], "v")
    model = fit(train, valid, max_latents=3, task="continuous", fixed_budget=True)
    assert len(model.latent_ids) == 3
    assert model.metadata["budget_selection"] == "fixed"


def test_ablation_keeps_original_residual_and_margin_definition():
    hidden = torch.tensor([[2.0, 3.0], [4.0, 1.0]])
    features = torch.tensor([[0.5, 0.8], [0.2, 0.4]])
    decoder = torch.tensor([[1.0, 2.0], [0.0, 1.0]])
    original = hidden.clone()
    changed = ablate_latents(hidden, features, decoder, [0])
    torch.testing.assert_close(changed, hidden - features[:, :1] @ decoder[:, :1].T)
    torch.testing.assert_close(hidden, original)
    weight = torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, 1.0]])
    head = lambda x: x @ weight
    ids = torch.tensor([0, 1])
    effects = readout_effects(hidden, features, decoder, [0], head, ids)
    expected = []
    for before, after, target in zip(head(hidden), head(changed), ids):
        competitors = [i for i in range(3) if i != target]
        expected.append((after[target] - after[competitors].logsumexp(0)) -
                        (before[target] - before[competitors].logsumexp(0)))
    torch.testing.assert_close(effects["delta_margin"], torch.stack(expected))
    assert (effects["kl_original_to_intervened"] >= -1e-7).all()


def test_batchtopk_is_global_not_per_token():
    sae = BatchTopKTiedSAE(2, 3, k=1)
    with torch.no_grad():
        sae.W.copy_(torch.tensor([[10.0, 9.0, 0.0], [0.0, 0.0, 1.0]]))
    encoded = sae.encode(torch.eye(2))
    assert (encoded[0] > 0).sum() == 2
    assert (encoded[1] > 0).sum() == 0


def test_shared_sets_require_same_dictionary():
    a = LinearReadout([2, 0], [1.0, 1.0], 0, 3, "d")
    b = LinearReadout([0, 1], [1.0, 1.0], 0, 3, "d")
    assert shared_latents([a, b]) == [0]
    b.dictionary_id = "other"
    with pytest.raises(ValueError):
        shared_latents([a, b])

