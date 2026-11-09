"""Clean and re-evaluate existing Fig2 candidates for every training source.

Reuses frozen training coefficients and verified prediction/activation caches.
Writes new splits/results; does not alter training, labels, or original assets.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from grabspark import fig2_clean_reselection as shared
from grabspark.raw_probes import confusion

FIG2 = shared.FIG2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def load_frame(path: Path, split: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame['source_row'] = np.arange(len(frame))
    frame['split'] = split
    frame['normalized_sequence'] = frame.sequence.map(shared.normalize_sequence)
    frame['sequence_sha256'] = frame.normalized_sequence.map(lambda s: hashlib.sha256(s.encode()).hexdigest())
    if any(len(s) != len(m) or set(m) - set('01') for s, m in zip(frame.sequence, frame.mask_protein_domains)):
        raise ValueError(f'Invalid masks: {path}')
    return frame


def candidate_models(experiment: str, base: Path, register) -> list[dict]:
    models, single_ids = [], set()
    for k in range(1, 11):
        path = register(base.with_name(base.stem + f'_{k}.json'))
        payload = json.loads(path.read_text())['features']['mask_protein_domains']
        combo = payload['combination']
        assert len(combo['neuron_ids']) == len(combo['coefficients']) == k
        models.append({'model_id': f'{experiment}|CrossLatent|{k}', 'method': 'CrossLatent', 'K': k,
                       'lambda': None, 'nonzero_count': k, 'coordinates': combo['neuron_ids'],
                       'coefficients': combo['coefficients'], 'intercept': combo['intercept_b'], 'source': str(path)})
        single_ids.add(int(payload['best_single_neuron']['neuron_id']))
    assert len(single_ids) == 1
    register(base)
    models.append({'model_id': f'{experiment}|best_single|1', 'method': 'best_single', 'K': 1, 'lambda': None,
                   'nonzero_count': 1, 'coordinates': [single_ids.pop()], 'coefficients': [1.], 'intercept': 0., 'source': str(base)})
    path = register(FIG2 / f'baselines/training/{experiment}/baseline_training_results.json')
    for item in json.loads(path.read_text())['independent_topk']:
        k = item['k']
        models.append({'model_id': f'{experiment}|Independent Top-K|{k}', 'method': 'Independent Top-K',
                       'K': k, 'lambda': None, 'nonzero_count': k, 'coordinates': item['neuron_ids'],
                       'coefficients': item['coefficients'], 'intercept': item['intercept'], 'source': str(path)})
    path = register(FIG2 / f'baselines/training/{experiment}/coordinate_lasso_budget10_results.json')
    for item in json.loads(path.read_text())['models']:
        alpha = item['alpha']
        models.append({'model_id': f'{experiment}|Screened Lasso|{alpha:.12g}', 'method': 'Screened Lasso',
                       'K': None, 'lambda': alpha, 'nonzero_count': item['nonzero_count'],
                       'coordinates': item['neuron_ids'], 'coefficients': item['coefficients_raw'],
                       'intercept': item['intercept_raw'], 'source': str(path)})
    return models


def cache_scores(cache: dict, baseline: dict, models: list[dict]) -> np.ndarray:
    columns = {int(v): i for i, v in enumerate(cache['latent_ids'])}
    baseline_columns = {str(v): i for i, v in enumerate(baseline['model_ids'])}
    scores = np.empty((len(cache['labels']), len(models)), dtype=np.float64)
    for i, model in enumerate(models):
        if model['method'] in ('CrossLatent', 'best_single'):
            selected = [columns[c] for c in model['coordinates']]
            scores[:, i] = model['intercept'] + cache['activations'][:, selected].astype(np.float64) @ np.asarray(model['coefficients'])
        else:
            scores[:, i] = baseline['scores'][:, baseline_columns[model['model_id']]]
    assert np.isfinite(scores).all()
    return scores


def main() -> None:
    output = parse_args().output.resolve()
    if output.exists():
        raise FileExistsError('Use a new output directory')
    output.mkdir(parents=True)
    fingerprints = {}
    def register(path):
        path = Path(path).resolve()
        if str(path) not in fingerprints:
            fingerprints[str(path)] = {'sha256': shared.digest(path), 'size_bytes': path.stat().st_size}
        return path
    manifest = pd.read_csv(register(FIG2 / 'same_clan_evaluation/experiment_manifest.csv'))
    audit_manifest = json.loads(register(REPO / 'fig/experiments/fig2_all_sources_overlap_audit_20261007/input_manifest.json').read_text())
    audit_inputs = {r['path']: r['sha256'] for r in audit_manifest}
    protocol = {'normalization': 'complete sequence; whitespace stripped; uppercase',
                'priority': 'train > valid > test; within-split duplicates retained',
                'selection': 'validation mean per-sequence two-class macro F1',
                'threshold_grid': shared.THRESHOLDS.tolist(),
                'ties': 'lower nonzero count, threshold closer to 0.5, lower threshold; stable Lasso candidate order',
                'training': 'all saved coefficients/training rows unchanged; no fitting or extraction',
                'test_policy': 'freeze clean validation settings before calculating test metrics',
                'scope': '11 sources; four established latent methods; Raw Group3 already cleaned separately',
                'external_policy': 'no row removal; apply newly frozen clean-validation settings',
                'uncertainty': 'point estimates; no new bootstrap or significance tests'}
    shared.save_json(output / 'protocol.json', protocol)
    frames, caches, baselines = {}, {}, {}
    for split in ('valid', 'test', 'PF00069', 'PF04055'):
        path = register(shared.SOURCES[split])
        assert shared.digest(path) == audit_inputs[str(path)]
        frames[split] = load_frame(path, split)
        activation = (FIG2 / f'same_clan_evaluation/activation_cache/{split}_needed_latents.npz'
                      if split in ('valid', 'test') else FIG2 / f'cross_clan/evaluation/activation_cache/{split}_needed_latents.npz')
        safe = {'test': 'cl0023_test'}.get(split, split.lower())
        baseline = FIG2 / f'baselines/evaluation/score_cache/{safe}_f0d48f35a99a.npz'
        for destination, p in ((caches, activation), (baselines, baseline)):
            with np.load(register(p), allow_pickle=True) as data:
                shared.check_cache(data, frames[split])
                destination[split] = {key: data[key] for key in data.files}
    old_same = pd.read_csv(register(FIG2 / 'same_clan_evaluation/results/frozen_test_summary.csv'))
    old_external = pd.read_csv(register(FIG2 / 'cross_clan/evaluation/results/cross_clan_frozen_summary.csv'))
    old_baseline = pd.read_csv(register(FIG2 / 'baselines/evaluation/results/frozen_test_summary.csv'))
    all_summaries, all_counts, all_overlaps, max_error = [], [], [], 0.
    for record in manifest.itertuples(index=False):
        experiment = record.experiment
        root = output / experiment
        (root / 'splits').mkdir(parents=True)
        base = register(Path(record.base_json))
        train_path = register(Path(json.loads(base.read_text())['config']['csv_path']))
        assert shared.digest(train_path) == audit_inputs[str(train_path)]
        local_frames = {'train': load_frame(train_path, 'train'), **frames}
        kept, deletions, conflicts = shared.clean_splits(local_frames)
        shared.write_csv(root / 'deletions.csv', pd.DataFrame(deletions))
        shared.write_csv(root / 'label_conflicts.csv', pd.DataFrame(conflicts))
        for split, frame in local_frames.items():
            shared.write_csv(root / f'splits/{split}.csv', frame.iloc[kept[split]])
            all_counts.append({'experiment': experiment, 'split': split, 'before_rows': len(frame),
                               'after_rows': len(kept[split]), 'deleted_rows': len(frame) - len(kept[split])})
        for left, right in itertools.combinations(local_frames, 2):
            before = len(set(local_frames[left].normalized_sequence) & set(local_frames[right].normalized_sequence))
            after = len(set(local_frames[left].iloc[kept[left]].normalized_sequence) & set(local_frames[right].iloc[kept[right]].normalized_sequence))
            assert after == 0, (experiment, left, right)
            all_overlaps.append({'experiment': experiment, 'left': left, 'right': right, 'before': before, 'after': after})
        models = candidate_models(experiment, base, register)
        shared.save_json(root / 'candidates.json', {'models': models})
        valid_scores = cache_scores(caches['valid'], baselines['valid'], models)
        grid, selections = shared.validation_grids(valid_scores, caches['valid']['labels'], caches['valid']['offsets'], models, kept['valid'])
        shared.write_csv(root / 'validation_grid.csv', pd.DataFrame(grid))
        shared.write_csv(root / 'frozen_settings.csv', pd.DataFrame(selections))
        freeze = {'frozen_at': datetime.now(timezone.utc).isoformat(), 'selections': selections,
                  'test_metrics_used': False, 'clean_valid_sha256': shared.digest(root / 'splits/valid.csv'),
                  'candidates_sha256': shared.digest(root / 'candidates.json'), 'protocol_sha256': shared.digest(output / 'protocol.json')}
        shared.save_json(root / 'frozen_settings.json', freeze)
        freeze_hash = shared.digest(root / 'frozen_settings.json')
        del valid_scores
        summaries, per_sequence = [], []
        for split in ('test', 'PF00069', 'PF04055'):
            scores = cache_scores(caches[split], baselines[split], models)
            labels, offsets = caches[split]['labels'], caches[split]['offsets']
            for stage, validation_stage, rows in [('before', 'original', np.arange(len(frames[split]))),
                                                ('clean_test_only', 'original', kept[split]),
                                                ('after_clean_reselect', 'clean', kept[split])]:
                for setting in [s for s in selections if s['validation_stage'] == validation_stage]:
                    values = []
                    for source_row in rows:
                        start, end = offsets[source_row:source_row + 2]
                        counts = confusion(labels[start:end], scores[start:end, setting['model_index']], setting['threshold'])
                        values.append(counts['macro_f1'])
                        row = frames[split].iloc[source_row]
                        per_sequence.append({'experiment': experiment, 'stage': stage, 'split': split, 'method': setting['method'],
                                             'source_row': int(source_row), 'uniprot_id': row.uniprot_id, 'sequence_sha256': row.sequence_sha256,
                                             'length': int(end - start), 'model_id': setting['model_id'], 'threshold': setting['threshold'], **counts})
                    value = float(np.mean(values))
                    if stage == 'before':
                        if setting['method'] in ('CrossLatent', 'best_single'):
                            old = old_same[old_same.experiment.eq(experiment) & old_same.method.eq(setting['method'])] if split == 'test' else old_external[old_external.experiment.eq(experiment) & old_external.method.eq(setting['method']) & old_external.external_dataset.eq(split)]
                            expected = old['test_sequence_macro_class_f1' if split == 'test' else 'sequence_macro_class_f1'].item()
                        else:
                            dataset = 'CL0023 test' if split == 'test' else split
                            old = old_baseline[old_baseline.experiment.eq(experiment) & old_baseline.method.eq(setting['method']) & old_baseline.evaluation_dataset.eq(dataset)]
                            expected = old.sequence_macro_class_f1.item()
                        error = abs(value - expected)
                        max_error = max(max_error, error)
                        assert error < 1e-10, (experiment, split, setting['method'], value, expected)
                    summaries.append({'experiment': experiment, 'stage': stage, 'split': split, 'method': setting['method'],
                                      'model_id': setting['model_id'], 'K': setting['K'], 'lambda': setting['lambda'],
                                      'nonzero_count': setting['nonzero_count'], 'threshold': setting['threshold'],
                                      'valid_f1': setting['valid_f1'], 'macro_f1': value, 'n_sequences': len(rows),
                                      'freeze_sha256': freeze_hash})
            del scores
        shared.write_csv(root / 'test_summary.csv', pd.DataFrame(summaries))
        shared.write_csv(root / 'test_per_sequence.csv', pd.DataFrame(per_sequence))
        shared.save_json(root / 'completion.json', {'status': 'PASS', 'finished_at': datetime.now(timezone.utc).isoformat(), 'freeze_sha256': freeze_hash})
        all_summaries.extend(summaries)
        print(f'{experiment}: valid {len(kept["valid"])}, test {len(kept["test"])}; original metrics reproduced', flush=True)
    table = pd.DataFrame(all_summaries)
    shared.write_csv(output / 'test_summary.csv', table)
    shared.write_csv(output / 'split_counts.csv', pd.DataFrame(all_counts))
    shared.write_csv(output / 'overlap_checks.csv', pd.DataFrame(all_overlaps))
    comparison = table.pivot(index=['experiment', 'split', 'method'], columns='stage', values='macro_f1').reset_index()
    comparison['delta_test_removal'] = comparison.clean_test_only - comparison.before
    comparison['delta_reselection'] = comparison.after_clean_reselect - comparison.clean_test_only
    comparison['delta_total'] = comparison.after_clean_reselect - comparison.before
    shared.write_csv(output / 'before_after_comparison.csv', comparison)
    # The repeated Group3 analysis must agree with its previously verified delivery.
    group3 = pd.read_csv(register(REPO / 'fig/experiments/fig2_group3_clean_reselection_20261006/test_summary.csv'))
    for row in table[table.experiment.eq('group3')].itertuples(index=False):
        match = group3[group3.stage.eq(row.stage) & group3.split.eq(row.split) & group3.method.eq(row.method)]
        assert abs(match.macro_f1.item() - row.macro_f1) < 1e-12
    assert all(shared.digest(Path(p)) == state['sha256'] for p, state in fingerprints.items())
    shared.save_json(output / 'input_fingerprints.json', fingerprints)
    shared.save_json(output / 'verification.json', {'status': 'PASS', 'sources': 11, 'methods': 4,
                     'original_metrics_reproduced': 132, 'original_max_error': max_error,
                     'all_110_post_clean_overlap_checks_zero': True, 'group3_previous_results_reproduced': True,
                     'original_inputs_unchanged': True, 'training_unchanged': True, 'labels_unchanged': True,
                     'selection_uses_test_metrics': False})
    print('PASS: 11 sources, 132 original metrics reproduced, 110 zero-overlap checks', flush=True)


if __name__ == '__main__':
    main()
