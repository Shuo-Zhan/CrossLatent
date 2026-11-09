"""Re-select existing Fig2 Group3 candidates after sequence overlap removal."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from grabspark.raw_probes import confusion
from grabspark.run_raw_probe_experiment import score_models

FIG2 = REPO / 'fig/paper_figures/fig2'
RAW = REPO / 'fig/experiments/raw_embedding_probes_20261006'
THRESHOLDS = np.arange(21) / 20
SOURCES = {
    'train': REPO / 'data/Protein/clan_CL0023/CL0023_group3.csv',
    'valid': FIG2 / 'data/CL0023_filtered_valid.csv',
    'test': FIG2 / 'data/CL0023_filtered_test.csv',
    'PF00069': FIG2 / 'cross_clan/data/PF00069_filtered.csv',
    'PF04055': FIG2 / 'cross_clan/data/PF04055_filtered.csv',
}


def normalize_sequence(sequence: str) -> str:
    """Uppercase complete sequence with whitespace removed; no residue remapping."""
    return ''.join(sequence.split()).upper()


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, float_format='%.15g')


def clean_splits(frames: dict[str, pd.DataFrame]) -> tuple[dict, list, list]:
    """Retain all train rows; train > valid > test for cross-split duplicates."""
    kept = {split: np.arange(len(frame), dtype=int) for split, frame in frames.items()}
    deletions, conflicts = [], []
    train = frames['train']
    valid = frames['valid']
    test = frames['test']
    train_keys = set(train.normalized_sequence)
    kept['valid'] = np.flatnonzero(~valid.normalized_sequence.isin(train_keys))
    valid_keys = set(valid.iloc[kept['valid']].normalized_sequence)
    kept['test'] = np.flatnonzero(~test.normalized_sequence.isin(train_keys | valid_keys))
    for split, priorities in [('valid', ['train']), ('test', ['train', 'valid'])]:
        removed = sorted(set(range(len(frames[split]))) - set(kept[split]))
        for index in removed:
            row = frames[split].iloc[index]
            keeper_split = next(s for s in priorities if row.normalized_sequence in set(frames[s].iloc[kept[s]].normalized_sequence))
            matches = frames[keeper_split].iloc[kept[keeper_split]]
            matches = matches[matches.normalized_sequence == row.normalized_sequence]
            deletions.append({'removed_split': split, 'source_row': index, 'uniprot_id': row.uniprot_id,
                              'normalized_sequence_sha256': row.sequence_sha256, 'length': len(row.normalized_sequence),
                              'reason': f'exact_sequence_in_{keeper_split}', 'retained_split': keeper_split,
                              'retained_source_rows': json.dumps(matches.source_row.tolist()),
                              'retained_uniprot_ids': json.dumps(matches.uniprot_id.tolist()),
                              'label_conflict_with_retained': any(row.mask_protein_domains != m for m in matches.mask_protein_domains)})
    # Audit every ORIGINAL cross-split overlapping pair, including rows removed from both valid and test.
    for left, right in itertools.combinations(['train', 'valid', 'test'], 2):
        shared = set(frames[left].normalized_sequence) & set(frames[right].normalized_sequence)
        for sequence in sorted(shared):
            a = frames[left][frames[left].normalized_sequence == sequence]
            b = frames[right][frames[right].normalized_sequence == sequence]
            for _, x in a.iterrows():
                for _, y in b.iterrows():
                    if x.mask_protein_domains == y.mask_protein_domains:
                        continue
                    positions = [i for i, (u, v) in enumerate(zip(x.mask_protein_domains, y.mask_protein_domains)) if u != v]
                    conflicts.append({'left_split': left, 'right_split': right, 'left_source_row': int(x.source_row),
                                      'right_source_row': int(y.source_row), 'left_id': x.uniprot_id, 'right_id': y.uniprot_id,
                                      'normalized_sequence_sha256': x.sequence_sha256, 'length': len(sequence),
                                      'mismatch_positions_zero_based': json.dumps(positions), 'mismatch_count': len(positions),
                                      'left_mask': x.mask_protein_domains, 'right_mask': y.mask_protein_domains,
                                      'left_retained': int(x.source_row) in set(kept[left]),
                                      'right_retained': int(y.source_row) in set(kept[right])})
    return kept, deletions, conflicts


def choose_setting(rows: list[dict]) -> dict:
    """Select only from validation, preserving legacy per-method tie rules."""
    if any(any(key.startswith('test') for key in row) for row in rows):
        raise ValueError('Test information cannot enter selection')
    def key(row):
        complexity = -row['lambda'] if row['method'] == 'Raw-Ridge' else row['nonzero_count']
        return (-row['valid_f1'], complexity, abs(row['threshold'] - .5), row['threshold'])
    return min(rows, key=key).copy()


def candidate_models(register) -> list[dict]:
    models = []
    directory = REPO / 'ckpt/json/Protein/protein_domains/group3_protein_domains'
    base = directory / 'ESM2_650M_33_mask_protein_domains_neuron_parameters_group3.json'
    single = None
    for k in range(1, 11):
        path = base.with_name(base.stem + f'_{k}.json')
        payload = json.loads(register(path).read_text())['features']['mask_protein_domains']
        combo = payload['combination']
        models.append({'model_id': f'group3|CrossLatent|{k}', 'method': 'CrossLatent', 'K': k,
                       'lambda': None, 'nonzero_count': k, 'coordinates': combo['neuron_ids'],
                       'coefficients': combo['coefficients'], 'intercept': combo['intercept_b'], 'source': str(path)})
        identifier = int(payload['best_single_neuron']['neuron_id'])
        if single is not None and single != identifier:
            raise ValueError('Training best-single ID changes with K')
        single = identifier
    register(base)
    models.append({'model_id': 'group3|best_single|1', 'method': 'best_single', 'K': 1, 'lambda': None,
                   'nonzero_count': 1, 'coordinates': [single], 'coefficients': [1.0], 'intercept': 0.0, 'source': str(base)})
    p = register(FIG2 / 'baselines/training/group3/baseline_training_results.json')
    for item in json.loads(p.read_text())['independent_topk']:
        k = item['k']
        models.append({'model_id': f'group3|Independent Top-K|{k}', 'method': 'Independent Top-K',
                       'K': k, 'lambda': None, 'nonzero_count': k, 'coordinates': item['neuron_ids'],
                       'coefficients': item['coefficients'], 'intercept': item['intercept'], 'source': str(p)})
    p = register(FIG2 / 'baselines/training/group3/coordinate_lasso_budget10_results.json')
    for item in json.loads(p.read_text())['models']:
        alpha = item['alpha']
        models.append({'model_id': f'group3|Screened Lasso|{alpha:.12g}', 'method': 'Screened Lasso',
                       'K': None, 'lambda': alpha, 'nonzero_count': item['nonzero_count'],
                       'coordinates': item['neuron_ids'], 'coefficients': item['coefficients_raw'],
                       'intercept': item['intercept_raw'], 'source': str(p)})
    p = register(RAW / 'models/fig2/Group3/all_candidates.json')
    for i, item in enumerate(json.loads(p.read_text())['models']):
        models.append({**item, 'model_id': f'group3|{item["method"]}|{i}', 'raw_index': i,
                       'nonzero_count': len(item['coordinates']), 'source': str(p)})
    return models


def check_cache(payload, frame: pd.DataFrame) -> None:
    ids_key = 'sequence_ids' if 'sequence_ids' in payload else 'uniprot_ids'
    if not np.array_equal(payload[ids_key].astype(str), frame.uniprot_id.to_numpy(dtype=str)):
        raise ValueError('Cached sequence IDs/order differ from original CSV')
    lengths = frame.sequence.str.len().to_numpy(dtype=int)
    expected_offsets = np.r_[0, np.cumsum(lengths)]
    labels = np.frombuffer(''.join(frame.mask_protein_domains).encode(), dtype=np.uint8) - ord('0')
    if not np.array_equal(payload['offsets'], expected_offsets) or not np.array_equal(payload['labels'], labels):
        raise ValueError('Cached token/label alignment differs')
    if 'sequence_sha256' in payload and not np.array_equal(payload['sequence_sha256'], frame.sequence_sha256):
        raise ValueError('Cached full sequence hashes differ')
    if 'source_csv' in payload and Path(str(payload['source_csv'])).resolve() != SOURCES[frame.split.iloc[0]].resolve():
        raise ValueError('Cached source path differs')


def load_scores(split, frame, models, register, output):
    """Reuse predictions, reconstruct linear scores only when no score cache exists."""
    ids = [m['model_id'] for m in models]
    scores = np.empty((int(frame.sequence.str.len().sum()), len(models)), dtype=np.float64)
    provenance = []
    activation_path = (FIG2 / 'same_clan_evaluation/activation_cache' / f'{split}_needed_latents.npz' if split in ('valid', 'test')
                       else FIG2 / 'cross_clan/evaluation/activation_cache' / f'{split}_needed_latents.npz')
    with np.load(register(activation_path), allow_pickle=True) as data:
        check_cache(data, frame)
        columns = {int(v): i for i, v in enumerate(data['latent_ids'])}
        activations = data['activations']
        for i, model in enumerate(models):
            if model['method'] not in ('CrossLatent', 'best_single'):
                continue
            selected = [columns[c] for c in model['coordinates']]
            scores[:, i] = model['intercept'] + activations[:, selected].astype(np.float64) @ np.asarray(model['coefficients'])
        provenance.append({'path': str(activation_path), 'role': 'legacy float32 latent cache; existing CrossLatent/best-single linear readout only'})
    del activations
    safe = {'test': 'cl0023_test'}.get(split, split.lower())
    baseline_path = FIG2 / 'baselines/evaluation/score_cache' / f'{safe}_f0d48f35a99a.npz'
    with np.load(register(baseline_path), allow_pickle=True) as data:
        check_cache(data, frame)
        columns = {str(v): i for i, v in enumerate(data['model_ids'])}
        baseline_scores = data['scores']
        for i, model in enumerate(models):
            if model['method'] in ('Independent Top-K', 'Screened Lasso'):
                scores[:, i] = baseline_scores[:, columns[model['model_id']]]
        provenance.append({'path': str(baseline_path), 'role': 'existing float32 per-candidate prediction cache, exact score values reused'})
    del baseline_scores
    raw_columns = [i for i, m in enumerate(models) if m['method'] in ('Raw-OLS', 'Raw-Ridge')]
    if split == 'valid':
        meta_path = register(RAW / 'cache/fig2/Group3/valid.metadata.json')
        meta = json.loads(meta_path.read_text())
        path = register(Path(meta['embedding_path']))
        if meta['csv_sha256'] != digest(SOURCES[split]) or digest(path) != meta['embedding_sha256']:
            raise ValueError('Original embedding identity differs')
        embedding = np.load(path, mmap_mode='r')
        scores[:, raw_columns] = score_models(embedding, [models[i] for i in raw_columns])
        provenance.append({'path': str(path), 'role': 'existing raw embedding + fixed saved coefficients; no extraction or fitting'})
        del embedding
    else:
        path = register(RAW / f'predictions/fig2/Group3/{split}.npz')
        with np.load(path, allow_pickle=False) as data:
            check_cache(data, frame)
            scores[:, raw_columns] = data['scores']
        provenance.append({'path': str(path), 'role': 'existing float64 raw per-candidate score cache'})
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite candidate predictions')
    offsets = np.r_[0, np.cumsum(frame.sequence.str.len())].astype(int)
    labels = np.frombuffer(''.join(frame.mask_protein_domains).encode(), dtype=np.uint8) - ord('0')
    np.savez(output / f'candidate_scores/{split}.npz', scores=scores, labels=labels, offsets=offsets,
             sequence_ids=frame.uniprot_id.to_numpy(dtype=str), sequence_sha256=frame.sequence_sha256.to_numpy(dtype=str),
             model_ids=np.asarray(ids))
    return scores, labels, offsets, provenance


def validation_grids(scores, labels, offsets, models, kept):
    grids = []
    for i, model in enumerate(models):
        per_sequence = []
        for start, end in zip(offsets[:-1], offsets[1:]):
            y = labels[start:end].astype(bool)[:, None]
            predictions = scores[start:end, i, None] >= THRESHOLDS
            tp, tn = np.sum(y & predictions, axis=0), np.sum(~y & ~predictions, axis=0)
            fp, fn = np.sum(~y & predictions, axis=0), np.sum(y & ~predictions, axis=0)
            pd_, nd = 2*tp+fp+fn, 2*tn+fp+fn
            per_sequence.append((np.divide(2*tp, pd_, out=np.zeros(21), where=pd_ != 0) +
                                 np.divide(2*tn, nd, out=np.zeros(21), where=nd != 0))/2)
        values = np.asarray(per_sequence)
        for stage, rows in [('original', np.arange(len(offsets)-1)), ('clean', kept)]:
            for threshold, value in zip(THRESHOLDS, values[rows].mean(axis=0)):
                grids.append({'validation_stage': stage, 'model_index': i, 'model_id': model['model_id'],
                              'method': model['method'], 'K': model['K'], 'lambda': model['lambda'],
                              'nonzero_count': model['nonzero_count'], 'threshold': float(threshold), 'valid_f1': float(value)})
    selected = []
    for stage in ('original', 'clean'):
        for method in dict.fromkeys(m['method'] for m in models):
            row = choose_setting([r for r in grids if r['validation_stage'] == stage and r['method'] == method])
            selected.append(row)
    return grids, selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Use a new output directory; existing results are never overwritten')
    output.mkdir(parents=True)
    for folder in ('splits', 'candidate_scores'):
        (output / folder).mkdir()
    fingerprints = {}
    def register(path):
        path = Path(path).resolve()
        if str(path) not in fingerprints:
            fingerprints[str(path)] = {'sha256': digest(path), 'size_bytes': path.stat().st_size}
        return path
    frames = {}
    original_manifest = pd.read_csv(register(RAW / 'input_manifest.csv'))
    for split, path in SOURCES.items():
        register(path)
        expected = original_manifest[(original_manifest.suite == 'fig2') & (original_manifest.split == split)].iloc[0]
        if digest(path) != expected.csv_sha256:
            raise ValueError('Original input CSV changed since prior evaluation')
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        frame['source_row'] = np.arange(len(frame))
        frame['split'] = split
        frame['normalized_sequence'] = frame.sequence.map(normalize_sequence)
        frame['sequence_sha256'] = frame.normalized_sequence.map(lambda s: hashlib.sha256(s.encode()).hexdigest())
        if any(len(s) != len(m) or set(m)-set('01') for s, m in zip(frame.sequence, frame.mask_protein_domains)):
            raise ValueError('Original label/sequence alignment invalid')
        frames[split] = frame
    kept, deletions, conflicts = clean_splits(frames)
    write_csv(output/'deletions.csv', pd.DataFrame(deletions))
    write_csv(output/'label_conflicts.csv', pd.DataFrame(conflicts))
    overlap_rows, counts = [], []
    for split, frame in frames.items():
        cleaned = frame.iloc[kept[split]]
        write_csv(output/f'splits/{split}.csv', cleaned)
        counts.append({'split': split, 'before_rows': len(frame), 'after_rows': len(cleaned), 'deleted_rows': len(frame)-len(cleaned)})
        # All original columns, including sequence and labels, must remain identical for retained rows.
        assert frame.iloc[kept[split]].drop(columns=['source_row','split','normalized_sequence','sequence_sha256']).equals(cleaned.drop(columns=['source_row','split','normalized_sequence','sequence_sha256']))
    assert len(kept['train']) == len(frames['train'])
    for a, b in itertools.combinations(frames, 2):
        before = len(set(frames[a].normalized_sequence) & set(frames[b].normalized_sequence))
        after = len(set(frames[a].iloc[kept[a]].normalized_sequence) & set(frames[b].iloc[kept[b]].normalized_sequence))
        overlap_rows.append({'left': a, 'right': b, 'before_shared_unique_sequences': before, 'after_shared_unique_sequences': after})
        if after:
            raise ValueError(f'Exact overlap remains: {a}/{b}')
    write_csv(output/'split_counts.csv', pd.DataFrame(counts))
    write_csv(output/'overlap_checks.csv', pd.DataFrame(overlap_rows))
    models = candidate_models(register)
    save_json(output/'candidates.json', {'models': models, 'policy': 'existing Group3 candidates only; no coefficient or neuron changes'})
    protocol = {'normalization': 'complete sequence, remove whitespace, uppercase; preserve ambiguous residues/gaps; no truncation',
                'priority': 'train > valid > test; within-split duplicates retained; external test families unchanged',
                'selection': 'validation mean per-sequence positive/negative class-macro F1',
                'threshold_grid': THRESHOLDS.tolist(), 'ties': 'lower K/nonzero_count, closer threshold .5, lower threshold; Raw-Ridge larger lambda before threshold',
                'lasso_ties': 'legacy stable input alpha order after equal F1, nonzero_count and threshold',
                'training': 'none; all existing training rows/labels and candidate coefficients preserved',
                'test_policy': 'cleaning/label-conflict QC may inspect source labels; no test scores/F1 used to select; freeze before test scoring',
                'scope': 'Fig2 Group3, six methods, CL0023 test/PF00069/PF04055', 'uncertainty': 'point estimates only'}
    save_json(output/'protocol.json', protocol)
    scores, labels, offsets, provenance = load_scores('valid', frames['valid'], models, register, output)
    grid, selections = validation_grids(scores, labels, offsets, models, kept['valid'])
    write_csv(output/'validation_grid.csv', pd.DataFrame(grid))
    freeze = {'frozen_at': datetime.now(timezone.utc).isoformat(), 'protocol_sha256': digest(output/'protocol.json'),
              'candidates_sha256': digest(output/'candidates.json'), 'selections': selections,
              'selection_inputs': {'valid_csv_sha256': digest(output/'splits/valid.csv'), 'candidate_score_sha256': digest(output/'candidate_scores/valid.npz')},
              'test_metrics_used': False}
    save_json(output/'frozen_settings.json', freeze)
    freeze_hash = digest(output/'frozen_settings.json')
    write_csv(output/'frozen_settings.csv', pd.DataFrame(selections).assign(freeze_sha256=freeze_hash))
    del scores
    summaries, sequence_results, prediction_checks = [], [], []
    provenance_rows = [{'split': 'valid', **r} for r in provenance]
    for split in ('test', 'PF00069', 'PF04055'):
        scores, labels, offsets, provenance = load_scores(split, frames[split], models, register, output)
        provenance_rows.extend({'split': split, **r} for r in provenance)
        for stage, validation_stage, rows in [('before', 'original', np.arange(len(frames[split]))),
                                               ('clean_test_only', 'original', kept[split]),
                                               ('after_clean_reselect', 'clean', kept[split])]:
            for setting in [s for s in selections if s['validation_stage'] == validation_stage]:
                values = []
                for source_row in rows:
                    row = frames[split].iloc[source_row]
                    start, end = offsets[source_row:source_row+2]
                    counts_ = confusion(labels[start:end], scores[start:end, setting['model_index']], setting['threshold'])
                    values.append(counts_['macro_f1'])
                    sequence_results.append({'stage': stage, 'split': split, 'method': setting['method'],
                                             'source_row': int(source_row), 'uniprot_id': row.uniprot_id, 'sequence_sha256': row.sequence_sha256,
                                             'length': int(end-start), 'model_id': setting['model_id'], 'threshold': setting['threshold'], **counts_})
                summaries.append({'stage': stage, 'split': split, 'method': setting['method'], 'model_id': setting['model_id'],
                                  'K': setting['K'], 'lambda': setting['lambda'], 'nonzero_count': setting['nonzero_count'],
                                  'threshold': setting['threshold'], 'valid_f1': setting['valid_f1'],
                                  'macro_f1': float(np.mean(values)), 'n_sequences': len(rows),
                                  'n_tokens': int(sum(offsets[r+1]-offsets[r] for r in rows)), 'freeze_sha256': freeze_hash})
        prediction_checks.append({'split': split, 'candidate_count': len(models), 'all_scores_finite': bool(np.isfinite(scores).all()),
                                  'test_scored_after_frozen_at': datetime.now(timezone.utc).isoformat()})
        del scores
    table = pd.DataFrame(summaries)
    write_csv(output/'test_summary.csv', table)
    write_csv(output/'test_per_sequence.csv', pd.DataFrame(sequence_results))
    historical = pd.read_csv(register(RAW/'all_method_comparison.csv'))
    before = table[table.stage == 'before'].copy()
    for _, row in before.iterrows():
        value = historical[(historical.suite=='fig2') & (historical.split==row.split)][row.method].iloc[0]
        if abs(value-row.macro_f1)>1e-10:
            raise ValueError(f'Original frozen result not reproduced: {row.method}/{row.split}: {value} vs {row.macro_f1}')
    comparison = table.pivot(index=['split','method'], columns='stage', values='macro_f1').reset_index()
    comparison['delta_test_removal'] = comparison.clean_test_only - comparison.before
    comparison['delta_reselection'] = comparison.after_clean_reselect - comparison.clean_test_only
    comparison['delta_total'] = comparison.after_clean_reselect - comparison.before
    comparison['n_before'] = comparison.split.map(before.groupby('split').n_sequences.first())
    comparison['n_after'] = comparison.split.map(table[table.stage=='after_clean_reselect'].groupby('split').n_sequences.first())
    write_csv(output/'before_after_comparison.csv', comparison)
    write_csv(output/'cache_provenance.csv', pd.DataFrame(provenance_rows))
    changed = [p for p, state in fingerprints.items() if digest(Path(p)) != state['sha256']]
    if changed:
        raise ValueError(f'Original files changed: {changed}')
    save_json(output/'source_fingerprints.json', fingerprints)
    verify = {'status': 'pass', 'candidate_count': len(models), 'methods': sorted(set(m['method'] for m in models)),
              'original_frozen_metrics_reproduced': len(before), 'original_max_f1_error': max(abs(float(historical[(historical.suite=='fig2')&(historical.split==row.split)][row.method].iloc[0])-row.macro_f1) for _,row in before.iterrows()),
              'all_post_clean_pair_overlaps_zero': True, 'overlap_pairs_checked': len(overlap_rows),
              'training_rows_preserved': True, 'labels_unchanged': True, 'no_retraining': True,
              'source_files_unchanged': len(fingerprints), 'freeze_sha256': freeze_hash, 'prediction_checks': prediction_checks,
              'deleted_rows': len(deletions), 'label_conflict_pairs': len(conflicts)}
    save_json(output/'verification.json', verify)
    (output/'report_zh.md').write_text('# Fig2精确重复清理与既有候选重新选参\n\n'
        '保留Group3训练集全部行。按规范化完整序列、train > valid > test优先级清理；不改标签，不重训，不覆盖旧结果。\n\n'
        '## 删除与重叠\n\n'+pd.DataFrame(counts).to_markdown(index=False)+'\n\n'
        '全部10组跨划分精确重叠均为0；不等同同源隔离。原始标签冲突及零起始位置见label_conflicts.csv；冲突没有通过修改标签解决。\n\n'
        '## 清理前后F1与分解\n\n'+comparison.to_markdown(index=False,floatfmt='.8f')+'\n\n'
        'before为原验证选参/原测试；clean_test_only保留原设置，仅删除测试重复；after_clean_reselect由清理验证重新选参/清理测试。'
        '两个外家族未删除任何行，其变化来自重新选参。所有指标为序列等权两类macro F1，未计算CI/P值。\n\n'
        '## 验证选择\n\n'+pd.DataFrame(selections)[['validation_stage','method','K','lambda','nonzero_count','threshold','valid_f1']].to_markdown(index=False,floatfmt='.8g')+'\n\n'
        '18个原冻结指标重现误差≤1e-10。逐候选验证网格、冻结JSON/hash、全候选分数和逐序列混淆计数均保留；source_fingerprints.json证明使用的旧文件执行前后SHA256不变。\n')
    (output/'README.md').write_text('# 复现与文件索引\n\n'
        '命令：\n```sh\nOPENBLAS_NUM_THREADS=4 /data1/zshuo23/anaconda3/envs/sparse_rna/bin/python grabspark/fig2_clean_reselection.py --output '+str(output)+'\n```\n\n'
        '运行要求一个尚不存在的新输出目录。此结果目录已有内容时，换用新的--output路径。无需GPU、模型提取或重新训练。\n\n'
        'deletions.csv：逐删除行及保留方对应；label_conflicts.csv：原重叠的标签冲突/位置；split_counts.csv：样本数；overlap_checks.csv：全部跨划分零重叠检查；'
        'splits/：保留原字段与标签的新CSV，加源行号及规范化身份；candidates.json：固定旧候选；validation_grid.csv：原/清理验证完整网格；'
        'frozen_settings.json/csv：持久化验证冻结；candidate_scores/：原始行顺序、全候选分数/标签/offset/ID/hash；'
        'test_summary.csv与test_per_sequence.csv：三个阶段评价；before_after_comparison.csv：样本清理和选参效应分解；'
        'verification.json、source_fingerprints.json、cache_provenance.csv：核验与旧文件身份；report_zh.md：中文结果。\n')
    print(pd.DataFrame(counts).to_string(index=False), flush=True)
    print(comparison.to_string(index=False), flush=True)
    print('PASS: all six methods; original metrics reproduced; exact overlaps zero; no old files changed', flush=True)


if __name__ == '__main__':
    main()
