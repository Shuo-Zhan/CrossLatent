"""Independent delivery checks for all-source Fig2 cleaned evaluation."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / 'fig/experiments/fig2_all_sources_clean_reselection_20261007'


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    table = pd.read_csv(OUT / 'test_summary.csv')
    counts = pd.read_csv(OUT / 'split_counts.csv')
    comparison_count, selection_count, max_error = 0, 0, 0.
    for experiment in table.experiment.unique():
        root = OUT / experiment
        per = pd.read_csv(root / 'test_per_sequence.csv')
        positive_denominator = 2 * per.tp + per.fp + per.fn
        negative_denominator = 2 * per.tn + per.fp + per.fn
        independent = (np.divide(2 * per.tp, positive_denominator, out=np.zeros(len(per)), where=positive_denominator.ne(0)) +
                       np.divide(2 * per.tn, negative_denominator, out=np.zeros(len(per)), where=negative_denominator.ne(0))) / 2
        assert np.max(np.abs(independent - per.macro_f1)) < 1e-12
        assert (per.tp + per.tn + per.fp + per.fn).equals(per.length)
        per['recomputed_f1'] = independent
        for row in table[table.experiment.eq(experiment)].itertuples(index=False):
            match = per[per.stage.eq(row.stage) & per.split.eq(row.split) & per.method.eq(row.method)]
            assert len(match) == row.n_sequences
            error = abs(match.recomputed_f1.mean() - row.macro_f1)
            assert error < 1e-12
            max_error = max(max_error, error)
            comparison_count += 1
        settings = pd.read_csv(root / 'frozen_settings.csv')
        grid = pd.read_csv(root / 'validation_grid.csv')
        freeze = json.loads((root / 'frozen_settings.json').read_text())
        finished = json.loads((root / 'completion.json').read_text())
        assert not freeze['test_metrics_used']
        assert freeze['frozen_at'] < finished['finished_at']
        assert sha256(root / 'splits/valid.csv') == freeze['clean_valid_sha256']
        assert sha256(root / 'candidates.json') == freeze['candidates_sha256']
        assert sha256(OUT / 'protocol.json') == freeze['protocol_sha256']
        assert sha256(root / 'frozen_settings.json') == finished['freeze_sha256']
        for row in settings.itertuples(index=False):
            candidates = grid[grid.validation_stage.eq(row.validation_stage) & grid.method.eq(row.method)].copy()
            candidates['distance_half'] = abs(candidates.threshold - .5)
            best = candidates.sort_values(['valid_f1', 'nonzero_count', 'distance_half', 'threshold'],
                                          ascending=[False, True, True, True], kind='stable').iloc[0]
            assert best.model_id == row.model_id and best.threshold == row.threshold
            selection_count += 1
        split_frames = {s: pd.read_csv(root / f'splits/{s}.csv', dtype=str, keep_default_na=False)
                        for s in ('train', 'valid', 'test', 'PF00069', 'PF04055')}
        for split, frame in split_frames.items():
            expected = counts[counts.experiment.eq(experiment) & counts.split.eq(split)].after_rows.item()
            assert len(frame) == expected
        for left in split_frames:
            for right in split_frames:
                if left < right:
                    assert not (set(split_frames[left].normalized_sequence) & set(split_frames[right].normalized_sequence))
    assert comparison_count == 396 and selection_count == 88
    for path, state in json.loads((OUT / 'input_fingerprints.json').read_text()).items():
        assert sha256(path) == state['sha256']
    result = {'status': 'PASS', 'independent_summary_checks': comparison_count,
              'independent_validation_selection_checks': selection_count,
              'max_summary_error': max_error, 'confusion_counts_cover_sequence_lengths': True,
              'freeze_identity_and_chronology_verified': True,
              'all_split_pair_intersections_zero': True, 'inputs_still_unchanged': True}
    (OUT / 'independent_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
