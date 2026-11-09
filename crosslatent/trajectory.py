"""Frozen-coefficient cumulative prediction trajectories, as used in Fig. 4."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .api import LinearReadout


def predict_steps(model: LinearReadout, activations: np.ndarray, *, dictionary_id: str,
                  lengths: Sequence[int] | None = None,
                  sequence_ids: Sequence[str] | None = None) -> dict:
    """Accumulate final-model terms in their saved selection order, without refitting.

    scores and labels include step zero (intercept only). Contributions, feature
    activations and transition flags contain steps 1..K. All arrays are token-major.
    Binary transitions compare each step to its actual preceding prediction.
    Continuous trajectories supply scores and signed contributions without labels.
    """
    if dictionary_id != model.dictionary_id:
        raise ValueError('dictionary_id does not match model')
    values = np.asarray(activations)
    # Reuse frozen prediction's shape and selected-column finiteness validation.
    final = model.predict(values)
    n = len(final)
    if n == 0:
        raise ValueError('trajectory requires at least one token')
    if lengths is None:
        lengths = np.array([n], dtype=np.int64)
    else:
        lengths = np.asarray(lengths)
    if (lengths.ndim != 1 or not np.issubdtype(lengths.dtype, np.integer)
            or np.any(lengths <= 0) or int(lengths.sum()) != n):
        raise ValueError('lengths must be positive integers summing to token count')
    if sequence_ids is None:
        sequence_ids = [f'sequence_{i+1}' for i in range(len(lengths))]
    else:
        sequence_ids = list(sequence_ids)
    if (len(sequence_ids) != len(lengths) or any(not isinstance(s, str) or not s for s in sequence_ids)
            or len(set(sequence_ids)) != len(sequence_ids)):
        raise ValueError('sequence_ids must be unique nonempty strings matching lengths')
    selected = np.asarray(values[:, model.latent_ids], dtype=np.float64)
    contributions = selected * np.asarray(model.coefficients, dtype=np.float64)
    scores = np.column_stack((np.full(n, model.intercept), model.intercept + np.cumsum(contributions, axis=1)))
    # Match the established frozen readout exactly at the final step, including rounding.
    scores[:, -1] = final
    if not np.isfinite(scores).all() or not np.isfinite(contributions).all() or not np.isfinite(final).all():
        raise ValueError('nonfinite trajectory scores or contributions')
    offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
    result = {'scores': scores, 'contributions': contributions, 'activations': selected,
              'latent_ids': np.asarray(model.latent_ids, dtype=np.int64),
              'coefficients': np.asarray(model.coefficients, dtype=np.float64),
              'intercept': np.asarray(model.intercept), 'offsets': offsets,
              'sequence_ids': np.asarray(sequence_ids), 'dictionary_id': np.asarray(dictionary_id),
              'task': np.asarray(model.task), 'steps': np.arange(len(model.latent_ids)+1)}
    if model.task == 'binary':
        labels = scores >= model.threshold
        previous, current = labels[:, :-1], labels[:, 1:]
        result.update({'threshold': np.asarray(model.threshold), 'labels': labels,
                       'retained': previous & current, 'added': ~previous & current,
                       'removed': previous & ~current})
    return result


def export_steps(model: LinearReadout, activations: np.ndarray, output: str | Path, *,
                 dictionary_id: str, lengths: Sequence[int] | None = None,
                 sequence_ids: Sequence[str] | None = None) -> dict:
    """Write NPZ, per-position CSV, sequence/step summary and explanatory Markdown."""
    trajectory = predict_steps(model, activations, dictionary_id=dictionary_id,
                               lengths=lengths, sequence_ids=sequence_ids)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/'trajectory.npz', **trajectory)
    model.save(output/'model.json')
    k = len(model.latent_ids)
    binary = model.task == 'binary'
    with (output/'positions.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['sequence_id','position','step','latent_id','coefficient','activation',
                         'contribution','score','label','change'])
        for sequence, start, end in zip(trajectory['sequence_ids'], trajectory['offsets'][:-1], trajectory['offsets'][1:]):
            for token in range(int(start), int(end)):
                for step in range(k+1):
                    change = ''
                    if binary:
                        if step == 0: change = 'baseline'
                        elif trajectory['added'][token,step-1]: change = 'added'
                        elif trajectory['removed'][token,step-1]: change = 'removed'
                        elif trajectory['retained'][token,step-1]: change = 'retained'
                        else: change = 'negative'
                    writer.writerow([sequence,token-int(start)+1,step,
                                     model.latent_ids[step-1] if step else '',
                                     model.coefficients[step-1] if step else '',
                                     trajectory['activations'][token,step-1] if step else '',
                                     trajectory['contributions'][token,step-1] if step else '',
                                     trajectory['scores'][token,step],
                                     int(trajectory['labels'][token,step]) if binary else '',change])
    with (output/'steps.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['sequence_id','step','latent_id','coefficient','positive_count','retained','added','removed'])
        for sequence,start,end in zip(trajectory['sequence_ids'],trajectory['offsets'][:-1],trajectory['offsets'][1:]):
            for step in range(k+1):
                counts = [int(trajectory['labels'][start:end,step].sum())] if binary else ['']
                counts += ([int(trajectory[key][start:end,step-1].sum()) for key in ['retained','added','removed']]
                           if binary and step else ['','',''])
                writer.writerow([sequence,step,model.latent_ids[step-1] if step else '',
                                 model.coefficients[step-1] if step else '',*counts])
    report = {'schema':'crosslatent-trajectory/v1','mode':'final-coefficient-prefix',
              'task':model.task,'dictionary_id':model.dictionary_id,'features':model.latent_ids,
              'threshold':model.threshold if binary else None,'tokens':len(trajectory['scores']),
              'sequences':len(trajectory['sequence_ids']),'feature_steps':k,
              'baseline_step':0,'position_indexing':'one-based within sequence',
              'transition_reference':'actual preceding step, including intercept-only step zero',
              'output':'trajectory.npz, positions.csv, steps.csv, model.json, README.md'}
    (output/'metadata.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'README.md').write_text(
        '# CrossLatent stepwise prediction\n\n'
        'This trajectory accumulates final-model coefficients in saved feature order. '
        'No coefficients, intercept or threshold are refitted or retuned. '
        'Step zero contains only the intercept; steps 1..K add one signed feature contribution.\n\n'
        '`positions.csv` contains per-sequence, one-based positions, scores, contributing feature IDs, '
        'coefficients, activations, contributions and binary label transitions. '
        '`steps.csv` summarizes each sequence and step. `trajectory.npz` retains all arrays without rounding. '
        'Scores/labels have K+1 columns; contributions/activations/transitions have K columns.\n\n'
        'Binary labels use the final frozen threshold. Added, retained and removed positives compare '
        'to the actual preceding step, including the intercept-only baseline. Continuous tasks contain '
        'scores and contributions; binary label/change fields are empty. Scores are raw linear outputs.\n\n'
        'Candidates remain available for reviewing and selecting a step; the export does not '
        'automatically select a step from test labels or perform expert corrections.\n')
    return report
