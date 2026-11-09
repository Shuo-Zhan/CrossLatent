# Stepwise prediction and annotation review

CrossLatent exposes a final prediction as an ordered trajectory with position scores, identifiable feature sources, signed contributions and candidate labels. This is the frozen-coefficient cumulative trajectory used in Fig. 4.

## Prediction rule

For the final selected model with K ordered features:

    score_s(position) = intercept + sum_{j=1..s} coefficient_j * activation_j(position)

Step zero is the intercept-only baseline. Steps 1..K add one feature contribution. The final coefficients, intercept, feature order and binary threshold remain frozen. The last score column exactly matches the existing frozen `predict` result, including numerical rounding. Binary labels use `score >= threshold`. Scores are raw linear outputs, not probabilities.

The fitting history in `metadata.validation_path` describes separately fitted budgets and validation choices. `predict-steps` uses the final model's coefficients at every prefix; it does not substitute the separately fitted budget models. A final model with K selected features produces K feature steps, even if training evaluated a larger maximum budget. Training history remains available separately.

## Command line

```sh
crosslatent predict-steps \
  --model model.json \
  --activations activations.npy \
  --dictionary-id fixed-dictionary-id \
  --lengths lengths.npy \
  --sequence-ids sequence_ids.json \
  --output trajectory_output
```

`activations.npy` is a token-by-dictionary activation matrix in the exact saved dictionary order. Remove special tokens and padding before export. `lengths.npy` contains positive integer token counts summing to the total number of tokens. `sequence_ids.json` is a JSON list of unique nonempty sequence identifiers. Lengths and identifiers are optional; without lengths, all tokens are treated as one sequence. The output directory must be new.

## Python API

```python
from crosslatent import LinearReadout, predict_steps, export_steps
import numpy as np

model = LinearReadout.load('model.json')
activations = np.load('activations.npy', mmap_mode='r')
lengths = np.load('lengths.npy')
trajectory = predict_steps(
    model, activations, dictionary_id=model.dictionary_id, lengths=lengths)

export_steps(
    model, activations, 'trajectory_output',
    dictionary_id=model.dictionary_id, lengths=lengths)
```

Use the actual identifier recorded when encoding your dictionary, rather than assigning a different dictionary the model's identifier. For large datasets, export manageable sequence batches: the trajectory retains token-by-step arrays in memory.

## Output files

| File | Contents |
|---|---|
| `positions.csv` | Sequence ID, one-based position, step, newly added feature ID, final coefficient, activation, signed contribution, cumulative score, binary label and transition |
| `steps.csv` | Sequence/step summary, new feature identity and counts of positive, retained, added and removed labels |
| `trajectory.npz` | Full-precision arrays, ordered feature IDs and coefficients, sequence offsets, dictionary ID and task |
| `model.json` | The frozen model used for this export |
| `metadata.json` | Trajectory schema, step convention, coordinate convention and frozen threshold |
| `README.md` | Explanation of this export |

The score and binary-label arrays have shape tokens × (K+1), including step zero. Activations, signed contributions and transition arrays have shape tokens × K. Columns in the latter arrays correspond to feature steps 1..K.

For binary predictions, `retained`, `added` and `removed` compare each candidate to its actual preceding step. Step 1 therefore compares to the intercept-only baseline; if that baseline already predicts positives, those positions can be retained or removed. `negative` in the CSV denotes a position negative at both steps. These transitions describe prediction changes, not correctness relative to ground truth.

Continuous DNA trajectories provide scores, feature responses and signed contributions. Binary labels and transition columns are left empty, and their arrays are omitted from the NPZ.

## Use in review

Compare intermediate candidates and inspect the scores and feature contributions at changing boundaries or positions. Independent sequence or structural evidence can then guide step selection or revision of individual labels. Positive and negative contributions explain how coverage expands or excessive labels are suppressed. The output makes these choices inspectable; it does not automatically perform expert corrections or select a step using test annotations.

No fitting, threshold search or external-model inference is performed by trajectory export. All preset fitting budgets remain in the original training history.
