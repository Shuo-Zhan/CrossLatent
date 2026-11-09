"""Verify the supplied numerical evidence and recompute saved per-sequence metrics."""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import runpy
from collections import defaultdict


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def check_confusion(path, summary):
    groups = defaultdict(list)
    maximum = 0.0
    for record in rows(path):
        tp, tn, fp, fn = [int(record[k]) for k in ['tp', 'tn', 'fp', 'fn']]
        pd, nd = 2*tp+fp+fn, 2*tn+fp+fn
        score = ((2*tp/pd if pd else 0) + (2*tn/nd if nd else 0))/2
        error = abs(score-float(record['macro_f1']))
        assert error < 2e-11, (path, record, error)
        maximum = max(maximum, error)
        groups[(record['split'], record['method'])].append(score)
    for record in rows(summary):
        scores = groups[(record['split'],record['method'])]
        error = abs(sum(scores)/len(scores)-float(record['macro_f1']))
        assert error < 2e-11, (summary, record, error)
        assert len(scores) == int(record['n_sequences'])
        maximum = max(maximum,error)
    return maximum


def verify(root):
    manifest = json.loads((root/'MANIFEST.json').read_text())
    for name, entry in manifest['files'].items():
        p = root/name
        assert p.is_file() and p.stat().st_size == entry['bytes'], name
        assert digest(p) == entry['sha256'], name
    runpy.run_path(str(root/'supplementary_data/verify_supplement.py'),run_name='__main__')
    source_count = 0
    worst = 0.0
    for source in sorted((root/'fig2').iterdir()):
        if not source.is_dir():
            continue
        identities = {}
        for split in ['train','valid','test']:
            samples = rows(source/'splits'/f'{split}.csv')
            ids = set()
            for sample in samples:
                seq = ''.join(sample['sequence'].split()).upper()
                mask = sample['mask_protein_domains']
                assert len(seq)==len(mask) and set(mask) <= {'0','1'}, (source,split)
                sha = hashlib.sha256(seq.encode()).hexdigest()
                assert sample['sequence_sha256']==sha
                ids.add(sha)
            identities[split] = ids
        for a,b in itertools.combinations(identities,2):
            assert not identities[a] & identities[b], (source,a,b)
        worst = max(worst,check_confusion(source/'test_per_sequence.csv',source/'test_summary.csv'))
        source_count += 1
    assert source_count==11
    g3=root/'fig2/group3'
    worst=max(worst,check_confusion(g3/'six_method_test_per_sequence.csv',g3/'six_method_test_summary.csv'))
    tasks=0
    for suite,expected in [('protein60',60),('rna40',40)]:
        task_dirs=[p for p in (root/'raw_probes'/suite).iterdir() if p.is_dir()]
        assert len(task_dirs)==expected
        for task in task_dirs:
            worst=max(worst,check_confusion(task/'test_per_sequence.csv',task/'test_summary.csv'))
            tasks+=1
    new_count=0
    sequences=set()
    for path in (root/'fig5_new800').glob('*_external_200.fasta'):
        records=[];sequence=[]
        for line in path.read_text().splitlines():
            if line.startswith('>'):
                if sequence:records.append(''.join(sequence));sequence=[]
            else:sequence.append(line.strip())
        if sequence:records.append(''.join(sequence))
        assert len(records)==200,path
        for seq in records:
            assert seq and seq not in sequences
            sequences.add(seq)
        new_count+=len(records)
    assert new_count==800
    import numpy as np
    expected_ids = json.loads((root/'fig5_new800/frozen_readouts.json').read_text())['latent_ids']
    lookup = {}
    for path in (root/'fig5_new800').glob('*_external_200.fasta'):
        identifier, chunks = None, []
        for line in path.read_text().splitlines() + ['>END']:
            if line.startswith('>'):
                if identifier is not None: lookup[identifier] = ''.join(chunks)
                identifier, chunks = line[1:].split()[0], []
            else: chunks.append(line.strip())
    for identifier, seq in lookup.items():
        with np.load(root/'fig5_new800/activations'/f'{identifier}.npz', allow_pickle=False) as array:
            assert array['activations'].shape == (len(seq),len(expected_ids))
            assert np.isfinite(array['activations']).all()
            assert array['latent_ids'].tolist() == expected_ids
            assert str(array['sequence_sha256']) == hashlib.sha256(seq.encode()).hexdigest()
    return {'payload_files':len(manifest['files']),'sha256_verified':True,
            'fig2_sources':source_count,'fig2_split_separation':True,
            'raw_tasks':tasks,'new_protein_sequences':new_count,'new_selected_activation_arrays_verified':800,
            'maximum_f1_recomputation_error':worst}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    report=verify(args.data_root.resolve())
    if args.output:args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
