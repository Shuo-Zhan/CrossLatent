"""Filter detected close homologs; deterministic clustering/taxon sampling."""
from pathlib import Path
import argparse
import hashlib
import json
import pandas as pd

OUT = Path(__file__).resolve().parent
FAMILIES = ['PF00004', 'PF00005', 'PF00069', 'PF07714']
COLS = ['accession','old_id','identity','query_coverage','target_coverage','evalue','bits']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-clustering', action='store_true')
    args = parser.parse_args()
    frame = pd.read_csv(OUT/'candidates.csv', dtype={'mask_target_domain':str,'taxid':str})
    near, best_parts = set(), []
    for hits in pd.read_csv(OUT/'old_similarity.tsv', sep='\t', names=COLS, chunksize=500000):
        near.update(hits.loc[(hits.identity >= .8) & (hits.query_coverage >= .5) & (hits.target_coverage >= .5), 'accession'])
        best_parts.append(hits.sort_values(['identity','bits'],ascending=False).drop_duplicates('accession'))
    best = pd.concat(best_parts).sort_values(['identity','bits'],ascending=False).drop_duplicates('accession')
    best.to_csv(OUT/'best_old_similarity.csv', index=False)
    frame = frame.merge(best, on='accession', how='left', validate='one_to_one')
    frame['excluded_near_old'] = frame.accession.isin(near)
    frame.to_csv(OUT/'homology_audit.csv', index=False)
    eligible = frame[~frame.excluded_near_old].copy()
    if args.prepare_clustering:
        with (OUT/'eligible.fasta').open('w') as handle:
            for row in eligible.itertuples():
                handle.write(f'>{row.accession}\n{row.sequence}\n')
        print(frame.groupby(['family','excluded_near_old']).size().to_string(), flush=True)
        return
    clusters = pd.read_csv(OUT/'new_clusters_cluster.tsv', sep='\t', names=['new_cluster','accession'])
    representatives = eligible.merge(clusters,on='accession',validate='one_to_one').sort_values('sort_key').drop_duplicates('new_cluster')
    counts = representatives.family.value_counts().reindex(FAMILIES,fill_value=0).to_dict()
    (OUT/'eligible_cluster_counts.json').write_text(json.dumps(counts, indent=2))
    print('Independent eligible representatives:', counts, flush=True)
    if any(n < 200 for n in counts.values()):
        raise SystemExit(2)
    selected = []
    for family in FAMILIES:
        group = representatives[representatives.family==family].sort_values('sort_key').copy()
        group['taxon_round'] = group.groupby('taxid').cumcount()
        selected.append(group.sort_values(['taxon_round','sort_key']).head(200))
    selected = pd.concat(selected,ignore_index=True)
    assert len(selected)==800 and selected.new_cluster.nunique()==800 and selected.sequence_sha256.nunique()==800
    selected.to_csv(OUT/'selected_sequences.csv',index=False)
    for family, subset in selected.groupby('family'):
        with (OUT/f'{family}_external_200.fasta').open('w') as handle:
            for row in subset.itertuples():
                handle.write(f'>{row.accession}\n{row.sequence}\n')
    selected[selected.family.isin(['PF00004','PF00005'])][['family','accession','sequence_sha256']].to_csv(OUT/'ploop_shared_cohort.csv',index=False)
    manifest = dict(selection_before_activations=True, n_sequences=800, n_by_family=selected.family.value_counts().to_dict(), n_taxa_by_family=selected.groupby('family').taxid.nunique().to_dict(), cohort_sha256=hashlib.sha256((OUT/'selected_sequences.csv').read_bytes()).hexdigest(), protocol_sha256=hashlib.sha256((OUT/'PROTOCOL.md').read_bytes()).hexdigest(), max_detected_old_identity=float(selected.identity.max()), raw_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((OUT/'raw').glob('*.json'))})
    (OUT/'selection_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(manifest['n_taxa_by_family'],flush=True)

if __name__=='__main__':
    main()
