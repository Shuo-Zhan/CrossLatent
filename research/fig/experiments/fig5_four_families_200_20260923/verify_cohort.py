"""Audit frozen cohort against acquisition records and pre-existing reference IDs."""
from pathlib import Path
import hashlib
import json
import re
import pandas as pd

OUT=Path(__file__).resolve().parent
PILOT=OUT.parent/'fig5_new_sequences_20260923'

def main():
    frame=pd.read_csv(OUT/'selected_sequences.csv',dtype={'taxid':str,'mask_target_domain':str})
    manifest=json.loads((OUT/'selection_manifest.json').read_text())
    assert hashlib.sha256((OUT/'selected_sequences.csv').read_bytes()).hexdigest()==manifest['cohort_sha256']
    assert len(frame)==800 and frame.accession.nunique()==800 and frame.sequence_sha256.nunique()==800
    assert (frame.groupby('family').size()==200).all()
    old_ids=set(json.loads((PILOT/'old_accessions.json').read_text()))
    pilot=pd.read_csv(PILOT/'selected_sequences.csv')
    assert not set(frame.accession)&(old_ids|set(pilot.accession))
    old_hashes=set(pilot.sequence_sha256)
    with (PILOT/'old_sequences.fasta').open() as handle:
        for line in handle:
            if line.startswith('>old_'):
                old_hashes.add(line.strip()[5:])
    assert not set(frame.sequence_sha256)&old_hashes
    selected=set(frame.accession)
    raw_sequences={}
    for name,digest in manifest['raw_sha256'].items():
        raw=(OUT/'raw'/name).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==digest
        for record in json.loads(raw)['results']:
            accession=record['metadata']['accession']
            if accession in selected:
                raw_sequences[accession]=record['extra_fields']['sequence']
    for row in frame.itertuples():
        assert row.sequence==raw_sequences[row.accession]
        assert row.sequence_sha256==hashlib.sha256(row.sequence.encode()).hexdigest()
        assert len(row.sequence)==row.length==len(row.mask_target_domain)
        assert re.fullmatch('[ACDEFGHIKLMNPQRSTVWY]+',row.sequence)
        assert 50<=row.length<=1022
        expected=['0']*row.length
        for a,b in json.loads(row.domains):
            assert 0<=a<b<=row.length
            expected[a:b]=['1']*(b-a)
        assert ''.join(expected)==row.mask_target_domain
    frozen=json.loads((OUT/'frozen_readouts.json').read_text())
    for source,digest in frozen['source_hashes'].items():
        assert hashlib.sha256(Path(source).read_bytes()).hexdigest()==digest
        assert hashlib.sha256((OUT/Path(source).name).read_bytes()).hexdigest()==digest
    ploop=pd.read_csv(OUT/'ploop_shared_cohort.csv')
    assert set(ploop.accession)==set(frame[frame.family.isin(['PF00004','PF00005'])].accession)
    clusters=pd.read_csv(OUT/'new_clusters_cluster.tsv',sep='\t',names=['cluster','accession'])
    assert clusters[clusters.accession.isin(selected)].cluster.nunique()==800
    assert not (frame.identity>=.8).any()
    result=dict(n_sequences=800,exact_old_or_pilot_sequence_overlap=0,old_or_pilot_accession_overlap=0,unique_new_clusters=800,canonical_full_length_sequences=True,all_sequences_match_raw_download=True,all_domain_masks_match_coordinates=True,frozen_readouts_unchanged=True,ploop_cohort_identical=True,n_ploop_sequences=len(ploop),max_detected_old_identity=float(frame.identity.max()))
    (OUT/'cohort_checks.json').write_text(json.dumps(result,indent=2))
    print(result)
if __name__=='__main__':
    main()
