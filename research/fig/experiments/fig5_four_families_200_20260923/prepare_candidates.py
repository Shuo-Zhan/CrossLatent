"""Prepare canonical new candidates, retaining motif-negative sequences."""
from pathlib import Path
import hashlib
import json
import re
import pandas as pd

OUT = Path(__file__).resolve().parent
PILOT = OUT.parent / 'fig5_new_sequences_20260923'
FAMILIES = ['PF00004', 'PF00005', 'PF00069', 'PF07714']

def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()

def main():
    old_hashes = set()
    with (PILOT / 'old_sequences.fasta').open() as handle:
        for line in handle:
            if line.startswith('>old_'):
                old_hashes.add(line.strip()[5:])
    old_ids = set(json.loads((PILOT / 'old_accessions.json').read_text()))
    pilot = pd.read_csv(PILOT / 'selected_sequences.csv')
    old_hashes.update(pilot.sequence_sha256)
    old_ids.update(pilot.accession)
    reference = OUT / 'reference_with_pilot.fasta'
    if not reference.exists():
        with reference.open('wb') as handle:
            with (PILOT / 'old_sequences.fasta').open('rb') as source:
                while chunk := source.read(8 * 1024 * 1024):
                    handle.write(chunk)
            for row in pilot.itertuples():
                handle.write(f'>pilot_{row.accession}\n{row.sequence}\n'.encode())
    kept, audit, seen_ids, seen_hashes = [], [], set(), set()
    for family in FAMILIES:
        for path in sorted((OUT / 'raw').glob(f'{family}_page*.json')):
            for record in json.loads(path.read_text())['results']:
                metadata = record['metadata']
                accession = metadata['accession']
                sequence = record.get('extra_fields', {}).get('sequence', '')
                digest = sha(sequence)
                domains = []
                for entry in record['entries']:
                    if entry['accession'] != family:
                        continue
                    for location in entry['entry_protein_locations']:
                        fragments = location['fragments']
                        if len(fragments) == 1 and fragments[0].get('dc-status') == 'CONTINUOUS':
                            domains.append((int(fragments[0]['start'])-1, int(fragments[0]['end'])))
                domains = sorted(set(domains))
                reason = ''
                if not 50 <= len(sequence) <= 1022 or len(sequence) != int(metadata['length']):
                    reason = 'length_outside_50_1022_or_mismatch'
                elif not re.fullmatch('[ACDEFGHIKLMNPQRSTVWY]+', sequence):
                    reason = 'noncanonical_residues'
                elif digest in old_hashes:
                    reason = 'exact_sequence_in_old_or_pilot'
                elif accession in old_ids:
                    reason = 'accession_in_old_or_pilot'
                elif accession in seen_ids or digest in seen_hashes:
                    reason = 'duplicate_new_sequence_or_accession'
                elif not domains or any(a < 0 or b > len(sequence) or a >= b for a,b in domains):
                    reason = 'no_valid_continuous_domain'
                audit.append(dict(family=family, accession=accession, reason=reason or 'eligible_before_homology', source=path.name))
                if reason:
                    continue
                seen_ids.add(accession)
                seen_hashes.add(digest)
                mask = ['0'] * len(sequence)
                for a,b in domains:
                    mask[a:b] = ['1'] * (b-a)
                taxon = metadata.get('source_organism', {})
                kept.append(dict(family=family, accession=accession, sequence=sequence, length=len(sequence), sequence_sha256=digest, domains=json.dumps(domains), mask_target_domain=''.join(mask), taxid=str(taxon.get('taxId','unknown')), organism=taxon.get('scientificName',''), sort_key=sha('20260923_four_family'+accession)))
    pd.DataFrame(audit).to_csv(OUT / 'candidate_audit.csv', index=False)
    pd.DataFrame(kept).to_csv(OUT / 'candidates.csv', index=False)
    with (OUT / 'candidates.fasta').open('w') as handle:
        for row in kept:
            handle.write(f'>{row["accession"]}\n{row["sequence"]}\n')
    print(pd.DataFrame(audit).groupby(['family','reason']).size().to_string(), flush=True)

if __name__ == '__main__':
    main()
