"""Cross-check reused ATPase masks against independently saved original labels."""
from types import SimpleNamespace
import json
import re
import pandas as pd
from evaluate_external import ROOT, OUT, labels_for

def main():
    checks=[]
    base=ROOT/'data/Protein/fig5_pfam00005_pfam00004_labels/full_alignment_sample/embeddings/ESM2_650M_layer33'
    for family in ['PF00004','PF00005']:
        frame=pd.read_csv(base/f'{family}_test.csv',dtype=str,keep_default_na=False)
        for r in frame.head(100).itertuples():
            # Original target domains in these rows do not overlap; infer spans from saved domain mask.
            domains=[(m.start(),m.end()) for m in re.finditer('1+',r.mask_target_domain)]
            row=SimpleNamespace(sequence=r.sequence,family=family,mask_target_domain=r.mask_target_domain,domains=json.dumps(domains))
            masks,_,_=labels_for(row)
            for name,mask in masks.items():
                actual=''.join('1' if x else '0' for x in mask)
                assert actual==getattr(r,name),(family,r.uniprot_id,name)
        checks.append({'family':family,'n_original_rows_checked':min(100,len(frame)),'all_three_original_masks_exact':True})
    (OUT/'label_rule_checks.json').write_text(json.dumps(checks,indent=2))
    print(checks)
if __name__=='__main__':
    main()
