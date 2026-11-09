from pathlib import Path
import hashlib
import json
import shutil
import sys
ROOT=Path('/data1/zshuo23/GraBSpArk')
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'fig/paper_figures/fig5'))
from analyze_domain_shared_latent_annotations import shared_latents
ANN=ROOT/'fig/paper_figures/fig5/domain_shared_latent_annotations'
SHARED=[7053,7911,6996,8702,2425,6964,9767,4424]

def main():
    files=[ANN/'atpase_group_definitions.json',ANN/'kinase_group_definitions.json',ROOT/'fig/paper_figures/submission_v2/fig5/reshuffled_holdout_seed20260910/frozen_train_scales.csv']
    for source in files:
        shutil.copyfile(source,OUT/source.name)
    groups={group:json.loads((OUT/f'{group}_group_definitions.json').read_text()) for group in ['atpase','kinase']}
    discovery={group:shared_latents(group) for group in groups}
    ids=SHARED+[7815,3963,6683,3625]
    ids+=sorted(set(sum(discovery.values(),[]))-set(ids))
    for group in groups.values():
        for definition in group.values():
            assert set(definition['latents']).issubset(ids)
    manifest={'latent_ids':ids,'ploop_shared8':SHARED,'domain_shared_latents':discovery,'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'source_annotation_script_sha256':hashlib.sha256((ROOT/'fig/paper_figures/fig5/analyze_domain_shared_latent_annotations.py').read_bytes()).hexdigest(),'source_atpase_label_script_sha256':hashlib.sha256((ROOT/'data/Protein/fig5_pfam00005_pfam00004_labels/download_fig5_labels.py').read_bytes()).hexdigest()}
    (OUT/'frozen_readouts.json').write_text(json.dumps(manifest,indent=2))
    print({k:len(v) for k,v in discovery.items()}, 'total IDs:',len(ids))
if __name__=='__main__':
    main()
