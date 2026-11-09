"""Frozen module and P-loop evaluation on the same independently selected proteins."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path('/data1/zshuo23/GraBSpArk')
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'fig/paper_figures/fig5'))
from analyze_domain_shared_latent_annotations import kinase_masks
spec = importlib.util.spec_from_file_location('original_atpase_labels', ROOT/'data/Protein/fig5_pfam00005_pfam00004_labels/download_fig5_labels.py')
atpase = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = atpase
spec.loader.exec_module(atpase)
READOUTS = json.loads((OUT/'frozen_readouts.json').read_text())
IDS = READOUTS['latent_ids']
SHARED = READOUTS['ploop_shared8']
PLOOP_IDS = SHARED + [7815,3963,6683,3625]
PRIMARY = {('c_core','mask_c_core_ps00017'), ('c_context','mask_c_context_proxy'), ('walker_b','mask_walker_b_proxy'), ('gly_loop_module','gly_loop_proxy_window3'), ('vaik_module','VAIK_like_window3')}

def metrics(y: np.ndarray, scores: np.ndarray) -> dict:
    both = y.any() and not y.all()
    return dict(auroc=float(roc_auc_score(y,scores)) if both else np.nan,
                auprc=float(average_precision_score(y,scores)) if both else np.nan,
                n_positive=int(y.sum()),n_tokens=len(y),prevalence=float(y.mean()))

def labels_for(row) -> tuple:
    sequence = row.sequence
    domain = np.array(list(row.mask_target_domain))=='1'
    assert len(domain)==len(sequence)
    if row.family in ['PF00069','PF07714']:
        masks = kinase_masks(sequence,row.mask_target_domain)
        masks['HRD_DFG_union_window3'] = masks['HRD_window3']|masks['DFG_window3']
        return masks, domain, []
    masks = {key:np.zeros(len(sequence),dtype=bool) for key in ['mask_c_core_ps00017','mask_c_context_proxy','mask_walker_b_proxy']}
    cores=[]
    for a,b in json.loads(row.domains):
        core=atpase.choose_p_loop(atpase.motif_spans(atpase.P_LOOP_RE,sequence,a,b),a)
        if core:
            cores.append(core)
            masks['mask_c_core_ps00017'][core[0]:core[1]]=True
            lo,hi=atpase.centered_window(core,(a,b),21)
            masks['mask_c_context_proxy'][lo:hi]=True
        walker=atpase.choose_walker_b(sequence,(a,b),core)
        if walker:
            masks['mask_walker_b_proxy'][walker[0]:walker[1]]=True
    assert all(not (mask & ~domain).any() for mask in masks.values())
    return masks,domain,sorted(set(cores))

def load_activation(row):
    z=np.load(OUT/f'activations/{row.accession}.npz')
    assert list(z['latent_ids'])==IDS and str(z['sequence_sha256'])==row.sequence_sha256
    a=z['activations']
    assert a.shape==(len(row.sequence),len(IDS)) and np.isfinite(a).all() and (a>=0).all()
    return a

def main():
    frame=pd.read_csv(OUT/'selected_sequences.csv',dtype={'mask_target_domain':str,'taxid':str})
    manifest=json.loads((OUT/'selection_manifest.json').read_text())
    assert hashlib.sha256((OUT/'selected_sequences.csv').read_bytes()).hexdigest()==manifest['cohort_sha256']
    assert len(frame)==800 and (frame.groupby('family').size()==200).all()
    scales=pd.read_csv(OUT/'frozen_train_scales.csv')
    definitions={g:json.loads((OUT/f'{g}_group_definitions.json').read_text()) for g in ['atpase','kinase']}
    module_rows, sequence_rows, coverage_rows, ploop_rows, ploop_sequence, core_rows, domain_rows = [],[],[],[],[],[],[]
    annotated=[]
    for family,subset in frame.groupby('family'):
        group='atpase' if family in ['PF00004','PF00005'] else 'kinase'
        arrays=[];all_masks={};domains=[];position_parts=[];core_blocks=[]
        seq_data=[]
        for row in subset.itertuples():
            a=load_activation(row)
            masks,domain,cores=labels_for(row)
            arrays.append(a);domains.append(domain)
            for label,mask in masks.items():
                all_masks.setdefault(label,[]).append(mask)
                coverage_rows.append(dict(family=family,accession=row.accession,taxid=row.taxid,label=label,n_positive=int(mask.sum()),n_domain_tokens=int(domain.sum()),detected=bool(mask.any())))
            if group=='atpase':
                positions=np.full(len(a),-1)
                for lo,hi in cores:
                    assert hi-lo==8
                    positions[lo:hi]=np.arange(8)
                    block=a[lo:hi][:,[IDS.index(n) for n in SHARED]]
                    core_blocks.append(block)
                    for j,latent in enumerate(SHARED):
                        score=block[:,j]
                        strict=score[j]>0 and (score==score.max()).sum()==1 and score.argmax()==j
                        core_rows.append(dict(family=family,accession=row.accession,taxid=row.taxid,start_0based=lo,end_exclusive=hi,latent_id=latent,expected_offset=j,predicted_offset=int(score.argmax()),strict_top1=bool(strict),expected_activation=float(score[j]),max_activation=float(score.max())))
                assert np.array_equal(positions>=0,masks['mask_c_core_ps00017'])
                position_parts.append(positions)
                for latent in PLOOP_IDS:
                    j=IDS.index(latent);m=metrics(positions>=0,a[:,j])
                    record=dict(family=family,accession=row.accession,taxid=row.taxid,latent_id=latent,n_cores=len(cores),core_auroc=m['auroc'],core_auprc=m['auprc'])
                    if latent in SHARED:
                        pos=SHARED.index(latent);pm=metrics(positions==pos,a[:,j])
                        record.update(position_auroc=pm['auroc'],position_auprc=pm['auprc'])
                    ploop_sequence.append(record)
            annotation=dict(family=family,accession=row.accession,core_spans=json.dumps(cores),n_cores=len(cores))
            annotation.update({k:''.join('1' if v else '0' for v in mask) for k,mask in masks.items()})
            annotated.append(annotation)
            seq_data.append((row,a,masks,domain))
        pooled=np.concatenate(arrays);domain=np.concatenate(domains)
        masks={k:np.concatenate(v) for k,v in all_masks.items()}
        scale=scales[scales.family==family].set_index('latent_id').train_q99_scale
        for name,definition in definitions[group].items():
            cols=[IDS.index(n) for n in definition['latents']]
            train_scales=scale.loc[definition['latents']].to_numpy(dtype=np.float32)
            assert np.isfinite(train_scales).all() and (train_scales>0).all()
            norm=np.clip(pooled[:,cols]/train_scales,0,1)
            scores={'combination_mean':norm.mean(1)}
            scores.update({f'single_{n}':norm[:,j] for j,n in enumerate(definition['latents'])})
            labels=definition.get('labels',[definition.get('label')])
            for label in labels:
                positive=masks[label][domain]
                for method,scores_ in scores.items():
                    selected=scores_[domain]
                    record=dict(task=group+'_domains',family=family,group=name,label=label,method=method,n_latents=len(cols) if method=='combination_mean' else 1,is_fig5b=(name,label) in PRIMARY,**metrics(positive,selected))
                    record['positive_mean']=float(selected[positive].mean()) if positive.any() else np.nan
                    record['background_mean']=float(selected[~positive].mean()) if (~positive).any() else np.nan
                    module_rows.append(record)
                for row,a,row_masks,row_domain in seq_data:
                    mean=np.clip(a[:,cols]/train_scales,0,1).mean(1)[row_domain]
                    sequence_rows.append(dict(family=family,accession=row.accession,taxid=row.taxid,group=name,label=label,**metrics(row_masks[label][row_domain],mean)))
        # Full-domain discovery features, preserved individually; no new top-feature selection.
        for latent in READOUTS['domain_shared_latents'][group]:
            domain_rows.append(dict(family=family,latent_id=latent,**metrics(domain,pooled[:,IDS.index(latent)])))
        if group=='atpase':
            positions=np.concatenate(position_parts)
            blocks=np.stack(core_blocks) if core_blocks else np.zeros((0,8,8))
            matrix=blocks.mean(0).T if len(blocks) else np.full((8,8),np.nan)
            pd.DataFrame(matrix,index=SHARED,columns=range(8)).to_csv(OUT/f'{family}_mean_position_response.csv')
            pd.DataFrame(matrix/np.maximum(matrix.max(1,keepdims=True),1e-12),index=SHARED,columns=range(8)).to_csv(OUT/f'{family}_row_normalized_position_response.csv')
            for latent in PLOOP_IDS:
                j=IDS.index(latent);m=metrics(positions>=0,pooled[:,j])
                record=dict(family=family,latent_id=latent,n_sequences=len(subset),n_cores=len(blocks),core_auroc=m['auroc'],core_auprc=m['auprc'],n_tokens=len(pooled),core_prevalence=m['prevalence'])
                if latent in SHARED:
                    pos=SHARED.index(latent);pm=metrics(positions==pos,pooled[:,j]);b=blocks[:,:,pos]
                    strict=(b.argmax(1)==pos)&((b==b.max(1,keepdims=True)).sum(1)==1)&(b[:,pos]>0)
                    record.update(expected_offset=pos,position_auroc=pm['auroc'],position_auprc=pm['auprc'],position_prevalence=pm['prevalence'],within_core_top1=float((b.argmax(1)==pos).mean()),strict_within_core_top1=float(strict.mean()),mean_response_peak_offset=int(matrix[pos].argmax()))
                ploop_rows.append(record)
        print(f'Evaluated {family}: {len(subset)} sequences, {len(pooled)} tokens, {domain.sum()} domain tokens',flush=True)
    outputs={'module_metrics':module_rows,'module_per_sequence':sequence_rows,'motif_coverage_per_sequence':coverage_rows,'ploop_metrics':ploop_rows,'ploop_per_sequence':ploop_sequence,'ploop_per_core':core_rows,'domain_shared_individual_metrics':domain_rows,'sequence_annotations':annotated}
    for name,records in outputs.items():
        pd.DataFrame(records).to_csv(OUT/f'{name}.csv',index=False)
    modules=pd.DataFrame(module_rows)
    main_results=modules[(modules.method=='combination_mean') & modules.is_fig5b]
    main_results.to_csv(OUT/'fig5b_external_metrics.csv',index=False)
    coverage=pd.DataFrame(coverage_rows).groupby(['family','label']).agg(n_sequences=('accession','size'),n_detected=('detected','sum'),n_positive=('n_positive','sum'),n_domain_tokens=('n_domain_tokens','sum')).reset_index()
    coverage['detection_fraction']=coverage.n_detected/coverage.n_sequences
    coverage.to_csv(OUT/'motif_coverage.csv',index=False)
    core_df=pd.DataFrame(core_rows)
    core_df[~core_df.strict_top1].to_csv(OUT/'ploop_within_core_failures.csv',index=False)
    # Descriptive taxon-balanced means/intervals of sequence-level metrics; not pooled-token CIs.
    intervals=[];rng=np.random.default_rng(20260923)
    for (family,name,label),g in pd.DataFrame(sequence_rows).groupby(['family','group','label']):
        for metric in ['auroc','auprc']:
            values=g.groupby('taxid')[metric].mean().dropna().to_numpy()
            if len(values):
                bootstrap=rng.choice(values,(2000,len(values)),replace=True).mean(1)
                intervals.append(dict(family=family,group=name,label=label,metric=metric,unit='taxon mean of defined sequence metrics',n_taxa=len(values),n_sequences_defined=int(g[metric].notna().sum()),mean=float(values.mean()),low=float(np.quantile(bootstrap,.025)),high=float(np.quantile(bootstrap,.975))))
    pd.DataFrame(intervals).to_csv(OUT/'taxon_equal_module_intervals.csv',index=False)
    old_dir=ROOT/'fig/paper_figures/submission_v2/fig5/reshuffled_holdout_seed20260910'
    old=pd.read_csv(old_dir/'module_metrics.csv')
    old=old[(old.partition=='new_excluding_train_duplicates')&(old.split=='test')&(old.method=='combination_mean')]
    modules[modules.method=='combination_mean'].merge(old,on=['family','group','label','method'],suffixes=('_external','_original'),validate='one_to_one').to_csv(OUT/'module_comparison_with_original_test.csv',index=False)
    old=pd.read_csv(old_dir/'same_core_metrics.csv')
    old=old[(old.partition=='new_excluding_train_duplicates')&(old.split=='test')]
    pd.DataFrame(ploop_rows).merge(old,on=['family','latent_id'],suffixes=('_external','_original'),validate='one_to_one').to_csv(OUT/'ploop_comparison_with_original_test.csv',index=False)
    checks=dict(n_sequences=800,n_by_family=frame.family.value_counts().to_dict(),ploop_accessions_identical_to_atpase_cohort=set(pd.DataFrame(ploop_sequence).accession)==set(frame[frame.family.isin(['PF00004','PF00005'])].accession),all_frozen_groups_evaluated=True,no_samples_removed_after_inference=True,conditional_within_core_failed_pairs=int((~core_df.strict_top1).sum()),cohort_sha256=manifest['cohort_sha256'])
    assert checks['ploop_accessions_identical_to_atpase_cohort']
    (OUT/'evaluation_checks.json').write_text(json.dumps(checks,indent=2))
    print(main_results[['family','group','label','auroc','auprc','prevalence']].to_string(index=False),flush=True)

if __name__=='__main__':
    main()
