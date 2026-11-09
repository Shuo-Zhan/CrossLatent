"""Frozen Fig5 holdout repartition, seed 20260910.

Keeps original inputs/results unchanged. Splits homology components (and exact
sequence unions) jointly across families within each experiment. ATPase domain
and same-core experiments reuse one assignment. No test-based seed search.
"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

REPO=Path('/data1/zshuo23/GraBSpArk')
FIG=REPO/'fig/paper_figures/fig5'
ANN=FIG/'domain_shared_latent_annotations'
OUT=Path(__file__).resolve().parent
SEED=20260910
SHARED=[7053,7911,6996,8702,2425,6964,9767,4424]
CORE_IDS=SHARED+[7815,6683,3963,3625]
COMP='homology_component_30pct'
sys.path.insert(0,str(REPO/'InterPLM'))
sys.path.insert(0,str(FIG))
from analyze_domain_shared_latent_annotations import kinase_masks


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flatten(frame: pd.DataFrame, label: str) -> np.ndarray:
    return np.fromiter((c=='1' for s in frame[label] for c in s),dtype=bool)


def read_frame(directory: Path, family: str, split: str) -> pd.DataFrame:
    path=directory/f'{family}_{split}.csv'
    d=pd.read_csv(path,dtype=str,keep_default_na=False)
    d['length']=d.length.astype(int)
    assert (d.sequence.str.len()==d.length).all()
    for c in d:
        if c.startswith('mask_'):assert (d[c].str.len()==d.length).all(),(path,c)
    assert (d.split==split).all()
    d['original_split']=split;d['original_row']=np.arange(len(d))
    d['sequence_sha256']=d.sequence.map(lambda s:hashlib.sha256(s.encode()).hexdigest())
    return d


def split_pool(directory: Path, families: list[str], name: str) -> dict[str,pd.DataFrame]:
    trains=[read_frame(directory,f,'train') for f in families]
    frames={f:pd.concat([read_frame(directory,f,s) for s in ['validation','test']],ignore_index=True) for f in families}
    all_rows=pd.concat(list(frames.values()),ignore_index=True)
    train=pd.concat(trains,ignore_index=True)
    assert not set(train[COMP])&set(all_rows[COMP]),'Original train/holdout component overlap'
    overlap=set(train.sequence_sha256)&set(all_rows.sequence_sha256)
    all_rows[all_rows.sequence_sha256.isin(overlap)].to_csv(OUT/f'{name}_original_train_overlap.csv',index=False)
    for d in frames.values():d['overlaps_original_train']=d.sequence_sha256.isin(overlap)
    # Link component IDs if identical sequences occur under different annotations.
    parent={v:v for v in all_rows[COMP].unique()}
    def find(v):
        while parent[v]!=v:
            parent[v]=parent[parent[v]];v=parent[v]
        return v
    for _,g in all_rows.groupby('sequence_sha256'):
        comps=list(g[COMP].unique())
        for c in comps[1:]:parent[find(c)]=find(comps[0])
    all_rows['split_group']=all_rows[COMP].map(find)
    counts=pd.crosstab(all_rows.split_group,all_rows.target_family).reindex(columns=families,fill_value=0)
    rng=np.random.default_rng(SEED)
    order=rng.permutation(sorted(counts.index))
    totals=counts.sum().to_numpy(float);v=np.zeros(len(families));t=np.zeros(len(families));assignment={}
    for group in order:
        n=counts.loc[group].to_numpy(float)
        # Sequence-count balance only. Labels and metrics are never consulted.
        cv=np.sum(((v+n-t)/totals)**2);ct=np.sum(((v-t-n)/totals)**2)
        choose_v=(cv<ct) if abs(cv-ct)>1e-15 else bool(rng.integers(2))
        assignment[group]='validation' if choose_v else 'test'
        if choose_v:v+=n
        else:t+=n
    manifest=[]
    for family,d in frames.items():
        d['split_group']=d[COMP].map(find);d['split']=d.split_group.map(assignment)
        assert set(d.split)=={'validation','test'}
        for split in ['validation','test']:
            part=d[d.split==split].sample(frac=1,random_state=SEED).reset_index(drop=True)
            dest=OUT/'splits'/name/f'{family}_{split}.csv';dest.parent.mkdir(parents=True,exist_ok=True)
            part.to_csv(dest,index=False)
            manifest.append({'pool':name,'family':family,'split':split,'n_sequences':len(part),'n_tokens':int(part.length.sum()),'n_components':part[COMP].nunique(),'n_split_groups':part.split_group.nunique(),'from_old_validation':int((part.original_split=='validation').sum()),'from_old_test':int((part.original_split=='test').sum())})
        assert not set(d[d.split=='validation'][COMP])&set(d[d.split=='test'][COMP])
        assert not set(d[d.split=='validation'].sequence_sha256)&set(d[d.split=='test'].sequence_sha256)
    rows=pd.concat(list(frames.values()),ignore_index=True)
    assert not set(rows[rows.split=='validation'].split_group)&set(rows[rows.split=='test'].split_group)
    rows[['target_family','uniprot_id','sequence_sha256',COMP,'split_group','original_split','original_row','split','overlaps_original_train']].to_csv(OUT/f'{name}_assignments.csv',index=False)
    pd.DataFrame(manifest).to_csv(OUT/f'{name}_split_summary.csv',index=False)
    print(pd.DataFrame(manifest).to_string(index=False),flush=True)
    return frames


class Projector:
    def __init__(self):
        torch.set_num_threads(4)
        self.path=REPO/'InterPLM/interplm/sae/migration/InterPLM-esm2-650m/ae_normalized.pt'
        self.sd=torch.load(self.path,map_location='cpu',weights_only=True)
        assert not bool(self.sd.get('normalize_to_sqrt_d',False))
    def encode(self, directory: Path, family: str, split: str, ids: list[int]) -> np.ndarray:
        path=directory/f'{family}_{split}_ESM2_650M_layer33.npy'
        metadata=json.loads(path.with_suffix('.metadata.json').read_text())
        assert metadata['status']=='complete'
        frame=read_frame(directory,family,split)
        if 'source_csv_sha256' in metadata:assert fingerprint(directory/f'{family}_{split}.csv')==metadata['source_csv_sha256']
        src=np.load(path,mmap_mode='r');assert src.shape==(frame.length.sum(),1280)
        cache=OUT/'activation_cache'/f'{family}_{split}_selected.npy'
        meta=cache.with_suffix('.json');cache.parent.mkdir(exist_ok=True)
        signature={'ids':ids,'csv_sha256':fingerprint(directory/f'{family}_{split}.csv'),'shape':[len(src),len(ids)]}
        if cache.exists() and meta.exists() and json.loads(meta.read_text())==signature:
            return np.load(cache,mmap_mode='r')
        print('Projecting',family,split,len(src),'tokens',len(ids),'latents',flush=True)
        out=np.lib.format.open_memmap(cache,mode='w+',dtype='float32',shape=(len(src),len(ids)))
        w=self.sd['encoder.weight'][ids].float();b=self.sd['encoder.bias'][ids].float();bias=self.sd['bias'].float()
        with torch.no_grad():
            for start in range(0,len(src),16384):
                x=torch.from_numpy(np.array(src[start:start+16384],dtype=np.float32,copy=True))
                out[start:start+len(x)]=torch.relu((x-bias)@w.T+b).numpy()
        out.flush();meta.write_text(json.dumps(signature,indent=2)+'\n')
        return out


def metrics(y: np.ndarray, s: np.ndarray) -> dict:
    if len(y)==0 or y.min()==y.max():return {'auprc':float('nan'),'auroc':float('nan'),'n_positive':int(y.sum()),'n_tokens':len(y),'prevalence':float(y.mean()) if len(y) else float('nan')}
    pos=float(s[y].mean());neg=float(s[~y].mean())
    return {'auprc':float(average_precision_score(y,s)),'auroc':float(roc_auc_score(y,s)),'n_positive':int(y.sum()),'n_tokens':len(y),'prevalence':float(y.mean()),'positive_mean':pos,'background_mean':neg,'within_domain_enrichment':pos/max(neg,1e-12)}


def module_metrics(frame, acts, scales, ids, groups, family, task):
    domain=flatten(frame,'mask_target_domain')
    if task=='atpase_domains':
        labels={g['label']:flatten(frame,g['label']) for g in groups.values()}
    else:
        collected={}
        for row in frame.itertuples():
            mm=kinase_masks(row.sequence,row.mask_target_domain);mm['HRD_DFG_union_window3']=mm['HRD_window3']|mm['DFG_window3']
            for k,m in mm.items():collected.setdefault(k,[]).append(m)
        labels={k:np.concatenate(v) for k,v in collected.items()}
    records=[]
    token_splits=np.repeat(frame.split.to_numpy(),frame.length.to_numpy())
    original_splits=np.repeat(frame.original_split.to_numpy(),frame.length.to_numpy())
    norm=np.clip(acts/np.asarray(scales),0,1)
    overlap=np.repeat(frame.overlaps_original_train.to_numpy(),frame.length.to_numpy())
    for label_version,split_ids in [('new',token_splits),('original',original_splits),('new_excluding_train_duplicates',token_splits)]:
      for split in ['validation','test']:
        keep=domain&(split_ids==split)
        if label_version=='new_excluding_train_duplicates':keep=keep&~overlap
        for group,spec in groups.items():
            cols=[ids.index(n) for n in spec['latents']]
            scores={**{f'single_{n}':norm[:,c] for n,c in zip(spec['latents'],cols)},'combination_mean':norm[:,cols].mean(1),'combination_max':norm[:,cols].max(1),'combination_coverage_0.1':(norm[:,cols]>=.1).mean(1)}
            labs=[spec['label']] if 'label' in spec else spec['labels']
            for lab in labs:
                for method,score in scores.items():
                    records.append({'task':task,'family':family,'partition':label_version,'split':split,'group':group,'label':lab,'method':method,'n_latents':1 if method.startswith('single_') else len(cols),**metrics(labels[lab][keep],score[keep])})
    return records


def evaluate_core(frame,acts,ids,family):
    offsets=np.r_[0,np.cumsum(frame.length)];n=len(acts)
    core=np.zeros(n,dtype=bool);position=np.full(n,-1,dtype=np.int8);blocks=[];block_split=[]
    for i,mask in enumerate(frame.mask_c_core_ps00017):
        diff=np.diff(np.r_[0,np.array(list(mask),dtype=np.int8),0])
        for start,end in zip(np.flatnonzero(diff==1),np.flatnonzero(diff==-1)):
            if end-start==8:
                ix=offsets[i]+np.arange(start,end);core[ix]=True;position[ix]=np.arange(8);blocks.append(acts[ix][:,[ids.index(k) for k in CORE_IDS]]);block_split.append(frame.iloc[i]['split'])
    blocks=np.stack(blocks);block_split=np.array(block_split)
    token_splits=np.repeat(frame.split.to_numpy(),frame.length.to_numpy());records=[]
    for split in ['validation','test']:
        keep=token_splits==split;bb=blocks[block_split==split];matrix=bb.astype(float).mean(0).T
        pd.DataFrame(matrix,index=CORE_IDS,columns=range(8)).to_csv(OUT/f'{family}_{split}_core_position_matrix.csv')
        for j,latent in enumerate(CORE_IDS):
            score=acts[keep,ids.index(latent)];cm=metrics(core[keep],score)
            r={'family':family,'split':split,'latent_id':latent,'n_sequences':int((frame.split==split).sum()),'n_core_occurrences':len(bb),'core_auprc':cm['auprc'],'core_auroc':cm['auroc'],'core_prevalence':cm['prevalence']}
            if latent in SHARED:
                offset=SHARED.index(latent);pm=metrics(position[keep]==offset,score)
                r.update({'expected_offset':offset,'position_auprc':pm['auprc'],'position_auroc':pm['auroc'],'within_core_top1_accuracy':float((bb[:,:,j].argmax(1)==offset).mean())})
            records.append(r)
    # Additional exact-duplicate sensitivity uses the same fixed masks/scores.
    overlap=np.repeat(frame.overlaps_original_train.to_numpy(),frame.length.to_numpy())
    for r in records:r['partition']='new'
    for split in ['validation','test']:
        keep=(token_splits==split)&~overlap
        for latent in CORE_IDS:
            score=acts[keep,ids.index(latent)];cm=metrics(core[keep],score)
            r={'family':family,'split':split,'partition':'new_excluding_train_duplicates','latent_id':latent,'core_auprc':cm['auprc'],'core_auroc':cm['auroc'],'core_prevalence':cm['prevalence']}
            if latent in SHARED:
                offset=SHARED.index(latent);pm=metrics(position[keep]==offset,score);r.update({'expected_offset':offset,'position_auprc':pm['auprc'],'position_auroc':pm['auroc']})
            records.append(r)
    return records


def comparisons(table):
    rows=[]
    for key,d in table.groupby(['task','family','partition','split','group','label']):
        single=d[d.method.str.startswith('single_')]
        for metric in ['auroc','auprc','within_domain_enrichment']:
            best=single.loc[single[metric].idxmax()]
            for method in ['combination_mean','combination_max','combination_coverage_0.1']:
                combo=d[d.method==method].iloc[0]
                rows.append({**dict(zip(['task','family','partition','split','group','label'],key)),'metric':metric,'best_single':best.method,'best_single_value':best[metric],'combination_method':method,'combination_value':combo[metric],'combination_minus_best_single':combo[metric]-best[metric]})
    return pd.DataFrame(rows)


def main():
    configs=[('atpase','fig5_pfam00005_pfam00004_labels',['PF00005','PF00004'],'atpase_group_definitions.json'),('kinase','fig5_kinase_pfam00069_pfam07714',['PF00069','PF07714'],'kinase_group_definitions.json')]
    inputs={};pools={}
    for name,folder,families,gfile in configs:
        directory=REPO/f'data/Protein/{folder}/full_alignment_sample/embeddings/ESM2_650M_layer33'
        pools[name]=split_pool(directory,families,name)
        for f in families:
            for split in ['train','validation','test']:
                path=directory/f'{f}_{split}.csv';inputs[str(path)]=fingerprint(path)
    projector=Projector();all_modules=[];all_core=[];scale_records=[]
    for name,folder,families,gfile in configs:
        directory=REPO/f'data/Protein/{folder}/full_alignment_sample/embeddings/ESM2_650M_layer33'
        groups=json.loads((ANN/gfile).read_text());group_ids=sorted({k for g in groups.values() for k in g['latents']})
        ids=sorted(set(group_ids+CORE_IDS)) if name=='atpase' else group_ids
        for family in families:
            if name=='atpase':
                train=read_frame(directory,family,'train');ta=projector.encode(directory,family,'train',ids)
                domain=flatten(train,'mask_target_domain');scales=np.maximum(np.quantile(ta[domain],.99,axis=0),1e-12)
                del ta
            else:
                saved=pd.read_csv(ANN/'kinase_train_q99_scales.csv');scales=saved[saved.family==family].set_index('latent_id').loc[ids].train_q99_scale.to_numpy()
            for n,v in zip(ids,scales):scale_records.append({'family':family,'latent_id':n,'train_q99_scale':v})
            acts=np.concatenate([projector.encode(directory,family,s,ids) for s in ['validation','test']])
            frame=pools[name][family]
            all_modules.extend(module_metrics(frame,acts,scales,ids,groups,family,f'{name}_domains'))
            if name=='atpase':all_core.extend(evaluate_core(frame,acts,ids,family))
            # Save completed families so progress survives interruption.
            pd.DataFrame(all_modules).to_csv(OUT/'module_metrics.csv',index=False)
            pd.DataFrame(all_core).to_csv(OUT/'same_core_metrics.csv',index=False)
            pd.DataFrame(scale_records).to_csv(OUT/'frozen_train_scales.csv',index=False)
            print('Completed metrics',family,flush=True)
    df=pd.DataFrame(all_modules);comparison=comparisons(df);comparison.to_csv(OUT/'module_comparison.csv',index=False)
    # Independent group-member baseline chosen on the new validation, frozen on new test.
    rows=[]
    for key,d in df[df.partition=='new'].groupby(['task','family','group','label']):
        val=d[(d.split=='validation')&d.method.str.startswith('single_')];test=d[d.split=='test']
        for metric in ['auroc','auprc']:
            method=val.loc[val[metric].idxmax(),'method'];sr=test[test.method==method].iloc[0];cr=test[test.method=='combination_mean'].iloc[0]
            rows.append({**dict(zip(['task','family','group','label'],key)),'metric':metric,'validation_selected_single':method,'single_test_value':sr[metric],'combination_test_value':cr[metric],'difference':cr[metric]-sr[metric]})
    pd.DataFrame(rows).to_csv(OUT/'validation_selected_single_test_comparison.csv',index=False)
    # Report agreement with original saved metrics, without overwriting any originals.
    audits=[]
    for task,file in [('atpase_domains','atpase_group_validation_metrics.csv'),('kinase_domains','kinase_group_heldout_metrics.csv')]:
        old=pd.read_csv(ANN/file)
        if 'split' not in old:old['split']='validation'
        new=df[(df.task==task)&(df.partition=='original')]
        m=new.merge(old,on=['family','split','group','label','method'],suffixes=('_recomputed','_saved'))
        for metric in ['auroc','auprc']:
            errors=(m[metric+'_recomputed']-m[metric+'_saved']).abs()
            audits.append({'task':task,'metric':metric,'n_rows':len(m),'max_absolute_difference':float(errors.max()),'mean_absolute_difference':float(errors.mean())})
    pd.DataFrame(audits).to_csv(OUT/'original_metric_reproduction.csv',index=False)
    assert all(fingerprint(Path(k))==v for k,v in inputs.items())
    manifest={'seed':SEED,'target_validation_fraction':.5,'unit':'homology components unioned by exact sequence','balance':'family sequence counts; no label or performance search','train_unchanged':True,'original_csv_sha256':inputs,'atpase_domain_and_core_share_assignment':True,'new_test_is_repartitioned_old_holdout_not_new_external_data':True,'original_metric_reproduction':audits}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('COMPLETE',OUT,flush=True);print(pd.DataFrame(audits).to_string(index=False),flush=True)

if __name__=='__main__':main()
