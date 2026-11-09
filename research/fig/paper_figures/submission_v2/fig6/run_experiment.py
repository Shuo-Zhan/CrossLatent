"""Fig6 final-layer interventions on Fig5's frozen seed-20260910 holdout.

Reuses original unmasked ESM2 embeddings by exact source row, rematches controls
on the new validation set, and recomputes all interventions on the new test set.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO = Path('/data1/zshuo23/GraBSpArk')
OUT = Path(__file__).resolve().parent
SOURCE = REPO/'data/Protein/fig5_pfam00005_pfam00004_labels'
EMBED = SOURCE/'full_alignment_sample/embeddings/ESM2_650M_layer33'
SPLITS = OUT.parent/'fig5/reshuffled_holdout_seed20260910/primary_splits/atpase'
spec = importlib.util.spec_from_file_location('original_fig6', SOURCE/'run_fig6_last_layer_ablation.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
SHARED = old.SHARED


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--batch-tokens', type=int, default=4096)
    parser.add_argument('--output-dir', type=Path, default=OUT/'experiment_primary')
    return parser.parse_args()


class Source:
    def __init__(self, family):
        self.frames = {}
        self.arrays = {}
        self.offsets = {}
        for split in ['validation', 'test']:
            self.frames[split] = pd.read_csv(EMBED/f'{family}_{split}.csv', dtype=str, keep_default_na=False)
            self.arrays[split] = np.load(EMBED/f'{family}_{split}_ESM2_650M_layer33.npy', mmap_mode='r')
            self.offsets[split] = np.r_[0, self.frames[split].length.astype(int).cumsum()]
            assert self.arrays[split].shape == (self.offsets[split][-1], 1280)

    def fetch(self, frame):
        arrays = []
        for _, row in frame.iterrows():
            s, i = row.original_split, int(row.original_row)
            original = self.frames[s].iloc[i]
            assert original.sequence == row.sequence and original.uniprot_id == row.uniprot_id
            for col in ['mask_c_core_ps00017', 'mask_target_domain']:
                assert original[col] == row[col] and len(row[col]) == len(row.sequence)
            a, b = self.offsets[s][i:i+2]
            arrays.append(self.arrays[s][a:b])
        return torch.from_numpy(np.concatenate(arrays).astype(np.float32))


def read_frame(family, split):
    d = pd.read_csv(SPLITS/f'{family}_{split}.csv', dtype=str, keep_default_na=False)
    d['length'] = d.length.astype(int)
    assert (d.split == split).all()
    return d


@torch.no_grad()
def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    dest = args.output_dir
    dest.mkdir(parents=True, exist_ok=True)
    weight_path = REPO/'InterPLM/interplm/sae/migration/InterPLM-esm2-650m/ae_normalized.pt'
    sd = torch.load(weight_path, map_location='cpu', weights_only=True)
    assert not bool(sd.get('normalize_to_sqrt_d', False))
    head = old.load_esm2_logits_head(map_location='cpu').eval()
    w, b, bias, decoder = [sd[k].float() for k in ['encoder.weight','encoder.bias','bias','decoder.weight']]
    sources = {fam: Source(fam) for fam in old.FAMILIES}
    frames = {(fam,s):read_frame(fam,s) for fam in old.FAMILIES for s in ['validation','test']}
    valid = pd.concat([frames[f,'validation'] for f in old.FAMILIES])
    test = pd.concat([frames[f,'test'] for f in old.FAMILIES])
    assert not set(valid.sequence_sha256) & set(test.sequence_sha256)
    assert not set(valid.split_group) & set(test.split_group)
    assert not (valid.overlaps_original_train == 'True').any()
    assert not (test.overlaps_original_train == 'True').any()
    stats_path = dest/'validation_activation_stats.npz'
    if not stats_path.exists():
        count = torch.zeros(len(b), dtype=torch.float64)
        total = torch.zeros_like(count)
        nt = 0
        for family in old.FAMILIES:
            frame = frames[family,'validation']; start = 0
            while start < len(frame):
                end = old.batch_row_end(frame.length.to_numpy(), start, args.batch_tokens)
                x = sources[family].fetch(frame.iloc[start:end])
                a = torch.relu((x-bias) @ w.T+b)
                count += (a>0).sum(0, dtype=torch.float64)
                total += a.sum(0, dtype=torch.float64)
                nt += len(x); start=end
                if start % 50 < 15 or start == len(frame):
                    print('Validation matching', family, start, '/',len(frame),flush=True)
            print('Validation completed',family,flush=True)
        np.savez(stats_path, frequency=(count/nt).numpy(), positive_mean=(total/count.clamp(min=1)).numpy(), n_tokens=nt)
    stats = np.load(stats_path)
    freq, pos = stats['frequency'], stats['positive_mean']
    norms = decoder.norm(dim=0).numpy()
    available = np.array([i for i in range(len(b)) if i not in old.read_selected_ids()])
    matched = []
    for target in SHARED:
        candidates = available[~np.isin(available,matched)]
        distance = (np.log10(freq[candidates]+1e-7)-np.log10(freq[target]+1e-7))**2+(np.log10(pos[candidates]+1e-7)-np.log10(pos[target]+1e-7))**2+(norms[candidates]-norms[target])**2
        matched.append(int(candidates[np.argmin(distance)]))
    (dest/'matched_random_latents.json').write_text(json.dumps({'selection_split':'train-disjoint validation from the repartitioned existing holdout pool, pooled across PF00005/PF00004','split_seed':20260910,'matching':'activation frequency, positive activation mean, decoder norm','excluded':'both original top-100 plus shared and family-specific latents','target_latent_ids':SHARED,'matched_latent_ids':matched,'target_stats':[dict(latent=i,frequency=float(freq[i]),positive_mean=float(pos[i]),decoder_norm=float(norms[i])) for i in SHARED],'matched_stats':[dict(latent=i,frequency=float(freq[i]),positive_mean=float(pos[i]),decoder_norm=float(norms[i])) for i in matched]},indent=2))
    print('Matched controls:',matched,flush=True)
    conditions = {f'ablate_shared_{n}_offset_{i}':[n] for i,n in enumerate(SHARED)}
    conditions.update(ablate_shared8_joint=SHARED, ablate_matched_random8=matched)
    ids = sorted(set(SHARED+matched))
    selected_w, selected_b = w[ids], b[ids]
    mapping = {n:i for i,n in enumerate(ids)}
    checks = {'test_validation_sequence_overlap':0,'test_validation_group_overlap':0,'test_train_exact_overlap':0,'validation_train_exact_overlap':0,'families':{},'head_factorization_max_abs_error':0.0}
    for fi,family in enumerate(old.FAMILIES):
        frame=frames[family,'test']; output=dest/f'{family}_test_sequence_level_effects.csv'
        if output.exists():
            print('Using completed experiment',family,flush=True)
            continue
        labels_by_row, core_offsets, domains, aa_matched = old.annotations(frame)
        rows=[]; start=0
        while start < len(frame):
            end=old.batch_row_end(frame.length.to_numpy(),start,args.batch_tokens)
            x=sources[family].fetch(frame.iloc[start:end])
            labels=torch.from_numpy(np.concatenate(labels_by_row[start:end]))
            activation=torch.relu((x-bias)@selected_w.T+selected_b)
            dense=head.dense(x)
            def from_dense(z):
                return head.decoder(head.layer_norm(F.gelu(z)))+head.bias
            logits=from_dense(dense)
            original_logp, original_correct_logp, original_margin=old.logit_metrics(logits,labels)
            original_top1=logits.argmax(-1)==labels
            boundaries=np.r_[0,np.cumsum(frame.length.to_numpy()[start:end])]
            joint_delta=None
            def record(name, perturbed):
                lp, cp, margin=old.logit_metrics(perturbed,labels)
                kl=(original_logp.exp()*(original_logp-lp)).sum(-1)
                old.append_sequence_rows(rows,family,'test',name,frame,start,boundaries,core_offsets,domains,aa_matched,cp-original_correct_logp,margin-original_margin,kl,original_top1,perturbed.argmax(-1)==labels)
            for name, latent_ids in conditions.items():
                a=activation[:,[mapping[n] for n in latent_ids]]
                delta=-(a@decoder[:,latent_ids].T)
                if name=='ablate_shared8_joint':joint_delta=delta
                # Linear algebra factorization avoids repeating a full dense projection.
                perturbed=from_dense(dense-a@(decoder[:,latent_ids].T@head.dense.weight.T))
                if start==0:
                    error=float((perturbed-head(x+delta)).abs().max())
                    checks['head_factorization_max_abs_error']=max(checks['head_factorization_max_abs_error'],error)
                    assert error<2e-4, error
                record(name,perturbed)
            gen=torch.Generator().manual_seed(20260821+fi*10000+1000+start)
            norm=joint_delta.norm(dim=-1,keepdim=True)
            for repeat in range(5):
                direction=torch.randn(x.shape,generator=gen)
                delta=direction/direction.norm(dim=-1,keepdim=True).clamp(min=1e-12)*norm
                record(f'equal_norm_random_direction_{repeat}',head(x+delta))
            start=end
            print('Test interventions',family,start,'/',len(frame),flush=True)
        pd.DataFrame(rows).to_csv(output,index=False)
        checks['families'][family]={'test_sequences':len(frame),'core_sequences':int(sum((a>=0).any() for a in core_offsets))}
    combined=pd.concat([pd.read_csv(dest/f'{f}_test_sequence_level_effects.csv') for f in old.FAMILIES],ignore_index=True)
    assert not combined.duplicated(['family','sequence_id','condition','region']).any()
    combined.to_csv(dest/'fig6_sequence_level_effects.csv',index=False)
    summary=combined.groupby(['family','split','condition','region'],as_index=False).agg(n_sequences=('sequence_id','nunique'),mean_delta_margin=('mean_delta_margin','mean'),original_top1=('original_top1_accuracy','mean'),intervened_top1=('intervened_top1_accuracy','mean'))
    summary.to_csv(dest/'fig6_ablation_summary.csv',index=False)
    checks['input_hashes']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [weight_path,REPO/'ckpt/ESM2_LogitsHead/model_weights.pt',*SPLITS.glob('*.csv')]}
    (dest/'checks.json').write_text(json.dumps(checks,indent=2))
    print(summary[(summary.condition=='ablate_shared8_joint')&(summary.region=='c_core_all')].to_string(index=False),flush=True)

if __name__=='__main__':
    main()
