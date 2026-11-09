"""Frozen CPU ESM2/SAE inference; selected IDs and cohort fixed before execution."""
from pathlib import Path
import json,time,hashlib,sys,argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer,EsmModel
ROOT=Path('/data1/zshuo23/GraBSpArk');OUT=Path(__file__).resolve().parent
IDS=json.loads((OUT/'frozen_readouts.json').read_text())['latent_ids']
def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--family',choices=['PF00004','PF00005','PF00069','PF07714'])
 parser.add_argument('--benchmark-only',action='store_true')
 args=parser.parse_args()
 suffix='_'+args.family if args.family else ''
 torch.set_num_threads(8);torch.set_num_interop_threads(1)
 model_path=ROOT/'InterPLM/esm2_model/esm2_650M_UR50D'
 tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
 model=EsmModel.from_pretrained(model_path,add_pooling_layer=False,local_files_only=True).eval()
 weights=ROOT/'InterPLM/interplm/sae/migration/InterPLM-esm2-650m/ae_normalized.pt'
 sd=torch.load(weights,map_location='cpu',weights_only=True)
 assert not sd.get('normalize_to_sqrt_d',False)
 w,b,bias=sd['encoder.weight'].float(),sd['encoder.bias'].float(),sd['bias'].float()
 selected_w=w[IDS];selected_b=b[IDS]
 def embed(seq):
  tokens=tokenizer([seq],return_tensors='pt',padding=False,truncation=False)
  assert tokens['input_ids'].shape[1]==len(seq)+2
  with torch.inference_mode():return model(**tokens).last_hidden_state[0,1:-1].float()
 # Check CPU extraction against one independently frozen old embedding before new inference.
 prefix=ROOT/'data/Protein/fig5_pfam00005_pfam00004_labels/full_alignment_sample/embeddings/ESM2_650M_layer33'
 old=pd.read_csv(prefix/'PF00005_test.csv',dtype=str,keep_default_na=False);idx=int(old.length.astype(int).idxmin());row=old.iloc[idx]
 start=old.iloc[:idx].length.astype(int).sum();ref=np.load(prefix/'PF00005_test_ESM2_650M_layer33.npy',mmap_mode='r')[start:start+len(row.sequence)].astype('float32')
 t=time.time();x=embed(row.sequence)
 diff=np.abs(x.numpy()-ref)
 with torch.inference_mode():
  projected=torch.relu((x-bias)@selected_w.T+selected_b)
  full=torch.relu((x-bias)@w.T+b)[:,IDS]
 checks={'reference_accession':row.uniprot_id,'reference_length':len(row.sequence),'reference_embedding_max_abs_error':float(diff.max()),'reference_embedding_mean_abs_error':float(diff.mean()),'selected_vs_full_sae_max_abs_error':float((full-projected).abs().max()),'reference_seconds':time.time()-t}
 print('Reference check:',checks,flush=True)
 assert diff.mean()<.01 and checks['selected_vs_full_sae_max_abs_error']<.001
 (OUT/f'inference_checks{suffix}.json').write_text(json.dumps(checks,indent=2))
 if '--benchmark-only' in sys.argv:return
 frame=pd.read_csv(OUT/'selected_sequences.csv',dtype={'mask_target_domain':str})
 assert len(frame)==800 and (frame.groupby('family').size()==200).all()
 assert hashlib.sha256((OUT/'selected_sequences.csv').read_bytes()).hexdigest()==json.loads((OUT/'selection_manifest.json').read_text())['cohort_sha256']
 if args.family:frame=frame[frame.family==args.family].reset_index(drop=True)
 cache=OUT/'activations';cache.mkdir(exist_ok=True)
 for i,row in frame.iterrows():
  p=cache/f'{row.accession}.npz'
  if p.exists():
   z=np.load(p);assert str(z['sequence_sha256'])==row.sequence_sha256 and list(z['latent_ids'])==IDS;continue
  t=time.time();x=embed(row.sequence)
  with torch.inference_mode():act=torch.relu((x-bias)@selected_w.T+selected_b).numpy()
  assert act.shape==(len(row.sequence),len(IDS))
  np.savez_compressed(p,activations=act,latent_ids=IDS,sequence_sha256=row.sequence_sha256)
  print(f'{i+1}/{len(frame)} {row.family} {row.accession} {len(row.sequence)} aa {time.time()-t:.1f}s',flush=True)
 manifest={'n_sequences':len(frame),'latent_ids':IDS,'cohort_sha256':hashlib.sha256((OUT/'selected_sequences.csv').read_bytes()).hexdigest(),'sae_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),'model_path':str(model_path),'model_weights_sha256':hashlib.sha256((model_path/'pytorch_model.bin').read_bytes()).hexdigest(),'cpu_threads':8,'model_dtype':'float32','layer':33,'new_latent_selection':False}
 (OUT/f'inference_manifest{suffix}.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
