#!/usr/bin/env python3
"""Serial resumable ERNIE-RNA embedding -> CrossLatent latent-10 pipeline for RNA40."""
from __future__ import annotations
import argparse,csv,json,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[3]
EXTRACTOR=ROOT/"scripts/extract_rna40_ernierna_embedding.py"
TRAINER=REPO/"grabspark/boost_pipeline/ERNIERNA_train_neuron_boosting.py"
DEFAULT_SAE=REPO/"ckpt/ERNIE-RNA/layer_12_token/relu_12x_dim768_20260602_013912/best.pt"
DEFAULT_EXTRACT_PYTHON=Path("/data1/zshuo23/anaconda3/envs/sparse_rna/bin/python")

def now():return datetime.now(timezone.utc).isoformat()
def atomic(path,payload):
    tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(payload,indent=2)+"\n");tmp.replace(path)
def logged(command,log,env):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open("w") as h:
        h.write("COMMAND: "+" ".join(command)+"\n");h.flush()
        subprocess.run(command,cwd=REPO,env=env,stdout=h,stderr=subprocess.STDOUT,check=True)
def validate(path,label,n):
    x=json.loads(path.read_text());feature=x["features"][label];ids=[int(v) for v in feature["combination"]["neuron_ids"]];steps=feature["boosting_steps"]
    if len(ids)!=n or len(set(ids))!=n:raise ValueError(f"{path}: expected {n} unique latents, found {len(ids)}")
    if len(steps)!=n+1 or int(steps[0]["neuron_id"])!=-1:raise ValueError(f"{path}: incomplete boosting steps")
    return {"selected_latents":ids,"composite_r2":feature["boosting_final_composite_r2"],
            "threshold":feature["combination"]["best_threshold"],"macro_f1":feature["combination"]["best_macro_f1"]}
def remove_embedding(npy):
    removed=[]
    for p in (npy,npy.with_suffix(".metadata.json")):
        if p.exists():p.unlink();removed.append(str(p))
    return removed

def main():
    p=argparse.ArgumentParser();p.add_argument("--manifest",type=Path,default=ROOT/"manifests/rna40_diverse_tasks.csv")
    p.add_argument("--split-dir",type=Path,default=ROOT/"splits/rna");p.add_argument("--work-dir",type=Path,default=ROOT/"work/rna40_latent10")
    p.add_argument("--result-dir",type=Path,default=ROOT/"results/rna40_latent10");p.add_argument("--sae-checkpoint",type=Path,default=DEFAULT_SAE)
    p.add_argument("--device",default="cuda:0");p.add_argument("--max-batch-tokens",type=int,default=128000)
    p.add_argument("--extract-python",type=Path,default=DEFAULT_EXTRACT_PYTHON,
                   help="Python environment containing ERNIE-RNA/fairseq")
    p.add_argument("--train-python",type=Path,default=Path(sys.executable),
                   help="Python environment used for CrossLatent boosting")
    p.add_argument("--num-latents",type=int,default=10);p.add_argument("--tasks",nargs="*");p.add_argument("--continue-on-error",action="store_true");a=p.parse_args()
    a.work_dir.mkdir(parents=True,exist_ok=True);a.result_dir.mkdir(parents=True,exist_ok=True);state_path=a.result_dir/"pipeline_state.json"
    state=json.loads(state_path.read_text()) if state_path.exists() else {"created_at":now(),"num_latents":a.num_latents,"tasks":{}}
    if int(state.get("num_latents",a.num_latents))!=a.num_latents:raise ValueError("Existing state uses another latent count")
    with a.manifest.open() as h:tasks=list(csv.DictReader(h))
    if a.tasks:
        wanted=set(a.tasks);tasks=[r for r in tasks if r["task"] in wanted];missing=wanted-{r["task"] for r in tasks}
        if missing:raise ValueError(f"Tasks absent from manifest: {sorted(missing)}")
    env=os.environ.copy()
    for k in ("HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy","all_proxy"):env.pop(k,None)
    if not a.extract_python.is_file():raise FileNotFoundError(f"Embedding Python not found: {a.extract_python}")
    if not a.train_python.is_file():raise FileNotFoundError(f"Training Python not found: {a.train_python}")
    for i,row in enumerate(tasks,1):
        task,label=row["task"],row["label_column"];work=a.work_dir/task;result=a.result_dir/task;work.mkdir(parents=True,exist_ok=True);result.mkdir(parents=True,exist_ok=True)
        csv_out=work/f"{task}_train.csv";npy=work/f"{task}_train_ERNIERNA_layer12.npy";json_out=result/f"{task}_{label}_latent{a.num_latents}.json";record=state["tasks"].setdefault(task,{})
        print(f"[{i}/{len(tasks)}] {task} label={label}",flush=True)
        try:
            if json_out.exists():
                metrics=validate(json_out,label,a.num_latents);removed=remove_embedding(npy);record.update(status="complete",verified_at=now(),result_json=str(json_out.resolve()),metrics=metrics,removed_embedding_files=removed);atomic(state_path,state);print("SKIP verified",task,flush=True);continue
            record.update(status="extracting",started_at=record.get("started_at",now()));atomic(state_path,state);started=time.monotonic()
            logged([str(a.extract_python.resolve()),"-u",str(EXTRACTOR),"--input-csv",str((a.split_dir/task/"train.csv").resolve()),"--output-csv",str(csv_out.resolve()),"--output-npy",str(npy.resolve()),"--device",a.device],result/"extract_embedding.log",env)
            if not npy.exists() or not npy.with_suffix(".metadata.json").exists():raise FileNotFoundError(f"Incomplete embedding {npy}")
            record.update(status="training",embedding_seconds=round(time.monotonic()-started,3),embedding_bytes=npy.stat().st_size);atomic(state_path,state);started=time.monotonic()
            logged([str(a.train_python.resolve()),"-u",str(TRAINER),"--csv_path",str(csv_out.resolve()),"--embedding_npy",str(npy.resolve()),"--sae_checkpoint",str(a.sae_checkpoint.resolve()),"--sae_type","relu","--label_column",label,"--num_iterations",str(a.num_latents),"--max_batch_tokens",str(a.max_batch_tokens),"--device",a.device,"--output_json",str(json_out.resolve())],result/"train.log",env)
            metrics=validate(json_out,label,a.num_latents);removed=remove_embedding(npy);record.update(status="complete",completed_at=now(),training_seconds=round(time.monotonic()-started,3),result_json=str(json_out.resolve()),metrics=metrics,removed_embedding_files=removed);atomic(state_path,state);print("COMPLETE",task,"embedding deleted",flush=True)
        except Exception as exc:
            record.update(status="failed",failed_at=now(),error=repr(exc));atomic(state_path,state);print("FAILED",task,repr(exc),"embedding retained",flush=True)
            if not a.continue_on_error:raise
    state["finished_at"]=now();atomic(state_path,state);print(f"PIPELINE COMPLETE: {len(tasks)} tasks",flush=True)

if __name__=="__main__":main()
