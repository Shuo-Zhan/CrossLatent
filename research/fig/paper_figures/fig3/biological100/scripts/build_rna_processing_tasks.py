#!/usr/bin/env python3
"""Build splice-site and polyadenylation tasks as gene-disjoint genomic RNA windows."""
from __future__ import annotations
import csv, gzip, hashlib, json, random, re
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; RAW=ROOT/"raw/rna"; WINDOW=401
TASKS=("splice_donor","splice_acceptor","polya_cleavage_site","intronic_polya_site","polya_signal","m6a_site")
SIZES={"train":3000,"valid":500,"test":500}
RC=str.maketrans("ACGTUN","TGCAAN")

def attrs(x): return {k:v for k,v in re.findall(r'(\w+) "([^"]+)"',x)}

def annotation(gtf):
    genes={}; transcripts={}; symbols={}
    with gzip.open(gtf,"rt") as h:
        for line in h:
            if line.startswith("#"):continue
            f=line.rstrip().split("\t")
            if len(f)!=9:continue
            a=attrs(f[8])
            if f[2]=="gene":
                genes[a["gene_id"]]=(f[0],int(f[3])-1,int(f[4]),f[6])
                if a.get("gene_name"):symbols[a["gene_name"]]=a["gene_id"]
            elif f[2]=="transcript" and 'tag "basic"' in f[8]:
                transcripts[a["transcript_id"]]={"gene":a["gene_id"],"chrom":f[0],"strand":f[6],"exons":[]}
    with gzip.open(gtf,"rt") as h:
        for line in h:
            if line.startswith("#"):continue
            f=line.rstrip().split("\t")
            if len(f)==9 and f[2]=="exon":
                tid=attrs(f[8]).get("transcript_id")
                if tid in transcripts:transcripts[tid]["exons"].append((int(f[3])-1,int(f[4])))
    return genes,transcripts,symbols

def splice_sites(transcripts):
    out=[];seen=set()
    for tid,r in transcripts.items():
        ex=sorted(r["exons"],reverse=r["strand"]=="-")
        for upstream,downstream in zip(ex,ex[1:]):
            if r["strand"]=="+": donor=(upstream[1],upstream[1]+2);acceptor=(downstream[0]-2,downstream[0])
            else: donor=(upstream[0]-2,upstream[0]);acceptor=(downstream[1],downstream[1]+2)
            for task,span in (("splice_donor",donor),("splice_acceptor",acceptor)):
                key=(task,r["gene"],r["chrom"],span,r["strand"])
                if span[0]>=0 and key not in seen:
                    seen.add(key);out.append({"task":task,"gene":r["gene"],"chrom":r["chrom"],"start":span[0],"end":span[1],"strand":r["strand"],"source_id":tid})
    return out

def gene_index(genes,bin_size=100000):
    bins=defaultdict(set)
    for gene,(chrom,start,end,strand) in genes.items():
        for b in range(max(0,start-2000)//bin_size,(end+1999)//bin_size+1):bins[(chrom,b)].add(gene)
    return bins

def polya_sites(path,genes,bins):
    out=[];op=gzip.open if path.suffix==".gz" else open
    with op(path,"rt") as h:
        for line in h:
            f=line.rstrip().split("\t")
            if len(f)<11:continue
            chrom="chr"+f[0] if not f[0].startswith("chr") else f[0];strand=f[5];ann=f[9]
            try:pos=int(f[3].split(":")[1])-1
            except Exception:continue
            candidates=[]
            for gene in bins.get((chrom,pos//100000),()):
                gc,gs,ge,gstrand=genes[gene]
                if gstrand!=strand:continue
                distance=0 if gs<=pos<ge else min(abs(pos-gs),abs(pos-ge))
                if distance<=2000:candidates.append((distance,gene))
            if not candidates:continue
            gene=min(candidates)[1];common={"gene":gene,"chrom":chrom,"strand":strand,"source_id":f[3]}
            if ann in {"TE","EX","IN","DS"}:
                out.append({**common,"task":"polya_cleavage_site","start":pos,"end":pos+1})
            if ann=="IN":out.append({**common,"task":"intronic_polya_site","start":pos,"end":pos+1})
            if f[10]!="NA":
                # Use the closest annotated PAS motif and its official absolute coordinate.
                candidates=[]
                for item in f[10].split(";"):
                    q=item.split("@")
                    if len(q)==3:
                        try:candidates.append((abs(int(q[1])),q[0],int(q[2])-1))
                        except ValueError:pass
                if candidates:
                    _,motif,mpos=min(candidates)
                    out.append({**common,"task":"polya_signal","start":mpos,"end":mpos+len(motif),"motif":motif})
    return out

def m6a_sites(path,symbols,genes):
    out=[]
    with path.open() as h:
        for r in csv.DictReader(h,delimiter="\t"):
            gene=symbols.get(r.get("GeneName",""))
            if not gene:continue
            try:pos=int(r["Position"])-1
            except (ValueError,KeyError):continue
            chrom=r["Seqname"];strand=r.get("Strand","+")
            if genes[gene][0]!=chrom or strand not in "+-":continue
            out.append({"task":"m6a_site","gene":gene,"chrom":chrom,"start":pos,"end":pos+1,
                        "strand":strand,"source_id":r["ID"]})
    return out

def preassign(rows,seed):
    by=defaultdict(list)
    for r in rows:by[r["gene"]].append(r)
    rng=random.Random(seed);groups=list(by);rng.shuffle(groups);groups.sort(key=lambda g:len(by[g]))
    assigned={s:[] for s in SIZES};cursor=0
    for part in ("test","valid"):
        n=0
        while n<1800 and cursor<len(groups):
            g=groups[cursor];cursor+=1;assigned[part].append(g);n+=len(by[g])
    assigned["train"]=groups[cursor:]
    selected=[]
    for part,target in SIZES.items():
        pool=[r for g in assigned[part] for r in by[g]];rng.shuffle(pool)
        for r in pool[:target*4]:selected.append({**r,"split":part})
    return selected

def genome_records(path):
    name=None;parts=[]
    with gzip.open(path,"rt") as h:
        for line in h:
            if line.startswith(">"):
                if name:yield name,"".join(parts).upper()
                name=line[1:].split()[0];parts=[]
            else:parts.append(line.strip())
        if name:yield name,"".join(parts).upper()

def extract(sites,genome):
    bychrom=defaultdict(list)
    for s in sites:bychrom[s["chrom"]].append(s)
    rows=defaultdict(lambda:defaultdict(list))
    for chrom,sequence in genome_records(genome):
        for s in bychrom.get(chrom,()):
            center=(s["start"]+s["end"])//2;ws=center-WINDOW//2;we=ws+WINDOW
            if ws<0 or we>len(sequence):continue
            seq=sequence[ws:we];lo=s["start"]-ws;hi=s["end"]-ws
            if s["strand"]=="-":seq=seq.translate(RC)[::-1];lo,hi=WINDOW-hi,WINDOW-lo
            seq=seq.replace("T","U")
            if seq.count("N")>WINDOW*.05:continue
            mask=["0"]*WINDOW;mask[lo:hi]="1"*(hi-lo);task=s["task"]
            if task=="splice_donor" and seq[lo:hi]!="GU":continue
            if task=="splice_acceptor" and seq[lo:hi]!="AG":continue
            if task=="polya_signal" and seq[lo:hi]!=s.get("motif","").replace("T","U"):continue
            if task=="m6a_site" and seq[lo:hi]!="A":continue
            digest=hashlib.md5(seq.encode()).hexdigest()
            unique_name=f"{s['source_id']}:{s['chrom']}:{s['start']}-{s['end']}:{s['strand']}"
            rows[task][s["split"]].append({"name":unique_name,"gene":s["gene"],"sequence":seq,task:"".join(mask),"sequence_md5":digest})
    return rows

def main():
    genes,transcripts,symbols=annotation(RAW/"gencode.v49.annotation.gtf.gz");bins=gene_index(genes)
    allsites=(splice_sites(transcripts)+polya_sites(RAW/"diversity_sources/polyasite_GRCh38.bed.gz",genes,bins)
              +m6a_sites(RAW/"diversity_sources/m6a_atlas_human_highres.tsv",symbols,genes))
    selected=[]
    for i,task in enumerate(TASKS):selected+=preassign([r for r in allsites if r["task"]==task],20260827+i)
    rows=extract(selected,RAW/"GRCh38.primary_assembly.genome.fa.gz");summary=[]
    for task in TASKS:
        out=ROOT/"splits/rna"/task;out.mkdir(parents=True,exist_ok=True);used=set()
        for part,n in SIZES.items():
            chosen=[]
            for r in rows[task][part]:
                if r["sequence_md5"] in used:continue
                used.add(r["sequence_md5"]);chosen.append(r)
                if len(chosen)==n:break
            if len(chosen)<n:raise RuntimeError(f"{task}/{part}: only {len(chosen)} after sequence QC")
            fields=["name","gene","sequence",task,"sequence_md5"]
            with (out/f"{part}.csv").open("w",newline="") as h:
                w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(chosen)
            summary.append([task,part,n,sum(r[task].count("1") for r in chosen)])
        (out/"definition.json").write_text(json.dumps({"task":task,"source":("GENCODE v49 gene annotation + m6A-Atlas 2.0 high-resolution human sites (>=2 techniques)" if task=="m6a_site" else "GENCODE v49 splice annotations" if task in ("splice_donor","splice_acceptor") else "GENCODE v49 + PolyASite 2.0 GRCh38.96"),"gene_disjoint":True,"window":WINDOW},indent=2))
        print("COMPLETE",task,flush=True)
    with (ROOT/"reports/rna_processing_task_summary.csv").open("w",newline="") as h:
        w=csv.writer(h);w.writerow(["task","split","rows","positive_tokens"]);w.writerows(summary)

if __name__=="__main__":main()
