#!/usr/bin/env python3
"""Build four homology-cluster-disjoint miRNA biogenesis tasks from miRBase."""
from __future__ import annotations
import argparse, csv, hashlib, html, random, re, subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw/rna/diversity_sources"
MMSEQS = Path("/data1/zshuo23/anaconda3/envs/hfds/bin/mmseqs")
TASKS = ("mirna_mature_5p", "mirna_mature_3p", "mirna_seed_region", "mirna_terminal_loop")
SIZES = {"train": 3000, "valid": 500, "test": 500}


def fasta_from_download_page(path: Path):
    text = path.read_text(errors="replace")
    if "<br>" in text or "<br/>" in text:
        text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
        text = html.unescape(re.sub(r"<[^>]+>", "", text))
    name = None; parts = []
    for line in text.splitlines():
        if line.startswith(">"):
            if name: yield name, "".join(parts).upper().replace("T", "U")
            name = line[1:].split()[0]; parts = []
        elif name: parts.append(line.strip())
    if name: yield name, "".join(parts).upper().replace("T", "U")


def map_matures(precursors, matures):
    k = 10; index = defaultdict(set)
    mature_seq = {}
    for name, seq in matures:
        if len(seq) >= k and set(seq) <= set("ACGUN"):
            mature_seq[name] = seq; index[seq[:k]].add(name)
    result = defaultdict(list)
    for pre_name, pre_seq in precursors.items():
        candidates = set()
        for i in range(max(0, len(pre_seq) - k + 1)): candidates.update(index.get(pre_seq[i:i+k], ()))
        for mature_name in candidates:
            seq = mature_seq[mature_name]; start = pre_seq.find(seq)
            if start >= 0: result[pre_name].append((start, start + len(seq), mature_name))
    return result


def cluster(precursors: dict[str, str], work: Path):
    work.mkdir(parents=True, exist_ok=True); fasta = work / "precursors.fa"
    with fasta.open("w") as handle:
        for name, seq in precursors.items(): handle.write(f">{name}\n{seq}\n")
    prefix = work / "linclust"; tmp = work / "tmp"
    subprocess.run([str(MMSEQS), "easy-linclust", str(fasta), str(prefix), str(tmp),
                    "--min-seq-id", "0.8", "-c", "0.8", "--cov-mode", "0",
                    "--kmer-per-seq-scale", "0.3", "--threads", "16"], check=True)
    mapping = {}
    with (work / "linclust_cluster.tsv").open() as handle:
        for line in handle:
            representative, member = line.rstrip().split("\t")[:2]; mapping[member] = representative
    return mapping


def make_rows(precursors, mapped, clusters):
    output = {t: [] for t in TASKS}
    for name, seq in precursors.items():
        hits = mapped.get(name, [])
        left = [h for h in hits if (h[0] + h[1]) / 2 <= len(seq) / 2]
        right = [h for h in hits if (h[0] + h[1]) / 2 > len(seq) / 2]
        left = min(left, key=lambda h: h[0], default=None)
        right = max(right, key=lambda h: h[1], default=None)
        digest = hashlib.md5(seq.encode()).hexdigest()
        base = {"name": name, "cluster": clusters.get(name, name), "species": name.split("-", 1)[0],
                "sequence": seq, "sequence_md5": digest}
        if left:
            mask = ["0"] * len(seq); mask[left[0]:left[1]] = "1" * (left[1]-left[0])
            output["mirna_mature_5p"].append({**base, "mirna_mature_5p": "".join(mask)})
        if right:
            mask = ["0"] * len(seq); mask[right[0]:right[1]] = "1" * (right[1]-right[0])
            output["mirna_mature_3p"].append({**base, "mirna_mature_3p": "".join(mask)})
        if left or right:
            mask = ["0"] * len(seq)
            for hit in (left, right):
                if hit: mask[hit[0]+1:min(hit[0]+8, hit[1])] = "1" * max(0, min(7, hit[1]-hit[0]-1))
            if "1" in mask: output["mirna_seed_region"].append({**base, "mirna_seed_region": "".join(mask)})
        if left and right and right[0] - left[1] >= 3:
            mask = ["0"] * len(seq); mask[left[1]:right[0]] = "1" * (right[0]-left[1])
            output["mirna_terminal_loop"].append({**base, "mirna_terminal_loop": "".join(mask)})
    return output


def split(rows, task, seed):
    by_group = defaultdict(list)
    for row in rows: by_group[row["cluster"]].append(row)
    rng = random.Random(seed); groups = list(by_group); rng.shuffle(groups); groups.sort(key=lambda g: len(by_group[g]))
    assigned = {s: [] for s in SIZES}; cursor = 0; reserved = set()
    for part in ("test", "valid"):
        hashes = set()
        while len(hashes) < 520 and cursor < len(groups):
            group = groups[cursor]; cursor += 1; assigned[part].append(group)
            hashes.update(r["sequence_md5"] for r in by_group[group] if r["sequence_md5"] not in reserved)
        reserved.update(hashes)
    assigned["train"] = groups[cursor:]; output = {}; used = set()
    for part, n in SIZES.items():
        pool = [r for g in assigned[part] for r in by_group[g]]; rng.shuffle(pool); selected = []
        for row in pool:
            if row["sequence_md5"] in used: continue
            used.add(row["sequence_md5"]); selected.append(row)
            if len(selected) == n: break
        if len(selected) < n: raise RuntimeError(f"{task}/{part}: only {len(selected)} unique rows")
        output[part] = selected
    return output


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, default=20260827); args = parser.parse_args()
    precursors = dict(fasta_from_download_page(RAW / "mirbase_hairpin.fa"))
    matures = list(fasta_from_download_page(RAW / "mirbase_mature.fa"))
    if len(precursors) < 4000 or len(matures) < 4000: raise RuntimeError(f"invalid miRBase downloads: {len(precursors)}, {len(matures)}")
    mapped = map_matures(precursors, matures); clusters = cluster(precursors, ROOT / "work/rna_diversity/mirbase_cluster")
    tasks = make_rows(precursors, mapped, clusters); summary = []
    for i, task in enumerate(TASKS):
        splits = split(tasks[task], task, args.seed+i); out = ROOT / "splits/rna" / task; out.mkdir(parents=True, exist_ok=True)
        for part, rows in splits.items():
            fields = ["name", "cluster", "species", "sequence", task, "sequence_md5"]
            with (out/f"{part}.csv").open("w", newline="") as handle:
                writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
            summary.append([task,part,len(rows),sum(r[task].count("1") for r in rows)])
        (out/"definition.json").write_text(__import__("json").dumps({"task":task,"source":"miRBase release 23","cluster_disjoint":True},indent=2))
        print("COMPLETE",task,flush=True)
    with (ROOT/"reports/mirbase_task_summary.csv").open("w",newline="") as h:
        w=csv.writer(h);w.writerow(["task","split","rows","positive_tokens"]);w.writerows(summary)

if __name__ == "__main__": main()
