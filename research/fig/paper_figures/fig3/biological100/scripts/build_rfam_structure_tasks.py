#!/usr/bin/env python3
"""Build ten biologically defined, family-disjoint RNA structure tasks from Rfam seed."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPLIT_SIZES = {"train": 3000, "valid": 500, "test": 500}
TASKS = (
    "paired_stem", "unpaired_region", "hairpin_loop", "bulge_loop",
    "internal_loop", "multibranch_loop", "external_region", "pseudoknot",
    "canonical_pair", "gu_wobble_pair",
)
OPEN_TO_CLOSE = {"<": ">", "(": ")", "[": "]", "{": "}"}
OPEN_TO_CLOSE.update({chr(c): chr(c + 32) for c in range(ord("A"), ord("Z") + 1)})
CLOSE_TO_OPEN = {v: k for k, v in OPEN_TO_CLOSE.items()}


def stockholm_records(path: Path):
    accession = ""; seqs: dict[str, str] = {}; ss_parts: list[str] = []
    with gzip.open(path, "rt", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith("#=GF AC"):
                accession = line.split()[2].split(".")[0]
            elif line.startswith("#=GC SS_cons"):
                ss_parts.append(line.split(None, 2)[2].strip())
            elif line == "//":
                if accession and ss_parts:
                    yield accession, seqs, "".join(ss_parts)
                accession = ""; seqs = {}; ss_parts = []
            elif line and not line.startswith("#"):
                fields = line.split()
                if len(fields) >= 2:
                    seqs[fields[0]] = seqs.get(fields[0], "") + fields[1]


def pair_table(ss: str) -> tuple[dict[int, int], set[int]]:
    stacks = {op: [] for op in OPEN_TO_CLOSE}
    pairs: dict[int, int] = {}; pseudoknot: set[int] = set()
    for i, char in enumerate(ss):
        if char in stacks:
            stacks[char].append(i)
        elif char in CLOSE_TO_OPEN:
            op = CLOSE_TO_OPEN[char]
            if stacks[op]:
                j = stacks[op].pop()
                pairs[i] = j; pairs[j] = i
                if op != "<": pseudoknot.update((i, j))
    return pairs, pseudoknot


def loop_contexts(ss: str, pairs: dict[int, int]) -> dict[str, set[int]]:
    """Classify unpaired consensus columns using the nested '<>' pairing tree."""
    primary = sorted((i, j) for i, j in pairs.items() if i < j and ss[i] == "<")
    children: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    roots = []
    stack: list[tuple[int, int]] = []
    for pair in primary:
        i, j = pair
        while stack and i > stack[-1][1]: stack.pop()
        if stack: children[stack[-1]].append(pair)
        else: roots.append(pair)
        stack.append(pair)
    labels = {k: set() for k in ("hairpin_loop", "bulge_loop", "internal_loop", "multibranch_loop")}
    for i, j in primary:
        kids = children[(i, j)]
        if not kids:
            labels["hairpin_loop"].update(k for k in range(i + 1, j) if k not in pairs)
        elif len(kids) == 1:
            k, l = kids[0]
            left = {x for x in range(i + 1, k) if x not in pairs}
            right = {x for x in range(l + 1, j) if x not in pairs}
            if left and right: labels["internal_loop"].update(left | right)
            elif left or right: labels["bulge_loop"].update(left | right)
        else:
            bounds = [(i, kids[0][0])]
            bounds += [(kids[x][1], kids[x + 1][0]) for x in range(len(kids) - 1)]
            bounds += [(kids[-1][1], j)]
            for a, b in bounds:
                labels["multibranch_loop"].update(x for x in range(a + 1, b) if x not in pairs)
    nested_positions = {x for p in primary for x in p}
    if roots:
        lo, hi = min(x[0] for x in roots), max(x[1] for x in roots)
        external = {x for x in range(len(ss)) if x not in pairs and (x < lo or x > hi)}
    else: external = {x for x in range(len(ss)) if x not in pairs}
    # WUSS explicitly reserves ',' for multibranch-loop residues and ':' for
    # external residues, including cases involving pseudoknotted branches.
    labels["multibranch_loop"].update(i for i, char in enumerate(ss) if char == ",")
    external.update(i for i, char in enumerate(ss) if char == ":")
    labels["external_region"] = external
    return labels


def make_rows(accession: str, seqs: dict[str, str], ss: str):
    pairs, pk = pair_table(ss); contexts = loop_contexts(ss, pairs)
    paired = set(pairs); unpaired = set(range(len(ss))) - paired
    for name, aligned in seqs.items():
        if len(aligned) != len(ss): continue
        sequence = "".join(c for c in aligned.upper().replace("T", "U") if c not in ".-")
        if not 20 <= len(sequence) <= 1022 or set(sequence) - set("ACGUN"): continue
        col_to_pos = {}; pos = 0
        for col, char in enumerate(aligned):
            if char not in ".-": col_to_pos[col] = pos; pos += 1
        masks = {task: ["0"] * len(sequence) for task in TASKS}
        for col in paired:
            if col in col_to_pos: masks["paired_stem"][col_to_pos[col]] = "1"
        for col in unpaired:
            if col in col_to_pos: masks["unpaired_region"][col_to_pos[col]] = "1"
        for task, cols in contexts.items():
            for col in cols:
                if col in col_to_pos: masks[task][col_to_pos[col]] = "1"
        for col in pk:
            if col in col_to_pos: masks["pseudoknot"][col_to_pos[col]] = "1"
        for col, mate in pairs.items():
            if col >= mate or col not in col_to_pos or mate not in col_to_pos: continue
            bases = aligned[col].upper().replace("T", "U") + aligned[mate].upper().replace("T", "U")
            task = "canonical_pair" if bases in {"AU", "UA", "GC", "CG"} else "gu_wobble_pair" if bases in {"GU", "UG"} else None
            if task:
                masks[task][col_to_pos[col]] = masks[task][col_to_pos[mate]] = "1"
        digest = hashlib.md5(sequence.encode()).hexdigest()
        yield {"name": name, "rfam_accession": accession, "sequence": sequence,
               "sequence_md5": digest, **{k: "".join(v) for k, v in masks.items()}}


def choose_splits(rows_by_family, task: str, seed: int):
    eligible = {f: [r for r in rows if "1" in r[task]] for f, rows in rows_by_family.items()}
    eligible = {f: rows for f, rows in eligible.items() if rows}
    rng = random.Random(seed); families = list(eligible); rng.shuffle(families)
    # Reserve many small families for held-out sets, leaving the large majority
    # for training. Count unique sequences during assignment, not raw rows.
    families.sort(key=lambda f: len(eligible[f]))
    assigned = {s: [] for s in SPLIT_SIZES}; cursor = 0
    reserved_hashes = set()
    for split in ("test", "valid"):
        split_hashes = set()
        while len(split_hashes) < 650 and cursor < len(families):
            family = families[cursor]; cursor += 1; assigned[split].append(family)
            split_hashes.update(r["sequence_md5"] for r in eligible[family] if r["sequence_md5"] not in reserved_hashes)
        reserved_hashes.update(split_hashes)
    assigned["train"] = families[cursor:]
    counts = {s: len({r["sequence_md5"] for f in fs for r in eligible[f]}) for s, fs in assigned.items()}
    if any(counts[s] < n for s, n in SPLIT_SIZES.items()):
        raise RuntimeError(f"{task}: insufficient family-disjoint unique positives {counts}")
    output = {}
    used_hashes = set()
    for split, target in SPLIT_SIZES.items():
        pool = [r for f in assigned[split] for r in eligible[f]]
        rng.shuffle(pool); selected = []
        for row in pool:
            if row["sequence_md5"] in used_hashes: continue
            selected.append(row); used_hashes.add(row["sequence_md5"])
            if len(selected) == target: break
        if len(selected) != target:
            raise RuntimeError(f"{task}/{split}: only {len(selected)} unique sequences")
        output[split] = selected
    return output, assigned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "raw/rna/Rfam.seed.gz")
    parser.add_argument("--output", type=Path, default=ROOT / "splits/rna")
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()
    rows_by_family = {}
    for accession, seqs, ss in stockholm_records(args.input):
        rows_by_family[accession] = list(make_rows(accession, seqs, ss))
    summary = []
    for index, task in enumerate(TASKS):
        splits, assigned = choose_splits(rows_by_family, task, args.seed + index)
        task_dir = args.output / task; task_dir.mkdir(parents=True, exist_ok=True)
        for split, rows in splits.items():
            fields = ["name", "rfam_accession", "sequence", task, "sequence_md5"]
            with (task_dir / f"{split}.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
                writer.writerows({k: row[k] for k in fields} for row in rows)
            summary.append({"task": task, "split": split, "rows": len(rows),
                            "positive_tokens": sum(r[task].count("1") for r in rows),
                            "families": len(assigned[split])})
        (task_dir / "definition.json").write_text(json.dumps({
            "task": task, "source": "Rfam.seed SS_cons projected to member sequences",
            "family_disjoint": True, "seed": args.seed + index,
            "families_by_split": assigned,
        }, indent=2))
        print(f"COMPLETE {task}", flush=True)
    report = ROOT / "reports/rfam_structure_task_summary.csv"
    with report.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0]); writer.writeheader(); writer.writerows(summary)


if __name__ == "__main__": main()
