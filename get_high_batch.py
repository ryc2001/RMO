#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_partition_and_reorder.py

1. Find batch partition that maximizes sum of variances.
2. Shuffle and flatten batches, then reorder a JSONL dataset accordingly.
Usage:
  python batch_partition_and_reorder.py \
    --margins logp_margin_llama_3b_after.txt \
    --partition_out batches.json \
    --train_in data/3b-ultra-filter-simpo/train.jsonl \
    --train_out data/3b-high-filter-simpo-ultra-r2/train.jsonl \
    --batch_size 16 \
    --restarts 200 \
    --iterations 100000
"""

import random
import json
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

Pair = Tuple[int, float]
Batch = List[Pair]

def read_values(path: str) -> List[Pair]:
    pairs = []
    with Path(path).open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                pairs.append((i, float(line)))
    return pairs

def save_partition(batches: List[Batch], var_sum: float, fname: str):
    out = {
        "batch_size": args.batch_size,
        "total_used": sum(len(b) for b in batches),
        "variance_sum": var_sum,
        "batches": [[idx for idx, _ in b] for b in batches],
    }
    with Path(fname).open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Saved results to {fname}.")

def init_partition(pairs: List[Pair], n: int) -> List[Batch]:
    shuffled = pairs[:]
    random.shuffle(shuffled)
    return [shuffled[i : i + n] for i in range(0, len(shuffled), n)]

def batch_stats(batches: List[Batch]):
    n   = len(batches[0])
    s   = np.array([sum(v for _, v in b) for b in batches], dtype=np.float64)
    s2  = np.array([sum(v*v for _, v in b) for b in batches], dtype=np.float64)
    var = s2 / n - (s / n) ** 2
    return s, s2, var, var.sum()

def improve_partition(batches: List[Batch], iterations: int) -> Tuple[List[Batch], float]:
    n = len(batches[0])
    k = len(batches)
    s, s2, var, best_score = batch_stats(batches)
    rng = random.Random()
    loop = tqdm(range(iterations), desc="optimizing") if args.show_inner else range(iterations)
    for _ in loop:
        i, j = rng.sample(range(k), 2)
        ai   = rng.randrange(n)
        bj   = rng.randrange(n)
        idx_i, val_i = batches[i][ai]
        idx_j, val_j = batches[j][bj]
        new_s_i   = s[i]  - val_i       + val_j
        new_s2_i  = s2[i] - val_i*val_i + val_j*val_j
        new_var_i = new_s2_i / n - (new_s_i / n) ** 2
        new_s_j   = s[j]  - val_j       + val_i
        new_s2_j  = s2[j] - val_j*val_j + val_i*val_i
        new_var_j = new_s2_j / n - (new_s_j / n) ** 2
        new_score = best_score - var[i] - var[j] + new_var_i + new_var_j
        if new_score > best_score:
            batches[i][ai], batches[j][bj] = batches[j][bj], batches[i][ai]
            s[i], s2[i], var[i] = new_s_i, new_s2_i, new_var_i
            s[j], s2[j], var[j] = new_s_j, new_s2_j, new_var_j
            best_score = new_score
    return batches, float(best_score)

def find_best_partition(pairs: List[Pair], batch_size: int, restarts: int, iterations: int) -> Tuple[List[Batch], float]:
    best_part, best_score = None, -np.inf
    for _ in tqdm(range(restarts), desc="restarts"):
        part = init_partition(pairs, batch_size)
        part, score = improve_partition(part, iterations)
        if score > best_score:
            best_part, best_score = part, score
    return best_part, best_score

def reorder_dataset(batch_json: str, train_in: str, train_out: str):
    info = json.load(open(batch_json, "r", encoding="utf-8"))
    batches = info["batches"]
    random.shuffle(batches)
    shuffled_indices = [idx for batch in batches for idx in batch]
    data = [json.loads(line) for line in open(train_in, "r", encoding="utf-8")]
    reordered = [data[i] for i in shuffled_indices]
    Path(train_out).parent.mkdir(parents=True, exist_ok=True)
    with open(train_out, "w", encoding="utf-8") as f:
        for item in reordered:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Partition and reorder")
    parser.add_argument("--margins",       type=str, required=True)
    parser.add_argument("--partition_out", type=str, required=True)
    parser.add_argument("--train_in",      type=str, required=True)
    parser.add_argument("--train_out",     type=str, required=True)
    parser.add_argument("--batch_size",    type=int, default=16)
    parser.add_argument("--restarts",      type=int, default=200)
    parser.add_argument("--iterations",    type=int, default=100000)
    parser.add_argument("--show_inner",    action="store_true")
    args = parser.parse_args()

    pairs = read_values(args.margins)
    usable = (len(pairs) // args.batch_size) * args.batch_size
    pairs = pairs[:usable]
    part, score = find_best_partition(pairs, args.batch_size, args.restarts, args.iterations)
    save_partition(part, score, args.partition_out)
    reorder_dataset(args.partition_out, args.train_in, args.train_out)

