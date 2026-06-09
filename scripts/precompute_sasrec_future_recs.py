#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预计算 SASRec “未来页面”推荐。

与 LOO 评测不同，本脚本把 train + valid + test 都视为用户已经发生的历史，
再用 SASRec 预测下一批推荐，并过滤所有历史 item。

在 recbole_platform 目录下运行示例:
  python scripts/precompute_sasrec_future_recs.py --top-k 100
  python scripts/precompute_sasrec_future_recs.py --user-cap 1000 --top-k 50

输出:
  results/sasrec_future/movies_tv_seq_sasrec_future_top100.jsonl
  results/sasrec_future/movies_tv_seq_sasrec_future_top100_manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd
import torch

PLATFORM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PLATFORM_ROOT not in sys.path:
    sys.path.insert(0, PLATFORM_ROOT)
os.chdir(PLATFORM_ROOT)

from scripts.eval_cascade_rank import load_model_bundle  # noqa: E402


BASE_DATASET = "movies_tv"
SEQ_DATASET = "movies_tv_seq"
MODEL_KEY = "sasrec"
SPLITS = ("train", "valid", "test")


def _read_inter_split(split: str, dataset: str = BASE_DATASET) -> pd.DataFrame:
    path = os.path.join(PLATFORM_ROOT, "datasets", dataset, f"{dataset}.{split}.inter")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(
        path,
        sep="\t",
        skiprows=1,
        names=["user_id", "item_id", "rating", "timestamp"],
        dtype={"user_id": str, "item_id": str},
    )


def resolve_default_checkpoint() -> str | None:
    candidates: list[str] = []
    logs_dir = os.path.join(PLATFORM_ROOT, "results", "logs")
    if os.path.isdir(logs_dir):
        names = [n for n in os.listdir(logs_dir) if "SASRec" in n and n.endswith(".json")]
        names.sort(key=lambda n: os.path.getmtime(os.path.join(logs_dir, n)), reverse=True)
        for name in names:
            path = os.path.join(logs_dir, name)
            try:
                with open(path, encoding="utf-8") as f:
                    payload = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            if payload.get("dataset") == SEQ_DATASET and payload.get("model") == "SASRec":
                ckpt = payload.get("best_ckpt")
                if ckpt:
                    candidates.append(str(ckpt))
    candidates.append(r"D:\recbole_checkpoints\movies_tv_seq\SASRec\best.pth")
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0] if candidates else None


def build_future_histories(
    *,
    dataset: str = BASE_DATASET,
    user_cap: int = 0,
) -> list[tuple[str, list[str]]]:
    """按 split 顺序拼接 train + valid + test，得到“当前已知完整历史”。"""
    histories: dict[str, list[str]] = defaultdict(list)
    for split in SPLITS:
        df = _read_inter_split(split, dataset)
        for uid, grp in df.groupby("user_id", sort=False):
            histories[str(uid)].extend(grp["item_id"].astype(str).tolist())

    rows = [(uid, items) for uid, items in histories.items() if items]
    rows.sort(key=lambda x: (-len(x[1]), x[0]))
    if user_cap and user_cap > 0:
        rows = rows[:user_cap]
    return rows


def token_to_internal(dataset, token: str) -> int | None:
    try:
        return int(dataset.token2id(dataset.iid_field, str(token)))
    except (KeyError, ValueError):
        return None


def internal_to_token(dataset, iid: int) -> str:
    try:
        return str(dataset.id2token(dataset.iid_field, int(iid)))
    except (KeyError, ValueError):
        return str(iid)


def histories_to_internal(bundle, token_histories: Iterable[list[str]]) -> list[list[int]]:
    out: list[list[int]] = []
    for hist in token_histories:
        ids: list[int] = []
        for token in hist:
            iid = token_to_internal(bundle.dataset, token)
            if iid is not None and iid > 0:
                ids.append(iid)
        out.append(ids)
    return out


@torch.no_grad()
def score_batch(bundle, histories: list[list[int]], top_k: int) -> list[list[tuple[int, float]]]:
    from recbole.data.interaction import Interaction

    model = bundle.model
    config = bundle.config
    device = next(model.parameters()).device
    max_len = int(config["MAX_ITEM_LIST_LENGTH"])
    seq_field = config["ITEM_ID_FIELD"] + config["LIST_SUFFIX"]
    len_field = config["ITEM_LIST_LENGTH_FIELD"]
    uid_field = config["USER_ID_FIELD"]

    bsz = len(histories)
    seq_batch = torch.zeros(bsz, max_len, dtype=torch.long, device=device)
    len_batch = torch.zeros(bsz, dtype=torch.long, device=device)
    for row, hist in enumerate(histories):
        h = hist[-max_len:]
        if h:
            seq_batch[row, : len(h)] = torch.tensor(h, dtype=torch.long, device=device)
            len_batch[row] = len(h)

    inter = Interaction(
        {
            uid_field: torch.arange(bsz, dtype=torch.long, device=device),
            seq_field: seq_batch,
            len_field: len_batch,
        }
    )
    scores = model.full_sort_predict(inter.to(device))
    if scores.dim() == 1:
        scores = scores.view(bsz, -1)
    scores = scores.clone()
    scores[:, 0] = -np.inf
    for row, hist in enumerate(histories):
        if hist:
            idx = torch.tensor(list(set(hist)), dtype=torch.long, device=device)
            idx = idx[(idx > 0) & (idx < scores.shape[1])]
            scores[row, idx] = -np.inf

    k = min(top_k, scores.shape[1] - 1)
    vals, idx = torch.topk(scores, k, dim=1)
    result: list[list[tuple[int, float]]] = []
    for row in range(bsz):
        result.append(
            [
                (int(i), float(s))
                for i, s in zip(idx[row].detach().cpu().tolist(), vals[row].detach().cpu().tolist())
                if np.isfinite(float(s))
            ]
        )
    return result


def write_jsonl(
    *,
    bundle,
    user_rows: list[tuple[str, list[str]]],
    out_path: str,
    top_k: int,
    batch_size: int,
) -> int:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n_written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for start in range(0, len(user_rows), batch_size):
            batch = user_rows[start : start + batch_size]
            internal_histories = histories_to_internal(bundle, [hist for _, hist in batch])
            rec_batches = score_batch(bundle, internal_histories, top_k)
            for (uid, hist_tokens), hist_internal, recs in zip(batch, internal_histories, rec_batches):
                payload = {
                    "user_id": uid,
                    "history_len": len(hist_tokens),
                    "history_used": min(len(hist_internal), int(bundle.config["MAX_ITEM_LIST_LENGTH"])),
                    "items": [internal_to_token(bundle.dataset, iid) for iid, _ in recs],
                    "scores": [score for _, score in recs],
                }
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                n_written += 1
            print(f"processed {min(start + batch_size, len(user_rows)):,}/{len(user_rows):,}")
    return n_written


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute SASRec future recommendations.")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--user-cap", type=int, default=0, help="0=all users; otherwise take most active users first")
    parser.add_argument("--checkpoint", default=None, help="Optional SASRec checkpoint path")
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSONL path. Default: results/sasrec_future/<dataset>_sasrec_future_topK.jsonl",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint or resolve_default_checkpoint()
    print(">>> load SASRec")
    print(f"checkpoint: {checkpoint or '(yaml default)'}")
    bundle = load_model_bundle(MODEL_KEY, checkpoint=checkpoint)
    rows = build_future_histories(user_cap=args.user_cap)
    out_path = args.out or os.path.join(
        PLATFORM_ROOT,
        "results",
        "sasrec_future",
        f"{SEQ_DATASET}_sasrec_future_top{args.top_k}.jsonl",
    )
    print(f"users: {len(rows):,}  top_k={args.top_k}  batch_size={args.batch_size}")
    print(f"out: {out_path}")

    n_written = write_jsonl(
        bundle=bundle,
        user_rows=rows,
        out_path=out_path,
        top_k=args.top_k,
        batch_size=args.batch_size,
    )
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": SEQ_DATASET,
        "base_dataset": BASE_DATASET,
        "model": "SASRec",
        "top_k": args.top_k,
        "n_users": n_written,
        "history_source": "train+valid+test",
        "checkpoint": checkpoint,
        "output": out_path,
    }
    manifest_path = os.path.splitext(out_path)[0] + "_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
