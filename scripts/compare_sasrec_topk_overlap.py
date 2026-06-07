#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单用户全库 Top-K 对比：SASRec K=50（原模型） vs SASRec K=5。

在 recbole_platform 目录下:
  python scripts/compare_sasrec_topk_overlap.py

输出：Top-K 交集、仅 K50 有、仅 K5 有、Jaccard、正例排名等。
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import torch

PLATFORM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PLATFORM_ROOT not in sys.path:
    sys.path.insert(0, PLATFORM_ROOT)
os.chdir(PLATFORM_ROOT)

_cascade_path = os.path.join(PLATFORM_ROOT, "scripts", "eval_cascade_rank.py")
_spec = importlib.util.spec_from_file_location("eval_cascade_rank", _cascade_path)
_cascade = importlib.util.module_from_spec(_spec)
sys.modules["eval_cascade_rank"] = _cascade
assert _spec.loader is not None
_spec.loader.exec_module(_cascade)
load_model_bundle = _cascade.load_model_bundle

# ========================= 可调区域 =========================
# user_id 为 .inter 里的 token（字符串）；None 则自动选 test 前历史>=MIN_HIST 的用户
USER_TOKEN: str | None = None
MIN_HIST = 10  # 自动选用户时，test 前历史至少这么长
AUTO_PICK_SEED = 42

TOPK = 100
PHASE = "test"  # test | valid
SAVE_JSON = True
# ============================================================


def _load_test_rows(phase: str) -> pd.DataFrame:
    path = os.path.join(
        PLATFORM_ROOT,
        "datasets",
        "movies_tv_seq",
        f"movies_tv_seq.{phase}.inter",
    )
    df = pd.read_csv(
        path,
        sep="\t",
        skiprows=1,
        names=["user_id", "item_id", "item_id_list", "item_length"],
    )
    df = df.drop_duplicates("user_id", keep="last")
    return df


def _parse_item_list(raw: Any) -> list[str]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    if isinstance(raw, (list, tuple)):
        return [str(x) for x in raw]
    return [x for x in str(raw).split() if x]


def _tokens_to_internal(dataset, tokens: list[str]) -> list[int]:
    field = dataset.iid_field
    out: list[int] = []
    for t in tokens:
        try:
            out.append(int(dataset.token2id(field, str(t))))
        except (KeyError, ValueError):
            continue
    return out


def _token_of(dataset, iid: int) -> str:
    field = dataset.iid_field
    try:
        return str(dataset.id2token(field, int(iid)))
    except (KeyError, ValueError):
        return str(iid)


def _pick_user(df: pd.DataFrame) -> str:
    rng = np.random.RandomState(AUTO_PICK_SEED)
    candidates: list[str] = []
    for _, row in df.iterrows():
        hist = _parse_item_list(row["item_id_list"])
        if len(hist) >= MIN_HIST:
            candidates.append(str(row["user_id"]))
    if not candidates:
        raise RuntimeError(f"没有 hist>={MIN_HIST} 的 test 用户")
    return str(rng.choice(candidates))


@torch.no_grad()
def _full_sort_scores(
    bundle,
    hist_internal: list[int],
    *,
    max_len: int,
    device: torch.device,
) -> torch.Tensor:
    from recbole.data.interaction import Interaction

    model = bundle.model
    config = bundle.config
    seq_field = config["ITEM_ID_FIELD"] + config["LIST_SUFFIX"]
    len_field = config["ITEM_LIST_LENGTH_FIELD"]
    uid_field = config["USER_ID_FIELD"]

    h = hist_internal[-max_len:]
    length = len(h)
    seq_batch = torch.zeros(1, max_len, dtype=torch.long, device=device)
    if length > 0:
        seq_batch[0, :length] = torch.tensor(h, dtype=torch.long, device=device)
    len_batch = torch.tensor([length], dtype=torch.long, device=device)
    uids = torch.zeros(1, dtype=torch.long, device=device)

    inter = Interaction(
        {uid_field: uids, seq_field: seq_batch, len_field: len_batch}
    )
    scores = model.full_sort_predict(inter.to(device))
    if scores.dim() == 1:
        scores = scores.view(1, -1)
    scores = scores[0].clone()
    scores[0] = -np.inf
    for iid in hist_internal:
        if 0 < iid < scores.numel():
            scores[iid] = -np.inf
    return scores


def _topk_items(scores: torch.Tensor, k: int) -> tuple[list[int], list[float]]:
    k = min(k, int((scores > -np.inf).sum().item()))
    if k <= 0:
        return [], []
    vals, idx = torch.topk(scores, k)
    return idx.cpu().tolist(), vals.cpu().tolist()


def compare_user(
    *,
    user_token: str,
    bundle_k50,
    bundle_k5,
    row: pd.Series,
    device: torch.device,
    topk: int,
) -> dict[str, Any]:
    hist_tokens = _parse_item_list(row["item_id_list"])
    target_token = str(row["item_id"])
    hist = _tokens_to_internal(bundle_k50.dataset, hist_tokens)
    target = int(bundle_k50.dataset.token2id(bundle_k50.dataset.iid_field, target_token))

    max50 = int(bundle_k50.config["MAX_ITEM_LIST_LENGTH"])
    max5 = int(bundle_k5.config["MAX_ITEM_LIST_LENGTH"])

    scores50 = _full_sort_scores(bundle_k50, hist, max_len=max50, device=device)
    scores5 = _full_sort_scores(bundle_k5, hist, max_len=max5, device=device)

    top50, sc50 = _topk_items(scores50, topk)
    top5, sc5 = _topk_items(scores5, topk)

    set50 = set(top50)
    set5 = set(top5)
    overlap = set50 & set5
    only50 = set50 - set5
    only5 = set5 - set50
    union = set50 | set5

    def _rank(scores: torch.Tensor, item: int) -> int | None:
        if item <= 0 or item >= scores.numel():
            return None
        s = scores[item].item()
        if not np.isfinite(s):
            return None
        return int((scores > s).sum().item()) + 1

    out: dict[str, Any] = {
        "user_token": user_token,
        "phase": PHASE,
        "hist_len": len(hist_tokens),
        "hist_used_k50": min(len(hist), max50),
        "hist_used_k5": min(len(hist), max5),
        "target_token": target_token,
        "target_internal_id": target,
        "topk": topk,
        "overlap_count": len(overlap),
        "only_k50_count": len(only50),
        "only_k5_count": len(only5),
        "overlap_ratio": len(overlap) / topk if topk else 0.0,
        "jaccard": len(overlap) / len(union) if union else 0.0,
        "rank_k50_target": _rank(scores50, target),
        "rank_k5_target": _rank(scores5, target),
        "topk_k50_tokens": [_token_of(bundle_k50.dataset, i) for i in top50[:20]],
        "topk_k5_tokens": [_token_of(bundle_k5.dataset, i) for i in top5[:20]],
        "only_k50_sample_tokens": [_token_of(bundle_k50.dataset, i) for i in list(only50)[:10]],
        "only_k5_sample_tokens": [_token_of(bundle_k5.dataset, i) for i in list(only5)[:10]],
    }
    return out


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    df = _load_test_rows(PHASE)
    user_token = USER_TOKEN or _pick_user(df)
    if user_token not in set(df["user_id"].astype(str)):
        raise ValueError(f"找不到用户 {user_token!r}")
    row = df[df["user_id"].astype(str) == user_token].iloc[0]

    print(">>> load SASRec K=50 (sasrec)")
    bundle_k50 = load_model_bundle("sasrec")
    print(">>> load SASRec K=5 (sasrec_k5)")
    bundle_k5 = load_model_bundle("sasrec_k5")

    result = compare_user(
        user_token=user_token,
        bundle_k50=bundle_k50,
        bundle_k5=bundle_k5,
        row=row,
        device=device,
        topk=TOPK,
    )

    print("\n" + "=" * 60)
    print(f"用户: {result['user_token']}")
    print(f"test 前历史: {result['hist_len']} 条 (K50 用 {result['hist_used_k50']}, K5 用 {result['hist_used_k5']})")
    print(f"test 正例 item: {result['target_token']}")
    print(f"Top-{TOPK} 对比:")
    print(f"  相同(交集):     {result['overlap_count']}  ({result['overlap_ratio']*100:.1f}%)")
    print(f"  仅 K50 模型有:  {result['only_k50_count']}")
    print(f"  仅 K5 模型有:   {result['only_k5_count']}")
    print(f"  Jaccard@{TOPK}: {result['jaccard']:.4f}")
    print(f"正例全库排名: K50=#{result['rank_k50_target']}  K5=#{result['rank_k5_target']}")
    print("K50 Top-5 tokens:", result["topk_k50_tokens"][:5])
    print("K5  Top-5 tokens:", result["topk_k5_tokens"][:5])
    print("=" * 60)

    if SAVE_JSON:
        out_dir = os.path.join(PLATFORM_ROOT, "results", "logs")
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(out_dir, f"sasrec_k50_vs_k5_top{TOPK}_{ts}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
