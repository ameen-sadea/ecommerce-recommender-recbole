# -*- coding: utf-8 -*-
"""CrossDomainNeuMF 全库排序评估（与 RecBole full_sort 协议对齐）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from crossdomain_neumf.metrics import hr_at_rank, mrr_at_rank, ndcg_at_rank, rank_from_scores
from crossdomain_neumf.model import CrossDomainNeuMF, UserHistoryCSR

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover

    def tqdm(it, **kwargs):
        return it


def build_user_history_from_frames(
    frames: list[pd.DataFrame], num_users: int
) -> UserHistoryCSR:
    if not frames:
        return UserHistoryCSR.from_interactions(
            np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32), num_users
        )
    users = np.concatenate(
        [f["global_user_id"].to_numpy(dtype=np.int32, copy=False) for f in frames]
    )
    items = np.concatenate(
        [f["global_item_id"].to_numpy(dtype=np.int32, copy=False) for f in frames]
    )
    return UserHistoryCSR.from_interactions(users, items, num_users)


@torch.no_grad()
def score_all_items_for_user(
    model: CrossDomainNeuMF,
    user_id: int,
    domain_id: int,
    num_items: int,
    device: torch.device,
    *,
    item_batch_size: int = 8192,
) -> np.ndarray:
    scores = np.full(num_items, -np.inf, dtype=np.float32)
    for start in range(1, num_items, item_batch_size):
        end = min(start + item_batch_size, num_items)
        n = end - start
        items = torch.arange(start, end, device=device, dtype=torch.long)
        u_t = torch.full((n,), user_id, device=device, dtype=torch.long)
        d_t = torch.full((n,), domain_id, device=device, dtype=torch.long)
        scores[start:end] = model(u_t, items, d_t).cpu().numpy()
    return scores


def _mask_history(scores: np.ndarray, user: int, history: UserHistoryCSR) -> None:
    start = history.indptr[user]
    end = history.indptr[user + 1]
    if start < end:
        scores[history.indices[start:end]] = -np.inf


def evaluate_crossdomain_full_catalog(
    model: CrossDomainNeuMF,
    eval_df: pd.DataFrame,
    history: UserHistoryCSR,
    *,
    num_items: int,
    device: torch.device,
    topk_list: list[int] | None = None,
    item_batch_size: int = 8192,
    show_progress: bool = True,
) -> dict[str, float]:
    """
    对 eval_df 每行 (user, pos_item) 在全部 item 上排序。
    history 应已含需 mask 的交互（test 用 train+valid，valid 用 train）。
    """
    if eval_df.empty:
        return {}

    topk_list = sorted({int(k) for k in (topk_list or [10, 50]) if int(k) > 0})
    max_k = max(topk_list)

    sums: dict[str, float] = {}
    for k in topk_list:
        sums[f"hr@{k}"] = 0.0
        sums[f"ndcg@{k}"] = 0.0
        sums[f"mrr@{k}"] = 0.0
    sums["meanrank"] = 0.0
    n = 0

    model.eval()
    rows = eval_df.itertuples(index=False)
    if show_progress:
        rows = tqdm(rows, total=len(eval_df), desc="full-catalog", unit="user")

    with torch.no_grad():
        for row in rows:
            u = int(row.global_user_id)
            pos = int(row.global_item_id)
            d = int(row.domain_id)

            scores = score_all_items_for_user(
                model,
                u,
                d,
                num_items,
                device,
                item_batch_size=item_batch_size,
            )
            scores[0] = -np.inf
            _mask_history(scores, u, history)

            rank = rank_from_scores(scores, pos)
            if rank <= 0:
                continue

            n += 1
            sums["meanrank"] += float(rank)
            for k in topk_list:
                sums[f"hr@{k}"] += hr_at_rank(rank, k)
                sums[f"ndcg@{k}"] += ndcg_at_rank(rank, k)
                sums[f"mrr@{k}"] += mrr_at_rank(rank, k)

    if n == 0:
        return {}

    out = {key: val / n for key, val in sums.items()}
    out["n_users"] = float(n)
    out["max_k"] = float(max_k)
    return out
