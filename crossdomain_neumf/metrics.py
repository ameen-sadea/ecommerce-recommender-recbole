# -*- coding: utf-8 -*-
"""与 crossdomain 脚本一致的 ranking 指标（1 正 + N 负候选集）。"""

from __future__ import annotations

import math

import numpy as np


def get_hit_ratio(top_items: list, item) -> float:
    return 1.0 if item in top_items else 0.0


def get_ndcg(top_items: list, item) -> float:
    if item not in top_items:
        return 0.0
    rank = top_items.index(item) + 1
    return 1.0 / math.log2(rank + 1)


def get_mrr(ranked_items: list, item) -> float:
    try:
        rank = ranked_items.index(item) + 1
        return 1.0 / rank
    except ValueError:
        return 0.0


def get_mean_rank(ranked_items: list, item) -> float:
    try:
        return float(ranked_items.index(item) + 1)
    except ValueError:
        return float(len(ranked_items) + 1)


def rank_from_scores(scores: np.ndarray, item: int) -> int:
    """1-based rank among all items; 0 if item masked or out of range."""
    if item <= 0 or item >= len(scores):
        return 0
    pos_score = float(scores[item])
    if not np.isfinite(pos_score):
        return 0
    return int(1 + np.sum(scores > pos_score))


def hr_at_rank(rank: int, k: int) -> float:
    return 1.0 if 0 < rank <= k else 0.0


def ndcg_at_rank(rank: int, k: int) -> float:
    if rank <= 0 or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def mrr_at_rank(rank: int, k: int) -> float:
    if rank <= 0 or rank > k:
        return 0.0
    return 1.0 / rank


def user_gauc_from_rank(rank: int, *, user_len: int, pos_len: int = 1) -> float:
    """
    与 RecBole GAUC 一致：每用户 1 正 + (user_len-1) 负，按降序 rank 算 AUC 再加权平均。
    rank 为 1-based 降序名次（1=最高分）。
    """
    if rank <= 0 or user_len <= pos_len:
        return 0.0
    neg_len = user_len - pos_len
    if neg_len <= 0 or pos_len <= 0:
        return 0.0
    pair_num = (user_len + 1) * pos_len - pos_len * (pos_len + 1) / 2.0 - float(rank)
    return pair_num / (neg_len * pos_len)


def mean_gauc_from_ranks(ranks: list[int], *, user_len: int, pos_len: int = 1) -> float:
    if not ranks:
        return 0.0
    return float(
        np.mean([user_gauc_from_rank(r, user_len=user_len, pos_len=pos_len) for r in ranks])
    )
