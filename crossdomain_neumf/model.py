# -*- coding: utf-8 -*-
"""CrossDomainNeuMF + 动态 BPR 训练 + 1正+99负评估（摘自 models_crossdomain）。"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from crossdomain_neumf.eval_sampling import (
    build_global_item_pool,
    negative_pool_for_domain,
    sample_eval_negatives,
)
from crossdomain_neumf.metrics import (
    get_hit_ratio,
    get_mean_rank,
    get_mrr,
    get_ndcg,
    mean_gauc_from_ranks,
)


def compute_implicit_train_loss(out: torch.Tensor, loss_type: str) -> torch.Tensor:
    """
    out: (batch, 1 + num_negatives)，第 0 列为正样本 logit。

    loss_type
        bce / bpr / bpr_weighted / bpr_mean
            bpr*：对全部负例的 BPR 项做 mean（train0531 同款，常叫「多负例平均 BPR」）
        bpr_max：仅对 batch 内得分最高的负例做 BPR（hard negative mining）
    """
    kind = str(loss_type).strip().lower()
    if kind == "bce":
        targets = torch.zeros_like(out)
        targets[:, 0] = 1.0
        return F.binary_cross_entropy_with_logits(out, targets)
    if kind in ("bpr", "bpr_weighted", "bpr_mean", "bpr_max"):
        pos_scores = out[:, 0]
        neg_scores = out[:, 1:]
        if kind == "bpr_max":
            hard_neg, _ = torch.max(neg_scores, dim=1)
            return -torch.mean(F.logsigmoid(pos_scores - hard_neg))
        return -torch.mean(F.logsigmoid(pos_scores.unsqueeze(1) - neg_scores))
    raise ValueError(
        f"未知 train_loss={loss_type!r}，请使用 bce / bpr / bpr_weighted / bpr_max"
    )


class CrossDomainNeuMF(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_domains: int,
        embed_dim: int = 64,
        domain_embed_dim: int = 16,
        mlp_hidden: tuple = (256, 128),
        dropout: float = 0.4,
        share_embeddings: bool = True,
        gmf_domain_aware: bool = True,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.domain_embed_dim = int(domain_embed_dim)
        self.gmf_domain_aware = bool(gmf_domain_aware)
        self.share_embeddings = bool(share_embeddings)

        if self.share_embeddings:
            self.user_emb = nn.Embedding(num_users, embed_dim)
            self.item_emb = nn.Embedding(num_items, embed_dim)
        else:
            self.user_emb_gmf = nn.Embedding(num_users, embed_dim)
            self.item_emb_gmf = nn.Embedding(num_items, embed_dim)
            self.user_emb_mlp = nn.Embedding(num_users, embed_dim)
            self.item_emb_mlp = nn.Embedding(num_items, embed_dim)

        self.domain_emb = nn.Embedding(num_domains, domain_embed_dim)
        if self.gmf_domain_aware:
            self.domain_proj_gmf = nn.Linear(domain_embed_dim, embed_dim)

        mlp_layers: list = []
        in_sz = embed_dim * 2 + domain_embed_dim
        for h in mlp_hidden:
            mlp_layers += [nn.Linear(in_sz, h), nn.ReLU(), nn.Dropout(p=dropout)]
            in_sz = h
        self.mlp = nn.Sequential(*mlp_layers)
        self.mlp_out_dim = in_sz
        self.prediction_layer = nn.Linear(embed_dim + self.mlp_out_dim, 1)
        self._init_weights()

    def forward(self, user_indices, item_indices, domain_indices):
        d_vec = self.domain_emb(domain_indices)
        if self.share_embeddings:
            u = self.user_emb(user_indices)
            it = self.item_emb(item_indices)
        else:
            u = self.user_emb_gmf(user_indices)
            it = self.item_emb_gmf(item_indices)
        if self.gmf_domain_aware:
            gmf_vector = torch.mul(torch.mul(u, it), self.domain_proj_gmf(d_vec))
        else:
            gmf_vector = torch.mul(u, it)
        mlp_input = torch.cat(
            [
                self.user_emb(user_indices) if self.share_embeddings else self.user_emb_mlp(user_indices),
                self.item_emb(item_indices) if self.share_embeddings else self.item_emb_mlp(item_indices),
                d_vec,
            ],
            dim=-1,
        )
        mlp_vector = self.mlp(mlp_input)
        cat_vector = torch.cat([gmf_vector, mlp_vector], dim=-1)
        return self.prediction_layer(cat_vector).squeeze(-1)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class UserHistoryCSR:
    def __init__(self, indptr: np.ndarray, indices: np.ndarray):
        self.indptr = indptr
        self.indices = indices

    @staticmethod
    def from_interactions(user_ids: np.ndarray, item_ids: np.ndarray, num_users: int):
        order = np.lexsort((item_ids, user_ids))
        u = user_ids[order]
        it = item_ids[order]
        if len(u) == 0:
            indptr = np.zeros(num_users + 1, dtype=np.int64)
            return UserHistoryCSR(indptr=indptr, indices=np.zeros(0, dtype=np.int32))
        keep = np.ones(len(u), dtype=bool)
        keep[1:] = (u[1:] != u[:-1]) | (it[1:] != it[:-1])
        u = u[keep]
        it = it[keep]
        counts = np.bincount(u, minlength=num_users).astype(np.int64)
        indptr = np.zeros(num_users + 1, dtype=np.int64)
        indptr[1:] = np.cumsum(counts)
        return UserHistoryCSR(indptr=indptr, indices=it.astype(np.int32))

    def contains(self, user: int, item: int) -> bool:
        start = self.indptr[user]
        end = self.indptr[user + 1]
        if start >= end:
            return False
        row = self.indices[start:end]
        return int(np.searchsorted(row, item)) < len(row) and row[np.searchsorted(row, item)] == item


class CrossDomainDataset(Dataset):
    def __init__(self, records, user_history, domain_item_pools, num_negatives: int = 8):
        self.records = records
        self.user_history = user_history
        self.domain_item_pools = domain_item_pools
        self.num_negatives = int(num_negatives)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        u, i, d = self.records[idx]
        users = [u] * (1 + self.num_negatives)
        items = [i]
        domains = [d] * (1 + self.num_negatives)
        pool = self.domain_item_pools[d]
        for _ in range(self.num_negatives):
            neg = random.choice(pool)
            while self.user_history.contains(u, neg):
                neg = random.choice(pool)
            items.append(neg)
        return (
            torch.tensor(users, dtype=torch.long),
            torch.tensor(items, dtype=torch.long),
            torch.tensor(domains, dtype=torch.long),
        )


def evaluate_crossdomain(
    model: CrossDomainNeuMF,
    eval_df: pd.DataFrame,
    user_history: UserHistoryCSR,
    domain_item_pools: dict,
    device: torch.device,
    *,
    top_k: int = 10,
    num_negatives: int = 99,
    same_domain_negatives: bool = True,
    global_item_pool: list | None = None,
) -> tuple[float, float, float, float, float]:
    if not same_domain_negatives and global_item_pool is None:
        global_item_pool = build_global_item_pool(domain_item_pools)

    hits, ndcgs, mrrs, mean_ranks, pos_ranks = [], [], [], [], []
    user_len = 1 + int(num_negatives)
    model.eval()
    with torch.no_grad():
        for row in eval_df.itertuples(index=False):
            u = int(row.global_user_id)
            gi = int(row.global_item_id)
            d = int(row.domain_id)
            interacted_contains = lambda item, _u=u: user_history.contains(_u, item)
            pool = negative_pool_for_domain(
                domain_item_pools, d, same_domain_negatives, global_item_pool
            )
            negatives = sample_eval_negatives(pool, num_negatives, interacted_contains)
            if len(negatives) < num_negatives:
                continue
            candidates = negatives + [gi]
            u_t = torch.tensor([u] * len(candidates), device=device)
            i_t = torch.tensor(candidates, device=device)
            d_t = torch.tensor([d] * len(candidates), device=device)
            scores = model(u_t, i_t, d_t).cpu().numpy()
            preds = sorted(zip(candidates, scores), key=lambda x: -x[1])
            ranked_items = [p[0] for p in preds]
            top_items = ranked_items[:top_k]
            hits.append(get_hit_ratio(top_items, gi))
            ndcgs.append(get_ndcg(top_items, gi))
            mrrs.append(get_mrr(ranked_items, gi))
            mean_ranks.append(get_mean_rank(ranked_items, gi))
            pos_ranks.append(int(ranked_items.index(gi)) + 1)

    if not hits:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    return (
        float(np.mean(hits)),
        float(np.mean(ndcgs)),
        float(np.mean(mrrs)),
        float(np.mean(mean_ranks)),
        mean_gauc_from_ranks(pos_ranks, user_len=user_len),
    )
