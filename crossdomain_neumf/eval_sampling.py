# -*- coding: utf-8 -*-
from __future__ import annotations

import random


def build_global_item_pool(domain_item_pools: dict) -> list:
    seen: set = set()
    out: list = []
    for pool in domain_item_pools.values():
        for i in pool:
            if i not in seen:
                seen.add(i)
                out.append(i)
    return out


def negative_pool_for_domain(
    domain_item_pools: dict,
    domain_id: int,
    same_domain_negatives: bool,
    global_item_pool: list | None,
) -> list:
    if same_domain_negatives:
        return domain_item_pools[domain_id]
    return global_item_pool or build_global_item_pool(domain_item_pools)


def sample_eval_negatives(
    pool: list,
    num_negatives: int,
    interacted_contains,
    *,
    max_tries: int = 5000,
) -> list:
    if len(pool) <= num_negatives:
        return []
    negatives: list = []
    tries = 0
    while len(negatives) < num_negatives and tries < max_tries:
        tries += 1
        cand = random.choice(pool)
        if interacted_contains(cand) or cand in negatives:
            continue
        negatives.append(cand)
    return negatives
