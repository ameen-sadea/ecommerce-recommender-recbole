#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多模型全库 Reciprocal Rank Fusion (RRF) 评估。

在 recbole_platform 目录下:
  python scripts/eval_rrf_rank.py

流程:
  1. 在 valid 上网格搜索 RRF 常数 k（可选模型权重）
  2. 用最优 k 在 test 上报 HR/NDCG/MRR@K
  3. 同协议输出各单模型基线便于对比

RRF: score(i) = sum_m w_m / (k + rank_m(i))，rank 为全库 1-based 排名（历史 item 已 mask）。
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

PLATFORM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PLATFORM_ROOT not in sys.path:
    sys.path.insert(0, PLATFORM_ROOT)
os.chdir(PLATFORM_ROOT)

import importlib.util

_cascade_path = os.path.join(PLATFORM_ROOT, "scripts", "eval_cascade_rank.py")
_spec = importlib.util.spec_from_file_location("eval_cascade_rank", _cascade_path)
_cascade = importlib.util.module_from_spec(_spec)
sys.modules["eval_cascade_rank"] = _cascade
assert _spec.loader is not None
_spec.loader.exec_module(_cascade)

METRIC_TOPK = _cascade.METRIC_TOPK
ModelBundle = _cascade.ModelBundle
_apply_full_catalog_phase = _cascade._apply_full_catalog_phase
_build_full_sort_loader = _cascade._build_full_sort_loader
_history_items_for_user = _cascade._history_items_for_user
_metrics_from_ranks = _cascade._metrics_from_ranks
_rank_positions = _cascade._rank_positions
_restore_phase = _cascade._restore_phase
load_model_bundle = _cascade.load_model_bundle

# ========================= 可调区域（只改这里）=========================
# 融合模型；weight 为 RRF 项系数（可不等权）
MODELS: list[dict[str, Any]] = [
    {"key": "sasrec", "weight": 1.0, "checkpoint": None},
    {"key": "lightgcn", "weight": 1.0, "checkpoint": None},
    {"key": "bpr", "weight": 1.0, "checkpoint": None},
    # {"key": "pop", "weight": 0.5, "checkpoint": None},  # 全库 Pop 较慢，默认关闭
]

# 全库 eval 以 SASRec 序列数据集上的 test/valid 用户为准
DRIVER = "sasrec"

USER_CAP = 20000  # 0=全量 test；20000=与主表 cap 一致
EVAL_SAMPLE_SEED = 42

# valid 上网格搜 k；test 用 TUNE_BEST 或固定 RRF_K
RRF_K_GRID = [10, 20, 40, 60, 80, 100, 120]
RRF_K = 60  # TUNE_ON_VALID=False 时直接用
TUNE_ON_VALID = True
TUNE_METRIC = "ndcg@10"  # ndcg@10 | hr@10 | mrr@10

# 粗排/融合均 mask：train+valid 历史 + 序列 item_seq（不含当前 test 正例）
MASK_INTERACTION_HISTORY = True

RUN_SINGLE_BASELINES = True  # 同 loader 下单模型分数（与 RRF 同 mask 规则）
SHOW_PROGRESS = True
SAVE_JSON = True
# ========================================================================


def _rank_from_scores(scores: torch.Tensor) -> torch.Tensor:
    """全库降序 1-based rank；scores 中 -inf 会落到较大 rank。"""
    order = scores.argsort(dim=1, descending=True)
    ranks = torch.empty_like(scores, dtype=torch.float32)
    n_items = scores.shape[1]
    idx = torch.arange(n_items, device=scores.device, dtype=torch.float32).unsqueeze(0)
    idx = idx.expand(scores.shape[0], -1) + 1.0
    ranks.scatter_(1, order, idx)
    return ranks


def _mask_history_on_scores(
    scores: torch.Tensor,
    *,
    bundle: ModelBundle,
    interaction,
    history_index,
    user_ids: torch.Tensor,
    positive_i: torch.Tensor,
    sampler,
) -> None:
    """就地 mask pad 与历史交互（test 正例保留）。"""
    device = scores.device
    scores[:, 0] = -np.inf

    if not MASK_INTERACTION_HISTORY:
        return

    if history_index is not None:
        hu, hi = history_index
        if hu.numel() > 0:
            scores[hu.to(device), hi.to(device)] = -np.inf
        return

    if sampler is not None and user_ids is not None and positive_i is not None:
        used_ids = sampler.used_ids
        for j in range(user_ids.shape[0]):
            uid = int(user_ids[j].item())
            pos = int(positive_i[j].item())
            hist = _history_items_for_user(used_ids, uid)
            hist.discard(pos)
            if hist:
                idx = torch.tensor(list(hist), device=device, dtype=torch.long)
                scores[j, idx] = -np.inf
        return

    if bundle.sequential:
        cfg = bundle.config
        list_suffix = cfg["LIST_SUFFIX"]
        seq_field = cfg["ITEM_ID_FIELD"] + list_suffix
        len_field = cfg["ITEM_LIST_LENGTH_FIELD"]
        item_seq = interaction[seq_field].to(device)
        lengths = interaction[len_field].to(device)
        targets = interaction[cfg["ITEM_ID_FIELD"]].to(device)
        batch_size = scores.shape[0]
        for j in range(batch_size):
            length = int(lengths[j].item())
            if length <= 0:
                continue
            hist = item_seq[j, :length]
            hist = hist[hist != targets[j]]
            hist = hist[hist > 0]
            if hist.numel() > 0:
                scores[j, hist] = -np.inf


@torch.no_grad()
def _full_sort_scores(
    bundle: ModelBundle,
    interaction,
    history_index,
    *,
    user_ids: torch.Tensor,
    positive_i: torch.Tensor,
    sampler,
) -> torch.Tensor:
    device = next(bundle.model.parameters()).device
    n_items = bundle.dataset.item_num

    if bundle.sequential:
        model_inter = interaction.to(device)
    else:
        from recbole.data.interaction import Interaction

        uid_field = bundle.config["USER_ID_FIELD"]
        model_inter = Interaction({uid_field: user_ids.to(device)})

    scores = bundle.model.full_sort_predict(model_inter)
    if scores.dim() == 1:
        scores = scores.view(-1, n_items)
    _mask_history_on_scores(
        scores,
        bundle=bundle,
        interaction=interaction,
        history_index=history_index,
        user_ids=user_ids,
        positive_i=positive_i,
        sampler=sampler,
    )
    return scores


def _prepare_general_samplers(
    bundles: list[ModelBundle],
    driver: ModelBundle,
    driver_built,
    phase: str,
) -> dict[str, Any]:
    from recbole.data.utils import create_samplers

    out: dict[str, Any] = {}
    for bundle in bundles:
        if bundle.sequential:
            continue
        if bundle.dataset is driver.dataset:
            built = driver_built
        else:
            built = bundle.dataset.build()
        _, valid_sampler, test_sampler = create_samplers(
            bundle.config, bundle.dataset, built
        )
        out[bundle.key] = test_sampler if phase == "test" else valid_sampler
    return out


@torch.no_grad()
def eval_rrf(
    *,
    bundles: list[ModelBundle],
    driver: ModelBundle,
    phase: str,
    user_cap: int,
    seed: int,
    rrf_k: float,
    model_weights: dict[str, float],
) -> dict[str, Any]:
    snaps = {
        b.key: _apply_full_catalog_phase(b.dataset, b.config, phase, user_cap, seed)
        for b in bundles
    }
    snap_drv = snaps[driver.key]
    loader, driver_built = _build_full_sort_loader(driver, phase)
    samplers = _prepare_general_samplers(bundles, driver, driver_built, phase)

    bundle_map = {b.key: b for b in bundles}
    device = driver.trainer.device
    all_ranks: list[int] = []

    desc = f"RRF k={rrf_k} ({phase})"
    batch_iter = loader
    if SHOW_PROGRESS:
        batch_iter = tqdm(loader, desc=desc, total=len(loader))

    for batched in batch_iter:
        interaction, history_index, _positive_u, positive_i = batched
        batch_size = interaction.length if driver.sequential else len(
            interaction[driver.config["USER_ID_FIELD"]]
        )
        user_ids = interaction[driver.config["USER_ID_FIELD"]]
        positive_i = positive_i.to(device)

        rrf_scores = torch.zeros(
            batch_size, driver.dataset.item_num, device=device, dtype=torch.float32
        )

        for spec in MODELS:
            key = str(spec["key"]).lower()
            weight = float(model_weights.get(key, spec.get("weight", 1.0)))
            if weight == 0.0:
                continue
            bundle = bundle_map[key]
            sampler = samplers.get(key)
            scores = _full_sort_scores(
                bundle,
                interaction,
                history_index,
                user_ids=user_ids,
                positive_i=positive_i,
                sampler=sampler,
            )
            ranks = _rank_from_scores(scores)
            rrf_scores += weight / (float(rrf_k) + ranks)

        _mask_history_on_scores(
            rrf_scores,
            bundle=driver,
            interaction=interaction,
            history_index=history_index,
            user_ids=user_ids,
            positive_i=positive_i,
            sampler=samplers.get(driver.key) if not driver.sequential else None,
        )

        order = torch.argsort(rrf_scores, dim=1, descending=True)
        ranks = _rank_positions(order, positive_i)
        for j in range(batch_size):
            all_ranks.append(int(ranks[j].item()))

    for b in bundles:
        _restore_phase(b.dataset, snaps[b.key])

    ranks_arr = np.array(all_ranks, dtype=np.int64)
    metrics = _metrics_from_ranks(ranks_arr, METRIC_TOPK)
    metrics["n_users"] = len(all_ranks)
    return {
        "mode": "rrf",
        "phase": phase,
        "rrf_k": float(rrf_k),
        "models": [m["key"] for m in MODELS],
        "weights": model_weights,
        "metrics": metrics,
    }


@torch.no_grad()
def eval_single_same_protocol(
    *,
    bundle: ModelBundle,
    driver: ModelBundle,
    phase: str,
    user_cap: int,
    seed: int,
) -> dict[str, Any]:
    """与 RRF 相同 loader / mask，单模型全库排序。"""
    snap_b = _apply_full_catalog_phase(bundle.dataset, bundle.config, phase, user_cap, seed)
    snap_d = _apply_full_catalog_phase(driver.dataset, driver.config, phase, user_cap, seed)
    loader, driver_built = _build_full_sort_loader(driver, phase)
    samplers = _prepare_general_samplers([bundle], driver, driver_built, phase)
    sampler = samplers.get(bundle.key)

    device = bundle.trainer.device
    all_ranks: list[int] = []
    batch_iter = loader
    if SHOW_PROGRESS:
        batch_iter = tqdm(loader, desc=f"single {bundle.key} ({phase})", total=len(loader))

    for batched in batch_iter:
        interaction, history_index, _positive_u, positive_i = batched
        batch_size = interaction.length if driver.sequential else len(
            interaction[driver.config["USER_ID_FIELD"]]
        )
        user_ids = interaction[driver.config["USER_ID_FIELD"]]
        positive_i = positive_i.to(device)

        scores = _full_sort_scores(
            bundle,
            interaction,
            history_index,
            user_ids=user_ids,
            positive_i=positive_i,
            sampler=sampler,
        )
        order = torch.argsort(scores, dim=1, descending=True)
        ranks = _rank_positions(order, positive_i)
        for j in range(batch_size):
            all_ranks.append(int(ranks[j].item()))

    _restore_phase(bundle.dataset, snap_b)
    _restore_phase(driver.dataset, snap_d)

    ranks_arr = np.array(all_ranks, dtype=np.int64)
    metrics = _metrics_from_ranks(ranks_arr, METRIC_TOPK)
    metrics["n_users"] = len(all_ranks)
    return {
        "mode": "single",
        "model": bundle.key,
        "phase": phase,
        "metrics": metrics,
    }


def _metric_value(metrics: dict[str, float], key: str) -> float:
    return float(metrics.get(key, metrics.get(key.replace("hit", "hr"), float("-inf"))))


def _print_summary(rows: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 78)
    print("RRF / 单模型全库对比（同 mask 协议）")
    print("=" * 78)
    header = f"{'名称':<28} {'阶段':<6} {'HR@10':>8} {'NDCG@10':>9} {'MRR@10':>8} {'HR@50':>8}"
    print(header)
    print("-" * len(header))
    for row in rows:
        name = row.get("name", "?")
        phase = row.get("phase", "?")
        m = row.get("metrics") or {}
        print(
            f"{name:<28} {phase:<6} "
            f"{m.get('hr@10', float('nan')):>8.4f} "
            f"{m.get('ndcg@10', float('nan')):>9.4f} "
            f"{m.get('mrr@10', float('nan')):>8.4f} "
            f"{m.get('hr@50', float('nan')):>8.4f}"
        )
        if row.get("rrf_k") is not None:
            print(f"    └ rrf_k={row['rrf_k']}")


def main() -> None:
    from run_train import _apply_recbole_dynamic_neg_device_fix, _apply_scipy_dok_compat

    _apply_scipy_dok_compat()
    _apply_recbole_dynamic_neg_device_fix()

    cap_note = "全量" if USER_CAP <= 0 else str(USER_CAP)
    model_keys = [str(m["key"]).lower() for m in MODELS]
    weights = {str(m["key"]).lower(): float(m.get("weight", 1.0)) for m in MODELS}

    print(
        f"RRF eval | driver={DRIVER} | models={model_keys} | "
        f"users={cap_note} | seed={EVAL_SAMPLE_SEED} | tune={TUNE_ON_VALID} | "
        f"mask_history={MASK_INTERACTION_HISTORY}"
    )

    bundles: list[ModelBundle] = []
    for spec in MODELS:
        key = str(spec["key"]).lower()
        print(f"\n>>> load {key}")
        bundles.append(load_model_bundle(key, spec.get("checkpoint")))
    driver_key = DRIVER.lower()
    driver = next((b for b in bundles if b.key == driver_key), None)
    if driver is None:
        print(f"\n>>> load driver {driver_key}")
        driver = load_model_bundle(driver_key, None)
        bundles.append(driver)

    results: list[dict[str, Any]] = []
    best_k = float(RRF_K)
    best_tune_score = float("-inf")

    if TUNE_ON_VALID:
        print(f"\n>>> tune RRF k on valid | grid={RRF_K_GRID} | metric={TUNE_METRIC}")
        tune_rows: list[tuple[float, dict[str, Any]]] = []
        for k in RRF_K_GRID:
            out = eval_rrf(
                bundles=bundles,
                driver=driver,
                phase="valid",
                user_cap=int(USER_CAP),
                seed=int(EVAL_SAMPLE_SEED),
                rrf_k=float(k),
                model_weights=weights,
            )
            score = _metric_value(out["metrics"], TUNE_METRIC)
            tune_rows.append((score, out))
            print(
                f"  k={k:>4} | {TUNE_METRIC}={score:.4f} | "
                f"HR@10={out['metrics'].get('hr@10', 0):.4f}"
            )
        tune_rows.sort(key=lambda x: x[0], reverse=True)
        best_tune_score, best_out = tune_rows[0]
        best_k = float(best_out["rrf_k"])
        best_out["name"] = f"RRF best k={best_k}"
        results.append(best_out)
        print(f"\n  best valid: k={best_k} | {TUNE_METRIC}={best_tune_score:.4f}")
    else:
        best_k = float(RRF_K)

    print(f"\n>>> RRF on test | k={best_k}")
    test_rrf = eval_rrf(
        bundles=bundles,
        driver=driver,
        phase="test",
        user_cap=int(USER_CAP),
        seed=int(EVAL_SAMPLE_SEED),
        rrf_k=best_k,
        model_weights=weights,
    )
    test_rrf["name"] = f"RRF k={best_k}"
    results.append(test_rrf)

    if RUN_SINGLE_BASELINES:
        print("\n>>> single-model baselines (same protocol, test)")
        for bundle in bundles:
            single = eval_single_same_protocol(
                bundle=bundle,
                driver=driver,
                phase="test",
                user_cap=int(USER_CAP),
                seed=int(EVAL_SAMPLE_SEED),
            )
            single["name"] = f"single {bundle.key}"
            single["phase"] = "test"
            results.append(single)

    _print_summary(results)

    if SAVE_JSON:
        out_dir = os.path.join(PLATFORM_ROOT, "results", "logs")
        os.makedirs(out_dir, exist_ok=True)
        tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(out_dir, f"rrf_rank_{tag}.json")
        payload = {
            "driver": DRIVER,
            "models": MODELS,
            "user_cap": USER_CAP,
            "eval_sample_seed": EVAL_SAMPLE_SEED,
            "rrf_k_grid": RRF_K_GRID,
            "best_rrf_k": best_k,
            "tune_on_valid": TUNE_ON_VALID,
            "tune_metric": TUNE_METRIC,
            "mask_interaction_history": MASK_INTERACTION_HISTORY,
            "runs": results,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        print(f"\njson: {out_path}")


if __name__ == "__main__":
    main()
