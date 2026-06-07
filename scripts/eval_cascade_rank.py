#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
粗排 → 精排 全库（或 capped 用户）评估。

在 recbole_platform 目录下:
  python scripts/eval_cascade_rank.py

可调区域支持:
  - 多组 RUNS 对比（cascade 或 single 全库基线）
  - 自选粗排/精排模型、checkpoint、粗排候选数 M
  - 分数融合 fusion（可选）
  - USER_CAP=0 全量用户，或 20000 等与主表一致
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

PLATFORM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PLATFORM_ROOT not in sys.path:
    sys.path.insert(0, PLATFORM_ROOT)
os.chdir(PLATFORM_ROOT)

# ========================= 可调区域（只改这里）=========================
PHASE = "test"  # test | valid
USER_CAP = 20000  # 0=全量 test 用户（~657k）；20000=与主表 cap 一致
EVAL_SAMPLE_SEED = 42

METRIC_TOPK = [10, 50]  # 输出 HR / NDCG / MRR @K

# 粗排未召回正例时：False=记 0（真实级联）；True=强制把正例并入候选（调试用）
ENSURE_POS_IN_CANDIDATES = False

# 多组实验；mode=cascade 需 coarse+fine；mode=single 只需 model
RUNS: list[dict[str, Any]] = [
    {
        "name": "Pop→SASRec@500",
        "mode": "cascade",
        "coarse": "pop",
        "fine": "sasrec",
        "coarse_topk": 500,
        "coarse_ckpt": None,  # None → yaml checkpoint_dir/best.pth
        "fine_ckpt": None,
        "fusion": None,  # 例: {"coarse": 0.2, "fine": 0.8} 对候选内归一化后加权
    },
]

CHUNK_USERS = 50000  # 分块用户数（0=不分块）
FINE_SCORE_BATCH = 512  # 精排对候选打分的 pair 数上限 / batch
SHOW_PROGRESS = True
SAVE_JSON = True  # results/logs/cascade_rank_<timestamp>.json
# ========================================================================

UNSUPPORTED = frozenset({"itemknn", "crossdomain_neumf"})


@dataclass
class ModelBundle:
    key: str
    config: Any
    dataset: Any
    model: torch.nn.Module
    trainer: Any
    sequential: bool


def _resolve_ckpt(model_key: str, checkpoint: str | None) -> str:
    from run_train import BEST_CKPT_NAME, load_yaml, pick_config

    if checkpoint:
        path = checkpoint
    else:
        cfg = load_yaml(pick_config(model_key))
        path = os.path.join(cfg["checkpoint_dir"], BEST_CKPT_NAME)
    if not os.path.isabs(path):
        path = os.path.join(PLATFORM_ROOT, path)
    return path


def load_model_bundle(model_key: str, checkpoint: str | None = None) -> ModelBundle:
    from run_train import (
        _apply_recbole_dynamic_neg_device_fix,
        _apply_scipy_dok_compat,
        _bind_single_best_checkpoint,
        _patch_neumf_full_sort_predict,
        _patch_pop_full_sort_eval_memory,
        apply_eval_user_caps,
        apply_negative_sampling_config,
        apply_sequential_model_config,
        ensure_sequential_benchmark_dataset,
        is_sequential_model,
        load_checkpoint_weights,
        merge_negative_sampling_defaults,
        pick_config,
        strip_platform_only_config_keys,
        load_yaml,
    )
    from recbole.config import Config
    from recbole.data import create_dataset
    from recbole.utils import get_model, get_trainer, init_logger, init_seed

    key = model_key.lower().strip()
    if key in UNSUPPORTED:
        raise ValueError(
            f"模型 {model_key!r} 暂不支持本脚本（请用 pop/bpr/neumf/lightgcn/sasrec）"
        )

    cfg = strip_platform_only_config_keys(
        apply_sequential_model_config(
            apply_negative_sampling_config(
                merge_negative_sampling_defaults(load_yaml(pick_config(key)))
            )
        )
    )
    ensure_sequential_benchmark_dataset(cfg)
    ckpt = _resolve_ckpt(key, checkpoint)
    if key == "neumf":
        from run_train import align_neumf_config_from_checkpoint

        cfg = align_neumf_config_from_checkpoint(cfg, ckpt)
    model_name = cfg["model"]
    config = Config(model=model_name, dataset=cfg["dataset"], config_dict=cfg)
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)

    dataset = create_dataset(config)
    apply_eval_user_caps(dataset, config)
    model = get_model(model_name)(config, dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], model_name)(config, model)
    _bind_single_best_checkpoint(trainer, config["checkpoint_dir"], model_name=model_name)
    _patch_neumf_full_sort_predict(model)
    _patch_pop_full_sort_eval_memory(model)

    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"{key}: 找不到权重 {ckpt}")
    load_checkpoint_weights(trainer, ckpt, verbose=True)

    model.eval()
    return ModelBundle(
        key=key,
        config=config,
        dataset=dataset,
        model=model,
        trainer=trainer,
        sequential=is_sequential_model(model_name),
    )


def _apply_full_catalog_phase(dataset_obj, config, phase: str, user_cap: int, seed: int):
    from run_train import _swap_inter_feat_for_full_catalog

    snap = _swap_inter_feat_for_full_catalog(
        dataset_obj,
        phase,
        user_cap=user_cap if user_cap > 0 else None,
        config=config,
    )
    return snap


def _restore_phase(dataset_obj, snap):
    from run_train import _restore_inter_feat_snapshot

    _restore_inter_feat_snapshot(dataset_obj, snap)


def _build_full_sort_loader(bundle: ModelBundle, phase: str, *, prebuilt=None):
    from recbole.data.utils import create_samplers, get_dataloader

    from run_train import _apply_full_catalog_eval_settings

    config = bundle.config
    dataset = bundle.dataset
    _apply_full_catalog_eval_settings(
        config, phase=phase, topk_list=[max(METRIC_TOPK)], metrics=["Hit"]
    )
    built = prebuilt if prebuilt is not None else dataset.build()
    train_dataset, valid_dataset, test_dataset = built
    dataset_map = {"train": train_dataset, "valid": valid_dataset, "test": test_dataset}
    train_sampler, valid_sampler, test_sampler = create_samplers(config, dataset, built)
    sampler_map = {
        "train": train_sampler,
        "valid": valid_sampler,
        "test": test_sampler,
    }
    loader_cls = get_dataloader(config, phase)
    loader = loader_cls(
        config, dataset_map[phase], sampler_map[phase], shuffle=False
    )
    return loader, built


def _history_items_for_user(used_ids, uid: int) -> set[int]:
    """RecBole sampler.used_ids[uid] 可能是 set / ndarray / list。"""
    raw = used_ids[uid]
    if isinstance(raw, set):
        return {int(x) for x in raw}
    if hasattr(raw, "tolist"):
        return {int(x) for x in raw.tolist()}
    return {int(x) for x in raw}


@torch.no_grad()
def _coarse_full_scores(
    bundle: ModelBundle,
    interaction,
    history_index,
    *,
    user_ids: torch.Tensor | None = None,
    positive_i: torch.Tensor | None = None,
    sampler=None,
) -> torch.Tensor:
    """返回 [batch_users, n_items] 粗排分（已 mask pad 与 history）。"""
    model = bundle.model
    device = next(model.parameters()).device
    n_items = bundle.dataset.item_num

    if bundle.sequential:
        scores = model.full_sort_predict(interaction.to(device))
        if scores.dim() == 1:
            scores = scores.view(-1, n_items)
    else:
        scores = model.full_sort_predict(interaction.to(device))
        scores = scores.view(-1, n_items)

    scores[:, 0] = -np.inf
    if history_index is not None:
        hu, hi = history_index
        if hu.numel() > 0:
            scores[hu.to(device), hi.to(device)] = -np.inf
    elif sampler is not None and user_ids is not None and positive_i is not None:
        used_ids = sampler.used_ids
        for j in range(user_ids.shape[0]):
            uid = int(user_ids[j].item())
            pos = int(positive_i[j].item())
            hist = _history_items_for_user(used_ids, uid)
            hist.discard(pos)
            if hist:
                idx = torch.tensor(list(hist), device=device, dtype=torch.long)
                scores[j, idx] = -np.inf
    return scores


@torch.no_grad()
def _fine_score_candidates_general(
    bundle: ModelBundle,
    user_ids: torch.Tensor,
    cand_items: torch.Tensor,
) -> torch.Tensor:
    """user_ids [B], cand_items [B, M] → fine scores [B, M]。"""
    model = bundle.model
    device = next(model.parameters()).device
    uid_field = bundle.config["USER_ID_FIELD"]
    iid_field = bundle.config["ITEM_ID_FIELD"]
    from recbole.data.interaction import Interaction

    b, m = cand_items.shape
    users = user_ids.unsqueeze(1).expand(b, m).reshape(-1)
    items = cand_items.reshape(-1)
    scores = torch.empty(b * m, device=device)
    bs = max(1, int(FINE_SCORE_BATCH))
    for start in range(0, b * m, bs):
        end = min(start + bs, b * m)
        inter = Interaction(
            {
                uid_field: users[start:end],
                iid_field: items[start:end],
            }
        )
        scores[start:end] = model.predict(inter.to(device))
    return scores.view(b, m)


@torch.no_grad()
def _fine_score_candidates_sequential(
    bundle: ModelBundle,
    base_interaction,
    cand_items: torch.Tensor,
) -> torch.Tensor:
    """base_interaction 为 SASRec eval 行；cand_items [B, M]。"""
    model = bundle.model
    device = next(model.parameters()).device
    iid_field = bundle.config["ITEM_ID_FIELD"]
    from recbole.data.interaction import Interaction

    b, m = cand_items.shape
    scores = torch.empty(b, m, device=device)
    for i in range(b):
        row = base_interaction[i : i + 1]
        items = cand_items[i]
        row_scores = []
        bs = max(1, int(FINE_SCORE_BATCH))
        for start in range(0, m, bs):
            end = min(m, start + bs)
            chunk_items = items[start:end]
            n = len(chunk_items)
            expanded = row.repeat(n)
            expanded[iid_field] = chunk_items.to(device)
            row_scores.append(model.predict(expanded.to(device)))
        scores[i] = torch.cat(row_scores)
    return scores


def _normalize_rows(x: torch.Tensor) -> torch.Tensor:
    xmin = x.min(dim=1, keepdim=True).values
    xmax = x.max(dim=1, keepdim=True).values
    return (x - xmin) / (xmax - xmin + 1e-12)


def _rank_positions(sorted_item_ids: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
    """每个用户正例在 sorted_item_ids [B, L] 中的 1-based rank；未命中为 0。"""
    b, length = sorted_item_ids.shape
    ranks = torch.zeros(b, dtype=torch.long, device=sorted_item_ids.device)
    for i in range(b):
        pos = positive[i].item()
        hit = (sorted_item_ids[i] == pos).nonzero(as_tuple=False)
        if hit.numel() > 0:
            ranks[i] = int(hit[0, 0].item()) + 1
    return ranks


def _metrics_from_ranks(ranks: np.ndarray, topk_list: list[int]) -> dict[str, float]:
    out: dict[str, float] = {}
    n = len(ranks)
    if n == 0:
        return out
    for k in topk_list:
        hit = (ranks > 0) & (ranks <= k)
        hr = float(hit.mean())
        ndcg_vals = np.zeros(n, dtype=np.float64)
        mrr_vals = np.zeros(n, dtype=np.float64)
        if hit.any():
            ndcg_vals[hit] = 1.0 / np.log2(ranks[hit] + 1)
            mrr_vals[hit] = 1.0 / ranks[hit]
        out[f"hr@{k}"] = hr
        out[f"ndcg@{k}"] = float(ndcg_vals.sum() / n)
        out[f"mrr@{k}"] = float(mrr_vals.sum() / n)
    return out


@torch.no_grad()
def eval_cascade_run(
    *,
    coarse: ModelBundle,
    fine: ModelBundle,
    coarse_topk: int,
    phase: str,
    user_cap: int,
    seed: int,
    fusion: dict[str, float] | None,
    ensure_pos: bool,
) -> dict[str, Any]:
    driver = fine if fine.sequential else coarse
    snap = _apply_full_catalog_phase(driver.dataset, driver.config, phase, user_cap, seed)
    loader, driver_built = _build_full_sort_loader(driver, phase)

    coarse_sampler = None
    if not coarse.sequential:
        from recbole.data.utils import create_samplers

        if coarse.dataset is driver.dataset:
            coarse_built = driver_built
        else:
            coarse_built = coarse.dataset.build()
        _, valid_sampler, test_sampler = create_samplers(
            coarse.config, coarse.dataset, coarse_built
        )
        coarse_sampler = test_sampler if phase == "test" else valid_sampler

    n_users = len(loader)
    print(
        f"  cascade: {coarse.key} → {fine.key} | M={coarse_topk} | "
        f"users={n_users} | phase={phase} | cap={user_cap or '全量'}"
    )

    all_ranks: list[int] = []
    coarse_hit = 0
    total = 0
    device = coarse.trainer.device

    batch_iter = loader
    if SHOW_PROGRESS:
        batch_iter = tqdm(loader, desc=f"{coarse.key}→{fine.key}", total=len(loader))

    for batched in batch_iter:
        interaction, history_index, positive_u, positive_i = batched
        if driver.sequential:
            batch_size = interaction.length
            user_ids = interaction[driver.config["USER_ID_FIELD"]]
        else:
            batch_size = len(interaction[driver.config["USER_ID_FIELD"]])
            user_ids = interaction[driver.config["USER_ID_FIELD"]]

        coarse_inter = interaction if coarse.sequential else interaction
        if not coarse.sequential and fine.sequential:
            uid_field = coarse.config["USER_ID_FIELD"]
            from recbole.data.interaction import Interaction

            coarse_inter = Interaction({uid_field: user_ids})

        coarse_scores = _coarse_full_scores(
            coarse,
            coarse_inter,
            history_index,
            user_ids=user_ids,
            positive_i=positive_i,
            sampler=coarse_sampler,
        )
        m = min(int(coarse_topk), coarse_scores.shape[1] - 1)
        _, coarse_top_idx = torch.topk(coarse_scores, k=m, dim=1)

        if ensure_pos:
            for j in range(batch_size):
                pos = positive_i[j].item()
                if pos not in coarse_top_idx[j].tolist():
                    coarse_top_idx[j, -1] = pos

        coarse_top_scores = coarse_scores.gather(1, coarse_top_idx)

        if fine.sequential:
            fine_scores = _fine_score_candidates_sequential(fine, interaction, coarse_top_idx)
        else:
            fine_scores = _fine_score_candidates_general(fine, user_ids.to(device), coarse_top_idx)

        if fusion:
            wc = float(fusion.get("coarse", 0.5))
            wf = float(fusion.get("fine", 0.5))
            combined = wc * _normalize_rows(coarse_top_scores) + wf * _normalize_rows(
                fine_scores
            )
            order = torch.argsort(combined, dim=1, descending=True)
        else:
            order = torch.argsort(fine_scores, dim=1, descending=True)

        sorted_cands = coarse_top_idx.gather(1, order)
        ranks = _rank_positions(sorted_cands, positive_i.to(device))

        for j in range(batch_size):
            r = int(ranks[j].item())
            all_ranks.append(r)
            pos = int(positive_i[j].item())
            if (coarse_top_idx[j] == pos).any():
                coarse_hit += 1
            total += 1

    _restore_phase(driver.dataset, snap)

    ranks_arr = np.array(all_ranks, dtype=np.int64)
    metrics = _metrics_from_ranks(ranks_arr, METRIC_TOPK)
    metrics["coarse_recall@M"] = coarse_hit / max(total, 1)
    metrics["n_users"] = total
    return {
        "mode": "cascade",
        "coarse": coarse.key,
        "fine": fine.key,
        "coarse_topk": coarse_topk,
        "fusion": fusion,
        "metrics": metrics,
    }


def eval_single_run(
    *,
    model_key: str,
    checkpoint: str | None,
    phase: str,
    user_cap: int,
    seed: int,
) -> dict[str, Any]:
    from run_train import build_full_catalog_metric_plan, run_full_catalog_hr_eval

    bundle = load_model_bundle(model_key, checkpoint)
    snap = _apply_full_catalog_phase(bundle.dataset, bundle.config, phase, user_cap, seed)
    plan = build_full_catalog_metric_plan(
        compute_hr_10=10 in METRIC_TOPK,
        compute_hr_50=50 in METRIC_TOPK,
        compute_ndcg=True,
        compute_mrr=True,
        compute_meanrank=False,
    )
    platform_eval = {
        "full_catalog_eval_user_cap": user_cap,
    }
    print(
        f"  single: {model_key} | users cap={user_cap or '全量'} | phase={phase}"
    )
    chunk = 0
    if CHUNK_USERS > 0 and (user_cap is None or int(user_cap) <= 0):
        chunk = int(CHUNK_USERS)
    result = run_full_catalog_hr_eval(
        bundle.trainer,
        bundle.dataset,
        bundle.config,
        metric_plan=plan,
        phases=[phase],
        show_progress=SHOW_PROGRESS,
        full_catalog_user_cap=platform_eval.get("full_catalog_eval_user_cap"),
        chunk_users=chunk,
        load_best_model=False,
    )
    _restore_phase(bundle.dataset, snap)
    metrics = result.get(phase, {})
    return {
        "mode": "single",
        "model": model_key,
        "metrics": metrics,
    }


def _print_summary(results: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 72)
    print("粗排精排 / 全库基线 对比")
    print("=" * 72)
    header = f"{'名称':<22} {'模式':<8} {'HR@10':>8} {'NDCG@10':>9} {'MRR@10':>8} {'HR@50':>8}"
    print(header)
    print("-" * len(header))
    for row in results:
        name = row.get("name", "?")
        mode = row.get("mode", "?")
        m = row.get("metrics") or {}
        print(
            f"{name:<22} {mode:<8} "
            f"{m.get('hr@10', m.get('hit@10', float('nan'))):>8.4f} "
            f"{m.get('ndcg@10', float('nan')):>9.4f} "
            f"{m.get('mrr@10', float('nan')):>8.4f} "
            f"{m.get('hr@50', m.get('hit@50', float('nan'))):>8.4f}"
        )
        if mode == "cascade" and "coarse_recall@M" in m:
            print(f"    └ coarse_recall@M = {m['coarse_recall@M']:.4f}")


def main() -> None:
    from run_train import _apply_recbole_dynamic_neg_device_fix, _apply_scipy_dok_compat

    _apply_scipy_dok_compat()
    _apply_recbole_dynamic_neg_device_fix()

    cap_note = "全量" if USER_CAP <= 0 else str(USER_CAP)
    print(
        f"cascade-rank eval | phase={PHASE} | users={cap_note} | "
        f"seed={EVAL_SAMPLE_SEED} | topk={METRIC_TOPK} | runs={len(RUNS)}"
    )

    results: list[dict[str, Any]] = []

    for spec in RUNS:
        name = spec.get("name") or "run"
        mode = str(spec.get("mode", "cascade")).lower()
        print(f"\n>>> [{name}]")
        try:
            if mode == "single":
                out = eval_single_run(
                    model_key=str(spec["model"]).lower(),
                    checkpoint=spec.get("checkpoint"),
                    phase=PHASE,
                    user_cap=int(USER_CAP),
                    seed=int(EVAL_SAMPLE_SEED),
                )
            elif mode == "cascade":
                coarse_key = str(spec["coarse"]).lower()
                fine_key = str(spec["fine"]).lower()
                coarse_b = load_model_bundle(coarse_key, spec.get("coarse_ckpt"))
                fine_b = load_model_bundle(fine_key, spec.get("fine_ckpt"))
                out = eval_cascade_run(
                    coarse=coarse_b,
                    fine=fine_b,
                    coarse_topk=int(spec.get("coarse_topk", 500)),
                    phase=PHASE,
                    user_cap=int(USER_CAP),
                    seed=int(EVAL_SAMPLE_SEED),
                    fusion=spec.get("fusion"),
                    ensure_pos=bool(ENSURE_POS_IN_CANDIDATES),
                )
            else:
                raise ValueError(f"未知 mode={mode!r}，应为 cascade 或 single")
            out["name"] = name
            results.append(out)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            results.append({"name": name, "mode": mode, "error": str(exc)})

    _print_summary(results)

    if SAVE_JSON:
        out_dir = os.path.join(PLATFORM_ROOT, "results", "logs")
        os.makedirs(out_dir, exist_ok=True)
        tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(out_dir, f"cascade_rank_{tag}.json")
        payload = {
            "phase": PHASE,
            "user_cap": USER_CAP,
            "eval_sample_seed": EVAL_SAMPLE_SEED,
            "metric_topk": METRIC_TOPK,
            "runs": results,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        print(f"\njson: {out_path}")


if __name__ == "__main__":
    main()
