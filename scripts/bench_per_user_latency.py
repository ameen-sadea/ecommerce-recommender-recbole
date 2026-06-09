#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单用户推理/排序耗时基准（直接看「每个用户要多久」）。

在 recbole_platform 目录下运行:
  python scripts/bench_per_user_latency.py

输出示例:
  模型                 任务            均值(ms)  中位(ms)  P90(ms)  吞吐(u/s)
  sasrec               全库排序         0.017     0.015     0.021    58.8
  bpr                  全库排序         0.120     0.118     0.135     8.3
  crossdomain_neumf    全库排序         8.512     8.401     9.102     0.12

说明:
  - 「全库排序」= 对该用户在全部 item 上打分（与 full-catalog 评估一致）
  - 「uni100」= 1 正例 + 100 负例候选打分（与主表 LOO 评估一致）
  - 计时含 GPU 同步；不含 DataLoader 取 batch 的开销
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import numpy as np
import torch

PLATFORM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PLATFORM_ROOT not in sys.path:
    sys.path.insert(0, PLATFORM_ROOT)
os.chdir(PLATFORM_ROOT)

# ========================= 可调区域（只改这里）=========================
# 要测的模型；crossdomain_neumf 单独走 CrossDomain 路径
MODELS: list[str] = [
    "pop",
    "lightgcn",
    "sasrec",
 
]

# full_catalog | uni100 | both
TASKS: list[str] = ["full_catalog", "uni100"]

PHASE = "test"  # test | valid
SAMPLE_USERS = 100  # 统计用样本用户数（不含 warmup）
WARMUP_USERS = 5  # 预热，不计入统计
EVAL_SAMPLE_SEED = 42

# crossdomain 全库打分的 item 批大小（与 eval_crossdomain_full_catalog 一致）
CROSSDOMAIN_ITEM_BATCH = 8192

# 各模型 checkpoint；None → 用 yaml 默认 best.pth / best_ckpt.pt
CHECKPOINTS: dict[str, str | None] = {}

SAVE_JSON = True
# ========================================================================


def _sync_device() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@dataclass
class LatencyStats:
    n: int
    mean_s: float
    median_s: float
    p90_s: float
    p99_s: float
    min_s: float
    max_s: float

    @property
    def users_per_sec(self) -> float:
        return 0.0 if self.mean_s <= 0 else 1.0 / self.mean_s


def _summarize(times: list[float]) -> LatencyStats | None:
    if not times:
        return None
    arr = np.asarray(times, dtype=np.float64)
    return LatencyStats(
        n=int(arr.size),
        mean_s=float(arr.mean()),
        median_s=float(np.median(arr)),
        p90_s=float(np.percentile(arr, 90)),
        p99_s=float(np.percentile(arr, 99)),
        min_s=float(arr.min()),
        max_s=float(arr.max()),
    )


def _print_header() -> None:
    print("\n" + "=" * 88)
    print("单用户推理耗时（per-user latency）")
    print("=" * 88)
    print(
        f"{'模型':<20} {'任务':<12} {'均值(ms)':>10} {'中位(ms)':>10} "
        f"{'P90(ms)':>10} {'P99(ms)':>10} {'吞吐(u/s)':>10} {'n':>6}"
    )
    print("-" * 88)


def _print_row(model: str, task: str, stats: LatencyStats, device: str) -> None:
    print(
        f"{model:<20} {task:<12} "
        f"{stats.mean_s * 1000:>10.3f} {stats.median_s * 1000:>10.3f} "
        f"{stats.p90_s * 1000:>10.3f} {stats.p99_s * 1000:>10.3f} "
        f"{stats.users_per_sec:>10.1f} {stats.n:>6}  [{device}]"
    )


@torch.no_grad()
def _recbole_score_one_user_full_catalog(
    bundle,
    interaction,
    history_index,
    user_idx: int,
) -> None:
    """对 batch 中第 user_idx 个用户做全库打分（含 history mask）。"""
    model = bundle.model
    device = bundle.trainer.device
    n_items = bundle.dataset.item_num

    row = interaction[user_idx : user_idx + 1]
    scores = model.full_sort_predict(row.to(device))
    if scores.dim() == 1:
        scores = scores.view(1, -1)
    else:
        scores = scores.view(-1, n_items)
    scores[:, 0] = -np.inf

    if history_index is not None:
        hu, hi = history_index
        mask = hu == user_idx
        if mask.any():
            items = hi[mask].to(device)
            scores[0, items] = -np.inf


@torch.no_grad()
def _recbole_score_one_user_uni100(
    bundle,
    interaction,
    user_idx: int,
    *,
    positive_u: torch.Tensor,
) -> None:
    """对 batch 中第 user_idx 个用户在 1+num_neg 候选上 predict（非序列模型）。"""
    model = bundle.model
    device = bundle.trainer.device
    uid_field = bundle.config["USER_ID_FIELD"]

    uids = interaction[uid_field]
    target_uid = positive_u[user_idx]
    mask = uids == target_uid
    pairs = interaction[mask]
    model.predict(pairs.to(device))


def bench_recbole(
    model_key: str,
    *,
    task: str,
    phase: str,
    n_users: int,
    warmup: int,
    seed: int,
    checkpoint: str | None,
) -> LatencyStats | None:
    from scripts.eval_cascade_rank import load_model_bundle

    from run_train import (
        _restore_eval_config,
        _restore_inter_feat_snapshot,
        build_full_catalog_dataloader,
        build_full_catalog_metric_plan,
    )

    bundle = load_model_bundle(model_key, checkpoint)
    config = bundle.config
    dataset = bundle.dataset
    device = str(config["device"])

    need = n_users + warmup
    times: list[float] = []
    seen = 0

    if task == "full_catalog":
        plan = build_full_catalog_metric_plan(
            compute_hr_10=True,
            compute_hr_50=False,
            compute_ndcg=False,
            compute_mrr=False,
            compute_meanrank=False,
        )
        loader, eval_snap, ds_snap = build_full_catalog_dataloader(
            config,
            dataset,
            phase=phase,
            topk_list=[10],
            full_catalog_user_cap=need,
            metric_plan=plan,
        )
        try:
            for batched in loader:
                interaction, history_index, _positive_u, _positive_i = batched
                uid_field = config["USER_ID_FIELD"]
                batch_size = (
                    interaction.length
                    if bundle.sequential
                    else len(interaction[uid_field])
                )
                for j in range(batch_size):
                    _sync_device()
                    t0 = time.perf_counter()
                    _recbole_score_one_user_full_catalog(
                        bundle, interaction, history_index, j
                    )
                    _sync_device()
                    dt = time.perf_counter() - t0

                    if seen < warmup:
                        seen += 1
                        continue
                    times.append(dt)
                    if len(times) >= n_users:
                        break
                if len(times) >= n_users:
                    break
        finally:
            _restore_eval_config(config, eval_snap)
            _restore_inter_feat_snapshot(dataset, ds_snap)

    elif task == "uni100":
        if bundle.sequential:
            raise NotImplementedError(
                "SASRec 等序列模型的 uni100 需逐候选 predict，"
                "请先用 full_catalog 任务看单用户耗时"
            )

        from recbole.data.utils import create_samplers, get_dataloader
        from run_train import apply_eval_user_caps

        cap_key = "eval_valid_user_cap" if phase == "valid" else "eval_test_user_cap"
        config.final_config_dict[cap_key] = need
        apply_eval_user_caps(dataset, config)

        built = dataset.build()
        train_ds, valid_ds, test_ds = built
        ds_map = {"train": train_ds, "valid": valid_ds, "test": test_ds}
        train_sampler, valid_sampler, test_sampler = create_samplers(
            config, dataset, built
        )
        sampler_map = {
            "train": train_sampler,
            "valid": valid_sampler,
            "test": test_sampler,
        }
        loader = get_dataloader(config, phase)(
            config, ds_map[phase], sampler_map[phase], shuffle=False
        )

        for batched in loader:
            if len(batched) == 4:
                interaction, _history_index, positive_u, positive_i = batched
            else:
                interaction, positive_u, positive_i = batched

            uid_field = config["USER_ID_FIELD"]
            if bundle.sequential:
                batch_size = interaction.length
            else:
                batch_size = len(positive_u)

            for j in range(batch_size):
                _sync_device()
                t0 = time.perf_counter()
                _recbole_score_one_user_uni100(
                    bundle,
                    interaction,
                    j,
                    positive_u=positive_u,
                )
                _sync_device()
                dt = time.perf_counter() - t0

                if seen < warmup:
                    seen += 1
                    continue
                times.append(dt)
                if len(times) >= n_users:
                    break
            if len(times) >= n_users:
                break
    else:
        raise ValueError(f"未知 task={task!r}")

    stats = _summarize(times)
    if stats is not None:
        _print_row(model_key, task, stats, device)
    else:
        print(f"{model_key:<20} {task:<12} （无有效样本）")
    return stats


def _load_crossdomain_bundle(checkpoint: str | None):
    from crossdomain_neumf.data_inter import load_splits_from_recbole_inter
    from crossdomain_neumf.model import CrossDomainNeuMF
    from crossdomain_neumf.platform_bridge import crossdomain_train_config_from_yaml
    from run_train import load_yaml, pick_config

    cfg_path = pick_config("crossdomain_neumf")
    raw_cfg = load_yaml(cfg_path)
    train_cfg = crossdomain_train_config_from_yaml(raw_cfg)

    data_path = raw_cfg.get("data_path", "datasets/")
    dataset_name = raw_cfg.get("dataset", "movies_tv")
    ckpt_dir = raw_cfg.get("checkpoint_dir", "")
    ckpt = checkpoint or os.path.join(ckpt_dir, "best_ckpt.pt")
    if not os.path.isabs(ckpt):
        ckpt = os.path.join(PLATFORM_ROOT, ckpt)
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"找不到权重: {ckpt}")

    train_df, valid_df, test_df, meta = load_splits_from_recbole_inter(
        data_path, dataset_name
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossDomainNeuMF(
        meta["num_global_users"],
        meta["num_global_items"],
        meta["num_domains"],
        embed_dim=int(train_cfg["embed_dim"]),
        domain_embed_dim=int(train_cfg["domain_embed_dim"]),
        mlp_hidden=tuple(train_cfg["mlp_hidden"]),
        dropout=float(train_cfg["dropout"]),
        share_embeddings=bool(train_cfg.get("share_embeddings", True)),
        gmf_domain_aware=bool(train_cfg.get("gmf_domain_aware", True)),
    ).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["state_dict"])
    model.eval()

    return {
        "model": model,
        "device": device,
        "train_df": train_df,
        "valid_df": valid_df,
        "test_df": test_df,
        "meta": meta,
        "checkpoint": ckpt,
    }


def bench_crossdomain(
    *,
    task: str,
    phase: str,
    n_users: int,
    warmup: int,
    seed: int,
    checkpoint: str | None,
) -> LatencyStats | None:
    from crossdomain_neumf.data_inter import cap_eval_users
    from crossdomain_neumf.eval_sampling import (
        build_global_item_pool,
        negative_pool_for_domain,
        sample_eval_negatives,
    )
    from crossdomain_neumf.trainer import _build_domain_pools
    from crossdomain_neumf.full_catalog_eval import (
        build_user_history_from_frames,
        score_all_items_for_user,
    )

    bundle = _load_crossdomain_bundle(checkpoint)
    model = bundle["model"]
    device = bundle["device"]
    meta = bundle["meta"]
    num_items = int(meta["num_global_items"])
    num_users = int(meta["num_global_users"])

    split_df = bundle["valid_df"] if phase == "valid" else bundle["test_df"]
    rng = np.random.RandomState(int(seed))
    need = n_users + warmup
    eval_df = cap_eval_users(split_df, "global_user_id", need, rng)

    history_frames = (
        [bundle["train_df"]]
        if phase == "valid"
        else [bundle["train_df"], bundle["valid_df"]]
    )
    history = build_user_history_from_frames(history_frames, num_users)

    from crossdomain_neumf.platform_bridge import crossdomain_train_config_from_yaml
    from run_train import load_yaml, pick_config

    domain_item_pools = _build_domain_pools(
        bundle["train_df"], bundle["valid_df"], bundle["test_df"]
    )
    global_pool = build_global_item_pool(domain_item_pools)
    train_cfg = crossdomain_train_config_from_yaml(
        load_yaml(pick_config("crossdomain_neumf"))
    )
    num_neg = int(train_cfg.get("eval_negatives", 100))

    times: list[float] = []
    seen = 0

    for row in eval_df.itertuples(index=False):
        u = int(row.global_user_id)
        gi = int(row.global_item_id)
        d = int(row.domain_id)

        if task == "full_catalog":
            _sync_device()
            t0 = time.perf_counter()
            scores = score_all_items_for_user(
                model,
                u,
                d,
                num_items,
                device,
                item_batch_size=int(CROSSDOMAIN_ITEM_BATCH),
            )
            scores[0] = -np.inf
            start = history.indptr[u]
            end = history.indptr[u + 1]
            if start < end:
                scores[history.indices[start:end]] = -np.inf
            _ = scores  # 避免被优化掉
            _sync_device()
            dt = time.perf_counter() - t0

        elif task == "uni100":
            interacted = lambda item, _u=u: history.contains(_u, item)
            pool = negative_pool_for_domain(
                domain_item_pools, d, True, global_pool
            )
            negatives = sample_eval_negatives(pool, num_neg, interacted)
            if len(negatives) < num_neg:
                continue
            candidates = negatives + [gi]
            u_t = torch.tensor([u] * len(candidates), device=device)
            i_t = torch.tensor(candidates, device=device)
            d_t = torch.tensor([d] * len(candidates), device=device)

            _sync_device()
            t0 = time.perf_counter()
            _ = model(u_t, i_t, d_t)
            _sync_device()
            dt = time.perf_counter() - t0
        else:
            raise ValueError(f"未知 task={task!r}")

        if seen < warmup:
            seen += 1
            continue
        times.append(dt)
        if len(times) >= n_users:
            break

    stats = _summarize(times)
    label = "crossdomain_neumf"
    if stats is not None:
        _print_row(label, task, stats, str(device))
    else:
        print(f"{label:<20} {task:<12} （无有效样本）")
    return stats


def main() -> None:
    from run_train import _apply_recbole_dynamic_neg_device_fix, _apply_scipy_dok_compat

    _apply_scipy_dok_compat()
    _apply_recbole_dynamic_neg_device_fix()

    tasks = list(TASKS)
    if len(tasks) == 1 and tasks[0].lower() == "both":
        tasks = ["full_catalog", "uni100"]

    cap_note = f"sample={SAMPLE_USERS} warmup={WARMUP_USERS} seed={EVAL_SAMPLE_SEED}"
    print(
        f"bench_per_user_latency | phase={PHASE} | tasks={tasks} | "
        f"models={MODELS} | {cap_note}"
    )

    _print_header()
    results: list[dict[str, Any]] = []

    for model_key in MODELS:
        key = model_key.lower().strip()
        ckpt = CHECKPOINTS.get(model_key) or CHECKPOINTS.get(key)
        for task in tasks:
            print(f"\n>>> {key} / {task}")
            try:
                if key == "crossdomain_neumf":
                    stats = bench_crossdomain(
                        task=task,
                        phase=PHASE,
                        n_users=int(SAMPLE_USERS),
                        warmup=int(WARMUP_USERS),
                        seed=int(EVAL_SAMPLE_SEED),
                        checkpoint=ckpt,
                    )
                else:
                    stats = bench_recbole(
                        key,
                        task=task,
                        phase=PHASE,
                        n_users=int(SAMPLE_USERS),
                        warmup=int(WARMUP_USERS),
                        seed=int(EVAL_SAMPLE_SEED),
                        checkpoint=ckpt,
                    )
                if stats is not None:
                    results.append(
                        {
                            "model": key,
                            "task": task,
                            "phase": PHASE,
                            "device": str(
                                torch.device(
                                    "cuda" if torch.cuda.is_available() else "cpu"
                                )
                            ),
                            **asdict(stats),
                            "mean_ms": stats.mean_s * 1000,
                            "median_ms": stats.median_s * 1000,
                            "users_per_sec": stats.users_per_sec,
                        }
                    )
            except Exception as exc:
                print(f"  [失败] {key}/{task}: {exc}")

    if SAVE_JSON and results:
        logs_dir = os.path.join(PLATFORM_ROOT, "results", "logs")
        os.makedirs(logs_dir, exist_ok=True)
        tag = f"per_user_latency_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log_path = os.path.join(logs_dir, f"{tag}.json")
        payload = {
            "phase": PHASE,
            "sample_users": SAMPLE_USERS,
            "warmup_users": WARMUP_USERS,
            "eval_sample_seed": EVAL_SAMPLE_SEED,
            "tasks": tasks,
            "models": MODELS,
            "results": results,
        }
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nlog: {log_path}")


if __name__ == "__main__":
    main()
