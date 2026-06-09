#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
仅做「全库排序」评估（不训练）。

用法（在 recbole_platform 目录下）:
  python scripts/eval_full_catalog_hr.py

改下面「可调区域」后运行。

SASRec 可选「推理期热度降温」:
  Adjusted_Score_i = Score_i - gamma * log(Count_i)
  Count_i 为 train 集 item 出现次数；在 full_sort 打分后、排序前扣除。
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

PLATFORM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PLATFORM_ROOT not in sys.path:
    sys.path.insert(0, PLATFORM_ROOT)
os.chdir(PLATFORM_ROOT)

# ========================= 可调区域（只改这里）=========================
MODEL = "sasrec"  # bpr | neumf | lightgcn | sasrec | sasrec_k5 | sasrec_k2 | pop | itemknn

ON = "test"  # test | valid | both

# --- 指标开关（关掉可少算、更快；meanrank 会多一次全库排名统计）---
COMPUTE_HR_10 = True
COMPUTE_HR_50 = True
COMPUTE_NDCG = True   # NDCG@K（K 与已开启的 HR@K 相同）
COMPUTE_MRR = True    # MRR@K
COMPUTE_MEANRANK = False  # 算术平均排名 meanrank（越小越好）

# 除 10/50 外还要 HR@K 时填写，如 [100]；需自行 True 上面 HR 开关或只加在列表里
EXTRA_HR_AT_K = []

# 全库评估用户数：
#   0      → test 全量（movies_tv 约 65.7 万）
#   20000  → 与 yaml 里 eval_test_user_cap 同量级（推荐先对比）
#   None   → 沿用配置文件 full_catalog_eval_user_cap
FULL_CATALOG_USER_CAP = 0

# Pop/BPR 全库：RecBole 约「每用户 1 步」× 65 万步；collector 指标会落到 CPU 缓冲（模型仍在 GPU）。
# 分块评估：每块跑完释放显存，指标按用户数加权合并（与一次跑完等价）。全量 Pop 建议开分块。
#   0     → 不分块（易 OOM；仅适合 cap 较小或 SASRec 等大 batch 模型）
#   50000 → 全量 test 约 14 块，显存峰值明显降低（默认用 GPU，无需强制 CPU）
EVAL_CHUNK_USERS = 0

# 每多少个 batch 调用一次 torch.cuda.empty_cache（Pop 步数多，可设 32~128）
VRAM_FLUSH_EVERY_BATCHES = 64

CHECKPOINT = None  # None → yaml 的 checkpoint_dir/best.pth
SHOW_PROGRESS = True

# --- SASRec 推理期热度降温（仅 MODEL=sasrec 且 POP_DEBIAS_ENABLED=True）---
# Adjusted_Score_i = Score_i - gamma * log(Count_i)，Count_i 来自 train 交互次数
POP_DEBIAS_ENABLED = False
POP_DEBIAS_GAMMA = [0.0, 0.05]  # 0.0=原始 logits 基线
POP_DEBIAS_TUNE_METRIC = "ndcg@10"  # valid 上选最优 gamma 的指标
# ========================================================================


def build_train_item_log_counts(dataset_obj) -> np.ndarray:
    """item_id -> log(train_count)；index 0(pad) 与未出现 item 为 0。"""
    iid_field = dataset_obj.iid_field
    inter = dataset_obj.inter_feat
    names = getattr(dataset_obj, "benchmark_filename_list", None)

    if names and "train" in names:
        offset = 0
        train_chunk = None
        for name, size in zip(names, dataset_obj.file_size_list):
            if name == "train":
                train_chunk = inter.iloc[offset : offset + size]
                break
            offset += size
        if train_chunk is None:
            raise ValueError("benchmark 中未找到 train 切片")
        counts = train_chunk[iid_field].value_counts()
    else:
        counts = inter[iid_field].value_counts()

    log_counts = np.zeros(int(dataset_obj.item_num), dtype=np.float32)
    for iid, cnt in counts.items():
        idx = int(iid)
        if 0 < idx < dataset_obj.item_num:
            log_counts[idx] = float(np.log(float(cnt)))
    return log_counts


def patch_trainer_pop_debias(trainer, log_counts: np.ndarray, gamma: float) -> None:
    """在 _full_sort_batch_eval 返回前扣除 gamma * log(count)。"""
    import torch

    if gamma <= 0:
        return

    device = trainer.device
    penalty = torch.tensor(log_counts, device=device, dtype=torch.float32) * float(gamma)
    orig = trainer._full_sort_batch_eval

    def _full_sort_with_pop_debias(batched_data):
        interaction, scores, positive_u, positive_i = orig(batched_data)
        scores = scores - penalty.unsqueeze(0)
        scores[:, 0] = -np.inf
        return interaction, scores, positive_u, positive_i

    trainer._full_sort_batch_eval = _full_sort_with_pop_debias


def _metric_get(metrics: dict[str, Any], key: str) -> float:
    if key in metrics:
        return float(metrics[key])
    alt = key.replace("hr@", "hit@")
    if alt in metrics:
        return float(metrics[alt])
    return float("nan")


def _print_metrics_block(label: str, result: dict[str, dict[str, Any]]) -> None:
    print(f"\n=== {label} ===")
    for phase, metrics in result.items():
        if not metrics:
            print(f"  [{phase}] （无输出）")
            continue
        hr_part = {k: v for k, v in metrics.items() if k.startswith(("hit@", "hr@"))}
        ndcg_part = {k: v for k, v in metrics.items() if k.startswith("ndcg@")}
        mrr_part = {k: v for k, v in metrics.items() if k.startswith("mrr@")}
        mr = metrics.get("meanrank") or metrics.get("avg_rank")
        if hr_part:
            print(f"  [{phase}] HR: {hr_part}")
        if ndcg_part:
            print(f"  [{phase}] NDCG: {ndcg_part}")
        if mrr_part:
            print(f"  [{phase}] MRR: {mrr_part}")
        if mr is not None:
            print(f"  [{phase}] meanrank = {mr}")


def _print_gamma_summary(rows: list[dict[str, Any]], tune_metric: str) -> None:
    print("\n" + "=" * 72)
    print("SASRec 热度降温 gamma 对比")
    print("=" * 72)
    header = (
        f"{'gamma':>6} {'phase':<6} {'HR@10':>8} {'NDCG@10':>9} "
        f"{'MRR@10':>8} {'HR@50':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        m = row.get("metrics") or {}
        print(
            f"{row['gamma']:>6.2f} {row['phase']:<6} "
            f"{_metric_get(m, 'hr@10'):>8.4f} "
            f"{_metric_get(m, 'ndcg@10'):>9.4f} "
            f"{_metric_get(m, 'mrr@10'):>8.4f} "
            f"{_metric_get(m, 'hr@50'):>8.4f}"
        )

    valid_rows = [r for r in rows if r.get("phase") == "valid"]
    if valid_rows:
        best = max(valid_rows, key=lambda r: _metric_get(r.get("metrics", {}), tune_metric))
        print(
            f"\nvalid 最优 gamma={best['gamma']:.2f} "
            f"({tune_metric}={_metric_get(best.get('metrics', {}), tune_metric):.4f})"
        )


def main() -> None:
    from run_train import (
        _apply_recbole_dynamic_neg_device_fix,
        _apply_scipy_dok_compat,
        apply_eval_user_caps,
        apply_negative_sampling_config,
        apply_sequential_model_config,
        build_full_catalog_metric_plan,
        ensure_sequential_benchmark_dataset,
        merge_negative_sampling_defaults,
        pick_config,
        pop_platform_eval_options,
        run_full_catalog_hr_eval,
        strip_platform_only_config_keys,
        load_checkpoint_weights,
        load_yaml,
    )
    from recbole.config import Config
    from recbole.data import create_dataset
    from recbole.utils import get_model, get_trainer, init_logger, init_seed
    from logging import getLogger

    _apply_scipy_dok_compat()
    _apply_recbole_dynamic_neg_device_fix()

    metric_plan = build_full_catalog_metric_plan(
        compute_hr_10=COMPUTE_HR_10,
        compute_hr_50=COMPUTE_HR_50,
        compute_ndcg=COMPUTE_NDCG,
        compute_mrr=COMPUTE_MRR,
        compute_meanrank=COMPUTE_MEANRANK,
        extra_hr_at_k=EXTRA_HR_AT_K,
    )

    cfg_path = pick_config(MODEL)
    cfg = strip_platform_only_config_keys(
        apply_sequential_model_config(
            apply_negative_sampling_config(
                merge_negative_sampling_defaults(load_yaml(cfg_path))
            )
        )
    )
    ensure_sequential_benchmark_dataset(cfg)

    cfg["full_catalog_eval_enabled"] = True
    cfg["full_catalog_eval_topk"] = metric_plan["topk_list"]
    cfg["full_catalog_eval_on"] = ON

    if FULL_CATALOG_USER_CAP is not None:
        cfg["full_catalog_eval_user_cap"] = int(FULL_CATALOG_USER_CAP)
    platform_eval = pop_platform_eval_options(cfg)
    cap = int(platform_eval["full_catalog_eval_user_cap"])
    cap_note = "全量用户" if cap <= 0 else f"抽样 {cap} 用户"

    use_pop_debias = (
        POP_DEBIAS_ENABLED
        and str(MODEL).lower().strip() == "sasrec"
        and len(POP_DEBIAS_GAMMA) > 0
    )
    gammas = [float(g) for g in POP_DEBIAS_GAMMA] if use_pop_debias else [0.0]

    on_parts = []
    if COMPUTE_HR_10:
        on_parts.append("HR@10")
    if COMPUTE_HR_50:
        on_parts.append("HR@50")
    for k in EXTRA_HR_AT_K:
        on_parts.append(f"HR@{k}")
    if COMPUTE_NDCG:
        on_parts.append("NDCG")
    if COMPUTE_MRR:
        on_parts.append("MRR")
    if COMPUTE_MEANRANK:
        on_parts.append("meanrank")
    debias_note = f" pop_debias gammas={gammas}" if use_pop_debias else ""
    print(
        f"full-catalog: model={MODEL} on={ON} metrics=[{', '.join(on_parts)}] "
        f"topk={metric_plan['topk_list']} users={cap} ({cap_note}){debias_note}"
    )
    strip_platform_only_config_keys(cfg)

    ckpt = CHECKPOINT or os.path.join(cfg.get("checkpoint_dir", ""), "best.pth")
    if not os.path.isabs(ckpt):
        ckpt = os.path.join(PLATFORM_ROOT, ckpt)
    if str(MODEL).lower().strip() == "neumf":
        from run_train import align_neumf_config_from_checkpoint

        cfg = align_neumf_config_from_checkpoint(cfg, ckpt)

    model_name = cfg["model"]
    config = Config(model=model_name, dataset=cfg["dataset"], config_dict=cfg)
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    getLogger().info("eval_full_catalog_hr.py — eval only")

    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"找不到权重: {ckpt}")
    dataset_obj = create_dataset(config)
    apply_eval_user_caps(dataset_obj, config)
    model = get_model(model_name)(config, dataset_obj).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], model_name)(config, model)

    load_checkpoint_weights(trainer, ckpt)
    print(f"checkpoint: {ckpt}")

    log_counts = None
    if use_pop_debias:
        log_counts = build_train_item_log_counts(dataset_obj)
        seen = int(np.count_nonzero(log_counts))
        print(
            f"pop debias: train log-count 非零 item 数={seen}, "
            f"log(count) max={log_counts.max():.3f}"
        )

    on = str(ON).lower()
    phases = ["valid", "test"] if on == "both" else [on]
    orig_full_sort = trainer._full_sort_batch_eval

    summary_rows: list[dict[str, Any]] = []
    all_results: dict[float, dict[str, dict[str, Any]]] = {}

    for gamma in gammas:
        trainer._full_sort_batch_eval = orig_full_sort
        if use_pop_debias and log_counts is not None and gamma > 0:
            patch_trainer_pop_debias(trainer, log_counts, gamma)
            print(f"\n>>> gamma={gamma:.2f} (Score -= {gamma} * log(train_count))")
        elif use_pop_debias:
            print("\n>>> gamma=0.00 (原始 logits 基线)")

        result = run_full_catalog_hr_eval(
            trainer,
            dataset_obj,
            config,
            metric_plan=metric_plan,
            phases=phases,
            show_progress=SHOW_PROGRESS,
            full_catalog_user_cap=platform_eval.get("full_catalog_eval_user_cap"),
            chunk_users=EVAL_CHUNK_USERS,
            vram_flush_every_batches=VRAM_FLUSH_EVERY_BATCHES,
            load_best_model=False,
        )
        all_results[gamma] = result
        for phase, metrics in result.items():
            summary_rows.append({"gamma": gamma, "phase": phase, "metrics": metrics})

    trainer._full_sort_batch_eval = orig_full_sort

    if use_pop_debias and len(gammas) > 1:
        _print_gamma_summary(summary_rows, POP_DEBIAS_TUNE_METRIC)
    else:
        _print_metrics_block("Full-catalog 结果", all_results.get(gammas[0], {}))


if __name__ == "__main__":
    main()
