#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CrossDomainNeuMF 全库排序评估（不走 RecBole full_sort）。

用法（在 recbole_platform 目录下）:
  python scripts/eval_crossdomain_full_catalog.py

改下面「可调区域」后运行。建议先 USER_CAP=20000 对比表7，再 USER_CAP=0 跑全量。
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime

PLATFORM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PLATFORM_ROOT not in sys.path:
    sys.path.insert(0, PLATFORM_ROOT)
os.chdir(PLATFORM_ROOT)

# ========================= 可调区域（只改这里）=========================
ON = "test"  # test | valid | both

COMPUTE_HR_10 = True
COMPUTE_HR_50 = True
COMPUTE_NDCG = True
COMPUTE_MRR = True
COMPUTE_MEANRANK = False
EXTRA_HR_AT_K: list[int] = []

# 0 → test 全量（movies_tv 约 65.7 万）；20000 → 与 yaml eval_test_user_cap 同量级
USER_CAP = 20000
EVAL_SAMPLE_SEED = 42

# 每用户分多少 item 一批打分（显存不够可降到 4096）
ITEM_BATCH_SIZE = 8192

CHECKPOINT = None  # None → yaml checkpoint_dir/best_ckpt.pt
SHOW_PROGRESS = True
# ========================================================================


def _topk_list() -> list[int]:
    ks: set[int] = set()
    if COMPUTE_HR_10:
        ks.add(10)
    if COMPUTE_HR_50:
        ks.add(50)
    ks.update(int(k) for k in EXTRA_HR_AT_K)
    if COMPUTE_NDCG or COMPUTE_MRR:
        ks.update({10, 50})
    return sorted(ks)


def main() -> None:
    import torch

    from crossdomain_neumf.data_inter import cap_eval_users, load_splits_from_recbole_inter
    from crossdomain_neumf.full_catalog_eval import (
        build_user_history_from_frames,
        evaluate_crossdomain_full_catalog,
    )
    from crossdomain_neumf.model import CrossDomainNeuMF
    from crossdomain_neumf.platform_bridge import crossdomain_train_config_from_yaml
    from run_train import load_yaml, pick_config

    topk_list = _topk_list()
    if not topk_list:
        raise ValueError("请至少开启 HR@10/HR@50 或 EXTRA_HR_AT_K")

    cfg_path = pick_config("crossdomain_neumf")
    raw_cfg = load_yaml(cfg_path)
    train_cfg = crossdomain_train_config_from_yaml(raw_cfg)

    data_path = raw_cfg.get("data_path", "datasets/")
    dataset = raw_cfg.get("dataset", "movies_tv")
    ckpt_dir = raw_cfg.get("checkpoint_dir", "")
    ckpt = CHECKPOINT or os.path.join(ckpt_dir, "best_ckpt.pt")
    if not os.path.isabs(ckpt):
        ckpt = os.path.join(PLATFORM_ROOT, ckpt)
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"找不到权重: {ckpt}")

    train_df, valid_df, test_df, meta = load_splits_from_recbole_inter(
        data_path, dataset
    )
    num_users = meta["num_global_users"]
    num_items = meta["num_global_items"]
    num_domains = meta["num_domains"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp_hidden = tuple(train_cfg["mlp_hidden"])
    model = CrossDomainNeuMF(
        num_users,
        num_items,
        num_domains,
        embed_dim=int(train_cfg["embed_dim"]),
        domain_embed_dim=int(train_cfg["domain_embed_dim"]),
        mlp_hidden=mlp_hidden,
        dropout=float(train_cfg["dropout"]),
        share_embeddings=bool(train_cfg.get("share_embeddings", True)),
        gmf_domain_aware=bool(train_cfg.get("gmf_domain_aware", True)),
    ).to(device)

    print(f"[load] {ckpt}")
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["state_dict"])
    model.eval()

    rng = __import__("numpy").random.RandomState(int(EVAL_SAMPLE_SEED))
    cap = int(USER_CAP)
    cap_note = "全量用户" if cap <= 0 else f"抽样 {cap} 用户 (seed={EVAL_SAMPLE_SEED})"
    print(
        f"CrossDomainNeuMF full-catalog | on={ON} topk={topk_list} "
        f"users={cap} ({cap_note}) device={device} item_batch={ITEM_BATCH_SIZE}"
    )

    history_train = build_user_history_from_frames([train_df], num_users)
    history_train_valid = build_user_history_from_frames(
        [train_df, valid_df], num_users
    )

    phases = ["valid", "test"] if str(ON).lower() == "both" else [str(ON).lower()]
    all_results: dict[str, dict] = {}

    for phase in phases:
        split_df = valid_df if phase == "valid" else test_df
        history = history_train if phase == "valid" else history_train_valid
        eval_df = cap_eval_users(split_df, "global_user_id", cap, rng)
        print(f"\n[{phase}] {len(eval_df)} users, mask={'train' if phase == 'valid' else 'train+valid'}")
        metrics = evaluate_crossdomain_full_catalog(
            model,
            eval_df,
            history,
            num_items=num_items,
            device=device,
            topk_list=topk_list,
            item_batch_size=int(ITEM_BATCH_SIZE),
            show_progress=SHOW_PROGRESS,
        )
        all_results[phase] = metrics

    print("\n=== CrossDomainNeuMF Full-catalog 结果 ===")
    for phase, metrics in all_results.items():
        if not metrics:
            print(f"  [{phase}] （无输出）")
            continue
        n_users = int(metrics.get("n_users", 0))
        print(f"  [{phase}] n_users={n_users}")
        if COMPUTE_HR_10 and "hr@10" in metrics:
            print(f"    HR@10  = {metrics['hr@10']:.4f}")
        if COMPUTE_HR_50 and "hr@50" in metrics:
            print(f"    HR@50  = {metrics['hr@50']:.4f}")
        for k in EXTRA_HR_AT_K:
            key = f"hr@{k}"
            if key in metrics:
                print(f"    HR@{k}  = {metrics[key]:.4f}")
        if COMPUTE_NDCG:
            for k in topk_list:
                key = f"ndcg@{k}"
                if key in metrics:
                    print(f"    NDCG@{k} = {metrics[key]:.4f}")
        if COMPUTE_MRR:
            for k in topk_list:
                key = f"mrr@{k}"
                if key in metrics:
                    print(f"    MRR@{k}  = {metrics[key]:.4f}")
        if COMPUTE_MEANRANK and "meanrank" in metrics:
            print(f"    meanrank = {metrics['meanrank']:.2f}")

    logs_dir = os.path.join(PLATFORM_ROOT, "results", "logs")
    os.makedirs(logs_dir, exist_ok=True)
    tag = f"crossdomain_full_catalog_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_path = os.path.join(logs_dir, f"{tag}.json")
    payload = {
        "model": "CrossDomainNeuMF",
        "checkpoint": ckpt,
        "on": ON,
        "user_cap": cap,
        "eval_sample_seed": EVAL_SAMPLE_SEED,
        "item_batch_size": ITEM_BATCH_SIZE,
        "topk_list": topk_list,
        "results": all_results,
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nlog: {log_path}")


if __name__ == "__main__":
    main()
