# -*- coding: utf-8 -*-
"""CrossDomainNeuMF 走 run_train.py 统一入口（不经过 RecBole Trainer）。"""

from __future__ import annotations

import copy
import os
from datetime import datetime

from crossdomain_neumf.trainer import run_crossdomain_training


def is_crossdomain_model(model_name: str | None) -> bool:
    return (model_name or "").strip().lower() in {
        "crossdomainneumf",
        "crossdomain_neumf",
    }


def crossdomain_train_config_from_yaml(cfg: dict) -> dict:
    """平台 yaml → CrossDomain 训练 dict。"""
    out = copy.deepcopy(cfg)
    if out.get("num_negatives_train") is not None:
        out["num_negatives"] = int(out.pop("num_negatives_train"))
    elif isinstance(out.get("train_neg_sample_args"), dict):
        out["num_negatives"] = int(
            out["train_neg_sample_args"].get("sample_num", 32)
        )
    if out.get("learning_rate") is not None:
        out["lr"] = float(out.pop("learning_rate"))
    if out.get("num_negatives_eval") is not None:
        out["eval_negatives"] = int(out.pop("num_negatives_eval"))
    elif "eval_negatives" not in out:
        out["eval_negatives"] = 100
    if out.get("eval_valid_user_cap") is not None:
        out["val_users"] = int(out.pop("eval_valid_user_cap"))
    if out.get("eval_test_user_cap") is not None:
        out["test_eval_users"] = int(out.pop("eval_test_user_cap"))
    out.setdefault("eval_sample_seed", out.get("seed", 42))
    if isinstance(out.get("mlp_hidden"), list):
        out["mlp_hidden"] = tuple(out["mlp_hidden"])
    return out


def resolve_crossdomain_resume(resume_from: str | None, ckpt_dir: str) -> str | None:
    """续训优先 best_ckpt.pt（含 optimizer）；.pth 仅权重时尝试同目录 best_ckpt.pt。"""
    if not resume_from:
        default = os.path.join(ckpt_dir, "best_ckpt.pt")
        return default if os.path.isfile(default) else None
    path = resume_from
    if path.endswith(".pth"):
        companion = os.path.join(os.path.dirname(path), "best_ckpt.pt")
        if os.path.isfile(companion):
            print(f">>> CrossDomain 续训使用 {companion}（.pth 无 optimizer 状态）")
            return companion
        raise FileNotFoundError(
            f"CrossDomain 续训需要 best_ckpt.pt（含 optimizer），同目录未找到: {companion}"
        )
    return path


def run_crossdomain_via_platform(
    *,
    cfg: dict,
    cfg_path: str,
    platform_root: str,
    ckpt_dir: str,
    tag: str,
    debug: bool,
    eval_only: bool,
    resume_from: str | None,
    platform_eval: dict,
    train_cfg: dict | None = None,
) -> None:
    dataset = cfg.get("dataset", "movies_tv")
    data_path = cfg.get("data_path", "datasets/")
    train_cfg = train_cfg or crossdomain_train_config_from_yaml(cfg)

    if debug:
        train_cfg["epochs"] = min(int(train_cfg.get("epochs", 40)), 5)
        train_cfg["patience"] = 2
        train_cfg["train_row_limit"] = 200_000

    resume = None
    if eval_only:
        resume = resolve_crossdomain_resume(
            os.path.join(ckpt_dir, "best_ckpt.pt"), ckpt_dir
        )
        if not resume or not os.path.isfile(resume):
            raise FileNotFoundError(
                f"EVAL_ONLY 需要 {ckpt_dir}/best_ckpt.pt，请先完整训练一轮"
            )
    elif resume_from:
        resume = resolve_crossdomain_resume(resume_from, ckpt_dir)

    if platform_eval.get("full_catalog_eval_enabled"):
        print(
            "note:    CrossDomainNeuMF 全库排序请用 "
            "scripts/eval_crossdomain_full_catalog.py（非 RecBole full_sort）"
        )

    print(">>> 训练后端: CrossDomainNeuMF（自定义 BPR 多负例），非 RecBole Trainer")
    print(f">>> 评估: 1正+{train_cfg.get('eval_negatives')}负 | "
          f"val_users={train_cfg.get('val_users')} test_users={train_cfg.get('test_eval_users')}")

    result = run_crossdomain_training(
        data_path=data_path,
        dataset_name=dataset,
        output_dir=ckpt_dir,
        config=train_cfg,
        resume=resume,
        tag=tag,
        eval_only=eval_only,
    )

    valid = result.get("best_valid_result") or {}
    test = result.get("test_result") or {}
    print("\n=== Summary (CrossDomainNeuMF) ===")
    print(f"best valid NDCG@10: {valid.get('ndcg@10')}")
    print(
        f"test  NDCG@10: {test.get('ndcg@10')}  HR@10: {test.get('hr@10')}  "
        f"GAUC: {test.get('gauc')}"
    )
    print(f"log: {os.path.join(platform_root, 'results', 'logs', tag + '.json')}")
    print(f"ckpt: {ckpt_dir}/best_ckpt.pt  (续训) | best.pt (仅权重)")
