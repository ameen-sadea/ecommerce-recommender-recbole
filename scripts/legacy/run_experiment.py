#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单模型训练 + 评估（封装 RecBole quick_start）。

用法（在 recbole_platform 目录下）:
  python scripts/legacy/run_experiment.py --dataset movies_tv --config configs/bpr_movies_tv_full.yaml
  python scripts/legacy/run_experiment.py --dataset movies_tv --config configs/neumf_movies_tv_full.yaml --epochs 30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import yaml
import torch

# RecBole checkpoint 在 PyTorch 2.6+ 需 weights_only=False
_orig_torch_load = torch.load


def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_compat  # noqa: A001

# 保证移动到 scripts/legacy 后仍能定位 recbole_platform 根目录
PLATFORM_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PLATFORM_ROOT not in sys.path:
    sys.path.insert(0, PLATFORM_ROOT)


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _merge_config(base_path: str, model_path: str, overrides: dict) -> dict:
    cfg = {}
    if base_path and os.path.isfile(base_path):
        cfg.update(_load_yaml(base_path))
    cfg.update(_load_yaml(model_path))
    cfg.update(overrides)
    return cfg


def run(
    dataset: str,
    config_path: str,
    base_config: str | None = None,
    extra_base: list[str] | str | None = None,
    epochs: int | None = None,
    gpu_id: str | None = None,
    results_subdir: str | None = None,
) -> dict:
    from recbole.quick_start import run_recbole

    os.chdir(PLATFORM_ROOT)

    model_path = os.path.abspath(config_path)
    extras: list[str] = []
    if extra_base:
        raw = extra_base if isinstance(extra_base, list) else [extra_base]
        for e in raw:
            p = e if os.path.isabs(e) else os.path.join(PLATFORM_ROOT, e)
            extras.append(p)
    if base_config and os.path.isfile(base_config):
        cfg = _merge_config(base_config, model_path, {"dataset": dataset})
    else:
        cfg = _load_yaml(model_path)
        cfg["dataset"] = dataset
    for p in extras:
        cfg.update(_load_yaml(p))

    if epochs is not None:
        cfg["epochs"] = epochs
    if gpu_id is not None:
        cfg["gpu_id"] = gpu_id

    # 每次实验单独子目录，避免覆盖
    tag = results_subdir or f"{dataset}_{cfg.get('model', 'model')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    cfg["checkpoint_dir"] = os.path.join("results", "checkpoints", tag)

    model_name = cfg.pop("model", None)
    if not model_name:
        raise ValueError("配置中缺少 model 字段，例如 model: BPR")

    print("=" * 60)
    print(f"数据集: {dataset}")
    print(f"模型:   {model_name}")
    print(f"配置:   {config_path}")
    print(f"输出:   {cfg['checkpoint_dir']}")
    print("=" * 60)

    # 在 RecBole 打日志前打印，避免被 tqdm 冲掉
    result = run_recbole(model=model_name, dataset=dataset, config_dict=cfg)

    out_dir = os.path.join(PLATFORM_ROOT, "results", "logs")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, f"{tag}.json")
    # result: (config, model, dataset, train_loss, valid_result, test_result)
    payload = {
        "dataset": dataset,
        "model": model_name,
        "config_path": config_path,
        "valid_result": result[4] if len(result) > 4 else None,
        "test_result": result[5] if len(result) > 5 else None,
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n指标已写入: {log_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="convert 时 --name 相同")
    parser.add_argument(
        "--config",
        required=True,
        help="完整模型 yaml，如 configs/bpr_movies_tv_full.yaml",
    )
    parser.add_argument("--base", default=None, help="可选：再叠加一层基础 yaml")
    parser.add_argument(
        "--extra-base",
        nargs="*",
        default=None,
        help="可选：额外叠加配置（可多个）",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--gpu-id", default=None, help="如 '0' 或 '' 表示 CPU")
    parser.add_argument("--tag", default=None, help="结果子目录名")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(PLATFORM_ROOT, config_path)

    run(
        dataset=args.dataset,
        config_path=config_path,
        base_config=args.base,
        extra_base=args.extra_base,
        epochs=args.epochs,
        gpu_id=args.gpu_id,
        results_subdir=args.tag,
    )


if __name__ == "__main__":
    main()
