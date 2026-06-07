#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
同一数据集上依次跑多个模型，便于快速对比。

用法:
  python scripts/legacy/run_benchmark.py --dataset movies_tv --configs configs/bpr_movies_tv_full.yaml configs/neumf_movies_tv_full.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys

PLATFORM_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PLATFORM_ROOT, "scripts", "legacy"))

from run_experiment import run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "configs/bpr_movies_tv_full.yaml",
            "configs/neumf_movies_tv_full.yaml",
            "configs/lightgcn_movies_tv_full.yaml",
        ],
    )
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    summary = []
    for cfg in args.configs:
        cfg_path = cfg if os.path.isabs(cfg) else os.path.join(PLATFORM_ROOT, cfg)
        print(f"\n>>> 开始: {cfg_path}\n")
        try:
            payload = run(
                dataset=args.dataset,
                config_path=cfg_path,
                epochs=args.epochs,
            )
            summary.append(payload)
        except Exception as e:
            print(f"失败 {cfg_path}: {e}")
            summary.append({"config_path": cfg_path, "error": str(e)})

    out = os.path.join(PLATFORM_ROOT, "results", f"benchmark_{args.dataset}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n对比汇总: {out}")


if __name__ == "__main__":
    main()
