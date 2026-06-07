#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将 BPR/NeuMF 的 item embedding 写入 SASRec/BERT4Rec，并保存为可续训 checkpoint。

与 run_train 中 yaml 的 init_item_embedding_from 逻辑相同；适合单独生成 init 权重。

用法（在 recbole_platform 目录）:
  python scripts/init_sequential_from_general.py
  python scripts/init_sequential_from_general.py --target bert4rec --source bpr
  python scripts/init_sequential_from_general.py --source bpr --source-ckpt D:/.../BPR/best.pth
"""

from __future__ import annotations

import argparse
import os
import sys

PLATFORM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PLATFORM_ROOT not in sys.path:
    sys.path.insert(0, PLATFORM_ROOT)
os.chdir(PLATFORM_ROOT)


def main() -> None:
    from logging import getLogger

    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import get_model, init_logger, init_seed

    from run_train import (
        _apply_recbole_dynamic_neg_device_fix,
        _apply_scipy_dok_compat,
        apply_init_item_embedding_from_general,
        apply_sequential_model_config,
        base_dataset_name,
        align_max_item_list_length_with_seq_files,
        ensure_sequential_benchmark_dataset,
        merge_negative_sampling_defaults,
        pick_config,
        pop_item_embedding_init_options,
        prepare_item_embedding_init,
        strip_platform_only_config_keys,
        load_yaml,
    )

    parser = argparse.ArgumentParser(description="Warm-start sequential item embedding")
    parser.add_argument(
        "--target",
        default="sasrec",
        choices=["sasrec", "bert4rec"],
        help="目标序列模型",
    )
    parser.add_argument(
        "--source",
        default="bpr",
        choices=["bpr", "neumf", "neumf_mlp"],
        help="源 general 模型",
    )
    parser.add_argument("--source-ckpt", default=None, help="源 best.pth 路径")
    parser.add_argument(
        "--out",
        default=None,
        help="输出路径（默认 <target_ckpt_dir>/init_from_<source>.pth）",
    )
    args = parser.parse_args()

    _apply_scipy_dok_compat()
    _apply_recbole_dynamic_neg_device_fix()

    cfg_path = pick_config(args.target)
    cfg = merge_negative_sampling_defaults(load_yaml(cfg_path))
    cfg = apply_sequential_model_config(cfg)
    ensure_sequential_benchmark_dataset(cfg)
    align_max_item_list_length_with_seq_files(cfg)
    cfg["init_item_embedding_from"] = args.source
    if args.source_ckpt:
        cfg["init_item_embedding_ckpt"] = args.source_ckpt
    init_opts = pop_item_embedding_init_options(cfg)
    prepare_item_embedding_init(
        cfg,
        init_opts,
        base_dataset=base_dataset_name(cfg.get("dataset", "movies_tv_seq")),
        model_name=cfg.get("model", ""),
    )
    strip_platform_only_config_keys(cfg)

    dataset = cfg.get("dataset", "movies_tv_seq")
    model_name = cfg.get("model")
    config = Config(model=model_name, dataset=dataset, config_dict=cfg)
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    getLogger().info("init_sequential_from_general.py")

    dataset_obj = create_dataset(config)
    train_data, _, _ = data_preparation(config, dataset_obj)
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])

    apply_init_item_embedding_from_general(
        model,
        init_opts=init_opts,
        base_dataset=base_dataset_name(dataset),
        target_model_name=model_name,
    )

    out_dir = config["checkpoint_dir"]
    os.makedirs(out_dir, exist_ok=True)
    out_path = args.out or os.path.join(out_dir, f"init_from_{args.source}.pth")
    if not os.path.isabs(out_path):
        out_path = os.path.join(PLATFORM_ROOT, out_path)

    payload = {
        "state_dict": model.state_dict(),
        "other_parameter": getattr(model, "other_parameter", None),
        "init_from": args.source,
        "target_model": model_name,
    }
    import torch

    torch.save(payload, out_path)
    print(f"\n>>> saved: {out_path}")
    print(">>> 训练时在 run_train.py 设 RESUME_FROM 为该路径，或把 init 逻辑留在 yaml 里从头训")


if __name__ == "__main__":
    main()
