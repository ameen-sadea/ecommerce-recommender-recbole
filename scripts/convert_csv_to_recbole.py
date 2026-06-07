#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将已有的 train.csv / valid.csv / test.csv 转为 RecBole atomic 格式。

期望 CSV 列（至少）: user_id, item_id
可选: rating, timestamp

用法:
  python scripts/convert_csv_to_recbole.py --src "D:/your/data/Books" --name books
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

# RecBole atomic 表头：列名:类型
INTER_HEADER = "user_id:token\titem_id:token\trating:float\ttimestamp:float"
SPLITS = ("train", "valid", "test")


def _read_split_csv(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"缺少文件: {path}")
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if "user_id" not in cols or "item_id" not in cols:
        raise ValueError(f"{path} 需要包含 user_id, item_id 列，当前: {list(df.columns)}")
    out = pd.DataFrame(
        {
            "user_id": df[cols["user_id"]].astype(str),
            "item_id": df[cols["item_id"]].astype(str),
        }
    )
    if "rating" in cols:
        out["rating"] = pd.to_numeric(df[cols["rating"]], errors="coerce").fillna(1.0)
    else:
        out["rating"] = 1.0
    if "timestamp" in cols:
        out["timestamp"] = pd.to_numeric(df[cols["timestamp"]], errors="coerce").fillna(0).astype(int)
    else:
        out["timestamp"] = 0
    return out


def convert(src_dir: str, dataset_name: str, out_root: str) -> str:
    src_dir = os.path.abspath(src_dir)
    out_dir = os.path.join(os.path.abspath(out_root), dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    for split in SPLITS:
        csv_path = os.path.join(src_dir, f"{split}.csv")
        df = _read_split_csv(csv_path)
        inter_path = os.path.join(out_dir, f"{dataset_name}.{split}.inter")
        with open(inter_path, "w", encoding="utf-8") as f:
            f.write(INTER_HEADER + "\n")
            df.to_csv(f, sep="\t", index=False, header=False)
        print(f"  [{split}] {len(df):,} 行 -> {inter_path}")

    print(f"\n完成。RecBole 配置中设置: dataset: {dataset_name}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="CSV -> RecBole atomic 数据集")
    parser.add_argument("--src", required=True, help="含 train/valid/test.csv 的目录")
    parser.add_argument("--name", required=True, help="数据集名（RecBole 的 dataset 字段）")
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets"),
        help="输出根目录，默认 recbole_platform/datasets",
    )
    args = parser.parse_args()
    print(f"源目录: {args.src}")
    print(f"输出:   {os.path.join(args.out, args.name)}\n")
    convert(args.src, args.name, args.out)


if __name__ == "__main__":
    main()
