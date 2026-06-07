#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将 movies_tv 的扁平 benchmark（train/valid/test.inter）转为 SASRec 所需的序列格式。

生成 datasets/<name>_seq/<name>_seq.{train,valid,test}.inter
列：user_id, item_id（预测目标）, item_id_list（空格分隔历史）, item_length

用法（在 recbole_platform 目录）:
  python scripts/build_sequential_dataset.py
  python scripts/build_sequential_dataset.py --dataset movies_tv --max-len 50
  python scripts/build_sequential_dataset.py --dataset movies_tv --max-len 5 --seq-name movies_tv_seq_k5
  python scripts/build_sequential_dataset.py --dataset movies_tv --max-len 2 --seq-name movies_tv_seq_k2
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

PLATFORM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTER_HEADER = (
    "user_id:token\titem_id:token\titem_id_list:token_seq\titem_length:float\n"
)


def _read_inter(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", skiprows=1, names=["user_id", "item_id", "rating", "timestamp"])
    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    return df.sort_values(["user_id", "timestamp"], kind="mergesort")


def _seq_str(items: list, max_len: int) -> str:
    if max_len > 0:
        items = items[-max_len:]
    return " ".join(str(x) for x in items)


def build_sequential_dataset(
    dataset: str,
    *,
    data_root: str | None = None,
    max_len: int = 50,
    seq_name: str | None = None,
) -> str:
    data_root = data_root or os.path.join(PLATFORM_ROOT, "datasets")
    src_dir = os.path.join(data_root, dataset)
    seq_name = seq_name or f"{dataset}_seq"
    out_dir = os.path.join(data_root, seq_name)
    os.makedirs(out_dir, exist_ok=True)

    train = _read_inter(os.path.join(src_dir, f"{dataset}.train.inter"))
    valid = _read_inter(os.path.join(src_dir, f"{dataset}.valid.inter"))
    test = _read_inter(os.path.join(src_dir, f"{dataset}.test.inter"))

    hist: dict[str, list[str]] = (
        train.groupby("user_id", sort=False)["item_id"].apply(list).to_dict()
    )
    valid_tgt = valid.groupby("user_id", sort=False)["item_id"].last().to_dict()

    train_rows: list[dict] = []
    for uid, items in hist.items():
        for i in range(1, len(items)):
            h = items[max(0, i - max_len) : i]
            train_rows.append(
                {
                    "user_id": uid,
                    "item_id": items[i],
                    "item_id_list": _seq_str(h, max_len),
                    "item_length": len(h),
                }
            )

    valid_rows: list[dict] = []
    for uid, tgt in valid_tgt.items():
        h = hist.get(uid, [])[-max_len:]
        if not h:
            continue
        valid_rows.append(
            {
                "user_id": uid,
                "item_id": tgt,
                "item_id_list": _seq_str(h, max_len),
                "item_length": len(h),
            }
        )

    test_rows: list[dict] = []
    for uid, grp in test.groupby("user_id", sort=False):
        tgt = grp["item_id"].iloc[-1]
        h = list(hist.get(uid, []))
        if uid in valid_tgt:
            h.append(valid_tgt[uid])
        h = h[-max_len:]
        if not h:
            continue
        test_rows.append(
            {
                "user_id": uid,
                "item_id": tgt,
                "item_id_list": _seq_str(h, max_len),
                "item_length": len(h),
            }
        )

    splits = {"train": train_rows, "valid": valid_rows, "test": test_rows}
    for split, rows in splits.items():
        out_path = os.path.join(out_dir, f"{seq_name}.{split}.inter")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(INTER_HEADER)
            pd.DataFrame(rows).to_csv(f, sep="\t", index=False, header=False)
        print(f"  [{split}] {len(rows):,} 行 -> {out_path}")

    print(f"\n完成。SASRec 配置请设 dataset: {seq_name}，MAX_ITEM_LIST_LENGTH: {max_len}")
    return seq_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="movies_tv")
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument(
        "--seq-name",
        default=None,
        help="输出目录/RecBole dataset 名，默认 <dataset>_seq",
    )
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    os.chdir(PLATFORM_ROOT)
    seq_name = args.seq_name or f"{args.dataset}_seq"
    print(f"源: datasets/{args.dataset}  max_len={args.max_len}  seq_name={seq_name}\n")
    build_sequential_dataset(
        args.dataset,
        data_root=args.data_root,
        max_len=args.max_len,
        seq_name=seq_name,
    )


if __name__ == "__main__":
    main()
