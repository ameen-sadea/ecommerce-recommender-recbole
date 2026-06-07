# -*- coding: utf-8 -*-
"""从 RecBole atomic .inter 加载单域数据（与 movies_tv 平台数据集一致）。"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


def _read_inter(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", skiprows=1, header=None)
    df.columns = ["user_id", "item_id", "rating", "timestamp"]
    df["user_id"] = df["user_id"].astype(int)
    df["item_id"] = df["item_id"].astype(int)
    return df


def load_splits_from_recbole_inter(
    data_path: str,
    dataset_name: str,
    *,
    train_row_limit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    root = os.path.join(os.path.abspath(data_path), dataset_name)
    train = _read_inter(os.path.join(root, f"{dataset_name}.train.inter"))
    valid = _read_inter(os.path.join(root, f"{dataset_name}.valid.inter"))
    test = _read_inter(os.path.join(root, f"{dataset_name}.test.inter"))

    if train_row_limit is not None and int(train_row_limit) > 0:
        train = train.iloc[: int(train_row_limit)].copy()

    for part in (train, valid, test):
        part["global_user_id"] = part["user_id"]
        part["global_item_id"] = part["item_id"]
        part["domain_id"] = 0

    num_users = int(max(train["global_user_id"].max(), valid["global_user_id"].max(), test["global_user_id"].max()) + 1)
    num_items = int(max(train["global_item_id"].max(), valid["global_item_id"].max(), test["global_item_id"].max()) + 1)
    meta = {
        "dataset": dataset_name,
        "num_global_users": num_users,
        "num_global_items": num_items,
        "num_domains": 1,
        "train_rows": len(train),
        "valid_rows": len(valid),
        "test_rows": len(test),
    }
    return train, valid, test, meta


def cap_eval_users(
    df: pd.DataFrame,
    uid_col: str,
    cap: int | None,
    rng: np.random.RandomState,
) -> pd.DataFrame:
    """
    与 run_train.apply_eval_user_caps 相同：从 split 中无放回抽 cap 个用户，保留其全部行。
    LOO 下每用户一行，等价于抽 cap 个用户。
    """
    if cap is None or int(cap) <= 0:
        return df
    users = df[uid_col].unique()
    n_keep = min(int(cap), len(users))
    if n_keep >= len(users):
        return df.reset_index(drop=True)
    chosen = rng.choice(users, size=n_keep, replace=False)
    return df[df[uid_col].isin(chosen)].reset_index(drop=True)
