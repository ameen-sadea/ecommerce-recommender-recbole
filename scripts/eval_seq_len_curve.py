#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SASRec 推理期历史截断实验（仅 SASRec，全库排序）。

在 recbole_platform 目录下:
  python scripts/eval_seq_len_curve.py

问题：训练好的 SASRec 在 test 上若只喂最近 K 条历史（而非全长），
全库 HR/NDCG 会不会更好？

协议:
  - 目标固定为 test 正例 item_id（与表 7 一致）
  - 输入为 test 前历史的截断：默认最近 K 条（TRUNCATE_MODE=last_k）
  - K=0 表示不截断，用完整 item_id_list（应对齐表 7 ~0.09 量级）
  - full_sort 对 ~20 万 item 排序；mask 已见历史 + pad
  - 抽样 USER_SAMPLE 个 test 用户（默认 uncapped 全量池；5-core 下 test 前历史 min=4，见下）

5-core 说明:
  - 5-core 指全表交互数 >= 5；test 行的 item_id_list 是「test 之前」的历史，
    LOO 划分下最少只有 4 条（5 次交互 = 4 次在 test 前 + 1 次 test 目标）。
  - 因此 hist>=5 表示至少 6 次总交互，不是「5-core 门槛」。
  - yaml eval_test_user_cap=20000 时 capped 池仅 2 万行，且 hist>=5 约 1.5 万；请开 USE_UNCAPPED_TEST。
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from datetime import datetime
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

PLATFORM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PLATFORM_ROOT not in sys.path:
    sys.path.insert(0, PLATFORM_ROOT)
os.chdir(PLATFORM_ROOT)

_cascade_path = os.path.join(PLATFORM_ROOT, "scripts", "eval_cascade_rank.py")
_spec = importlib.util.spec_from_file_location("eval_cascade_rank", _cascade_path)
_cascade = importlib.util.module_from_spec(_spec)
sys.modules["eval_cascade_rank"] = _cascade
assert _spec.loader is not None
_spec.loader.exec_module(_cascade)
load_model_bundle = _cascade.load_model_bundle

# ========================= 可调区域（只改这里）=========================
USER_SAMPLE = 0  # 0=用全部 test 用户；勿超过 uncapped test 总数（~657k）
SAMPLE_SEED = 42

# True=读 yaml cap 前的全量 test（~657k）；False=仅 capped test（yaml 里通常 2 万）
USE_UNCAPPED_TEST = True

# K=0 → 完整历史（表 7 对照）；其余为截断长度
SEQUENCE_LENGTHS = [0, 1, 2, 3, 5]

# last_k: 只保留最近 K 条（推理期常见）；first_k: 保留最早 K 条
TRUNCATE_MODE = "last_k"  # last_k | first_k

METRIC_TOPK = [10, 50]
FULL_CATALOG = True  # 本脚本默认全库；False 则退化为脚内 uni100 近似（非 RecBole 官方）

SASREC_CKPT = None
SASREC_BATCH = 256
SHOW_PROGRESS = True
SAVE_JSON = True
# ========================================================================


def _parse_item_list(raw, length_hint: int | None) -> list[int]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        items: list[int] = []
    elif isinstance(raw, (list, tuple, np.ndarray)):
        items = [int(x) for x in raw]
    else:
        items = [int(x) for x in str(raw).strip().split()]
    if length_hint is not None and length_hint > 0:
        items = items[: int(length_hint)]
    return [i for i in items if i > 0]


def build_test_cases(dataset, config, *, use_uncapped: bool) -> list[dict[str, Any]]:
    """test 行 → {uid, hist, target, hist_len}。"""
    uid_field = dataset.uid_field
    iid_field = dataset.iid_field
    list_field = config["ITEM_ID_FIELD"] + config["LIST_SUFFIX"]
    len_field = config["ITEM_LIST_LENGTH_FIELD"]

    test_df = None
    source = "capped inter_feat"
    if use_uncapped:
        uncapped = getattr(dataset, "_benchmark_uncapped", None) or {}
        if "test" in uncapped:
            test_df = uncapped["test"]
            source = "uncapped test (~657k)"

    if test_df is None:
        inter = dataset.inter_feat
        names = getattr(dataset, "benchmark_filename_list", None)
        if not names or "test" not in names:
            raise ValueError("需要 movies_tv_seq benchmark test 切片")
        offset = 0
        for name, size in zip(names, dataset.file_size_list):
            if name == "test":
                test_df = inter.iloc[offset : offset + size]
                break
            offset += size
        if test_df is None:
            raise ValueError("未找到 test 分片")

    rows: list[dict[str, Any]] = []
    for idx in range(len(test_df)):
        uid = int(test_df.iloc[idx][uid_field])
        target = int(test_df.iloc[idx][iid_field])
        len_hint = int(test_df.iloc[idx][len_field]) if len_field in test_df.columns else None
        hist = _parse_item_list(test_df.iloc[idx][list_field], len_hint)
        if not hist:
            continue
        rows.append(
            {
                "uid": uid,
                "hist": hist,
                "target": target,
                "hist_len": len(hist),
            }
        )
    print(f"test pool: {len(rows)} users ({source})")
    return rows


def sample_users(
    rows: list[dict[str, Any]], n: int, seed: int, min_hist_len: int = 1
) -> list[dict[str, Any]]:
    eligible = [r for r in rows if r["hist_len"] >= min_hist_len]
    if not eligible:
        raise ValueError(f"无 hist_len>={min_hist_len} 的用户")
    if n <= 0 or n >= len(eligible):
        if 0 < n < len(eligible):
            pass
        elif n > len(eligible):
            print(
                f"  提示: USER_SAMPLE={n} > 可用 {len(eligible)}，"
                f"改用全部 {len(eligible)} 人"
            )
        return list(eligible)
    rng = np.random.RandomState(seed)
    pick = rng.choice(len(eligible), size=n, replace=False)
    return [eligible[int(i)] for i in pick]


def print_hist_pool_stats(rows: list[dict[str, Any]]) -> None:
    lens = np.array([r["hist_len"] for r in rows], dtype=np.int64)
    print(
        f"  hist_len: min={lens.min()} median={int(np.median(lens))} max={lens.max()} | "
        f">=4 (5-core 最少 test 前): {(lens >= 4).sum()} | "
        f">=5: {(lens >= 5).sum()}"
    )


def truncate_history(hist: list[int], k: int, mode: str) -> list[int]:
    """k=0 表示全长；k>0 截断为 first_k 或 last_k。"""
    if k <= 0 or len(hist) <= k:
        return list(hist)
    if mode == "first_k":
        return hist[:k]
    return hist[-k:]


def _metrics_from_ranks(ranks: np.ndarray, topk_list: list[int]) -> dict[str, float]:
    out: dict[str, float] = {}
    n = len(ranks)
    if n == 0:
        return out
    for k in topk_list:
        hit = (ranks > 0) & (ranks <= k)
        hr = float(hit.mean())
        ndcg = np.zeros(n, dtype=np.float64)
        mrr = np.zeros(n, dtype=np.float64)
        if hit.any():
            ndcg[hit] = 1.0 / np.log2(ranks[hit] + 1)
            mrr[hit] = 1.0 / ranks[hit]
        out[f"hr@{k}"] = hr
        out[f"ndcg@{k}"] = float(ndcg.sum() / n)
        out[f"mrr@{k}"] = float(mrr.sum() / n)
    out["n_users"] = float(n)
    return out


def _rank_from_scores(scores: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    order = scores.argsort(dim=1, descending=True)
    b = scores.shape[0]
    ranks = torch.zeros(b, dtype=torch.long, device=scores.device)
    for i in range(b):
        pos = int(target[i].item())
        hit = (order[i] == pos).nonzero(as_tuple=False)
        if hit.numel() > 0:
            ranks[i] = int(hit[0, 0].item()) + 1
    return ranks


def _mask_histories(scores: torch.Tensor, histories: list[list[int]]) -> None:
    scores[:, 0] = -np.inf
    for j, hist in enumerate(histories):
        if not hist:
            continue
        idx = torch.tensor(hist, device=scores.device, dtype=torch.long)
        scores[j, idx] = -np.inf


@torch.no_grad()
def score_sasrec_batch(bundle, histories: list[list[int]], device: torch.device) -> torch.Tensor:
    model = bundle.model
    config = bundle.config
    max_len = int(config["MAX_ITEM_LIST_LENGTH"])
    seq_field = config["ITEM_ID_FIELD"] + config["LIST_SUFFIX"]
    len_field = config["ITEM_LIST_LENGTH_FIELD"]
    uid_field = config["USER_ID_FIELD"]
    from recbole.data.interaction import Interaction

    b = len(histories)
    n_items = bundle.dataset.item_num
    seq_batch = torch.zeros(b, max_len, dtype=torch.long, device=device)
    len_batch = torch.zeros(b, dtype=torch.long, device=device)
    uids = torch.arange(b, dtype=torch.long, device=device)

    for i, hist in enumerate(histories):
        h = hist[-max_len:]
        length = len(h)
        len_batch[i] = length
        if length > 0:
            seq_batch[i, :length] = torch.tensor(h, dtype=torch.long, device=device)

    inter = Interaction(
        {
            uid_field: uids,
            seq_field: seq_batch,
            len_field: len_batch,
        }
    )
    scores = model.full_sort_predict(inter)
    if scores.dim() == 1:
        scores = scores.view(b, n_items)
    return scores


def _build_uni100_scores(
    full_scores: torch.Tensor,
    targets: torch.Tensor,
    n_neg: int,
    rng: np.random.RandomState,
) -> torch.Tensor:
    b, n_items = full_scores.shape
    out = torch.full((b, n_neg + 1), -np.inf, device=full_scores.device)
    for i in range(b):
        pos = int(targets[i].item())
        out[i, 0] = full_scores[i, pos]
        picked: list[int] = []
        tries = 0
        while len(picked) < n_neg and tries < n_neg * 20:
            c = int(rng.randint(1, n_items))
            tries += 1
            if c != pos and c not in picked:
                picked.append(c)
        for j, c in enumerate(picked[:n_neg]):
            out[i, j + 1] = full_scores[i, c]
    return out


@torch.no_grad()
def eval_at_k(
    *,
    users: list[dict[str, Any]],
    k_hist: int,
    bundle,
    device: torch.device,
    topk_list: list[int],
    full_catalog: bool,
    truncate_mode: str,
    batch_size: int,
    show_progress: bool,
) -> dict[str, Any]:
    histories: list[list[int]] = []
    targets: list[int] = []

    for row in users:
        hist = row["hist"]
        if k_hist > 0 and len(hist) < k_hist:
            continue
        histories.append(truncate_history(hist, k_hist, truncate_mode))
        targets.append(int(row["target"]))

    if not histories:
        return {"k": k_hist, "metrics": {}, "n_users": 0}

    all_ranks: list[int] = []
    rng = np.random.RandomState(SAMPLE_SEED + int(k_hist))

    batch_ranges = range(0, len(histories), batch_size)
    label = "full" if k_hist <= 0 else str(k_hist)
    if show_progress:
        batch_ranges = tqdm(
            batch_ranges,
            desc=f"K={label}",
            total=(len(histories) + batch_size - 1) // batch_size,
        )

    for start in batch_ranges:
        end = min(start + batch_size, len(histories))
        h_batch = histories[start:end]
        t_batch = torch.tensor(targets[start:end], device=device, dtype=torch.long)

        scores = score_sasrec_batch(bundle, h_batch, device)
        _mask_histories(scores, h_batch)

        if not full_catalog:
            scores = _build_uni100_scores(scores, t_batch, 99, rng)

        all_ranks.extend(_rank_from_scores(scores, t_batch).tolist())

    metrics = _metrics_from_ranks(np.array(all_ranks, dtype=np.int64), topk_list)
    return {
        "k": k_hist,
        "k_label": label,
        "truncate_mode": truncate_mode,
        "target": "test_item",
        "ranking": "full_catalog" if full_catalog else "uni100_approx",
        "n_users": len(all_ranks),
        "metrics": metrics,
    }


def _print_curve(rows: list[dict[str, Any]], topk: int = 10) -> None:
    key = f"hr@{topk}"
    ndcg_key = f"ndcg@{topk}"
    rank_note = rows[0].get("ranking", "?") if rows else "?"
    print("\n" + "=" * 72)
    print(f"SASRec 历史截断 vs 全库 HR@{topk}（目标=test 正例，排序={rank_note}）")
    print("=" * 72)
    header = f"{'K':>6} {'HR@10':>8} {'NDCG@10':>9} {'MRR@10':>8} {'HR@50':>8} {'n':>6}"
    print(header)
    print("-" * len(header))
    for row in rows:
        m = row.get("metrics") or {}
        k_label = row.get("k_label", row.get("k"))
        print(
            f"{str(k_label):>6} "
            f"{m.get('hr@10', float('nan')):>8.4f} "
            f"{m.get('ndcg@10', float('nan')):>9.4f} "
            f"{m.get('mrr@10', float('nan')):>8.4f} "
            f"{m.get('hr@50', float('nan')):>8.4f} "
            f"{int(row.get('n_users', 0)):>6}"
        )


def main() -> None:
    from run_train import _apply_recbole_dynamic_neg_device_fix, _apply_scipy_dok_compat

    _apply_scipy_dok_compat()
    _apply_recbole_dynamic_neg_device_fix()

    rank_note = "full_catalog (~20万 item)" if FULL_CATALOG else "uni100_approx"
    pool_note = "uncapped test" if USE_UNCAPPED_TEST else "capped test (yaml eval_test_user_cap)"

    print(
        f"SASRec trunc eval | users={USER_SAMPLE or 'ALL'} | K in {SEQUENCE_LENGTHS} "
        f"(0=全长) | mode={TRUNCATE_MODE} | {rank_note} | pool={pool_note} | seed={SAMPLE_SEED}"
    )
    print("目标=test 正例；K>0 时截断历史后 full_sort。")

    print("\n>>> load SASRec")
    sasrec = load_model_bundle("sasrec", SASREC_CKPT)
    device = sasrec.trainer.device

    cases = build_test_cases(sasrec.dataset, sasrec.config, use_uncapped=USE_UNCAPPED_TEST)
    print_hist_pool_stats(cases)
    users = sample_users(cases, USER_SAMPLE, SAMPLE_SEED, min_hist_len=1)
    lens = [u["hist_len"] for u in users]
    print(
        f"sampled {len(users)} users | hist_len min/median/max = "
        f"{min(lens)}/{int(np.median(lens))}/{max(lens)}"
    )

    results: list[dict[str, Any]] = []
    for k_hist in SEQUENCE_LENGTHS:
        row = eval_at_k(
            users=users,
            k_hist=int(k_hist),
            bundle=sasrec,
            device=device,
            topk_list=METRIC_TOPK,
            full_catalog=FULL_CATALOG,
            truncate_mode=TRUNCATE_MODE,
            batch_size=SASREC_BATCH,
            show_progress=SHOW_PROGRESS,
        )
        results.append(row)
        m = row.get("metrics") or {}
        label = row.get("k_label", k_hist)
        print(
            f"  K={label!s:>4} | HR@10={m.get('hr@10', 0):.4f} | "
            f"NDCG@10={m.get('ndcg@10', 0):.4f} | n={row.get('n_users', 0)}"
        )

    _print_curve(results, topk=10)

    if SAVE_JSON:
        out_dir = os.path.join(PLATFORM_ROOT, "results", "logs")
        os.makedirs(out_dir, exist_ok=True)
        tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(out_dir, f"sasrec_trunc_curve_{tag}.json")
        payload = {
            "user_sample": USER_SAMPLE,
            "sample_seed": SAMPLE_SEED,
            "use_uncapped_test": USE_UNCAPPED_TEST,
            "sequence_lengths": SEQUENCE_LENGTHS,
            "truncate_mode": TRUNCATE_MODE,
            "full_catalog": FULL_CATALOG,
            "protocol": {
                "target": "test_item",
                "ranking": "full_catalog" if FULL_CATALOG else "uni100_approx",
                "mask": "truncated_history + pad0",
                "k0": "full history before test",
            },
            "results": results,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        print(f"\njson: {out_path}")


if __name__ == "__main__":
    main()
