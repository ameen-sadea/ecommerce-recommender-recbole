# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
import json
import os
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from crossdomain_neumf.data_inter import cap_eval_users, load_splits_from_recbole_inter
from crossdomain_neumf.model import (
    CrossDomainDataset,
    CrossDomainNeuMF,
    UserHistoryCSR,
    compute_implicit_train_loss,
    evaluate_crossdomain,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x


DEFAULT_CONFIG = {
    "embed_dim": 64,
    "domain_embed_dim": 16,
    "mlp_hidden": (256, 128),
    "dropout": 0.4,
    "num_negatives": 32,
    "train_loss": "bpr",
    "gmf_domain_aware": True,
    "share_embeddings": True,
    "batch_size": 512,
    "lr": 0.001,
    "weight_decay": 1e-5,
    "epochs": 40,
    "patience": 4,
    "val_users": 20000,
    "test_eval_users": 20000,
    "eval_negatives": 100,
    "eval_sample_seed": 42,
    "eval_same_domain_negatives": True,
    "grad_clip": 5.0,
    "scheduler_patience": 3,
    "checkpoint_every": 10,
    "amp": True,
    "num_workers": 2,
    "seed": 42,
}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_domain_pools(train_df, valid_df, test_df) -> dict:
    all_items = pd.concat(
        [
            train_df[["global_item_id", "domain_id"]],
            valid_df[["global_item_id", "domain_id"]],
            test_df[["global_item_id", "domain_id"]],
        ],
        ignore_index=True,
    )
    pools: dict = {}
    for d in all_items["domain_id"].unique():
        pools[int(d)] = (
            all_items[all_items["domain_id"] == d]["global_item_id"].unique().tolist()
        )
    return pools


def _metric_result(hr: float, ndcg: float, mrr: float, gauc: float | None = None) -> dict:
    """与 RecBole test_result 字段一致，便于 results/logs JSON 对比。"""
    out = {
        "hit@10": hr,
        "hr@10": hr,
        "ndcg@10": ndcg,
        "mrr@10": mrr,
        "recall@10": hr,
    }
    if gauc is not None:
        gauc_str = f"{gauc:.4f}"
        out["gauc"] = gauc_str
        out["auc"] = gauc_str
    return out


def run_crossdomain_training(
    *,
    data_path: str,
    dataset_name: str,
    output_dir: str,
    config: dict | None = None,
    resume: str | None = None,
    tag: str | None = None,
    eval_only: bool = False,
) -> dict:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    _set_seed(int(cfg["seed"]))

    train_df, valid_df, test_df, meta = load_splits_from_recbole_inter(
        data_path,
        dataset_name,
        train_row_limit=cfg.get("train_row_limit"),
    )
    num_users = meta["num_global_users"]
    num_items = meta["num_global_items"]
    num_domains = meta["num_domains"]

    u_np = train_df["global_user_id"].to_numpy(dtype=np.int32, copy=False)
    i_np = train_df["global_item_id"].to_numpy(dtype=np.int32, copy=False)
    user_history = UserHistoryCSR.from_interactions(u_np, i_np, num_users)
    domain_item_pools = _build_domain_pools(train_df, valid_df, test_df)

    eval_num_neg = int(cfg["eval_negatives"])
    eval_same_domain = bool(cfg["eval_same_domain_negatives"])
    global_item_pool = None

    train_records = list(
        train_df[["global_user_id", "global_item_id", "domain_id"]].itertuples(
            index=False, name=None
        )
    )
    train_neg = int(cfg.get("num_negatives", 32))
    dataset = CrossDomainDataset(
        train_records, user_history, domain_item_pools, train_neg
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 0)),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp_hidden = tuple(cfg["mlp_hidden"])
    model = CrossDomainNeuMF(
        num_users,
        num_items,
        num_domains,
        embed_dim=int(cfg["embed_dim"]),
        domain_embed_dim=int(cfg["domain_embed_dim"]),
        mlp_hidden=mlp_hidden,
        dropout=float(cfg["dropout"]),
        share_embeddings=bool(cfg.get("share_embeddings", True)),
        gmf_domain_aware=bool(cfg.get("gmf_domain_aware", True)),
    ).to(device)

    train_loss_type = str(cfg.get("train_loss", "bpr")).lower()
    optimizer = optim.Adam(
        model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"])
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=int(cfg.get("scheduler_patience", 3)),
    )

    start_epoch = 0
    best_score = -1.0
    best_epoch = -1
    no_improve = 0
    best_model_state = None

    if resume and os.path.isfile(resume):
        print(f"[resume] {resume}")
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        if not eval_only:
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_score = float(ckpt.get("best_val_hr", -1.0))
            best_epoch = int(ckpt.get("best_epoch", -1))
            no_improve = int(ckpt.get("no_improve", 0))
            if ckpt.get("optimizer_state"):
                optimizer.load_state_dict(ckpt["optimizer_state"])
            if config and "lr" in config:
                for g in optimizer.param_groups:
                    g["lr"] = float(cfg["lr"])
                print(f"  覆盖 lr={cfg['lr']:.2e}")
            print(f"  续训 epoch={start_epoch}, best Val HR@10={best_score:.4f}")
        else:
            best_model_state = copy.deepcopy(model.state_dict())
            print(">>> EVAL_ONLY：跳过训练，仅评估")

    eval_seed = int(cfg.get("eval_sample_seed", cfg["seed"]))
    eval_rng = np.random.RandomState(eval_seed)
    val_sample = cap_eval_users(
        valid_df, "global_user_id", cfg.get("val_users"), eval_rng
    )
    test_eval_cap = cfg.get("test_eval_users")
    log_path = os.path.join(output_dir, "training_log.jsonl")
    epochs = int(cfg["epochs"])
    patience = int(cfg["patience"])
    ckpt_every = int(cfg.get("checkpoint_every", 10))

    print(
        f"CrossDomainNeuMF | device={device} users={num_users:,} items={num_items:,} "
        f"train={len(train_df):,} valid={len(valid_df):,} test={len(test_df):,}"
    )
    print(
        f"embed={cfg['embed_dim']} domain={cfg['domain_embed_dim']} mlp={mlp_hidden} "
        f"train_loss={train_loss_type} train_neg={cfg['num_negatives']} "
        f"eval_neg={eval_num_neg} batch={cfg['batch_size']}"
    )

    scaler = GradScaler(
        device="cuda",
        enabled=bool(cfg.get("amp", False)) and device.type == "cuda",
    )

    interrupted = False
    best_val_ndcg = 0.0
    best_val_mrr = 0.0
    if not eval_only:
        try:
            for ep in range(start_epoch, epochs):
                model.train()
                total_loss = 0.0
                n_batches = 0
                t0 = time.time()
                use_amp = bool(cfg.get("amp", False)) and device.type == "cuda"

                for users_b, items_b, domains_b in tqdm(
                    dataloader, desc=f"Epoch {ep + 1}/{epochs}"
                ):
                    users_b = users_b.to(device)
                    items_b = items_b.to(device)
                    domains_b = domains_b.to(device)
                    optimizer.zero_grad()
                    with autocast(device_type="cuda", enabled=use_amp):
                        out = model(users_b, items_b, domains_b)
                        loss = compute_implicit_train_loss(out, train_loss_type)
                    scaler.scale(loss).backward()
                    if cfg.get("grad_clip"):
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), float(cfg["grad_clip"])
                        )
                    scaler.step(optimizer)
                    scaler.update()
                    total_loss += loss.item()
                    n_batches += 1

                avg_loss = total_loss / max(n_batches, 1)
                current_lr = optimizer.param_groups[0]["lr"]
                train_time = time.time() - t0

                t1 = time.time()
                val_hr, val_ndcg, val_mrr, _, val_gauc = evaluate_crossdomain(
                    model,
                    val_sample,
                    user_history,
                    domain_item_pools,
                    device,
                    num_negatives=eval_num_neg,
                    same_domain_negatives=eval_same_domain,
                    global_item_pool=global_item_pool,
                )
                eval_time = time.time() - t1
                scheduler.step(val_hr)

                print(
                    f"Epoch {ep + 1:02d}/{epochs} | Loss: {avg_loss:.4f} | "
                    f"Val HR@10: {val_hr:.4f} | Val NDCG@10: {val_ndcg:.4f} | Val MRR: {val_mrr:.4f} | "
                    f"Val GAUC: {val_gauc:.4f} | "
                    f"lr: {current_lr:.2e} | train_s: {train_time:.1f} eval_s: {eval_time:.1f}"
                )
                with open(log_path, "a", encoding="utf-8") as lf:
                    lf.write(
                        json.dumps(
                            {
                                "epoch": ep + 1,
                                "loss": round(avg_loss, 6),
                                "val_hr10": round(val_hr, 6),
                                "val_ndcg10": round(val_ndcg, 6),
                                "val_mrr": round(val_mrr, 6),
                                "lr": current_lr,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                if ckpt_every > 0 and (ep + 1) % ckpt_every == 0:
                    torch.save(
                        {
                            "epoch": ep,
                            "state_dict": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "best_val_hr": best_score,
                            "best_epoch": best_epoch,
                            "no_improve": no_improve,
                            "config": cfg,
                            "meta": meta,
                        },
                        os.path.join(output_dir, f"checkpoint_ep{ep + 1:02d}.pt"),
                    )

                if val_hr > best_score:
                    best_score = val_hr
                    best_val_ndcg = val_ndcg
                    best_val_mrr = val_mrr
                    best_epoch = ep
                    no_improve = 0
                    best_model_state = copy.deepcopy(model.state_dict())
                    torch.save(
                        {
                            "epoch": ep,
                            "state_dict": best_model_state,
                            "optimizer_state": optimizer.state_dict(),
                            "best_val_hr": best_score,
                            "best_epoch": best_epoch,
                            "no_improve": 0,
                            "config": cfg,
                            "meta": meta,
                        },
                        os.path.join(output_dir, "best_ckpt.pt"),
                    )
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(
                            f"Early stopping @ epoch {ep + 1}, "
                            f"best={best_epoch + 1} Val HR@10={best_score:.4f}"
                        )
                        break
        except KeyboardInterrupt:
            interrupted = True
            print("\n>>> 收到 Ctrl+C，将用当前 best 权重做 Test...")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.eval()

    test_sample = cap_eval_users(test_df, "global_user_id", test_eval_cap, eval_rng)
    n_test = len(test_sample)
    print(
        f"\nTest eval ({n_test} users, 1+{eval_num_neg} neg, seed={eval_seed})..."
    )
    test_hr, test_ndcg, test_mrr, test_mean_rank, test_gauc = evaluate_crossdomain(
        model,
        test_sample,
        user_history,
        domain_item_pools,
        device,
        num_negatives=eval_num_neg,
        same_domain_negatives=eval_same_domain,
        global_item_pool=global_item_pool,
    )
    print(
        f"TEST -> HR@10: {test_hr:.4f}  NDCG@10: {test_ndcg:.4f}  "
        f"MRR: {test_mrr:.4f}  GAUC: {test_gauc:.4f}  MeanRank: {test_mean_rank:.2f}"
    )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "num_users": num_users,
            "num_items": num_items,
            "num_domains": num_domains,
            "config": cfg,
            "meta": meta,
        },
        os.path.join(output_dir, "best.pt"),
    )

    tag = tag or f"{dataset_name}_CrossDomainNeuMF_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    best_valid_result = _metric_result(best_score, best_val_ndcg, best_val_mrr)
    test_result = _metric_result(test_hr, test_ndcg, test_mrr, test_gauc)
    result = {
        "tag": tag,
        "dataset": dataset_name,
        "model": "CrossDomainNeuMF",
        "best_epoch": best_epoch + 1,
        "best_valid_score": best_val_ndcg,
        "best_valid_result": best_valid_result,
        "test_result": test_result,
        "best_valid_hr10": best_score,
        "test_hr10": test_hr,
        "test_ndcg10": test_ndcg,
        "test_mrr": test_mrr,
        "test_meanrank": test_mean_rank,
        "test_gauc": test_gauc,
        "eval_negatives": eval_num_neg,
        "checkpoint_dir": output_dir,
        "interrupted": interrupted,
        "eval_only": eval_only,
        "config": cfg,
    }
    with open(os.path.join(output_dir, "model_evaluation.json"), "w", encoding="utf-8") as f:
        json.dump({"CrossDomainNeuMF": result}, f, indent=2, ensure_ascii=False)

    platform_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logs_dir = os.path.join(platform_root, "results", "logs")
    os.makedirs(logs_dir, exist_ok=True)
    platform_log = os.path.join(logs_dir, f"{tag}.json")
    with open(platform_log, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n指标已写入: {platform_log}")
    return result
