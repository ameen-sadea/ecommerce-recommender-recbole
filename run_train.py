#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
一个脚本启动训练/评估（替代一堆命令行参数）。

你只需要改最下面的“可调区域”，然后运行：
  python run_train.py
"""

from __future__ import annotations

import copy
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import yaml

# PyTorch 2.6+ 默认 weights_only=True 会导致 RecBole 读取 checkpoint 报错
_orig_torch_load = torch.load


def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_compat  # type: ignore[assignment]


def _apply_scipy_dok_compat() -> None:
    """scipy>=1.14 移除 dok_matrix._update；RecBole 图模型（LightGCN 等）仍调用它。"""
    import scipy.sparse as sp

    if hasattr(sp.dok_matrix, "_update"):
        return

    def _dok_update(self, data_dict):  # type: ignore[no-untyped-def]
        for (i, j), val in data_dict.items():
            self[i, j] = val

    sp.dok_matrix._update = _dok_update  # type: ignore[attr-defined]


def _apply_recbole_dynamic_neg_device_fix() -> None:
    """
    RecBole 在 GPU + dynamic 难负例时，indices 在 CUDA、neg_candidate_ids 在 CPU 会报错。
    将 indices 移到与 neg_candidate_ids 相同设备后再索引。
    """
    import copy

    from recbole.data.dataloader.abstract_dataloader import NegSampleDataLoader
    from recbole.data.interaction import Interaction

    if getattr(NegSampleDataLoader, "_platform_dynamic_neg_patched", False):
        return

    _orig_neg_sampling = NegSampleDataLoader._neg_sampling

    def _neg_sampling(self, inter_feat):  # type: ignore[no-untyped-def]
        if not self.neg_sample_args.get("dynamic", False):
            return _orig_neg_sampling(self, inter_feat)

        candidate_num = self.neg_sample_args["candidate_num"]
        user_ids = inter_feat[self.uid_field].numpy()
        item_ids = inter_feat[self.iid_field].numpy()
        neg_candidate_ids = self._sampler.sample_by_user_ids(
            user_ids, item_ids, self.neg_sample_num * candidate_num
        )
        self.model.eval()
        interaction = copy.deepcopy(inter_feat).to(self.model.device)
        interaction = interaction.repeat(self.neg_sample_num * candidate_num)
        neg_item_feat = Interaction(
            {self.iid_field: neg_candidate_ids.to(self.model.device)}
        )
        interaction.update(neg_item_feat)
        scores = self.model.predict(interaction).reshape(candidate_num, -1)
        indices = torch.max(scores, dim=0)[1].detach()
        if indices.device != neg_candidate_ids.device:
            indices = indices.to(neg_candidate_ids.device)
        neg_candidate_ids = neg_candidate_ids.reshape(candidate_num, -1)
        neg_item_ids = neg_candidate_ids[
            indices, [i for i in range(neg_candidate_ids.shape[1])]
        ].view(-1)
        self.model.train()
        return self.sampling_func(inter_feat, neg_item_ids)

    NegSampleDataLoader._neg_sampling = _neg_sampling  # type: ignore[assignment]
    NegSampleDataLoader._platform_dynamic_neg_patched = True


def _sanitize_training_config(cfg: dict, model_name: str) -> list[str]:
    """
    消除常见「双重正则」：LightGCN 等用 reg_weight（EmbLoss），optimizer 的 weight_decay 应置 0。
    返回发给用户的提示信息列表。
    """
    notes: list[str] = []
    model_upper = (model_name or "").upper()
    reg_weight = float(cfg.get("reg_weight") or 0)
    weight_decay = cfg.get("weight_decay")
    if weight_decay is None:
        return notes
    wd = float(weight_decay)
    graph_models = {"LIGHTGCN", "NGCF", "NCL", "SGL", "DGCF", "SPECTRALCF"}
    if model_upper in graph_models and reg_weight > 0 and wd > 0:
        cfg["weight_decay"] = 0.0
        notes.append(
            f"{model_name}: 已把 weight_decay 从 {wd:g} 改为 0 "
            f"（与 reg_weight={reg_weight:g} 重复，RecBole 会双重正则）"
        )
    if not is_sequential_model(model_name):
        tna = cfg.get("train_neg_sample_args") or {}
        if tna.get("dynamic") and int(tna.get("candidate_num") or 0) <= 0:
            notes.append(
                "neg_sampling_dynamic=true 但 candidate_num=0，"
                "难负例几乎无效，建议设为 100~500"
            )
    return notes


# RecBole 序列模型（MODEL_TYPE=SEQUENTIAL）；训练多为 CE，不用 BPR 式 train 负采样
SEQUENTIAL_MODEL_NAMES = frozenset(
    {
        "SASREC",
        "BERT4REC",
        "GRU4REC",
        "NARM",
        "SRGNN",
        "FPMC",
        "STAMP",
        "CORE",
        "HGN",
        "HPMN",
        "LIGHTSAN",
        "SASRECPR",
        "FEAREC",
    }
)

TRADITIONAL_MODEL_NAMES = frozenset({"POP", "ITEMKNN", "EASE", "SLIMElastic", "ADMMSLIM"})

MODEL_CONFIG_KEYS = {
    "bpr": "configs/bpr_movies_tv_full.yaml",
    "neumf": "configs/neumf_movies_tv_full.yaml",
    "lightgcn": "configs/lightgcn_movies_tv_full.yaml",
    "sasrec": "configs/sasrec_movies_tv_full.yaml",
    "sasrec_k5": "configs/sasrec_movies_tv_k5.yaml",
    "sasrec_k2": "configs/sasrec_movies_tv_k2.yaml",
    "bert4rec": "configs/bert4rec_movies_tv_full.yaml",
    "pop": "configs/pop_movies_tv_full.yaml",
    "itemknn": "configs/itemknn_movies_tv_full.yaml",
    "crossdomain_neumf": "configs/crossdomain_neumf_movies_tv_full.yaml",
}

# general 模型 → state_dict 中 item embedding 键名
_GENERAL_ITEM_EMBED_KEYS = {
    "bpr": "item_embedding.weight",
    "neumf": "item_mf_embedding.weight",
    "neumf_mlp": "item_mlp_embedding.weight",
}
_GENERAL_MODEL_CKPT_DIRS = {
    "bpr": "BPR",
    "neumf": "NeuMF",
    "neumf_mlp": "NeuMF",
}


def is_sequential_model(model_name: str | None) -> bool:
    return (model_name or "").upper() in SEQUENTIAL_MODEL_NAMES


def is_traditional_model(model_name: str | None) -> bool:
    return (model_name or "").upper() in TRADITIONAL_MODEL_NAMES


def supported_model_keys() -> list[str]:
    return sorted(MODEL_CONFIG_KEYS.keys())


PLATFORM_ROOT = os.path.dirname(os.path.abspath(__file__))

# D 盘权重根目录：CHECKPOINT_ROOT/<dataset>/<ModelName>/best.pth
# 例：D:\recbole_checkpoints\movies_tv\BPR\best.pth
CHECKPOINT_ROOT = r"D:\recbole_checkpoints"
BEST_CKPT_NAME = "best.pth"
BEST_META_NAME = "best_meta.json"


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge_dict(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, val in override.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(val, dict)
        ):
            out[key] = _deep_merge_dict(out[key], val)
        else:
            out[key] = val
    return out


def merge_negative_sampling_defaults(cfg: dict) -> dict:
    """合并 configs/negative_sampling_defaults.yaml（模型 yaml 优先）。"""
    shared_path = os.path.join(PLATFORM_ROOT, "configs/negative_sampling_defaults.yaml")
    if not os.path.isfile(shared_path):
        return cfg
    shared = load_yaml(shared_path)
    return _deep_merge_dict(shared, cfg)


# 平台自定义键（不会传给 RecBole Config；负采样别名在 apply_negative_sampling_config 里 pop）
_PLATFORM_ONLY_KEYS = frozenset(
    {
        "num_negatives_train",
        "num_negatives_eval",
        "eval_neg_mode",
        "neg_sampling_distribution",
        "neg_sampling_dynamic",
        "neg_sampling_candidate_num",
        "neg_sampling_alpha",
        "test_full_sort_enabled",
        "test_full_sort_topk",
        "full_catalog_eval_enabled",
        "full_catalog_eval_topk",
        "full_catalog_eval_on",
        "full_catalog_eval_user_cap",
        # CrossDomainNeuMF（非 RecBole Config）
        "embed_dim",
        "domain_embed_dim",
        "mlp_hidden",
        "dropout",
        "eval_negatives",
        "val_users",
        "test_eval_users",
        "eval_sample_seed",
        "gmf_domain_aware",
        "share_embeddings",
        "train_loss",
        "patience",
        "batch_size",
        "amp",
        "grad_clip",
        "scheduler_patience",
        "checkpoint_every",
        "num_workers",
        "train_row_limit",
        "init_item_embedding_from",
        "init_item_embedding_ckpt",
    }
)


def normalize_hr_topk_list(topk) -> list[int]:
    """单个 K 或列表 [10, 20, 50] → 去重排序。"""
    if topk is None:
        return [50]
    if isinstance(topk, (list, tuple)):
        return sorted({int(x) for x in topk})
    return [int(topk)]


def build_full_catalog_metric_plan(
    *,
    compute_hr_10: bool = True,
    compute_hr_50: bool = True,
    compute_ndcg: bool = True,
    compute_mrr: bool = True,
    compute_meanrank: bool = True,
    extra_hr_at_k: list[int] | None = None,
) -> dict:
    """
    全库评估指标开关 → RecBole metrics / topk。
    HR@K 对应 Hit@K（输出时复制为 hr@K）。
    meanrank 需 metrics 含 GAUC 以采集 rec.meanrank。
    """
    hr_topk = []
    if compute_hr_10:
        hr_topk.append(10)
    if compute_hr_50:
        hr_topk.append(50)
    for k in extra_hr_at_k or []:
        ki = int(k)
        if ki not in hr_topk:
            hr_topk.append(ki)

    ranking_topk = set(hr_topk)
    if compute_ndcg or compute_mrr:
        ranking_topk |= set(hr_topk)

    metrics: list[str] = []
    if hr_topk:
        metrics.append("Hit")
    if compute_ndcg and ranking_topk:
        metrics.append("NDCG")
    if compute_mrr and ranking_topk:
        metrics.append("MRR")
    if compute_meanrank:
        metrics.append("GAUC")

    if not metrics:
        raise ValueError(
            "至少开启一项指标：compute_hr_10 / compute_hr_50 / compute_meanrank 等"
        )
    if ("Hit" in metrics or "NDCG" in metrics or "MRR" in metrics) and not ranking_topk:
        raise ValueError("计算 HR/NDCG/MRR@K 时至少开启 COMPUTE_HR_10 或 COMPUTE_HR_50")
    if compute_meanrank and not ranking_topk:
        ranking_topk.add(10)

    return {
        "topk_list": sorted(ranking_topk),
        "metrics": metrics,
        "compute_meanrank": bool(compute_meanrank),
        "compute_hit": bool(hr_topk),
        "compute_ndcg": bool(compute_ndcg and ranking_topk),
        "compute_mrr": bool(compute_mrr and ranking_topk),
        "hr_topk_list": sorted(hr_topk),
    }


def _default_full_catalog_metric_plan(topk_list: list[int]) -> dict:
    """未显式传 plan 时：对 topk_list 中每个 K 算 Hit/NDCG/MRR，并算 meanrank。"""
    ks = normalize_hr_topk_list(topk_list)
    return build_full_catalog_metric_plan(
        compute_hr_10=10 in ks or not ks,
        compute_hr_50=50 in ks,
        compute_ndcg=True,
        compute_mrr=True,
        compute_meanrank=True,
        extra_hr_at_k=[k for k in ks if k not in (10, 50)],
    )


def ensure_sequential_benchmark_dataset(cfg: dict) -> str | None:
    """
    SASRec + benchmark_filename 需要 item_id_list 列。
    若 datasets/<dataset>_seq 不存在则从扁平 movies_tv 自动生成。
    返回提示信息（已存在 / 刚生成），无改动则返回 None。
    """
    model = cfg.get("model")
    if not is_sequential_model(model):
        return None
    if not cfg.get("benchmark_filename"):
        return None

    data_path = cfg.get("data_path", "datasets/")
    base_dataset = str(cfg.get("dataset", "movies_tv"))
    if "_seq" in base_dataset:
        src_dataset = base_dataset[: base_dataset.index("_seq")]
        seq_name = base_dataset
    else:
        src_dataset = base_dataset
        seq_name = f"{src_dataset}_seq"
        cfg["dataset"] = seq_name
    if not os.path.isabs(data_path):
        data_path = os.path.join(PLATFORM_ROOT, data_path)
    marker = os.path.join(data_path, seq_name, f"{seq_name}.train.inter")
    if os.path.isfile(marker):
        return f"sequential data: {seq_name} (cached)"

    scripts_dir = os.path.join(PLATFORM_ROOT, "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from build_sequential_dataset import build_sequential_dataset  # noqa: E402

    max_len = int(cfg.get("MAX_ITEM_LIST_LENGTH", 50))
    flat_dir = os.path.join(data_path, src_dataset)
    if not os.path.isdir(flat_dir):
        raise FileNotFoundError(
            f"序列数据需从扁平集 {src_dataset} 生成，目录不存在: {flat_dir}"
        )
    print(f"\n>>> 正在生成序列数据集 {seq_name}（来自 {src_dataset}）...")
    build_sequential_dataset(
        src_dataset, data_root=data_path, max_len=max_len, seq_name=seq_name
    )
    return f"sequential data: built {seq_name} from {src_dataset}"


def apply_sequential_model_config(cfg: dict) -> dict:
    """序列模型：保证 MAX_ITEM_LIST_LENGTH；训练阶段不用 BPR 负采样。"""
    model = cfg.get("model")
    if not is_sequential_model(model):
        return cfg
    cfg.setdefault("MAX_ITEM_LIST_LENGTH", 50)
    eval_args = dict(cfg.get("eval_args") or {})
    eval_args.setdefault("order", "TO")
    eval_args.setdefault("group_by", "user")
    cfg["eval_args"] = eval_args
    # CE 训练时 RecBole 要求 train_neg_sample_args 为 None（共享 defaults 里会有 dict）
    if str(cfg.get("loss_type", "CE")).upper() == "CE":
        cfg["train_neg_sample_args"] = None
    return cfg


def apply_negative_sampling_config(cfg: dict) -> dict:
    """
    将 yaml 里的「负例个数 / 动态难负例」转为 RecBole 字段，并移除平台别名键。

    RecBole 原名（也可直接写在 yaml，与本节合并，本节别名优先）：
      train_neg_sample_args: {sample_num, distribution, dynamic, candidate_num, alpha}
      eval_args.mode: {valid: uni100 | full | pop100, test: ...}

    序列模型（SASRec 等，loss_type=CE）：仅处理 eval 的 uni{N}，忽略 train 负采样别名。
    """
    sequential = is_sequential_model(cfg.get("model"))
    train_n = cfg.pop("num_negatives_train", None)
    eval_n = cfg.pop("num_negatives_eval", None)
    eval_mode = cfg.pop("eval_neg_mode", None)  # full | uni | pop
    dist = cfg.pop("neg_sampling_distribution", None)
    dynamic = cfg.pop("neg_sampling_dynamic", None)
    candidate = cfg.pop("neg_sampling_candidate_num", None)
    alpha = cfg.pop("neg_sampling_alpha", None)

    if sequential and any(
        x is not None for x in (train_n, dist, dynamic, candidate, alpha)
    ):
        pass  # SASRec(CE) 训练不读 train_neg_sample_args
    elif any(
        x is not None
        for x in (train_n, dist, dynamic, candidate, alpha)
    ):
        tna = dict(cfg.get("train_neg_sample_args") or {})
        if train_n is not None:
            tna["sample_num"] = int(train_n)
        if dist is not None:
            tna["distribution"] = str(dist)
        if dynamic is not None:
            tna["dynamic"] = bool(dynamic)
        if candidate is not None:
            tna["candidate_num"] = int(candidate)
        if alpha is not None:
            tna["alpha"] = float(alpha)
        cfg["train_neg_sample_args"] = {
            "distribution": tna.get("distribution", "uniform"),
            "sample_num": tna.get("sample_num", 1),
            "alpha": tna.get("alpha", 1.0),
            "dynamic": tna.get("dynamic", False),
            "candidate_num": tna.get("candidate_num", 0),
        }

    eval_args = dict(cfg.get("eval_args") or {})
    mode = eval_args.get("mode")
    if isinstance(mode, str):
        mode = {"valid": mode, "test": mode}
    elif not isinstance(mode, dict):
        mode = {}

    if eval_mode is not None:
        em = str(eval_mode).lower()
        if em == "full":
            mode["valid"] = mode["test"] = "full"
        elif em in ("uni", "pop"):
            if eval_n is None:
                raise ValueError(
                    f"eval_neg_mode={eval_mode!r} 时需要 num_negatives_eval（负例个数 N）"
                )
            prefix = em
            n = int(eval_n)
            mode["valid"] = mode["test"] = f"{prefix}{n}"
        else:
            raise ValueError(
                f"未知 eval_neg_mode={eval_mode!r}，应为 full | uni | pop"
            )
    elif eval_n is not None:
        n = int(eval_n)
        mode["valid"] = mode["test"] = f"uni{n}"

    if mode:
        eval_args["mode"] = mode
        cfg["eval_args"] = eval_args

    return cfg


def pop_platform_eval_options(cfg: dict) -> dict:
    """
    全库 HR@K 评估（对所有 item 排序，非 uni100）。

    yaml 字段（推荐）：
      full_catalog_eval_enabled: true
      full_catalog_eval_topk: [10, 20, 50]   # 或单个 50
      full_catalog_eval_on: test              # test | valid | both
      full_catalog_eval_user_cap: 0           # 0=全库用该 split 全部用户；训练期仍可用 eval_*_user_cap

    兼容旧名：test_full_sort_enabled / test_full_sort_topk
    """
    enabled = cfg.pop("full_catalog_eval_enabled", None)
    if enabled is None:
        enabled = cfg.pop("test_full_sort_enabled", False)
    topk = cfg.pop("full_catalog_eval_topk", None)
    if topk is None:
        topk = cfg.pop("test_full_sort_topk", 50)
    on = cfg.pop("full_catalog_eval_on", "test")
    # 0 或缺省 = 全库评估用该 split 的全部用户（不受 eval_test_user_cap 限制）
    user_cap = cfg.pop("full_catalog_eval_user_cap", 0)
    topk_list = normalize_hr_topk_list(topk)
    on = str(on).lower()
    if on not in ("test", "valid", "both"):
        raise ValueError(f"full_catalog_eval_on 应为 test|valid|both，当前: {on!r}")
    phases = ["valid", "test"] if on == "both" else [on]
    return {
        "full_catalog_eval_enabled": bool(enabled),
        "full_catalog_eval_topk": topk_list,
        "full_catalog_eval_on": on,
        "full_catalog_eval_phases": phases,
        "full_catalog_eval_user_cap": user_cap,
        # 兼容旧代码路径
        "test_full_sort_enabled": bool(enabled),
        "test_full_sort_topk": topk_list[-1] if topk_list else 50,
    }


def strip_platform_only_config_keys(cfg: dict) -> dict:
    """去掉 RecBole 不认识的自定义键。"""
    for key in list(cfg.keys()):
        if key in _PLATFORM_ONLY_KEYS:
            cfg.pop(key, None)
    return cfg


def base_dataset_name(dataset: str) -> str:
    """movies_tv_seq / movies_tv_seq_k5 → movies_tv（general 模型 checkpoint 在扁平集目录下）。"""
    ds = (dataset or "").strip()
    if "_seq" in ds:
        return ds[: ds.index("_seq")]
    return ds


def detect_seq_list_width(inter_path: str, *, sample_lines: int = 5000) -> int:
    """从 .inter 的 item_id_list 列估计序列宽度（空格分隔 token 数）。"""
    max_w = 0
    with open(inter_path, encoding="utf-8") as f:
        next(f, None)
        for i, line in enumerate(f):
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3 and parts[2]:
                max_w = max(max_w, len(parts[2].split()))
            if i + 1 >= sample_lines:
                break
    return max_w


def align_max_item_list_length_with_seq_files(cfg: dict) -> str | None:
    """
    已生成的 *_seq 文件若比 yaml MAX_ITEM_LIST_LENGTH 更长，RecBole 仍会读全宽序列，
    但 BERT4Rec/SASRec 的 position_embedding 只有 max_seq_length 行 → CUDA 越界。
    自动把 MAX_ITEM_LIST_LENGTH 提升到与数据一致。
    """
    if not is_sequential_model(cfg.get("model")):
        return None
    data_path = cfg.get("data_path", "datasets/")
    seq_name = str(cfg.get("dataset", ""))
    if "_seq" not in seq_name:
        return None
    if not os.path.isabs(data_path):
        data_path = os.path.join(PLATFORM_ROOT, data_path)
    marker = os.path.join(data_path, seq_name, f"{seq_name}.train.inter")
    if not os.path.isfile(marker):
        return None
    width = detect_seq_list_width(marker)
    if width <= 0:
        return None
    configured = int(cfg.get("MAX_ITEM_LIST_LENGTH", 50))
    if width <= configured:
        return None
    cfg["MAX_ITEM_LIST_LENGTH"] = width
    return (
        f"MAX_ITEM_LIST_LENGTH {configured} → {width} "
        f"（{seq_name} 序列文件更宽，已自动对齐以免 position_embedding 越界）"
    )


def resolve_checkpoint_dir(cfg_source: dict, dataset: str, model_name: str) -> str:
    """优先 yaml 里的 checkpoint_dir；否则 CHECKPOINT_ROOT/<base_dataset>/<Model>/。"""
    raw = cfg_source.get("checkpoint_dir")
    if raw:
        path = str(raw).strip()
        if not os.path.isabs(path):
            path = os.path.join(PLATFORM_ROOT, path)
        return path
    return checkpoint_dir_for(base_dataset_name(dataset), model_name)


def pop_item_embedding_init_options(cfg: dict) -> dict:
    """
    序列模型 warm-start：从 BPR/NeuMF 拷贝 item embedding。
    yaml: init_item_embedding_from: bpr | neumf | neumf_mlp | null
          init_item_embedding_ckpt: null → CHECKPOINT_ROOT/<base_dataset>/<Model>/best.pth
    """
    source = cfg.pop("init_item_embedding_from", None)
    ckpt = cfg.pop("init_item_embedding_ckpt", None)
    if source is None or str(source).strip().lower() in (
        "",
        "false",
        "none",
        "null",
        "0",
        "off",
    ):
        return {"enabled": False}
    source_key = str(source).strip().lower()
    if source_key not in _GENERAL_ITEM_EMBED_KEYS:
        raise ValueError(
            f"init_item_embedding_from 应为 bpr|neumf|neumf_mlp|null，当前: {source!r}"
        )
    return {
        "enabled": True,
        "source": source_key,
        "ckpt": ckpt.strip() if isinstance(ckpt, str) and ckpt.strip() else None,
    }


def resolve_general_item_embedding_ckpt(source: str, base_dataset: str, ckpt: str | None) -> str:
    if ckpt:
        path = ckpt
        if not os.path.isabs(path):
            path = os.path.join(PLATFORM_ROOT, path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"init_item_embedding_ckpt 不存在: {path}")
        return path
    model_dir = _GENERAL_MODEL_CKPT_DIRS[source]
    path = os.path.join(CHECKPOINT_ROOT, base_dataset, model_dir, BEST_CKPT_NAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"找不到 {source.upper()} 权重: {path}\n"
            f"请先训练 BPR/NeuMF，或在 yaml 设置 init_item_embedding_ckpt"
        )
    return path


def load_general_item_embedding_meta(
    init_opts: dict, base_dataset: str
) -> dict:
    """读取源 checkpoint 的 item embedding 形状（用于自动对齐 hidden_size）。"""
    source = init_opts["source"]
    ckpt_path = resolve_general_item_embedding_ckpt(
        source, base_dataset, init_opts.get("ckpt")
    )
    state_key = _GENERAL_ITEM_EMBED_KEYS[source]
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    if state_key not in state_dict:
        raise KeyError(f"{ckpt_path} 缺少 {state_key}")
    weight = state_dict[state_key]
    return {
        "source": source,
        "ckpt_path": ckpt_path,
        "state_key": state_key,
        "embed_dim": int(weight.shape[1]),
        "num_item_rows": int(weight.shape[0]),
        "weight": weight,
    }


def _pick_n_heads(hidden_size: int, preferred: int) -> int:
    """选不超过 preferred 且能整除 hidden_size 的最大 n_heads。"""
    preferred = max(1, int(preferred))
    for n in range(preferred, 0, -1):
        if hidden_size % n == 0:
            return n
    return 1


def prepare_item_embedding_init(
    cfg: dict,
    init_opts: dict,
    *,
    base_dataset: str,
    model_name: str,
) -> list[str]:
    """
    启用 init_item_embedding_from 时，在创建模型前：
    - 从源 checkpoint 读取 embedding 维数
    - 自动把 hidden_size 改成与源一致
    - 必要时调整 n_heads / inner_size
    """
    notes: list[str] = []
    if not init_opts.get("enabled") or not is_sequential_model(model_name):
        return notes

    meta = load_general_item_embedding_meta(init_opts, base_dataset)
    init_opts["meta"] = meta
    src_dim = meta["embed_dim"]
    cur_hidden = int(cfg.get("hidden_size", src_dim))
    if cur_hidden != src_dim:
        cfg["hidden_size"] = src_dim
        notes.append(
            f"init_item_embedding_from={meta['source']}: hidden_size "
            f"{cur_hidden} → {src_dim}（与 {meta['ckpt_path']} 对齐）"
        )

    cur_heads = int(cfg.get("n_heads", 2))
    new_heads = _pick_n_heads(src_dim, cur_heads)
    if new_heads != cur_heads:
        cfg["n_heads"] = new_heads
        notes.append(f"n_heads {cur_heads} → {new_heads}（须整除 hidden_size={src_dim}）")

    inner = int(cfg.get("inner_size", src_dim * 4))
    min_inner = max(src_dim * 2, src_dim)
    if inner < min_inner:
        cfg["inner_size"] = src_dim * 4
        notes.append(f"inner_size {inner} → {cfg['inner_size']}")

    notes.append(
        f"源 embedding: {meta['source'].upper()} dim={src_dim} "
        f"({meta['num_item_rows']} rows) @ {meta['ckpt_path']}"
    )
    return notes


def apply_init_item_embedding_from_general(
    model,
    *,
    init_opts: dict,
    base_dataset: str,
    target_model_name: str,
) -> None:
    """将 general 模型的 item embedding 写入 SASRec/BERT4Rec 的 item_embedding。"""
    if not init_opts.get("enabled"):
        return
    if not is_sequential_model(target_model_name):
        print(
            f">>> init_item_embedding_from 仅用于序列模型，跳过 ({target_model_name})"
        )
        return
    if not hasattr(model, "item_embedding"):
        raise AttributeError(f"{target_model_name} 无 item_embedding，无法 warm-start")

    meta = init_opts.get("meta")
    if meta is None:
        meta = load_general_item_embedding_meta(init_opts, base_dataset)
        init_opts["meta"] = meta

    src_weight = meta["weight"]
    tgt_weight = model.item_embedding.weight.data
    if src_weight.shape[1] != tgt_weight.shape[1]:
        raise ValueError(
            f"item embedding 维数仍不一致: src={src_weight.shape[1]} "
            f"tgt={tgt_weight.shape[1]}；请检查 prepare_item_embedding_init 是否已执行"
        )

    target_upper = (target_model_name or "").upper()
    if target_upper == "BERT4REC" and tgt_weight.shape[0] == src_weight.shape[0] + 1:
        n_rows = src_weight.shape[0]
        extra_note = "（保留 BERT4Rec mask token 行随机初始化）"
    else:
        n_rows = min(src_weight.shape[0], tgt_weight.shape[0])
        extra_note = ""

    with torch.no_grad():
        tgt_weight[:n_rows].copy_(src_weight[:n_rows])

    print(
        f">>> warm-start item_embedding: {meta['source'].upper()} → {target_model_name} "
        f"({n_rows} items × {src_weight.shape[1]} dims) from {meta['ckpt_path']}"
        f"{extra_note}"
    )


def summarize_negative_sampling(cfg: dict) -> str:
    modes = (cfg.get("eval_args") or {}).get("mode", {})
    if isinstance(modes, str):
        modes = {"valid": modes, "test": modes}
    if is_traditional_model(cfg.get("model")):
        model = cfg.get("model", "")
        extra = ""
        if (model or "").upper() == "ITEMKNN":
            extra = (
                f" k={cfg.get('k')} knn_method={cfg.get('knn_method', 'item')} "
                f"shrink={cfg.get('shrink', 0)} |"
            )
        return (
            f"traditional {model}: train=统计/预计算（通常 {cfg.get('epochs', 1)} epoch）"
            f"{extra} eval_mode: valid={modes.get('valid')} test={modes.get('test')}"
        )
    if is_sequential_model(cfg.get("model")):
        loss = cfg.get("loss_type", "CE")
        return (
            f"sequential train: loss_type={loss}, "
            f"MAX_ITEM_LIST_LENGTH={cfg.get('MAX_ITEM_LIST_LENGTH', 50)} | "
            f"eval_mode: valid={modes.get('valid')} test={modes.get('test')}"
        )
    tna = cfg.get("train_neg_sample_args") or {}
    return (
        f"train_neg: sample_num={tna.get('sample_num')} "
        f"dynamic={tna.get('dynamic')} candidate_num={tna.get('candidate_num')} | "
        f"eval_mode: valid={modes.get('valid')} test={modes.get('test')}"
    )


def pick_config(model_key: str) -> str:
    key = model_key.lower().strip()
    if key not in MODEL_CONFIG_KEYS:
        raise ValueError(
            f"未知模型 {model_key!r}. 可选: {supported_model_keys()}"
        )
    return os.path.join(PLATFORM_ROOT, MODEL_CONFIG_KEYS[key])


def format_hr_alias(result: dict, k: int) -> dict:
    """
    你熟悉的 HR@K 在 RecBole 里对应 Hit@K。
    这里把 hit@10 复制成 hr@10，便于看结果。
    """
    hit_key = f"hit@{k}"
    hr_key = f"hr@{k}"
    if hit_key in result and hr_key not in result:
        result = dict(result)
        result[hr_key] = result[hit_key]
    return result


def format_auc_alias(result: dict) -> dict:
    """
    RecBole 在 ranking 评估下不能与普通 AUC 混用；推荐场景用 GAUC（按用户 AUC）。
    输出时复制 gauc -> auc，便于与常见「验证 AUC」表述对齐。
    """
    if not result:
        return result
    out = dict(result)
    gauc = out.get("gauc")
    if gauc is not None and "auc" not in out:
        out["auc"] = gauc
    return out


def format_metric_aliases(result: dict, k: int) -> dict:
    """Hit->HR@k，GAUC->auc。"""
    if not result:
        return result
    return format_auc_alias(format_hr_alias(dict(result), k))


def format_hr_aliases(result: dict, ks: list[int]) -> dict:
    out = dict(result) if result else {}
    for ki in ks:
        out = format_hr_alias(out, int(ki))
    return format_auc_alias(out)


def _arithmetic_mean_rank_from_eval_struct(dataobject) -> float | None:
    """
    算术平均排名（Mean Rank）：每个用户真实 item 在全库降序排名中的平均名次（1-based，越小越好）。
    依赖 collector 的 rec.meanrank（需 metrics 含 GAUC 等会触发 meanrank 采集的指标）。
    """
    import numpy as np

    try:
        mean_rank = dataobject.get("rec.meanrank").numpy()
    except (KeyError, AttributeError, TypeError):
        return None
    pos_rank_sum, _user_len, pos_len = np.split(mean_rank, 3, axis=1)
    pos_rank_sum = pos_rank_sum.squeeze(-1)
    pos_len = pos_len.squeeze(-1)
    valid = pos_len > 0
    if not np.any(valid):
        return None
    ranks = pos_rank_sum[valid] / pos_len[valid]
    return float(np.mean(ranks))


def _patch_evaluator_mean_rank(trainer) -> None:
    """全库评估：在 Hit@K 之外附加 meanrank（算术平均排名）。"""
    if getattr(trainer.evaluator, "_platform_mean_rank_patched", False):
        return
    orig_evaluate = trainer.evaluator.evaluate
    decimal = int(trainer.config["metric_decimal_place"])

    def _evaluate_with_mean_rank(dataobject):
        result = orig_evaluate(dataobject)
        mr = _arithmetic_mean_rank_from_eval_struct(dataobject)
        if mr is None:
            return result
        out = dict(result) if result else {}
        v = round(mr, decimal)
        out["meanrank"] = v
        out["avg_rank"] = v
        # GAUC 仅用于触发 rec.meanrank 采集，全库输出里不保留
        out.pop("gauc", None)
        out.pop("auc", None)
        return out

    trainer.evaluator.evaluate = _evaluate_with_mean_rank  # type: ignore[method-assign]
    trainer.evaluator._platform_mean_rank_patched = True


def _format_full_catalog_metrics(
    result: dict | None,
    topk_list: list[int],
    metric_plan: dict | None = None,
) -> dict:
    out = format_hr_aliases(dict(result) if result else {}, topk_list)
    out.pop("gauc", None)
    out.pop("auc", None)
    if not metric_plan:
        return out

    filtered: dict = {}
    if metric_plan.get("compute_hit"):
        for k in metric_plan.get("hr_topk_list") or []:
            for prefix in ("hit@", "hr@"):
                key = f"{prefix}{k}"
                if key in out:
                    filtered[key] = out[key]
    if metric_plan.get("compute_ndcg"):
        for k in metric_plan.get("topk_list") or []:
            key = f"ndcg@{k}"
            if key in out:
                filtered[key] = out[key]
    if metric_plan.get("compute_mrr"):
        for k in metric_plan.get("topk_list") or []:
            key = f"mrr@{k}"
            if key in out:
                filtered[key] = out[key]
    if metric_plan.get("compute_meanrank"):
        for key in ("meanrank", "avg_rank"):
            if key in out:
                filtered[key] = out[key]
    return filtered


def _patch_pop_full_sort_eval_memory(model) -> None:
    """
    仅全库评估用：Pop 原生 full_sort 用 float64 + repeat_interleave，每用户复制 ~20 万维易顶显存。
    训练 / uni100 不调用此补丁。
    """
    if model.__class__.__name__ != "Pop":
        return
    if getattr(model, "_platform_pop_eval_full_sort_patched", False):
        return

    def full_sort_predict(self, interaction):
        batch_users = interaction[self.USER_ID].shape[0]
        scores = (
            self.item_cnt.squeeze(-1).to(dtype=torch.float32)
            / self.max_cnt.to(dtype=torch.float32).clamp(min=1e-12)
        )
        return scores.unsqueeze(0).expand(batch_users, -1).reshape(-1)

    import types

    model.full_sort_predict = types.MethodType(full_sort_predict, model)
    model._platform_pop_eval_full_sort_patched = True


def _patch_collector_store_on_cpu(collector) -> None:
    """全库评估：collector 累计张量全程在 CPU，避免 GPU cat 越积越大；模型仍在 GPU。"""
    if getattr(collector, "_platform_cpu_store_patched", False):
        return
    data_struct = collector.data_struct
    orig_update = data_struct.update_tensor

    def _update_tensor_cpu(name: str, value: torch.Tensor):
        if isinstance(value, torch.Tensor) and value.is_cuda:
            value = value.detach().cpu()
        existing = data_struct._data_dict.get(name)
        if isinstance(existing, torch.Tensor) and existing.is_cuda:
            data_struct._data_dict[name] = existing.detach().cpu()
        orig_update(name, value)

    data_struct.update_tensor = _update_tensor_cpu  # type: ignore[method-assign]
    collector._platform_cpu_store_patched = True


def _patch_collector_periodic_vram_flush(collector, *, every_batches: int = 64) -> None:
    """全库评估：每 N 批 empty_cache，缓解显存碎片与峰值只升不降。"""
    if every_batches <= 0:
        return
    if getattr(collector, "_platform_vram_flush_patched", False):
        return
    orig_collect = collector.eval_batch_collect
    state = {"n": 0}

    def _collect_with_flush(*args, **kwargs):
        orig_collect(*args, **kwargs)
        state["n"] += 1
        if state["n"] % every_batches == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    collector.eval_batch_collect = _collect_with_flush  # type: ignore[method-assign]
    collector._platform_vram_flush_patched = True


def _patch_trainer_evaluate_memory_release(trainer) -> None:
    """全库评估结束后释放 item_tensor 并 empty_cache。"""
    if getattr(trainer, "_platform_eval_release_patched", False):
        return
    orig_evaluate = trainer.evaluate

    def _evaluate_with_release(eval_data, load_best_model=True, model_file=None, show_progress=False):
        try:
            return orig_evaluate(
                eval_data,
                load_best_model=load_best_model,
                model_file=model_file,
                show_progress=show_progress,
            )
        finally:
            trainer.item_tensor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc

            gc.collect()

    trainer.evaluate = _evaluate_with_release  # type: ignore[method-assign]
    trainer._platform_eval_release_patched = True


def _merge_weighted_full_catalog_metrics(
    parts: list[dict],
    weights: list[int],
) -> dict:
    """多段全库评估按用户数加权合并 Hit/NDCG/MRR/meanrank。"""
    if not parts:
        return {}
    total_w = sum(int(w) for w in weights)
    if total_w <= 0:
        return parts[0]

    keys: set[str] = set()
    for p in parts:
        keys.update(p.keys())

    merged: dict = {}
    for key in sorted(keys):
        acc = 0.0
        has = False
        for p, w in zip(parts, weights):
            if key not in p:
                continue
            acc += float(p[key]) * int(w)
            has = True
        if has:
            merged[key] = acc / total_w
    return merged


def _get_benchmark_phase_chunk(
    dataset_obj,
    phase: str,
    *,
    user_cap,
    config,
) -> pd.DataFrame | None:
    """取某 split 的 interaction 表（优先未 cap 的 benchmark 切片）。"""
    names = getattr(dataset_obj, "benchmark_filename_list", None)
    if not names or phase not in names:
        return None
    uncapped_map = getattr(dataset_obj, "_benchmark_uncapped", None) or {}
    inter = dataset_obj.inter_feat
    if not isinstance(inter, pd.DataFrame):
        return None

    offset = 0
    for name, size in zip(names, dataset_obj.file_size_list):
        if name == phase:
            chunk = inter.iloc[offset : offset + size]
            if name in uncapped_map:
                chunk = uncapped_map[name]
            if user_cap is not None and int(user_cap) > 0:
                seed = int(
                    config.final_config_dict.get("eval_sample_seed", config["seed"])
                )
                rng = np.random.RandomState(seed)
                chunk = _cap_users_in_chunk(
                    chunk, dataset_obj.uid_field, int(user_cap), rng
                )
            return chunk.reset_index(drop=True)
        offset += size
    return None


def _iter_user_id_chunks(user_ids: np.ndarray, chunk_size: int):
    if chunk_size <= 0 or len(user_ids) <= chunk_size:
        yield user_ids
        return
    for start in range(0, len(user_ids), chunk_size):
        yield user_ids[start : start + chunk_size]


def _patch_neumf_full_sort_predict(model) -> None:
    """NeuMF 无原生 full_sort_predict；补矩阵打分以便全库 test。"""
    if model.__class__.__name__ != "NeuMF":
        return
    if getattr(model, "_platform_full_sort_patched", False):
        return

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        batch_users = user.size(0)
        device = user.device
        all_items = torch.arange(self.n_items, device=device)
        users = user.unsqueeze(1).expand(batch_users, self.n_items).reshape(-1)
        items = all_items.unsqueeze(0).expand(batch_users, self.n_items).reshape(-1)
        scores = self.sigmoid(self.forward(users, items))
        return scores.view(-1)

    import types

    model.full_sort_predict = types.MethodType(full_sort_predict, model)
    model._platform_full_sort_patched = True


def _snapshot_eval_config(config) -> dict:
    fcd = config.final_config_dict
    return {
        "eval_args": copy.deepcopy(fcd["eval_args"]),
        "topk": copy.deepcopy(fcd["topk"]),
        "metrics": copy.deepcopy(fcd["metrics"]),
    }


def _restore_eval_config(config, snapshot: dict) -> None:
    for key, val in snapshot.items():
        config.final_config_dict[key] = val


def _apply_full_catalog_eval_settings(
    config,
    *,
    phase: str,
    topk_list: list[int],
    metrics: list[str] | None = None,
) -> None:
    """临时把某 split 设为 full 排序，并合并 topk（不新建 Config，避免 nproc 等 KeyError）。"""
    fcd = config.final_config_dict
    eval_args = copy.deepcopy(dict(fcd["eval_args"]))
    modes = eval_args.get("mode")
    if isinstance(modes, str):
        modes = {"valid": modes, "test": modes}
    else:
        modes = dict(modes or {})
    modes[phase] = "full"
    eval_args["mode"] = modes

    fcd["eval_args"] = eval_args
    fcd["topk"] = sorted({int(x) for x in topk_list})
    fcd["metrics"] = metrics or ["Hit", "NDCG", "MRR", "GAUC"]


def _swap_inter_feat_for_full_catalog(
    dataset_obj,
    phase: str,
    *,
    user_cap,
    config,
    allowed_user_ids=None,
) -> dict | None:
    """
    全库评估前把 valid/test 换回未 cap 的 benchmark 切片（若曾 apply_eval_user_caps）。
    user_cap: 0/None=该 split 全部用户；>0 则在未 cap 切片上再随机抽用户。
    """
    names = getattr(dataset_obj, "benchmark_filename_list", None)
    if not names or phase not in names:
        return None

    uncapped_map = getattr(dataset_obj, "_benchmark_uncapped", None) or {}
    inter = dataset_obj.inter_feat
    if not isinstance(inter, pd.DataFrame):
        return None

    chunks: list[pd.DataFrame] = []
    offset = 0
    for name, size in zip(names, dataset_obj.file_size_list):
        chunk = inter.iloc[offset : offset + size]
        offset += size
        if name == phase:
            if name in uncapped_map:
                chunk = uncapped_map[name].copy()
            if user_cap is not None and int(user_cap) > 0:
                seed = int(
                    config.final_config_dict.get("eval_sample_seed", config["seed"])
                )
                rng = np.random.RandomState(seed)
                chunk = _cap_users_in_chunk(
                    chunk, dataset_obj.uid_field, int(user_cap), rng
                )
            if allowed_user_ids is not None and name == phase:
                uid_field = dataset_obj.uid_field
                chunk = chunk[chunk[uid_field].isin(allowed_user_ids)].reset_index(
                    drop=True
                )
        if name == phase:
            chunks.append(chunk.reset_index(drop=True))
        else:
            chunks.append(chunk.copy())

    new_inter = pd.concat(chunks, ignore_index=True)
    if len(new_inter) == len(inter) and new_inter.equals(inter):
        return None

    snapshot = {
        "inter_feat": dataset_obj.inter_feat,
        "file_size_list": list(dataset_obj.file_size_list),
    }
    dataset_obj.inter_feat = new_inter
    dataset_obj.file_size_list = [len(c) for c in chunks]
    return snapshot


def _restore_inter_feat_snapshot(dataset_obj, snapshot: dict | None) -> None:
    if not snapshot:
        return
    dataset_obj.inter_feat = snapshot["inter_feat"]
    dataset_obj.file_size_list = snapshot["file_size_list"]


def build_full_catalog_dataloader(
    config,
    dataset_obj,
    *,
    phase: str,
    topk_list: list[int],
    full_catalog_user_cap=None,
    metric_plan: dict | None = None,
    allowed_user_ids=None,
):
    """构建 valid/test 全库排序 DataLoader（eval_args.mode[phase]=full）。"""
    from recbole.data.utils import create_samplers, get_dataloader

    ds_snapshot = _swap_inter_feat_for_full_catalog(
        dataset_obj,
        phase,
        user_cap=full_catalog_user_cap,
        config=config,
        allowed_user_ids=allowed_user_ids,
    )

    eval_snapshot = _snapshot_eval_config(config)
    plan = metric_plan or _default_full_catalog_metric_plan(topk_list)
    _apply_full_catalog_eval_settings(
        config,
        phase=phase,
        topk_list=plan["topk_list"],
        metrics=plan["metrics"],
    )

    built = dataset_obj.build()
    train_dataset, valid_dataset, test_dataset = built
    dataset_map = {"train": train_dataset, "valid": valid_dataset, "test": test_dataset}
    train_sampler, valid_sampler, test_sampler = create_samplers(
        config, dataset_obj, built
    )
    sampler_map = {
        "train": train_sampler,
        "valid": valid_sampler,
        "test": test_sampler,
    }
    phase_dataset = dataset_map[phase]
    phase_sampler = sampler_map[phase]
    loader = get_dataloader(config, phase)(
        config, phase_dataset, phase_sampler, shuffle=False
    )
    return loader, eval_snapshot, ds_snapshot


def run_full_catalog_hr_eval(
    trainer,
    dataset_obj,
    config,
    *,
    topk_list: list[int] | None = None,
    phases: list[str] | None = None,
    show_progress: bool = True,
    full_catalog_user_cap=None,
    metric_plan: dict | None = None,
    chunk_users: int = 0,
    vram_flush_every_batches: int = 64,
    load_best_model: bool = False,
) -> dict:
    """
    在全部 item 上排序，按 metric_plan 计算 Hit/HR@K、NDCG@K、MRR@K、meanrank。

    metric_plan: build_full_catalog_metric_plan(...) 的返回值；为 None 时用 topk_list 全开。
    full_catalog_user_cap: 0/None=该 split 全部用户；>0 仅全库阶段再抽样。
    chunk_users: >0 时按用户数分块评估，每块结束后释放 collector 显存（Pop 全量推荐）。
    load_best_model: 默认 False（调用方已 load checkpoint 时不要重复加载）。
    """
    from logging import getLogger

    from recbole.evaluator import Collector, Evaluator

    logger = getLogger()
    _patch_neumf_full_sort_predict(trainer.model)
    _patch_pop_full_sort_eval_memory(trainer.model)
    _patch_trainer_evaluate_memory_release(trainer)
    phases = phases or ["test"]
    if metric_plan is None:
        if topk_list is None:
            topk_list = [10, 50]
        plan = _default_full_catalog_metric_plan(topk_list)
    else:
        plan = metric_plan
    topk_list = plan["topk_list"]
    n_items = dataset_obj.item_num
    out: dict = {}

    enabled = []
    if plan.get("compute_hit"):
        enabled.append("HR@" + ",".join(str(k) for k in plan.get("hr_topk_list", [])))
    if plan.get("compute_ndcg"):
        enabled.append("NDCG")
    if plan.get("compute_mrr"):
        enabled.append("MRR")
    if plan.get("compute_meanrank"):
        enabled.append("meanrank")
    enabled_note = "+".join(enabled) if enabled else "none"

    orig_evaluator = trainer.evaluator
    orig_collector = trainer.eval_collector
    for phase in phases:
        phase_chunk = _get_benchmark_phase_chunk(
            dataset_obj,
            phase,
            user_cap=full_catalog_user_cap,
            config=config,
        )
        n_rows = len(phase_chunk) if phase_chunk is not None else 0
        fcd = config.final_config_dict
        ks = ",".join(str(k) for k in topk_list)
        cap_note = (
            "全量用户"
            if full_catalog_user_cap is None or int(full_catalog_user_cap) <= 0
            else f"抽样 {int(full_catalog_user_cap)} 用户"
        )
        chunk_note = ""
        if chunk_users and int(chunk_users) > 0:
            chunk_note = f" | 分块={int(chunk_users)} 用户/块（块间释放显存）"
        logger.info(
            f"Full-catalog eval [{phase}]: n_samples={n_rows}, "
            f"items={n_items}, enabled=[{enabled_note}], topk=[{ks}], users={cap_note}, "
            f"model={fcd.get('model')}, MODEL_TYPE={fcd.get('MODEL_TYPE')}, "
            f"eval_batch_size={fcd.get('eval_batch_size')}"
        )
        print(
            f"\n>>> 全库排序 [{phase}] {enabled_note} | topk=[{ks}] | "
            f"评估样本数={n_rows}（各模型应一致）| items={n_items} | "
            f"{cap_note}{chunk_note}"
        )
        uid_field = dataset_obj.uid_field
        user_chunks: list[np.ndarray] = []
        if phase_chunk is not None and chunk_users and int(chunk_users) > 0:
            all_uids = phase_chunk[uid_field].unique()
            user_chunks = list(_iter_user_id_chunks(all_uids, int(chunk_users)))
        else:
            user_chunks = [None]

        chunk_metrics: list[dict] = []
        chunk_weights: list[int] = []
        for chunk_idx, uid_slice in enumerate(user_chunks):
            if uid_slice is not None:
                n_chunk_users = len(uid_slice)
                print(
                    f"    全库分块 {chunk_idx + 1}/{len(user_chunks)}: "
                    f"{n_chunk_users} 用户"
                )
            loader, eval_snapshot, ds_snapshot = build_full_catalog_dataloader(
                config,
                dataset_obj,
                phase=phase,
                topk_list=topk_list,
                full_catalog_user_cap=full_catalog_user_cap,
                metric_plan=plan,
                allowed_user_ids=uid_slice,
            )
            trainer.item_tensor = None
            trainer.evaluator = Evaluator(config)
            trainer.eval_collector = Collector(config)
            _patch_collector_store_on_cpu(trainer.eval_collector)
            _patch_collector_periodic_vram_flush(
                trainer.eval_collector, every_batches=vram_flush_every_batches
            )
            if plan.get("compute_meanrank"):
                _patch_evaluator_mean_rank(trainer)
            try:
                result = trainer.evaluate(
                    loader,
                    load_best_model=load_best_model,
                    show_progress=show_progress,
                )
                part = _format_full_catalog_metrics(
                    result, topk_list, metric_plan=plan
                )
                chunk_metrics.append(part)
                chunk_weights.append(_eval_dataloader_user_count(loader))
            finally:
                _restore_eval_config(config, eval_snapshot)
                _restore_inter_feat_snapshot(dataset_obj, ds_snapshot)
                trainer.evaluator = orig_evaluator
                trainer.eval_collector = orig_collector
                trainer.item_tensor = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        phase_metrics = _merge_weighted_full_catalog_metrics(
            chunk_metrics, chunk_weights
        )
        mr = phase_metrics.get("meanrank")
        if mr is not None:
            logger.info(f"Full-catalog [{phase}] meanrank (算术平均排名): {mr}")
            print(f"    算术平均排名 meanrank = {mr}（越小越好，1=完美）")
        out[phase] = phase_metrics

    return out


def run_test_full_sort_hr(
    trainer,
    dataset_obj,
    config,
    *,
    topk: int | list[int] = 50,
    show_progress: bool = True,
) -> dict:
    """兼容旧接口：仅 test 全库 HR@K。"""
    topk_list = normalize_hr_topk_list(topk)
    all_res = run_full_catalog_hr_eval(
        trainer,
        dataset_obj,
        config,
        topk_list=topk_list,
        phases=["test"],
        show_progress=show_progress,
    )
    return all_res.get("test", {})


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def _cap_users_in_chunk(chunk: pd.DataFrame, uid_field: str, cap: int | None, rng: np.random.RandomState):
    if cap is None or int(cap) <= 0:
        return chunk
    users = chunk[uid_field].unique()
    n_keep = min(int(cap), len(users))
    if n_keep >= len(users):
        return chunk
    chosen = rng.choice(users, size=n_keep, replace=False)
    return chunk[chunk[uid_field].isin(chosen)].reset_index(drop=True)


def apply_eval_user_caps(dataset_obj, config) -> None:
    """
    读取 yaml 中的 eval_valid_user_cap / eval_test_user_cap，
    在 benchmark train|valid|test 上对 valid/test 随机抽用户（RecBole 无内置开关）。
    """
    cap_valid = config.final_config_dict.get("eval_valid_user_cap")
    cap_test = config.final_config_dict.get("eval_test_user_cap")
    if (cap_valid is None or int(cap_valid) <= 0) and (cap_test is None or int(cap_test) <= 0):
        return
    names = getattr(dataset_obj, "benchmark_filename_list", None)
    if not names:
        return

    seed = int(config.final_config_dict.get("eval_sample_seed", config["seed"]))
    rng = np.random.RandomState(seed)
    uid = dataset_obj.uid_field
    inter = dataset_obj.inter_feat
    if not isinstance(inter, pd.DataFrame):
        inter = getattr(inter, "inter_feat", inter)
    if hasattr(inter, "to_df"):
        inter = inter.to_df()
    if not isinstance(inter, pd.DataFrame):
        return

    uncapped: dict[str, pd.DataFrame] = {}
    chunks = []
    offset = 0
    for name, size in zip(names, dataset_obj.file_size_list):
        chunk = inter.iloc[offset : offset + size].copy()
        offset += size
        if name in ("valid", "test"):
            uncapped[name] = chunk.copy()
        if name == "valid":
            chunk = _cap_users_in_chunk(chunk, uid, cap_valid, rng)
        elif name == "test":
            chunk = _cap_users_in_chunk(chunk, uid, cap_test, rng)
        chunks.append(chunk)
    dataset_obj._benchmark_uncapped = uncapped

    dataset_obj.inter_feat = pd.concat(chunks, ignore_index=True)
    dataset_obj.file_size_list = [len(c) for c in chunks]

    from logging import getLogger

    sizes = {name: len(c) for name, c in zip(names, chunks)}
    getLogger().info(
        "Eval user caps applied: "
        f"valid_cap={cap_valid} -> {sizes.get('valid')} rows, "
        f"test_cap={cap_test} -> {sizes.get('test')} rows, seed={seed}"
    )
    print(
        f"eval user cap: valid={sizes.get('valid')} rows, "
        f"test={sizes.get('test')} rows (seed={seed})"
    )


def checkpoint_dir_for(dataset: str, model_name: str) -> str:
    """按数据集 + 模型分文件夹，各模型目录下只保留一个 best.pth。"""
    return os.path.join(CHECKPOINT_ROOT, base_dataset_name(dataset), model_name)


def _describe_full_catalog_loader(loader, config) -> dict:
    """
    全库评估规模说明（用于跨模型对比）。

    RecBole 的 tqdm 总数 = len(DataLoader) 的 batch 数，不是用户数：
    - 序列模型（SASRec）：约 ceil(用户数 / eval_batch_size) 个 batch（如 321）
    - 通用模型（Pop/BPR）：item 很多时 step=1，约 1 用户 / batch（如 657203）
    指标 HR@K 的分母仍是同一批 test 用户（sample_size），与 batch 数无关。
    """
    n_samples = _eval_dataloader_user_count(loader)
    n_batches = len(loader) if hasattr(loader, "__len__") else None
    fcd = config.final_config_dict
    return {
        "n_eval_samples": n_samples,
        "n_batches": n_batches,
        "eval_batch_size": fcd.get("eval_batch_size"),
        "model_type": fcd.get("MODEL_TYPE"),
        "model": fcd.get("model"),
    }


def _eval_dataloader_user_count(loader) -> int:
    """FullSortEvalDataLoader：sample_size = 待评估样本数（LOO test 下≈用户数）。"""
    if hasattr(loader, "sample_size"):
        return int(loader.sample_size)
    ds = getattr(loader, "dataset", None)
    if ds is None:
        return 0
    inter_feat = getattr(ds, "inter_feat", None)
    if inter_feat is not None and hasattr(inter_feat, "shape"):
        return int(inter_feat.shape[0])
    return len(ds)


def align_neumf_config_from_checkpoint(cfg: dict, ckpt_path: str) -> dict:
    """NeuMF：yaml 的 mlp_hidden_size 与 best.pth 不一致时，按 checkpoint 自动修正。"""
    if str(cfg.get("model", "")).lower() != "neumf":
        return cfg
    if not ckpt_path or not os.path.isfile(ckpt_path):
        return cfg
    try:
        sd = torch.load(ckpt_path, map_location="cpu").get("state_dict") or {}
    except Exception:
        return cfg
    pred_key = "predict_layer.weight"
    if pred_key not in sd:
        return cfg

    linear_out_dims: list[int] = []
    for key, tensor in sd.items():
        if not key.startswith("mlp_layers.mlp_layers.") or not key.endswith(".weight"):
            continue
        layer_idx = int(key.split(".")[2])
        if layer_idx % 3 != 0:
            continue
        linear_out_dims.append(int(tensor.shape[0]))
    if not linear_out_dims:
        return cfg

    out = dict(cfg)
    current = list(out.get("mlp_hidden_size") or [])
    inferred = linear_out_dims
    if current != inferred:
        print(
            f">>> NeuMF: yaml mlp_hidden_size={current} 与 ckpt 不一致，"
            f"已改为 {inferred}"
        )
        out["mlp_hidden_size"] = inferred
    return out


def load_checkpoint_weights(trainer, ckpt_path: str, *, verbose: bool = True) -> None:
    """仅加载模型权重做评估（不恢复 optimizer / epoch，避免误触续训）。"""
    checkpoint = torch.load(ckpt_path, map_location=trainer.device)
    trainer.model.load_state_dict(checkpoint["state_dict"])
    trainer.model.load_other_parameter(checkpoint.get("other_parameter"))
    trainer.saved_model_file = ckpt_path
    if verbose:
        print(f"loaded weights for eval: {ckpt_path}")


def _skip_heavy_checkpoint_save(model_name: str | None) -> bool:
    """ItemKNN 的 pred_mat 含全用户×物品打分，torch.save 易 MemoryError。"""
    return (model_name or "").upper() == "ITEMKNN"


def _bind_single_best_checkpoint(
    trainer, checkpoint_dir: str, *, model_name: str | None = None
) -> str:
    """固定保存为 best.pth（valid 提升时覆盖），避免 RecBole 时间戳文件名。"""
    _ensure_dir(checkpoint_dir)
    best_path = os.path.join(checkpoint_dir, BEST_CKPT_NAME)
    trainer.saved_model_file = best_path
    if _skip_heavy_checkpoint_save(model_name):
        trainer._save_checkpoint = lambda *a, **k: None  # type: ignore[method-assign]
        print(
            ">>> ItemKNN: 训练过程不写 best.pth（pred_mat 过大，避免 MemoryError）；"
            "test 使用内存中的模型。"
        )
        return best_path
    orig_save = trainer._save_checkpoint

    def _save_checkpoint(epoch, verbose=True, **kwargs):
        kwargs["saved_model_file"] = best_path
        return orig_save(epoch, verbose=verbose, **kwargs)

    trainer._save_checkpoint = _save_checkpoint  # type: ignore[method-assign]
    return best_path


def _apply_resume_optimizer_overrides(trainer, *, lr=None, weight_decay=None) -> None:
    """续训加载 checkpoint 后覆盖 optimizer 超参（否则会沿用 checkpoint 内旧值）。"""
    if lr is None and weight_decay is None:
        return
    for g in trainer.optimizer.param_groups:
        if lr is not None:
            g["lr"] = float(lr)
        if weight_decay is not None:
            g["weight_decay"] = float(weight_decay)
    parts = []
    if lr is not None:
        parts.append(f"lr={lr}")
    if weight_decay is not None:
        parts.append(f"weight_decay={weight_decay}")
    print("续训 optimizer 已覆盖: " + ", ".join(parts))


def _write_best_meta(
    checkpoint_dir: str,
    *,
    tag: str,
    dataset: str,
    model_name: str,
    best_valid_score,
    best_valid_result,
    test_result,
    resumed_from: str | None,
) -> None:
    _write_json(
        os.path.join(checkpoint_dir, BEST_META_NAME),
        {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "tag": tag,
            "dataset": dataset,
            "model": model_name,
            "best_ckpt": os.path.join(checkpoint_dir, BEST_CKPT_NAME),
            "resumed_from": resumed_from,
            "best_valid_score": best_valid_score,
            "best_valid_result": dict(best_valid_result) if best_valid_result else None,
            "test_result": dict(test_result) if test_result else None,
        },
    )


def _finalize_training_run(
    *,
    trainer,
    test_data,
    dataset_obj,
    config,
    platform_eval: dict,
    k: int,
    tag: str,
    dataset: str,
    model_name: str,
    cfg_path: str,
    resume_path: str | None,
    interrupted: bool,
    eval_only: bool,
    get_environment,
    logger,
    load_best_model: bool = True,
) -> None:
    """常规 test（uni100）+ 可选全库 HR@K，写 json/meta 并打印 Summary。"""
    test_result = trainer.evaluate(
        test_data,
        load_best_model=load_best_model,
        show_progress=config["show_progress"],
    )

    full_catalog_result = None
    if platform_eval["full_catalog_eval_enabled"]:
        try:
            full_catalog_result = run_full_catalog_hr_eval(
                trainer,
                dataset_obj,
                config,
                topk_list=platform_eval["full_catalog_eval_topk"],
                phases=platform_eval["full_catalog_eval_phases"],
                show_progress=config["show_progress"],
                full_catalog_user_cap=platform_eval.get("full_catalog_eval_user_cap"),
            )
        except KeyboardInterrupt:
            print("\n>>> 全库 HR 评估被中断，已保留上方 uni100 结果。")
        except Exception as exc:
            print(f"\n>>> 全库 HR 评估失败: {exc}")
            logger.exception("full_catalog_eval failed")
    test_full_sort_result = (
        full_catalog_result.get("test") if isinstance(full_catalog_result, dict) else None
    )

    env_tb = get_environment(config)
    logger.info(
        "The running environment of this training is as follows:\n" + env_tb.draw()
    )

    best_valid_score = trainer.best_valid_score
    best_valid_result = trainer.best_valid_result

    results = {
        "best_valid_score": best_valid_score,
        "valid_score_bigger": config["valid_metric_bigger"],
        "best_valid_result": best_valid_result,
        "test_result": test_result,
        "interrupted": interrupted,
        "eval_only": eval_only,
    }

    if isinstance(results.get("best_valid_result"), dict):
        results["best_valid_result"] = format_metric_aliases(
            results["best_valid_result"], k
        )
    if isinstance(results.get("test_result"), dict):
        results["test_result"] = format_metric_aliases(results["test_result"], k)
    if full_catalog_result:
        results["full_catalog_result"] = full_catalog_result
    if test_full_sort_result:
        results["test_full_sort_result"] = test_full_sort_result

    _ensure_dir(os.path.join(PLATFORM_ROOT, "results", "logs"))
    out_json = os.path.join(PLATFORM_ROOT, "results", "logs", f"{tag}.json")
    _write_json(
        out_json,
        {
            "dataset": dataset,
            "model": model_name,
            "config_path": os.path.relpath(cfg_path, PLATFORM_ROOT),
            "checkpoint_dir": config["checkpoint_dir"],
            "best_ckpt": os.path.join(config["checkpoint_dir"], BEST_CKPT_NAME),
            "best_valid_score": best_valid_score,
            "best_valid_result": results["best_valid_result"],
            "test_result": results["test_result"],
            "test_full_sort_result": test_full_sort_result,
            "full_catalog_result": full_catalog_result,
            "interrupted": interrupted,
            "eval_only": eval_only,
        },
    )
    _write_best_meta(
        config["checkpoint_dir"],
        tag=tag,
        dataset=dataset,
        model_name=model_name,
        best_valid_score=best_valid_score,
        best_valid_result=results["best_valid_result"],
        test_result=results["test_result"],
        resumed_from=resume_path,
    )

    print("\n=== Summary ===")
    if interrupted:
        print("(训练被 Ctrl+C 中断，以下为 best 权重上的 test)")
    if eval_only:
        print("(EVAL_ONLY，未训练)")
    print("best_valid:", results["best_valid_result"])
    print("test:      ", results["test_result"])
    if full_catalog_result:
        print("full catalog metrics:", full_catalog_result)
    elif test_full_sort_result:
        print("test full: ", test_full_sort_result)
    print(f"\nmetrics json: {out_json}")
    print(f"best ckpt:     {os.path.join(config['checkpoint_dir'], BEST_CKPT_NAME)}")
    print(f"best meta:     {os.path.join(config['checkpoint_dir'], BEST_META_NAME)}")


def main() -> None:
    _apply_scipy_dok_compat()
    _apply_recbole_dynamic_neg_device_fix()

    from logging import getLogger

    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import (
        get_environment,
        get_model,
        get_trainer,
        init_logger,
        init_seed,
    )

    os.chdir(PLATFORM_ROOT)

    # =========================
    # 可调区域（你只改这里）
    # =========================
    MODEL = "sasrec_k2"  # bpr | neumf | lightgcn | sasrec | sasrec_k5 | sasrec_k2 | bert4rec | pop | itemknn | crossdomain_neumf
    DEBUG = False   # True: 少 epoch，先跑通；False: 正式训练

    # 仅在 DEBUG 时覆盖（不想覆盖就设为 None）
    DEBUG_EPOCHS = 3
    DEBUG_BATCH = 4096

    # 日志/指标 json 用的运行标签（与 checkpoint 目录无关）
    TAG = None

    # 断点续训：填 .pth 路径；None=从头训练
    # 续训 20 轮 BPR 示例见文件末尾注释
    RESUME_FROM = None
    # r"D:\recbole_checkpoints\movies_tv\BPR\best.pth"
    #r"D:\recbole_checkpoints\movies_tv\CrossDomainNeuMF\best_ckpt.pt"
    # 续训时总 epoch 上限（须 > checkpoint 里已完成的 epoch+1，例如上次跑到 19 则设 40）
    # None=使用 yaml 里的 epochs
    EPOCHS = None

    # 续训时建议略降学习率；None=沿用 checkpoint 内 optimizer（非 yaml）
    RESUME_LR = 1e-4

    # 续训时 weight_decay；None=沿用 checkpoint。原训练为 0，续训可设 1e-5
    RESUME_WEIGHT_DECAY = 1e-5

    # True：不训练，仅用 checkpoint_dir/best.pth 跑 test（补评中断的训练）
    # 注意：EVAL_ONLY 时 RESUME_FROM 会被忽略（勿留 CrossDomain 路径）；权重固定读 ckpt 行打印的 best.pth
    # 改 1+99：在 configs/*_full.yaml 写 num_negatives_eval: 99（不是 eval_negatives）；启动看 neg: 行是否 uni99
    EVAL_ONLY = True

    # =========================

    cfg_path = pick_config(MODEL)
    cfg_source = load_yaml(cfg_path)
    cfg = merge_negative_sampling_defaults(copy.deepcopy(cfg_source))
    cfg = apply_sequential_model_config(cfg)
    init_embed_opts = pop_item_embedding_init_options(cfg)
    _model_for_route = cfg.get("model")
    init_embed_notes: list[str] = []
    if init_embed_opts.get("enabled"):
        init_embed_notes = prepare_item_embedding_init(
            cfg,
            init_embed_opts,
            base_dataset=base_dataset_name(cfg.get("dataset", "movies_tv")),
            model_name=_model_for_route or "",
        )
    if not (
        (_model_for_route or "").strip().lower()
        in ("crossdomainneumf", "crossdomain_neumf")
    ):
        cfg = apply_negative_sampling_config(cfg)
    seq_data_note = ensure_sequential_benchmark_dataset(cfg)
    seq_len_note = align_max_item_list_length_with_seq_files(cfg)
    platform_eval = pop_platform_eval_options(cfg)
    # CrossDomain 须在 strip 前转换 yaml（num_negatives_eval 等会被 strip 掉）
    crossdomain_train_cfg = None
    if (_model_for_route or "").strip().lower() in (
        "crossdomainneumf",
        "crossdomain_neumf",
    ):
        from crossdomain_neumf.platform_bridge import crossdomain_train_config_from_yaml

        crossdomain_train_cfg = crossdomain_train_config_from_yaml(cfg)
    strip_platform_only_config_keys(cfg)

    if DEBUG:
        if DEBUG_EPOCHS is not None:
            cfg["epochs"] = int(DEBUG_EPOCHS)
        if DEBUG_BATCH is not None:
            cfg["train_batch_size"] = int(DEBUG_BATCH)
            cfg["eval_batch_size"] = int(DEBUG_BATCH)
    if EPOCHS is not None:
        cfg["epochs"] = int(EPOCHS)
    if RESUME_LR is not None:
        cfg["learning_rate"] = float(RESUME_LR)
    if RESUME_WEIGHT_DECAY is not None:
        cfg["weight_decay"] = float(RESUME_WEIGHT_DECAY)

    dataset = cfg.get("dataset", "unknown_dataset")
    model_name_early = cfg.get("model")
    config_notes = _sanitize_training_config(cfg, model_name_early or "")
    model_name = cfg.get("model")
    if not model_name:
        raise ValueError(f"{cfg_path} 缺少 model: XXX")

    topk = cfg.get("topk", [10])
    k = int(topk[0]) if isinstance(topk, list) and topk else 10

    ckpt_dir = resolve_checkpoint_dir(cfg_source, dataset, model_name)
    cfg["checkpoint_dir"] = ckpt_dir
    tag = TAG or f"{dataset}_{model_name}_{'debug' if DEBUG else 'full'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    resume_path = None
    if RESUME_FROM:
        resume_path = RESUME_FROM
        if not os.path.isabs(resume_path):
            resume_path = os.path.join(PLATFORM_ROOT, resume_path)
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"找不到续训权重: {resume_path}")

    print("=" * 70)
    print(f"dataset: {dataset}")
    print(f"model:   {model_name}")
    print(f"config:  {os.path.relpath(cfg_path, PLATFORM_ROOT)}")
    print(f"neg:     {summarize_negative_sampling(cfg)}")
    if platform_eval["full_catalog_eval_enabled"]:
        ks = platform_eval["full_catalog_eval_topk"]
        on = platform_eval["full_catalog_eval_on"]
        print(f"full-catalog HR@{ks} on [{on}]: on")
    else:
        print("full-catalog HR@K: off")
    print(f"debug:   {DEBUG}")
    print(f"ckpt:    {ckpt_dir}\\{BEST_CKPT_NAME}")
    if init_embed_opts.get("enabled"):
        print(
            f"init_emb: {init_embed_opts['source']} → {model_name}"
            + (
                f" ({init_embed_opts['ckpt']})"
                if init_embed_opts.get("ckpt")
                else ""
            )
        )
        for note in init_embed_notes:
            print(f"note:    {note}")
    if resume_path:
        print(f"resume:  {resume_path}")
    for note in config_notes:
        print(f"note:    {note}")
    if seq_data_note:
        print(f"note:    {seq_data_note}")
    if seq_len_note:
        print(f"note:    {seq_len_note}")
    print("=" * 70)

    from crossdomain_neumf.platform_bridge import (
        is_crossdomain_model,
        run_crossdomain_via_platform,
    )

    if is_crossdomain_model(model_name):
        run_crossdomain_via_platform(
            cfg=cfg,
            cfg_path=cfg_path,
            platform_root=PLATFORM_ROOT,
            ckpt_dir=ckpt_dir,
            tag=tag,
            debug=bool(DEBUG),
            eval_only=bool(EVAL_ONLY),
            resume_from=resume_path,
            platform_eval=platform_eval,
            train_cfg=crossdomain_train_cfg,
        )
        return

    # === RecBole 原生流程（支持 resume_checkpoint） ===
    config = Config(model=model_name, dataset=dataset, config_dict=cfg)
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    dataset_obj = create_dataset(config)
    apply_eval_user_caps(dataset_obj, config)
    logger.info(dataset_obj)
    train_data, valid_data, test_data = data_preparation(config, dataset_obj)

    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    logger.info(model)

    if (
        init_embed_opts.get("enabled")
        and not EVAL_ONLY
        and not resume_path
    ):
        apply_init_item_embedding_from_general(
            model,
            init_opts=init_embed_opts,
            base_dataset=base_dataset_name(dataset),
            target_model_name=model_name,
        )

    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    best_ckpt_path = _bind_single_best_checkpoint(
        trainer, config["checkpoint_dir"], model_name=model_name
    )
    print(f"best checkpoint: {best_ckpt_path}")
    skip_itemknn_ckpt = _skip_heavy_checkpoint_save(model_name)

    if resume_path and EVAL_ONLY:
        print(
            "\n>>> EVAL_ONLY=True：忽略 RESUME_FROM（续训路径与当前模型无关时会导致误加载）"
        )
    elif resume_path:
        trainer.resume_checkpoint(resume_path)
        _apply_resume_optimizer_overrides(
            trainer, lr=RESUME_LR, weight_decay=RESUME_WEIGHT_DECAY
        )

    interrupted = False
    if EVAL_ONLY:
        print("\n>>> EVAL_ONLY=True：跳过训练，仅用 best.pth 做评估")
        if not os.path.isfile(best_ckpt_path):
            raise FileNotFoundError(
                f"EVAL_ONLY 需要权重文件: {best_ckpt_path}\n"
                f"请从 results/checkpoints/... 拷到上述路径，或改 yaml checkpoint_dir"
            )
        load_checkpoint_weights(trainer, best_ckpt_path)
        best_valid_score = trainer.best_valid_score
        best_valid_result = trainer.best_valid_result
    else:
        try:
            best_valid_score, best_valid_result = trainer.fit(
                train_data,
                valid_data,
                saved=True,
                show_progress=config["show_progress"],
            )
        except KeyboardInterrupt:
            interrupted = True
            best_valid_score = trainer.best_valid_score
            best_valid_result = trainer.best_valid_result
            done = len(getattr(trainer, "train_loss_dict", {}) or {})
            print(
                f"\n>>> 收到 Ctrl+C（已完成约 {done} 个 epoch），"
                f"将加载 best 权重继续跑 test..."
            )

    _finalize_training_run(
        trainer=trainer,
        test_data=test_data,
        dataset_obj=dataset_obj,
        config=config,
        platform_eval=platform_eval,
        k=k,
        tag=tag,
        dataset=dataset,
        model_name=model_name,
        cfg_path=cfg_path,
        resume_path=resume_path,
        interrupted=interrupted,
        eval_only=bool(EVAL_ONLY),
        get_environment=get_environment,
        logger=logger,
        load_best_model=not skip_itemknn_ckpt,
    )


if __name__ == "__main__":
    main()

# ── 续训示例（上次 BPR 跑满 20 epoch，最后一轮为 epoch 19）────────────────────
# 1) 把旧权重拷到新目录（只需做一次）：
#    copy recbole_platform\results\checkpoints\movies_tv_BPR_full_20260602_222030\BPR-*.pth
#         D:\recbole_checkpoints\movies_tv\BPR\best.pth
# 2) run_train.py 顶部设置：
#    MODEL = "bpr"
#    RESUME_FROM = r"D:\recbole_checkpoints\movies_tv\BPR\best.pth"
#    EPOCHS = 40          # 从 epoch 20 继续训到 39，共再训 20 轮
#    RESUME_LR = 0.0005
#    RESUME_WEIGHT_DECAY = 1e-5   # 原训练多为 0，续训略加正则
#    DEBUG = False
# 3) python run_train.py

