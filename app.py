"""
Movies & TV 个性化推荐演示前端（Streamlit + SASRec）
运行方式：在 recbole_platform 目录执行 `python app.py`

新品类数据准备（首次使用）：
    python src/setup_category.py --category toys
    python src/setup_category.py --category sports
    python src/setup_category.py --category clothing
    python src/setup_category.py --category electronics
"""

import html
import importlib
import os
import sys
import json
import pickle
import random
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
try:
    _models_modern = importlib.import_module("models_modern")
    _categories = importlib.import_module("categories")
    _rec_diversify = importlib.import_module("rec_diversify")
    _demo_profile = importlib.import_module("demo_profile")
    _cold_start_infer = importlib.import_module("cold_start_infer")

    NeuMF = _models_modern.NeuMF
    CATEGORIES = _categories.CATEGORIES
    CATEGORY_KEYS = _categories.CATEGORY_KEYS
    data_dir = _categories.data_dir
    results_dir = _categories.results_dir
    stable_seed = _rec_diversify.stable_seed
    debias_popularity = _rec_diversify.debias_popularity
    stochastic_topk = _rec_diversify.stochastic_topk
    exploration_candidate_pool = _rec_diversify.exploration_candidate_pool
    load_reviewer_profiles = _demo_profile.load_reviewer_profiles
    list_demo_reviewers = _demo_profile.list_demo_reviewers
    reviewer_label = _demo_profile.reviewer_label
    build_user_recommendation_plan = _demo_profile.build_user_recommendation_plan
    preference_weights = _demo_profile.preference_weights
    user_cross_explore_ratio = _demo_profile.user_cross_explore_ratio
    interleave_recommendation_feed = _demo_profile.interleave_recommendation_feed
    _largest_remainder = _demo_profile._largest_remainder
    rank_cold_start = _cold_start_infer.rank_cold_start
except ModuleNotFoundError:
    # 当前 SASRec 演示入口不依赖旧的 NeuMF/BPR 多品类模块。
    # 保留旧函数定义，避免缺少旧 src 文件时阻塞页面启动。
    NeuMF = None
    CATEGORIES = {}
    CATEGORY_KEYS = []
    data_dir = results_dir = None
    stable_seed = debias_popularity = stochastic_topk = exploration_candidate_pool = None
    load_reviewer_profiles = list_demo_reviewers = reviewer_label = None
    build_user_recommendation_plan = preference_weights = user_cross_explore_ratio = None
    interleave_recommendation_feed = _largest_remainder = None
    rank_cold_start = None

# ──────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────
TRAD_MODEL_KEYS = {"SVD (Tuned)", "User-based CF", "Item-based CF"}

MODEL_OPTIONS = {
    "NeuMF（深度学习）":                          "NeuMF",
    "BPR（隐式矩阵分解）":                        "BPR",
    "SVD（矩阵分解）★ 预计算·仅20用户":            "SVD (Tuned)",
    "User-based CF（用户协同过滤）★ 预计算·仅20用户": "User-based CF",
    "Item-based CF（物品协同过滤）★ 预计算·仅20用户": "Item-based CF",
}


# ──────────────────────────────────────────────
# 路径辅助（按品类动态生成）
# ──────────────────────────────────────────────
def get_paths(cat: str) -> dict:
    d = data_dir(cat)
    r = results_dir(cat)
    return {
        "pt":         os.path.join(r, "neucf_best.pt"),
        "bpr":        os.path.join(r, "bpr_model.pkl"),
        "cfg":        os.path.join(r, "model_config.json"),
        "history":    os.path.join(r, "user_history.json"),
        "stats":      os.path.join(r, "item_stats.csv"),
        "eval":       os.path.join(r, "model_evaluation.json"),
        "item_map":   os.path.join(d, "item_mapping.json"),
        "meta":       os.path.join(d, "item_metadata.csv"),
        "user_names": os.path.join(d, "user_display_names.json"),
        "trad_recs":  os.path.join(r, "traditional_recs.json"),
        "trad_users": os.path.join(r, "traditional_users.json"),
        "test_items": os.path.join(r, "test_items.json"),
    }


def is_category_ready(cat: str) -> bool:
    """检测品类的深度学习模型是否已训练完毕。"""
    p = get_paths(cat)
    return os.path.exists(p["pt"]) and os.path.exists(p["bpr"])


# ──────────────────────────────────────────────
# 缓存加载函数（以品类为 key，自动隔离不同品类缓存）
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="正在加载深度学习模型权重…")
def load_models(cat: str):
    p = get_paths(cat)
    if not os.path.exists(p["pt"]) or not os.path.exists(p["bpr"]):
        return None, None, None
    cfg    = json.load(open(p["cfg"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ncf    = NeuMF(cfg["num_users"], cfg["num_items"],
                   embed_dim=cfg["embed_dim"],
                   mlp_hidden=tuple(cfg.get("mlp_hidden", [128, 64, 32])),
                   dropout=cfg.get("dropout", 0.3)).to(device)
    ckpt   = torch.load(p["pt"], map_location=device)
    ncf.load_state_dict(ckpt["state_dict"])
    ncf.eval()
    with open(p["bpr"], "rb") as f:
        bpr = pickle.load(f)
    return ncf, bpr, device


@st.cache_data(show_spinner="正在加载数据集…")
def load_data(cat: str):
    p = get_paths(cat)
    item_stats = (
        pd.read_csv(p["stats"]).set_index("item_id")
        if os.path.exists(p["stats"]) else pd.DataFrame()
    )
    with open(p["history"]) as f:
        user_history = {int(k): v for k, v in json.load(f).items()}

    with open(p["item_map"]) as f:
        raw = json.load(f)
    id_to_asin = {int(v): k for k, v in raw.items()}

    meta = (
        pd.read_csv(p["meta"], index_col="item_id")
        if os.path.exists(p["meta"]) else pd.DataFrame()
    )

    user_names: dict = {}
    if os.path.exists(p["user_names"]):
        user_names = {int(k): v for k, v in json.load(open(p["user_names"])).items()}

    return item_stats, user_history, id_to_asin, meta, user_names


@st.cache_data(show_spinner="正在加载传统模型预计算结果…")
def load_trad_data(cat: str):
    p = get_paths(cat)
    trad_recs  = json.load(open(p["trad_recs"]))  if os.path.exists(p["trad_recs"])  else {}
    trad_users = json.load(open(p["trad_users"])) if os.path.exists(p["trad_users"]) else []
    test_items = json.load(open(p["test_items"])) if os.path.exists(p["test_items"]) else {}
    return trad_recs, trad_users, test_items


@st.cache_data
def load_eval_results(cat: str):
    p = get_paths(cat)
    if not os.path.exists(p["eval"]):
        return {}
    with open(p["eval"]) as f:
        return json.load(f)


# ──────────────────────────────────────────────
# 推理函数（深度学习）
# ──────────────────────────────────────────────
def neucf_predict(ncf, device, user_id: int, item_ids: list) -> np.ndarray:
    with torch.no_grad():
        u = torch.tensor([user_id] * len(item_ids), dtype=torch.long).to(device)
        i = torch.tensor(item_ids, dtype=torch.long).to(device)
        return ncf(u, i).cpu().numpy()


def neucf_coldstart_predict(ncf, device, liked_item_ids: list, candidate_ids: list) -> np.ndarray:
    """
    冷启动：取喜好商品 embedding 均值作为伪用户向量，直接代入 NeuMF 双塔打分。
    GMF 路径：pseudo_u_gmf = mean(item_emb_gmf[liked])
    MLP 路径：pseudo_u_mlp = mean(item_emb_mlp[liked])
    """
    with torch.no_grad():
        liked_t      = torch.tensor(liked_item_ids, dtype=torch.long).to(device)
        pseudo_u_gmf = ncf.item_emb_gmf(liked_t).mean(dim=0)
        pseudo_u_mlp = ncf.item_emb_mlp(liked_t).mean(dim=0)

        cand_t  = torch.tensor(candidate_ids, dtype=torch.long).to(device)
        n       = len(candidate_ids)
        i_gmf   = ncf.item_emb_gmf(cand_t)
        i_mlp   = ncf.item_emb_mlp(cand_t)
        gmf_vec = pseudo_u_gmf.unsqueeze(0).expand(n, -1) * i_gmf
        mlp_vec = ncf.mlp(torch.cat([pseudo_u_mlp.unsqueeze(0).expand(n, -1), i_mlp], dim=-1))
        scores  = ncf.prediction_layer(torch.cat([gmf_vec, mlp_vec], dim=-1)).squeeze(-1)
        return scores.cpu().numpy()


def bpr_predict(bpr, user_id: int, item_ids: list) -> np.ndarray:
    return np.dot(bpr.item_factors[np.array(item_ids)], bpr.user_factors[user_id])


def bpr_coldstart_predict(bpr, liked_item_ids: list, candidate_ids: list) -> np.ndarray:
    pseudo_u = bpr.item_factors[np.array(liked_item_ids)].mean(axis=0)
    return np.dot(bpr.item_factors[np.array(candidate_ids)], pseudo_u)


def get_topk_recs(model_key, ncf, bpr, device,
                  user_id: int, all_item_ids: list,
                  history: set, top_k: int,
                  item_stats: pd.DataFrame | None = None,
                  diversify_seed: int | None = None) -> list:
    candidates = [i for i in all_item_ids if i not in history]
    if not candidates:
        return []
    scores = (neucf_predict(ncf, device, user_id, candidates)
              if model_key == "NeuMF" else bpr_predict(bpr, user_id, candidates))
    if item_stats is not None and not item_stats.empty:
        scores = debias_popularity(candidates, scores, item_stats)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    if diversify_seed is not None:
        return stochastic_topk(ranked, top_k, diversify_seed)
    return ranked[:top_k]


def get_coldstart_recs(model_key, ncf, bpr, device,
                       liked_ids: list, all_item_ids: list,
                       history: set, top_k: int,
                       item_stats: pd.DataFrame | None = None,
                       diversify_seed: int | None = None) -> list:
    candidates = [i for i in all_item_ids if i not in history]
    if not liked_ids or not candidates:
        return []
    scores = (neucf_coldstart_predict(ncf, device, liked_ids, candidates)
              if model_key == "NeuMF" else bpr_coldstart_predict(bpr, liked_ids, candidates))
    if item_stats is not None and not item_stats.empty:
        scores = debias_popularity(candidates, scores, item_stats)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    if diversify_seed is not None:
        return stochastic_topk(ranked, top_k, diversify_seed)
    return ranked[:top_k]


def exploration_recs(item_stats: pd.DataFrame, all_item_ids: list,
                     exclude: set, top_k: int,
                     diversify_key: str | None = None) -> list:
    """跨域探索：从扩大候选池按用户种子抽样，避免全站同一批爆款。"""
    if item_stats.empty:
        pool = [i for i in all_item_ids if i not in exclude]
        seed = stable_seed(diversify_key or "explore", str(len(pool)))
        rng = random.Random(seed)
        rng.shuffle(pool)
        return [(i, 0.0) for i in pool[:top_k]]

    pool = exploration_candidate_pool(item_stats, exclude, pool_factor=12, k=top_k)
    if not pool:
        return []
    if diversify_key is not None:
        return stochastic_topk(pool, top_k, stable_seed(diversify_key, "explore"))
    return pool[:top_k]


@st.cache_data(show_spinner="构建跨区用户画像…")
def get_reviewer_profiles(_version: str = "v3"):
  # _version 用于缓存失效；v3=时间衰减权重 + 新探索比例
    return load_reviewer_profiles()


def infer_recs(pref_source: str, model_key: str, ncf, bpr, device,
               user_id, liked_ids: list, all_item_ids: list,
               exclude: set, top_k: int,
               item_stats: pd.DataFrame | None = None,
               diversify_key: str | None = None) -> list:
    """统一推理入口：训练用户向量 / 勾选冷启动。"""
    div_seed = stable_seed(diversify_key, model_key) if diversify_key else None
    if pref_source == "勾选商品（冷启动，可实时变）" and liked_ids:
        return get_coldstart_recs(
            model_key, ncf, bpr, device, liked_ids, all_item_ids, exclude, top_k,
            item_stats=item_stats, diversify_seed=div_seed,
        )
    if pref_source == "训练用户向量（本区有 ID）" and user_id is not None:
        return get_topk_recs(
            model_key, ncf, bpr, device, user_id, all_item_ids, exclude, top_k,
            item_stats=item_stats, diversify_seed=div_seed,
        )
    if liked_ids:
        return get_coldstart_recs(
            model_key, ncf, bpr, device, liked_ids, all_item_ids, exclude, top_k,
            item_stats=item_stats, diversify_seed=div_seed,
        )
    return []


# ──────────────────────────────────────────────
# 展示辅助
# ──────────────────────────────────────────────
def item_display_name(item_id: int, id_to_asin: dict, meta: pd.DataFrame) -> str:
    if not meta.empty and item_id in meta.index:
        title = meta.loc[item_id].get("title", "")
        if pd.notna(title) and str(title).strip():
            return str(title)[:65]
    asin = id_to_asin.get(item_id, str(item_id))
    return f"商品 #{item_id}  (ASIN: {asin})"


def item_tag(item_id: int, item_stats: pd.DataFrame) -> str:
    if item_stats.empty or item_id not in item_stats.index:
        return ""
    row = item_stats.loc[item_id]
    cnt, avg = row["review_count"], row["avg_rating"]
    if cnt >= 50 and avg >= 4.5:
        return "🔥 热门高分"
    if cnt < 10 and avg >= 4.5:
        return "💎 高分小众"
    if cnt >= 50:
        return "📦 热门"
    return ""


def get_item_img(item_id: int, meta: pd.DataFrame):
    if meta.empty or item_id not in meta.index:
        return None
    url = meta.loc[item_id].get("imUrl", "")
    return str(url) if pd.notna(url) and str(url).strip() else None


def domain_badge(cat_key: str | None) -> str:
    if not cat_key:
        return ""
    cfg = CATEGORIES.get(cat_key, {})
    return f"{cfg.get('icon', '')} {cfg.get('label', cat_key)}"


def inject_product_grid_css():
    """电商风商品栅格：每次 rerun 都注入 CSS（避免滑块拖动后样式丢失）。"""
    st.markdown(
        """
        <style>
        header[data-testid="stHeader"] {
            display: none;
        }
        .stApp {
            background: #f5f6f8;
        }
        .block-container {
            padding-top: 0.25rem;
        }
        .store-topbar {
            background: #2f3338;
            color: #fff;
            border-radius: 0 0 12px 12px;
            padding: 12px 18px;
            margin: -0.25rem 0 16px 0;
            font-size: 1.35rem;
            font-weight: 750;
            letter-spacing: 0.02em;
            box-shadow: 0 2px 10px rgba(15, 23, 42, 0.12);
        }
        div[data-testid="column"] {
            min-width: 0 !important;
        }
        .product-card {
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            padding: 8px 8px 10px;
            margin-bottom: 4px;
            background: #fff;
            height: 100%;
            min-height: 350px;
            box-sizing: border-box;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }
        .product-card--hit {
            border-color: #22c55e;
            background: linear-gradient(180deg, #f0fdf4 0%, #fff 28%);
        }
        .product-card--prime {
            border-color: #00a8e1;
            background: linear-gradient(180deg, #effaff 0%, #fff 38%);
            box-shadow: 0 1px 8px rgba(0, 168, 225, 0.10);
        }
        .product-img-box {
            width: 100%;
            aspect-ratio: 2 / 3;
            position: relative;
            border-radius: 8px;
            background: #f3f4f6;
            overflow: hidden;
            margin-bottom: 6px;
        }
        .product-img-box--prime {
            background: #dff5ff;
        }
        .product-img-bg {
            position: absolute;
            inset: -12%;
            width: 124%;
            height: 124%;
            object-fit: cover;
            object-position: center center;
            filter: blur(12px) saturate(1.15);
            opacity: 0.24;
            transform: scale(1.04);
        }
        .product-img-box img {
            position: absolute;
            inset: 0;
            width: 100%;
            height: 100%;
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            object-position: center center;
            display: block;
        }
        .product-img-box img.product-img-bg {
            inset: -12%;
            width: 124%;
            height: 124%;
            object-fit: cover;
            filter: blur(12px) saturate(1.15);
            opacity: 0.24;
            transform: scale(1.04);
            z-index: 0;
        }
        .product-img-box img:not(.product-img-bg) {
            z-index: 1;
        }
        .product-img-placeholder {
            position: absolute;
            inset: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2rem;
            color: #9ca3af;
        }
        .product-domain {
            font-size: 0.72rem;
            color: #6b7280;
            margin: 0 0 2px 0;
            line-height: 1.2;
        }
        .product-title {
            font-size: 0.82rem;
            font-weight: 600;
            color: #111827;
            line-height: 1.35;
            margin: 0 0 4px 0;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
            min-height: 2.2em;
        }
        .product-meta {
            font-size: 0.70rem;
            color: #6b7280;
            margin: 0;
            line-height: 1.3;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .product-review {
            font-size: 0.74rem;
            color: #4b5563;
            margin-top: auto;
            line-height: 1.35;
            white-space: nowrap;
        }
        .product-rank {
            font-size: 0.7rem;
            color: #9ca3af;
            margin-bottom: 4px;
        }
        .product-prime-badge {
            display: inline-block;
            align-self: flex-start;
            font-size: 0.68rem;
            font-weight: 700;
            color: #0f172a;
            background: linear-gradient(90deg, #7dd3fc 0%, #38bdf8 100%);
            border-radius: 999px;
            padding: 1px 7px;
            margin: 0 0 4px 0;
            letter-spacing: 0.01em;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _product_image_html(img_url: str | None) -> str:
    if img_url:
        safe = html.escape(img_url, quote=True)
        return (
            f'<div class="product-img-box"><img src="{safe}" alt="" '
            f'referrerpolicy="no-referrer" loading="lazy" decoding="async"/></div>'
        )
    return '<div class="product-img-box"><span class="product-img-placeholder">📦</span></div>'


def _sasrec_product_image_html(img_url: str | None, *, prime: bool = False) -> str:
    if not img_url:
        return '<div class="product-img-box"><span class="product-img-placeholder">📦</span></div>'
    safe = html.escape(img_url, quote=True)
    box_cls = "product-img-box product-img-box--prime" if prime else "product-img-box"
    bg = f'<img class="product-img-bg" src="{safe}" alt="" referrerpolicy="no-referrer" loading="lazy" decoding="async"/>'
    return (
        f'<div class="{box_cls}">{bg}<img src="{safe}" alt="" '
        f'referrerpolicy="no-referrer" loading="lazy" decoding="async"/></div>'
    )


def _product_card_html(
    *,
    img_url: str | None,
    title: str,
    domain: str = "",
    meta_line: str = "",
    rank: int | None = None,
    score: float | None = None,
    highlight: bool = False,
    badge: str = "",
) -> str:
    cls = "product-card product-card--hit" if highlight else "product-card"
    parts = [f'<div class="{cls}">']
    if rank is not None:
        parts.append(f'<div class="product-rank">#{rank}</div>')
    parts.append(_product_image_html(img_url))
    if domain:
        parts.append(f'<div class="product-domain">{html.escape(domain)}</div>')
    parts.append(f'<div class="product-title">{html.escape(title)}</div>')
    if meta_line:
        parts.append(f'<div class="product-review">{html.escape(meta_line)}</div>')
    if badge:
        parts.append(f'<div class="product-meta">{html.escape(badge)}</div>')
    if score is not None:
        parts.append(f'<div class="product-meta">推荐分 {score:.3f}</div>')
    parts.append("</div>")
    return "".join(parts)


def _item_meta_line(item_id: int, item_stats: pd.DataFrame) -> str:
    if item_stats.empty or item_id not in item_stats.index:
        return ""
    row = item_stats.loc[item_id]
    tag = item_tag(item_id, item_stats)
    base = f"⭐ {row['avg_rating']:.1f} · {int(row['review_count'])}评"
    return f"{tag} {base}".strip() if tag else base


def render_product_tile_in_column(
    col,
    item_id: int,
    id_to_asin: dict,
    meta: pd.DataFrame,
    item_stats: pd.DataFrame,
    *,
    cat_key: str | None = None,
    rank: int | None = None,
    score: float | None = None,
    highlight: bool = False,
    kind: str = "",
    show_score: bool = True,
):
    """单列商品卡（正方图 + 标题），放入 st.columns 的某一列。"""
    name = item_display_name(item_id, id_to_asin, meta)
    img_url = get_item_img(item_id, meta)
    domain = domain_badge(cat_key) if cat_key else ""
    meta_line = _item_meta_line(item_id, item_stats)
    badge = kind.strip() if kind else ""
    with col:
        st.markdown(
            _product_card_html(
                img_url=img_url,
                title=name[:72],
                domain=domain,
                meta_line=meta_line,
                rank=rank,
                score=score if show_score else None,
                highlight=highlight,
                badge=badge,
            ),
            unsafe_allow_html=True,
        )


def render_product_grid(
    items: list[dict],
    cat_cache: dict[str, tuple],
    *,
    cols_per_row: int = 4,
    show_score: bool = True,
):
    """
    items 元素字段：item_id, cat(可选), rank, score, highlight, kind
    cat_cache: cat -> (item_stats, id_to_asin, meta)
    """
    inject_product_grid_css()
    if not items:
        return
    for i in range(0, len(items), cols_per_row):
        row_cols = st.columns(cols_per_row)
        for col, entry in zip(row_cols, items[i : i + cols_per_row]):
            cat = entry.get("cat")
            iid = entry["item_id"]
            if cat and cat not in cat_cache:
                ist, _, id_map, meta_c, _ = load_data(cat)
                cat_cache[cat] = (ist, id_map, meta_c)
            if cat:
                ist, id_map, meta_c = cat_cache[cat]
            else:
                ist, id_map, meta_c = next(iter(cat_cache.values()))
            render_product_tile_in_column(
                col, iid, id_map, meta_c, ist,
                cat_key=cat,
                rank=entry.get("rank"),
                score=entry.get("score"),
                highlight=entry.get("highlight", False),
                kind=entry.get("kind", ""),
                show_score=show_score,
            )


def render_pickable_product_grid(
    showcase: list[tuple[str, int]],
    seed: int,
) -> dict[str, list[int]]:
    """四列勾选墙，返回 {cat: [item_id, ...]}。"""
    inject_product_grid_css()
    liked_by_cat: dict[str, list[int]] = {}
    for i in range(0, len(showcase), 4):
        row_cols = st.columns(4)
        for col, pair in zip(row_cols, showcase[i : i + 4]):
            if not pair:
                continue
            cat, iid = pair
            item_stats, _, id_to_asin, meta, _ = load_data(cat)
            name = item_display_name(iid, id_to_asin, meta)
            img_url = get_item_img(iid, meta)
            domain = domain_badge(cat)
            meta_line = _item_meta_line(iid, item_stats)
            with col:
                st.markdown(
                    _product_card_html(
                        img_url=img_url,
                        title=name[:72],
                        domain=domain,
                        meta_line=meta_line,
                    ),
                    unsafe_allow_html=True,
                )
                if st.checkbox("感兴趣", key=f"gc_{cat}_{iid}_{seed}"):
                    liked_by_cat.setdefault(cat, []).append(iid)
    return liked_by_cat


def _uid_for_cat(user_ids: dict, cat: str):
    if cat in user_ids:
        return int(user_ids[cat])
    if str(cat) in user_ids:
        return int(user_ids[str(cat)])
    return None


def generate_blended_recommendations(
    reviewer: str,
    profiles: dict,
    ready_cats: list[str],
    model_key: str,
    top_k: int,
    cross_ratio: float | None,
    pref_source: str,
    liked_ids_current_cat: list,
    current_cat: str,
) -> tuple[list[dict], dict, dict, float]:
    """按该用户历史占比 + 其专属跨域探索比例生成混合推荐。"""
    prof = profiles[reviewer]
    counts = {c: int(prof["counts"].get(c, 0)) for c in ready_cats}
    weights = preference_weights(prof, ready_cats)
    user_ids = prof["user_ids"]
    if cross_ratio is None:
        cross_ratio = user_cross_explore_ratio(counts, ready_cats, weights)
    plan = build_user_recommendation_plan(
        counts, ready_cats, top_k, cross_ratio=cross_ratio, weights=weights,
    )
    main_alloc, cross_alloc = plan["main_alloc"], plan["cross_alloc"]

    blended: list[dict] = []

    for cat, k in main_alloc.items():
        if k <= 0:
            continue
        ncf, bpr, device = load_models(cat)
        if ncf is None:
            continue
        item_stats, user_history, id_to_asin, meta, _ = load_data(cat)
        all_items = list(id_to_asin.keys())
        uid = _uid_for_cat(user_ids, cat)
        liked = list(liked_ids_current_cat) if cat == current_cat else []
        exclude = set(liked)
        if uid is not None:
            exclude |= set(user_history.get(uid, []))
        div_key = f"{reviewer}:{cat}:cf"
        recs = infer_recs(
            pref_source, model_key, ncf, bpr, device, uid, liked, all_items, exclude, k,
            item_stats=item_stats, diversify_key=div_key,
        )
        for rank_in_cat, (iid, score) in enumerate(recs, 1):
            blended.append({
                "cat": cat, "item_id": iid, "score": score,
                "kind": "协同", "model": model_key, "rank_in_cat": rank_in_cat,
            })

    for cat, k in cross_alloc.items():
        if k <= 0:
            continue
        item_stats, user_history, id_to_asin, meta, _ = load_data(cat)
        all_items = list(id_to_asin.keys())
        uid = _uid_for_cat(user_ids, cat)
        exclude = set(user_history.get(uid, [])) if uid is not None else set()
        if cat == current_cat:
            exclude |= set(liked_ids_current_cat)
        recs = exploration_recs(
            item_stats, all_items, exclude, k,
            diversify_key=f"{reviewer}:{cat}:explore",
        )
        for rank_in_cat, (iid, score) in enumerate(recs, 1):
            blended.append({
                "cat": cat, "item_id": iid, "score": score,
                "kind": "探索", "model": "热度", "rank_in_cat": rank_in_cat,
            })

    slot_weights = {c: main_alloc.get(c, 0) + cross_alloc.get(c, 0) for c in ready_cats}
    blended = interleave_recommendation_feed(blended, slot_weights)

    return blended, main_alloc, cross_alloc, cross_ratio


# def _user_roster_df(...):  # 已精简：跨区名册表不再展示

def _explored_cats_for_reviewer(profiles: dict, reviewer: str, ready_cats: list[str]) -> list[str]:
    counts = profiles[reviewer].get("counts", {})
    return [c for c in ready_cats if counts.get(c, 0) > 0]


def render_cross_recommendations(
    reviewer: str, profiles: dict, ready_cats: list[str], top_k: int,
):
    """按该用户购买占比交织后的推荐（四列商品栅格）。"""
    explored = _explored_cats_for_reviewer(profiles, reviewer, ready_cats)
    blended, _, _, _ = generate_blended_recommendations(
        reviewer, profiles, ready_cats, "NeuMF", top_k, None,
        "训练用户向量（本区有 ID）", [], explored[0] if explored else ready_cats[0],
    )
    if not blended:
        return

    cat_cache: dict[str, tuple] = {}
    grid_items = [
        {
            "cat": row["cat"],
            "item_id": row["item_id"],
            "rank": idx,
            "kind": row.get("kind", ""),
        }
        for idx, row in enumerate(blended, 1)
    ]
    render_product_grid(grid_items, cat_cache, show_score=False)


@st.cache_data(show_spinner="抽样展示商品…")
def sample_global_showcase(ready_cats: tuple, per_cat: int, seed: int) -> list[tuple[str, int]]:
    """每区抽 per_cat 件（热门+随机各半），返回 [(category, item_id), …]。"""
    rng = random.Random(seed)
    out: list[tuple[str, int]] = []
    for cat in ready_cats:
        item_stats, _, id_to_asin, _, _ = load_data(cat)
        pool = list(id_to_asin.keys())
        if not pool:
            continue
        picks: list[int] = []
        if not item_stats.empty:
            hot = item_stats.sort_values("review_count", ascending=False).head(per_cat * 3).index.tolist()
            rng.shuffle(hot)
            picks.extend(hot[: max(1, per_cat // 2)])
        rest = [i for i in pool if i not in picks]
        need = per_cat - len(picks)
        if need > 0 and rest:
            picks.extend(rng.sample(rest, min(need, len(rest))))
        for iid in picks[:per_cat]:
            out.append((cat, int(iid)))
    rng.shuffle(out)
    return out


def render_global_coldstart_tab(ready_cats: list[str], top_k: int):
    """五类随机商品勾选 → 融合冷启动推荐（按勾选品类分槽）。"""
    st.subheader("🆕 全站新用户冷启动")
    st.caption(
        "从五个分区随机抽取商品供勾选；推荐算法为 **NeuMF 伪用户 + BPR 均值 + BPR 物品相似度** "
        "三路 z-score 融合（专为本页设计，无需用户 ID）。"
    )

    if "showcase_seed" not in st.session_state:
        st.session_state.showcase_seed = random.randint(0, 2**31 - 1)

    c1, c2, _ = st.columns([1, 1, 2])
    if c1.button("🎲 换一批展示商品", use_container_width=True):
        st.session_state.showcase_seed += 1
        st.rerun()
    per_cat = c2.slider("每区展示件数", 4, 10, 6, key="showcase_per_cat")

    seed = int(st.session_state.showcase_seed)
    showcase = sample_global_showcase(tuple(ready_cats), per_cat, seed)

    st.markdown("#### 勾选感兴趣的商品（至少 1 件）")
    liked_by_cat = render_pickable_product_grid(showcase, seed)

    if not st.button("✨ 生成全站冷启动推荐", type="primary"):
        return

    active = {c: ids for c, ids in liked_by_cat.items() if ids}
    if not active:
        st.warning("请至少勾选一件商品。")
        return

    slot_w = {c: float(len(ids)) for c, ids in active.items()}
    alloc = _largest_remainder(top_k, slot_w)

    st.success(f"已在 **{len(active)}** 个分区识别偏好，共 **{sum(len(v) for v in active.values())}** 件勾选商品")
    blended: list[dict] = []

    for cat, k in alloc.items():
        if k <= 0:
            continue
        ncf, bpr, device = load_models(cat)
        if ncf is None:
            continue
        item_stats, _, id_to_asin, meta, _ = load_data(cat)
        all_items = list(id_to_asin.keys())
        exclude = set(active[cat])
        with st.spinner(f"推理 {CATEGORIES[cat]['label']}…"):
            recs = rank_cold_start(
                ncf, bpr, device, active[cat], all_items, exclude, k,
                debias_fn=(
                    (lambda c, s, st=item_stats: debias_popularity(c, s, st))
                    if not item_stats.empty else None
                ),
            )
        for rank, (iid, score) in enumerate(recs, 1):
            blended.append({"cat": cat, "item_id": iid, "score": score, "rank_in_cat": rank})

    slot_weights = {c: alloc.get(c, 0) for c in ready_cats}
    feed = interleave_recommendation_feed(
        [{"cat": r["cat"], "item_id": r["item_id"], "score": r["score"],
          "kind": "冷启动", "model": "NeuMF+BPR+Sim", "rank_in_cat": r["rank_in_cat"]}
         for r in blended],
        slot_weights,
    )

    st.markdown("#### 推荐结果")
    cat_cache: dict = {}
    grid_items = [
        {
            "cat": row["cat"],
            "item_id": row["item_id"],
            "rank": i,
            "kind": "冷启动",
        }
        for i, row in enumerate(feed, 1)
    ]
    render_product_grid(grid_items, cat_cache, show_score=False)


def render_loo_hits_tab(cat: str, ready_cats: list[str]):
    """展示单域 NeuMF 全量排序 Top-20 命中 LOO 目标的案例。"""
    st.subheader("🎯 单域 LOO 命中案例（Top-20 · 全量候选）")
    st.caption(
        "与性能页 **HR@10（1 正 + 99 负）** 不同：此处对**全商品候选**排序。"
        "若 HR@10≈42%，相当于在「百里挑一」量级的困难任务里约四成用户命中。"
    )

    path = os.path.join(results_dir(cat), "loo_hits_top20.json")
    if not os.path.exists(path):
        st.warning(
            f"尚未生成案例文件。请在项目根目录运行：\n\n"
            f"`python src/extract_loo_hits.py --category {cat}`\n\n"
            f"或 `python src/extract_loo_hits.py --all-ready --top-k 20`"
        )
        if st.button("查看当前区 HR@10 指标"):
            ev = load_eval_results(cat)
            neu = ev.get("NeuMF (Dynamic+BPR)", {})
            st.metric("HR@10（1+99 负采样）", f"{neu.get('HR@10', 0):.4f}")
        return

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    meta = data.get("meta", {})
    hits = data.get("hits", [])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"HR@{meta.get('top_k', 20)}（全量）", f"{meta.get('hr_at_k', 0):.2%}")
    m2.metric("命中用户数", meta.get("n_hits", 0))
    m3.metric("评测用户", meta.get("n_users", 0))
    m4.metric("未进 Top-K", meta.get("n_misses", 0))

    st.info(meta.get("note", ""))

    st.markdown("#### 命中案例（按排名从优到劣）")
    for h in hits[:40]:
        rank = h["rank"]
        with st.container(border=True):
            st.markdown(
                f"**用户 {h['user_id']}** · 历史 {h['history_len']} 条 · "
                f"目标排在 **#{rank}**"
            )
            st.markdown(f"🎯 **{h['target_title']}**")
            st.caption("Top 推荐：" + " | ".join(
                f"#{i+1} {r['title'][:40]}" for i, r in enumerate(h.get("top_recommendations", [])[:5])
            ))

    with st.expander("五区 HR@20 汇总（若已生成）"):
        rows = []
        for c in ready_cats:
            p = os.path.join(results_dir(c), "loo_hits_top20.json")
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    m = json.load(f).get("meta", {})
                rows.append({
                    "分区": CATEGORIES[c]["label"],
                    "HR@20": m.get("hr_at_k"),
                    "命中": m.get("n_hits"),
                    "用户": m.get("n_users"),
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)


def render_metrics_tab(cat: str):
    """单区模型性能对比（图表 + 表）。"""
    cat_cfg = CATEGORIES[cat]
    eval_results = load_eval_results(cat)
    st.subheader(f"📊 {cat_cfg['icon']} {cat_cfg['label']} · 所有模型性能对比")

    if not eval_results:
        st.warning(
            f"未找到 `{os.path.join(results_dir(cat), 'model_evaluation.json')}`，"
            f"请先运行：\n`python src/setup_category.py --category {cat}`"
        )
        return

    df_eval = pd.DataFrame(eval_results).T.reset_index().rename(columns={"index": "Model"})
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Ranking 指标（越高越好）")
        df_rank = df_eval[df_eval["HR@10"] > 0][["Model", "HR@10", "NDCG@10"]]
        df_melt = df_rank.melt(id_vars="Model", var_name="Metric", value_name="Score")
        fig_rank = px.bar(
            df_melt, x="Model", y="Score", color="Metric",
            barmode="group", text_auto=".3f",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_rank.update_layout(xaxis_tickangle=-30, legend_title="指标")
        st.plotly_chart(fig_rank, use_container_width=True)

    with col_b:
        st.markdown("#### Rating 预测误差（越低越好）")
        df_err = df_eval[df_eval["RMSE"] > 0][["Model", "RMSE", "MAE"]]
        if df_err.empty:
            st.info("BPR、NeuMF 为隐式反馈模型，无评分误差指标。")
        else:
            df_err_melt = df_err.melt(id_vars="Model", var_name="Metric", value_name="Score")
            fig_err = px.bar(
                df_err_melt, x="Model", y="Score", color="Metric",
                barmode="group", text_auto=".3f",
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig_err.update_layout(xaxis_tickangle=-30, legend_title="指标")
            st.plotly_chart(fig_err, use_container_width=True)

    st.markdown("#### 完整数据表")
    st.dataframe(
        df_eval.set_index("Model").style.format("{:.4f}"),
        use_container_width=True,
    )
    st.divider()
    st.markdown(
        """
        **指标说明**
        - **HR@10**（Hit Ratio @10）：目标商品命中 Top-10 的用户比例，越高越好
        - **NDCG@10**（归一化折扣累积增益）：在 HR 基础上对排名位置加权，越高越好
        - **RMSE / MAE**：评分预测误差，仅对传统评分模型（SVD/CF）有意义，越低越好
        """
    )


def render_classic_recommendation_tab(ready_cats: list[str], pending_cats: list[str]):
    """经典单区推荐：分区、模型、用户选择与结果展示（无侧边栏）。"""
    st.caption("从已训练分区中选择用户，生成个性化 Top-K 推荐（新用户请用「新用户冷启动」页）。")

    c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
    with c1:
        cat = st.selectbox(
            "目标分区",
            ready_cats,
            format_func=lambda k: f"{CATEGORIES[k]['icon']} {CATEGORIES[k]['label']}",
            key="classic_cat",
        )
    with c2:
        model_label = st.selectbox(
            "推荐模型",
            list(MODEL_OPTIONS.keys()),
            key="classic_model",
        )
    with c3:
        top_k = st.slider("推荐数 K", 5, 20, 10, key="classic_top_k")
    with c4:
        model_key = MODEL_OPTIONS[model_label]
        is_trad = model_key in TRAD_MODEL_KEYS
        is_dl = not is_trad
        diversity_mode = False
        if is_dl:
            diversity_mode = st.checkbox(
                "多样性",
                value=False,
                key="classic_diversity",
                help="热门降权 + 固定种子抽样；关闭时纯分数排序，便于对照 LOO 命中。",
            )

    if pending_cats:
        with st.expander("ℹ️ 未激活的分区"):
            st.caption(
                "运行下列命令训练后刷新页面：\n\n"
                + "\n".join(
                    f"`python src/setup_category.py --category {k}`" for k in pending_cats
                )
            )

    if is_trad:
        trad_recs_path = get_paths(cat)["trad_recs"]
        if not os.path.exists(trad_recs_path):
            st.warning(
                f"传统模型尚未预计算，请运行：\n`python src/precompute_traditional.py --category {cat}`"
            )
        else:
            trad_users_path = get_paths(cat)["trad_users"]
            n_trad = len(json.load(open(trad_users_path))) if os.path.exists(trad_users_path) else 0
            st.info(f"📦 当前模型已为 **{n_trad}** 名用户预计算推荐（查表）。")

    cat_cfg = CATEGORIES[cat]
    cat_icon, cat_name = cat_cfg["icon"], cat_cfg["label"]

    ncf, bpr, device = load_models(cat)
    if ncf is None:
        st.error(f"未找到 **{cat_name}** 的已训练模型，请先运行 setup_category。")
        return

    item_stats, user_history, id_to_asin, meta, user_names = load_data(cat)
    trad_recs, trad_users, test_items = load_trad_data(cat)
    all_item_ids = list(id_to_asin.keys())
    has_trad = bool(trad_recs)

    if not meta.empty:
        st.caption(f"已加载 {len(meta)} 件商品 metadata（含图片）")

    u1, u2 = st.columns([3, 1])
    with u1:
        if is_trad and has_trad:
            available_users = [u for u in trad_users if u in user_history]
            st.caption(f"可选用户（预计算）：{len(available_users)} 名")
        else:
            available_users = sorted(user_history.keys())
        if not available_users:
            st.warning("该分区无可用用户。")
            return
        selected_user_id = st.selectbox(
            "选择用户",
            available_users,
            format_func=lambda u: (
                f"{user_names.get(u, f'User {u}')}  ({len(user_history[u])} 条记录)"
            ),
            key="classic_user",
        )
    with u2:
        run_btn = st.button("🔍 生成推荐", type="primary", use_container_width=True)

    if not run_btn:
        st.info("选择用户后点击「生成推荐」。")
        return

    uid = selected_user_id
    hist_list = user_history.get(uid, [])
    display_name = user_names.get(uid, f"User {uid}")
    last_item = test_items.get(str(uid))
    if last_item is not None:
        last_item = int(last_item)
    train_hist = set(hist_list) - ({last_item} if last_item else set())

    with st.expander(f"👤 {display_name} · 购买历史", expanded=False):
        st.caption(f"用户 ID: {uid} · {cat_icon} {cat_name}")
        all_counts = [len(v) for v in user_history.values()]
        pct = int(np.mean([c <= len(hist_list) for c in all_counts]) * 100)
        m1, m2 = st.columns(2)
        m1.metric("历史购买数", len(hist_list))
        m2.metric("活跃度", f"Top {100 - pct}%")
        if last_item is not None:
            st.markdown("**🎯 测试目标（最后一次购买）**")
            st.caption(item_display_name(last_item, id_to_asin, meta))
        for iid in hist_list[-12:]:
            st.caption(f"· {item_display_name(iid, id_to_asin, meta)[:50]}")

    title_suffix = " · 多样性" if diversity_mode and is_dl else ""
    st.subheader(f"🎯 Top-{top_k} · {model_key}{title_suffix}")
    if diversity_mode and is_dl:
        st.caption("热门降权 + 固定种子抽样")

    if is_trad:
        if not has_trad:
            st.error(f"请先运行：`python src/precompute_traditional.py --category {cat}`")
            return
        if model_key not in trad_recs or str(uid) not in trad_recs[model_key]:
            st.warning("此用户没有预计算数据。")
            return
        recs = [(int(iid), float(s)) for iid, s in trad_recs[model_key][str(uid)][:top_k]]
    else:
        div_seed = stable_seed(f"{cat}:{uid}", model_key) if diversity_mode else None
        stats_rec = item_stats if diversity_mode else None
        with st.spinner("推理中…"):
            recs = get_topk_recs(
                model_key, ncf, bpr, device,
                uid, all_item_ids, train_hist, top_k,
                item_stats=stats_rec,
                diversify_seed=div_seed,
            )

    if last_item is not None and not any(iid == last_item for iid, _ in recs):
        st.caption("💬 测试目标未出现在本次推荐列表中")
    if last_item is not None and any(iid == last_item for iid, _ in recs):
        st.success("🎯 测试目标已出现在推荐列表中")

    cat_cache = {cat: (item_stats, id_to_asin, meta)}
    grid_items = [
        {
            "cat": cat,
            "item_id": iid,
            "rank": rank,
            "score": score,
            "highlight": last_item is not None and iid == last_item,
        }
        for rank, (iid, score) in enumerate(recs, 1)
    ]
    render_product_grid(grid_items, cat_cache)


def render_smart_demo(ready_cats: list, top_k: int):
    """跨区展示：仅选人 + 推荐列表。"""
    if len(ready_cats) < 2:
        return

    profiles = get_reviewer_profiles()
    cross_reviewers = list_demo_reviewers(
        profiles, ready_cats, min_cats=2, limit=500, cross_domain_first=True
    )
    if not cross_reviewers:
        return

    reviewer = st.selectbox(
        "选择用户",
        cross_reviewers,
        format_func=lambda r: reviewer_label(profiles, r, ready_cats),
        key="cross_user_pick",
    )
    render_cross_recommendations(reviewer, profiles, ready_cats, top_k)

    # st.header("🌐 跨区用户 · 逐人推荐")
    # st.caption(...)
    # search / roster dataframe / compare_dl tabs / st.container(border=True) ...


# ──────────────────────────────────────────────
# SASRec 演示适配层（当前主入口）
# ──────────────────────────────────────────────
PLATFORM_ROOT = os.path.dirname(os.path.abspath(__file__))
SASREC_DATASET = "movies_tv_seq"
SASREC_BASE_DATASET = "movies_tv"
SASREC_MODEL_KEY = "sasrec"
SASREC_LABEL = "Movies & TV"
SASREC_DISPLAY_DIR = os.path.join(PLATFORM_ROOT, "datasets", SASREC_BASE_DATASET, "display")
SASREC_FUTURE_DIR = os.path.join(PLATFORM_ROOT, "results", "sasrec_future")


def _parse_seq_tokens(raw) -> list[str]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    if isinstance(raw, (list, tuple, np.ndarray)):
        return [str(x) for x in raw if str(x)]
    return [x for x in str(raw).strip().split() if x]


def _seq_inter_path(split: str = "test") -> str:
    return os.path.join(
        PLATFORM_ROOT,
        "datasets",
        SASREC_DATASET,
        f"{SASREC_DATASET}.{split}.inter",
    )


@st.cache_data(show_spinner="正在加载 SASRec 用户序列…")
def load_sasrec_user_sequences(split: str = "test") -> pd.DataFrame:
    path = _seq_inter_path(split)
    if not os.path.exists(path):
        return pd.DataFrame(columns=["user_id", "item_id", "item_id_list", "item_length", "hist_len"])
    df = pd.read_csv(
        path,
        sep="\t",
        skiprows=1,
        names=["user_id", "item_id", "item_id_list", "item_length"],
        dtype={"user_id": str, "item_id": str, "item_id_list": str},
    )
    df = df.drop_duplicates("user_id", keep="last").reset_index(drop=True)
    df["hist_len"] = df["item_id_list"].map(lambda x: len(_parse_seq_tokens(x)))
    return df


@st.cache_data(show_spinner="正在统计 POP 热门项目…")
def load_popularity_by_token() -> pd.Series:
    counts: dict[str, int] = {}
    data_dir_path = os.path.join(PLATFORM_ROOT, "datasets", SASREC_BASE_DATASET)
    for split in ("train", "valid", "test"):
        path = os.path.join(data_dir_path, f"{SASREC_BASE_DATASET}.{split}.inter")
        if not os.path.exists(path):
            continue
        for chunk in pd.read_csv(
            path,
            sep="\t",
            skiprows=1,
            names=["user_id", "item_id", "rating", "timestamp"],
            usecols=["item_id"],
            dtype={"item_id": str},
            chunksize=500_000,
        ):
            vc = chunk["item_id"].value_counts()
            for token, cnt in vc.items():
                counts[str(token)] = counts.get(str(token), 0) + int(cnt)
    return pd.Series(counts, dtype="int64").sort_values(ascending=False)


@st.cache_data(show_spinner="正在统计影片评分…")
def load_item_rating_stats_by_token() -> pd.DataFrame:
    cache_path = os.path.join(SASREC_DISPLAY_DIR, "item_rating_stats.csv")
    if os.path.exists(cache_path):
        cached = pd.read_csv(cache_path, dtype={"item_id": str})
        if "item_id" in cached.columns:
            return cached.set_index("item_id")

    stats: dict[str, list[float]] = {}
    data_dir_path = os.path.join(PLATFORM_ROOT, "datasets", SASREC_BASE_DATASET)
    for split in ("train", "valid", "test"):
        path = os.path.join(data_dir_path, f"{SASREC_BASE_DATASET}.{split}.inter")
        if not os.path.exists(path):
            continue
        for chunk in pd.read_csv(
            path,
            sep="\t",
            skiprows=1,
            names=["user_id", "item_id", "rating", "timestamp"],
            usecols=["item_id", "rating"],
            dtype={"item_id": str},
            chunksize=500_000,
        ):
            grouped = chunk.groupby("item_id")["rating"].agg(["sum", "count"])
            for token, row in grouped.iterrows():
                current = stats.setdefault(str(token), [0.0, 0.0])
                current[0] += float(row["sum"])
                current[1] += float(row["count"])
    if not stats:
        return pd.DataFrame(columns=["avg_rating", "review_count"])
    rows = [
        {
            "item_id": token,
            "avg_rating": total / count if count else 0.0,
            "review_count": int(count),
        }
        for token, (total, count) in stats.items()
    ]
    out = pd.DataFrame(rows)
    os.makedirs(SASREC_DISPLAY_DIR, exist_ok=True)
    out.to_csv(cache_path, index=False, encoding="utf-8")
    return out.set_index("item_id")


def display_metadata_version() -> tuple[float, float, float]:
    item_path = os.path.join(SASREC_DISPLAY_DIR, "item_metadata.csv")
    user_path = os.path.join(SASREC_DISPLAY_DIR, "user_profiles.csv")
    enrich_path = os.path.join(SASREC_DISPLAY_DIR, "prime_video_enrichment.json")
    return (
        os.path.getmtime(item_path) if os.path.exists(item_path) else 0.0,
        os.path.getmtime(user_path) if os.path.exists(user_path) else 0.0,
        os.path.getmtime(enrich_path) if os.path.exists(enrich_path) else 0.0,
    )


def _apply_prime_enrichment(item_meta: pd.DataFrame) -> pd.DataFrame:
    enrich_path = os.path.join(SASREC_DISPLAY_DIR, "prime_video_enrichment.json")
    if item_meta.empty or not os.path.exists(enrich_path):
        return item_meta
    with open(enrich_path, encoding="utf-8") as f:
        cache = json.load(f)
    if not cache:
        return item_meta
    out = item_meta.copy()
    if "asin" not in out.columns:
        return out
    for item_id, row in out.iterrows():
        extra = cache.get(str(row.get("asin", "")), {})
        if not extra:
            continue
        current_title = str(row.get("title", "") or "").strip()
        if extra.get("title") and (not current_title or current_title.startswith("Prime Video ·")):
            out.at[item_id, "title"] = extra["title"]
        current_url = str(row.get("imUrl", "") or "")
        if extra.get("imUrl") and ("_RI_" in current_url or not current_url.strip()):
            out.at[item_id, "imUrl"] = extra["imUrl"]
    return out


@st.cache_data(show_spinner="正在加载展示 metadata…")
def load_sasrec_display_metadata(_version: tuple[float, float, float] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    item_path = os.path.join(SASREC_DISPLAY_DIR, "item_metadata.csv")
    user_path = os.path.join(SASREC_DISPLAY_DIR, "user_profiles.csv")
    item_meta = pd.DataFrame()
    user_profiles = pd.DataFrame()
    if os.path.exists(item_path):
        item_meta = pd.read_csv(item_path, dtype={"item_id": str, "asin": str})
        if "item_id" in item_meta.columns:
            item_meta = item_meta.drop_duplicates("item_id").set_index("item_id")
        item_meta = _apply_prime_enrichment(item_meta)
    if os.path.exists(user_path):
        user_profiles = pd.read_csv(user_path, dtype={"user_id": str})
        if "user_id" in user_profiles.columns:
            user_profiles = user_profiles.drop_duplicates("user_id").set_index("user_id")
    return item_meta, user_profiles


def current_display_metadata() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_sasrec_display_metadata(display_metadata_version())


def default_future_recs_path(top_k: int = 100) -> str:
    return os.path.join(
        SASREC_FUTURE_DIR,
        f"{SASREC_DATASET}_sasrec_future_top{top_k}.jsonl",
    )


@st.cache_data(show_spinner="正在查询离线 future 推荐缓存…")
def lookup_future_recs_by_user(user_id: str, top_k: int, path: str | None = None) -> dict | None:
    rec_path = path or default_future_recs_path(100)
    if not os.path.exists(rec_path):
        return None
    wanted = str(user_id)
    with open(rec_path, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(row.get("user_id")) == wanted:
                return {
                    "items": list(row.get("items", []))[:top_k],
                    "scores": list(row.get("scores", []))[:top_k],
                    "history_len": row.get("history_len"),
                    "path": rec_path,
                }
    return None


def _read_sasrec_log_candidates() -> list[str]:
    out: list[str] = []
    logs_dir = os.path.join(PLATFORM_ROOT, "results", "logs")
    if os.path.isdir(logs_dir):
        names = [
            n for n in os.listdir(logs_dir)
            if "SASRec" in n and n.endswith(".json")
        ]
        names.sort(key=lambda n: os.path.getmtime(os.path.join(logs_dir, n)), reverse=True)
        for name in names:
            try:
                with open(os.path.join(logs_dir, name), encoding="utf-8") as f:
                    payload = json.load(f)
                if payload.get("dataset") != SASREC_DATASET or payload.get("model") != "SASRec":
                    continue
                ckpt = payload.get("best_ckpt")
                if ckpt:
                    out.append(str(ckpt))
            except (OSError, json.JSONDecodeError):
                continue
    meta_path = os.path.join(os.path.dirname(PLATFORM_ROOT), "SASrec_meta_0.79.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as f:
                payload = json.load(f)
            ckpt = payload.get("best_ckpt")
            if ckpt:
                out.append(str(ckpt))
        except (OSError, json.JSONDecodeError):
            pass
    out.extend([
        r"D:\recbole_checkpoints\movies_tv_seq\SASRec\best.pth",
    ])
    return list(dict.fromkeys(out))


def resolve_sasrec_checkpoint() -> tuple[str, bool, list[str]]:
    candidates = _read_sasrec_log_candidates()
    for path in candidates:
        if os.path.exists(path):
            return path, True, candidates
    return (candidates[0] if candidates else "", False, candidates)


@st.cache_resource(show_spinner="正在加载 RecBole SASRec 模型…")
def load_sasrec_bundle_cached(checkpoint_path: str):
    if PLATFORM_ROOT not in sys.path:
        sys.path.insert(0, PLATFORM_ROOT)
    from scripts.eval_cascade_rank import load_model_bundle

    return load_model_bundle(SASREC_MODEL_KEY, checkpoint=checkpoint_path)


def token_to_internal(bundle, token: str) -> int | None:
    try:
        return int(bundle.dataset.token2id(bundle.dataset.iid_field, str(token)))
    except (KeyError, ValueError):
        return None


def internal_to_token(bundle, item_id: int) -> str:
    try:
        return str(bundle.dataset.id2token(bundle.dataset.iid_field, int(item_id)))
    except (KeyError, ValueError):
        return str(item_id)


def tokens_to_internal(bundle, tokens: list[str]) -> list[int]:
    out: list[int] = []
    for token in tokens:
        iid = token_to_internal(bundle, token)
        if iid is not None and iid > 0:
            out.append(iid)
    return out


def pop_counts_internal(bundle, pop_counts: pd.Series, limit: int = 2000) -> dict[int, int]:
    out: dict[int, int] = {}
    for token, count in pop_counts.head(limit).items():
        iid = token_to_internal(bundle, str(token))
        if iid is not None and iid > 0:
            out[iid] = int(count)
    return out


def sasrec_item_title(bundle, item_id: int, item_meta: pd.DataFrame | None = None) -> str:
    token = internal_to_token(bundle, item_id)
    if item_meta is not None and not item_meta.empty and token in item_meta.index:
        row = item_meta.loc[token]
        title = row.get("title", "")
        if pd.notna(title) and str(title).strip() and str(title).strip().lower() != "nan":
            title_text = str(title).strip()
            if title_text.startswith("Prime Video ·"):
                title_text = title_text.replace("Prime Video ·", "", 1).strip()
            return title_text[:72]
        brand = row.get("brand", "")
        if pd.notna(brand) and str(brand).strip():
            return str(brand)[:72]
        starring = row.get("details_starring", "")
        if pd.notna(starring) and str(starring).strip():
            return str(starring)[:72]
        main_category = row.get("main_category", "")
        if pd.notna(main_category) and str(main_category).strip():
            return str(main_category)[:72]
        categories = row.get("categories", "")
        if pd.notna(categories) and str(categories).strip():
            return str(categories).split(">")[-1].strip()[:72]
        asin = row.get("asin", "")
        if pd.notna(asin) and str(asin).strip():
            return f"ASIN {str(asin)}"
    return f"Item {token}"


def sasrec_item_image(bundle, item_id: int, item_meta: pd.DataFrame | None = None) -> str | None:
    token = internal_to_token(bundle, item_id)
    if item_meta is None or item_meta.empty or token not in item_meta.index:
        return None
    for col in ("imUrl", "image", "imageURL"):
        if col in item_meta.columns:
            url = item_meta.loc[token].get(col, "")
            if pd.notna(url) and str(url).strip():
                return str(url)
    return None

def _product_card_text_html(
    *,
    title: str,
    domain: str = "",
    meta_line: str = "",
    rank: int | None = None,
    score: float | None = None,
    badge: str = "",
    prime_badge: bool = False,
) -> str:
    parts: list[str] = []
    if rank is not None:
        parts.append(f'<div class="product-rank">#{rank}</div>')
    if prime_badge:
        parts.append('<div class="product-prime-badge">prime video</div>')
    if domain:
        parts.append(f'<div class="product-domain">{html.escape(domain)}</div>')
    parts.append(f'<div class="product-title">{html.escape(title)}</div>')
    if meta_line:
        parts.append(f'<div class="product-meta">{html.escape(meta_line)}</div>')
    if badge:
        parts.append(f'<div class="product-meta">{html.escape(badge)}</div>')
    return "".join(parts)


def sasrec_cover_image_html(img_url: str | None, *, prime: bool = False) -> str:
    # Do not synchronously download covers during render; let the browser lazy-load them.
    return _sasrec_product_image_html(img_url, prime=prime)


def sasrec_item_review_line(
    bundle,
    item_id: int,
    item_meta: pd.DataFrame | None,
    pop_counts: dict[int, int],
    rating_stats: pd.DataFrame | None,
) -> str:
    token = internal_to_token(bundle, item_id)
    if item_meta is not None and not item_meta.empty and token in item_meta.index:
        row = item_meta.loc[token]
        avg = row.get("average_rating", "")
        count = row.get("rating_number", "")
        try:
            avg_float = float(avg)
            count_int = int(float(count))
            if avg_float > 0 and count_int > 0:
                return f"{avg_float:.1f}⭐ · {count_int:,}评价"
        except (TypeError, ValueError):
            pass
    if rating_stats is not None and not rating_stats.empty and token in rating_stats.index:
        row = rating_stats.loc[token]
        avg = float(row.get("avg_rating", 0) or 0)
        count = int(row.get("review_count", 0) or 0)
        if count and avg > 1.01:
            return f"{avg:.1f}⭐ · {count:,}评价"
    if item_id in pop_counts:
        return f"{pop_counts[item_id]:,}评价"
    return ""


def render_sasrec_product_tile_content(
    *,
    bundle,
    item_id: int,
    item_meta: pd.DataFrame,
    pop_counts: dict[int, int],
    rating_stats: pd.DataFrame | None = None,
    rank: int | None = None,
    score: float | None = None,
    highlight: bool = False,
    badge: str = "",
    show_score: bool = True,
) -> None:
    meta_line = sasrec_item_review_line(bundle, item_id, item_meta, pop_counts, rating_stats)
    prime_badge = is_prime_video_item(bundle, item_id, item_meta)
    card_cls = "product-card"
    if highlight:
        card_cls += " product-card--hit"
    if prime_badge:
        card_cls += " product-card--prime"
    st.markdown(
        "".join(
            [
                f'<div class="{card_cls}">',
                sasrec_cover_image_html(sasrec_item_image(bundle, item_id, item_meta), prime=prime_badge),
                _product_card_text_html(
                    title=sasrec_item_title(bundle, item_id, item_meta),
                    domain="",
                    meta_line=meta_line,
                    rank=rank,
                    score=score if show_score else None,
                    badge=badge,
                    prime_badge=prime_badge,
                ),
                "</div>",
            ]
        ),
        unsafe_allow_html=True,
    )


def render_sasrec_product_tile_in_column(
    col,
    *,
    bundle,
    item_id: int,
    item_meta: pd.DataFrame,
    pop_counts: dict[int, int],
    rating_stats: pd.DataFrame | None = None,
    rank: int | None = None,
    score: float | None = None,
    highlight: bool = False,
    badge: str = "",
    show_score: bool = True,
) -> None:
    with col:
        render_sasrec_product_tile_content(
            bundle=bundle,
            item_id=item_id,
            item_meta=item_meta,
            pop_counts=pop_counts,
            rating_stats=rating_stats,
            rank=rank,
            score=score,
            highlight=highlight,
            badge=badge,
            show_score=show_score,
        )


def sasrec_user_label(user_id: str, user_profiles: pd.DataFrame) -> str:
    if not user_profiles.empty and user_id in user_profiles.index:
        row = user_profiles.loc[user_id]
        name = row.get("display_name", f"User {user_id}")
        name_text = str(name).strip()
        if not name_text or name_text == f"User {user_id}" or name_text.startswith("Reviewer "):
            name_text = f"用户 {user_id}"
        cnt = int(row.get("rating_count", 0) or 0)
        avg = float(row.get("avg_rating", 0) or 0)
        return f"{name_text}  ({cnt} 条记录 · 均分 {avg:.2f})"
    return f"用户 {user_id}"


def sasrec_raw_reviewer_id(user_id: str, user_profiles: pd.DataFrame) -> str:
    if not user_profiles.empty and user_id in user_profiles.index:
        raw_user = user_profiles.loc[user_id].get("raw_user_id", "")
        if pd.notna(raw_user) and str(raw_user).strip():
            return str(raw_user)
    return ""


def is_prime_video_item(bundle, item_id: int, item_meta: pd.DataFrame | None = None) -> bool:
    token = internal_to_token(bundle, item_id)
    if item_meta is None or item_meta.empty or token not in item_meta.index:
        return False
    row = item_meta.loc[token]
    main_category = str(row.get("main_category", "") or "").strip().lower()
    categories = str(row.get("categories", "") or "").strip().lower()
    url = str(row.get("imUrl", "") or "").strip().lower()
    return main_category == "prime video" or "prime video" in categories or "pv-target-images" in url


def render_sasrec_product_grid(
    items: list[dict],
    bundle,
    pop_counts: dict[int, int],
    *,
    cols_per_row: int = 6,
    show_score: bool = True,
):
    inject_product_grid_css()
    item_meta, _ = current_display_metadata()
    rating_stats = load_item_rating_stats_by_token()
    if not items:
        st.info("暂无可展示项目。")
        return
    for i in range(0, len(items), cols_per_row):
        row_cols = st.columns(cols_per_row)
        for col, entry in zip(row_cols, items[i : i + cols_per_row]):
            render_sasrec_product_tile_in_column(
                col,
                bundle=bundle,
                item_id=int(entry["item_id"]),
                item_meta=item_meta,
                pop_counts=pop_counts,
                rating_stats=rating_stats,
                rank=entry.get("rank"),
                score=entry.get("score"),
                highlight=entry.get("highlight", False),
                badge=entry.get("kind", ""),
                show_score=show_score,
            )


def toggle_session_bool(key: str) -> None:
    st.session_state[key] = not st.session_state.get(key, False)


def render_sasrec_pickable_grid(
    item_ids: list[int],
    bundle,
    pop_counts: dict[int, int],
    *,
    key_prefix: str,
    cols_per_row: int = 6,
) -> list[int]:
    inject_product_grid_css()
    item_meta, _ = current_display_metadata()
    rating_stats = load_item_rating_stats_by_token()
    order_key = f"{key_prefix}_ordered_items"
    previous_order = [int(x) for x in st.session_state.get(order_key, [])]
    checked_now: list[int] = []
    for i in range(0, len(item_ids), cols_per_row):
        row_cols = st.columns(cols_per_row)
        for col, iid in zip(row_cols, item_ids[i : i + cols_per_row]):
            with col:
                render_sasrec_product_tile_content(
                    bundle=bundle,
                    item_id=iid,
                    item_meta=item_meta,
                    pop_counts=pop_counts,
                    rating_stats=rating_stats,
                    show_score=False,
                )
                if st.checkbox("感兴趣", key=f"{key_prefix}_{iid}"):
                    checked_now.append(iid)
    checked_set = set(checked_now)
    ordered = [iid for iid in previous_order if iid in checked_set]
    ordered.extend([iid for iid in checked_now if iid not in set(ordered)])
    st.session_state[order_key] = ordered
    return ordered


@torch.no_grad()
def sasrec_rank_from_history(
    bundle,
    history_internal: list[int],
    *,
    top_k: int,
    exclude: set[int] | None = None,
) -> list[tuple[int, float]]:
    from recbole.data.interaction import Interaction

    model = bundle.model
    device = next(model.parameters()).device
    config = bundle.config
    max_len = int(config["MAX_ITEM_LIST_LENGTH"])
    seq_field = config["ITEM_ID_FIELD"] + config["LIST_SUFFIX"]
    len_field = config["ITEM_LIST_LENGTH_FIELD"]
    uid_field = config["USER_ID_FIELD"]

    hist = [int(i) for i in history_internal if int(i) > 0][-max_len:]
    length = len(hist)
    seq_batch = torch.zeros(1, max_len, dtype=torch.long, device=device)
    if length:
        seq_batch[0, :length] = torch.tensor(hist, dtype=torch.long, device=device)
    inter = Interaction({
        uid_field: torch.zeros(1, dtype=torch.long, device=device),
        seq_field: seq_batch,
        len_field: torch.tensor([length], dtype=torch.long, device=device),
    })

    scores = model.full_sort_predict(inter.to(device))
    if scores.dim() == 1:
        scores = scores.view(1, -1)
    scores = scores[0].clone()
    scores[0] = -np.inf
    for iid in exclude or set(hist):
        if 0 <= int(iid) < scores.numel():
            scores[int(iid)] = -np.inf
    k = min(top_k, int(torch.isfinite(scores).sum().item()))
    if k <= 0:
        return []
    vals, idx = torch.topk(scores, k)
    return [(int(i), float(s)) for i, s in zip(idx.cpu().tolist(), vals.cpu().tolist())]


def top_pop_items(pop_internal: dict[int, int], top_k: int, exclude: set[int] | None = None) -> list[tuple[int, float]]:
    blocked = exclude or set()
    rows = [(iid, float(cnt)) for iid, cnt in pop_internal.items() if iid not in blocked]
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:top_k]


def top_pop_items_by_prime(
    pop_internal: dict[int, int],
    bundle,
    item_meta: pd.DataFrame,
    *,
    top_k: int,
    prime: bool,
    exclude: set[int] | None = None,
) -> list[tuple[int, float]]:
    blocked = exclude or set()
    rows = [
        (iid, float(cnt))
        for iid, cnt in pop_internal.items()
        if iid not in blocked and is_prime_video_item(bundle, iid, item_meta) == prime
    ]
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:top_k]


def sasrec_metric_summary() -> dict:
    preferred = os.path.join(
        PLATFORM_ROOT,
        "results",
        "logs",
        "movies_tv_seq_SASRec_full_20260604_125756.json",
    )
    if not os.path.exists(preferred):
        return {}
    with open(preferred, encoding="utf-8") as f:
        return json.load(f)


def render_sasrec_history_panel(
    *,
    user_id: str,
    reviewer_id: str,
    bundle,
    history_tokens: list[str],
    pop_internal: dict[int, int],
) -> None:
    history_internal = tokens_to_internal(bundle, history_tokens)
    if not history_internal:
        st.info("该用户暂无可展示历史。")
        return

    st.markdown("#### 最近历史")
    recent = history_internal[-12:]
    grid_items = [
        {"item_id": iid, "rank": rank, "kind": "历史"}
        for rank, iid in enumerate(recent, 1)
    ]
    render_sasrec_product_grid(grid_items, bundle, pop_internal, show_score=False)


def render_sasrec_existing_user_page(bundle, pop_internal: dict[int, int]):
    user_df = load_sasrec_user_sequences("test")
    _, user_profiles = current_display_metadata()
    if user_df.empty:
        st.error("未找到 SASRec test 序列文件。")
        return

    candidates = (
        user_df[user_df["hist_len"] >= 5]
        .sort_values("hist_len", ascending=False)
        .head(500)
    )
    _, user_col, k_col, history_col = st.columns([4.2, 2.4, 1.0, 1.1])
    with user_col:
        picked = st.selectbox(
            "用户",
            candidates["user_id"].tolist(),
            format_func=lambda u: sasrec_user_label(str(u), user_profiles),
            key="sasrec_user_pick",
            label_visibility="collapsed",
        )
    user_id = str(picked)
    history_state_key = f"sasrec_show_history_{user_id}"
    with k_col:
        top_k = st.slider("数量", 5, 50, 30, key="sasrec_existing_k", label_visibility="collapsed")
    with history_col:
        is_history_page = st.session_state.get(history_state_key, False)
        st.button(
            "返回" if is_history_page else "查看历史",
            use_container_width=True,
            on_click=toggle_session_bool,
            args=(history_state_key,),
        )

    row_match = user_df[user_df["user_id"] == user_id]
    if row_match.empty:
        st.warning(f"找不到用户 `{user_id}`。")
        return
    row = row_match.iloc[0]
    hist_tokens = _parse_seq_tokens(row["item_id_list"])
    future_hist_tokens = hist_tokens + [str(row["item_id"])]
    future_hist_internal = tokens_to_internal(bundle, future_hist_tokens)
    target_internal = token_to_internal(bundle, str(row["item_id"]))
    reviewer_id = sasrec_raw_reviewer_id(user_id, user_profiles)

    if st.session_state.get(history_state_key, False):
        render_sasrec_history_panel(
            user_id=user_id,
            reviewer_id=reviewer_id,
            bundle=bundle,
            history_tokens=future_hist_tokens,
            pop_internal=pop_internal,
        )
        return

    cached = lookup_future_recs_by_user(user_id, top_k)
    if cached:
        cached_items = tokens_to_internal(bundle, [str(x) for x in cached["items"]])
        scores = [float(x) for x in cached["scores"]]
        recs = list(zip(cached_items, scores[: len(cached_items)]))
    else:
        with st.spinner("正在生成推荐…"):
            recs = sasrec_rank_from_history(
                bundle,
                future_hist_internal,
                top_k=top_k,
                exclude=set(future_hist_internal),
            )

    grid_items = [
        {
            "item_id": iid,
            "rank": rank,
            "score": score,
            "highlight": target_internal is not None and iid == target_internal,
        }
        for rank, (iid, score) in enumerate(recs, 1)
    ]
    render_sasrec_product_grid(grid_items, bundle, pop_internal)


def render_sasrec_new_user_page(bundle, pop_internal: dict[int, int]):
    item_meta, _ = current_display_metadata()

    title_col, pool_col, k_col = st.columns([5.0, 1.25, 1.25])
    with title_col:
        st.markdown("### 热门发现")
    with pool_col:
        show_n = st.slider("展示", 6, 50, 30, step=2, key="sasrec_cold_pool")
    with k_col:
        top_k = st.slider("推荐", 5, 50, 30, key="sasrec_cold_k")

    prime_showcase = [
        iid
        for iid, _ in top_pop_items_by_prime(
            pop_internal,
            bundle,
            item_meta,
            top_k=show_n,
            prime=True,
        )
    ]
    regular_showcase = [
        iid
        for iid, _ in top_pop_items_by_prime(
            pop_internal,
            bundle,
            item_meta,
            top_k=show_n,
            prime=False,
        )
    ]
    with st.form(f"sasrec_cold_form_{show_n}"):
        tab_prime, tab_regular = st.tabs(["Prime Video", "Movies & TV"])
        with tab_prime:
            selected_prime = render_sasrec_pickable_grid(
                prime_showcase,
                bundle,
                pop_internal,
                key_prefix=f"sasrec_cold_prime_{show_n}",
            )
        with tab_regular:
            selected_regular = render_sasrec_pickable_grid(
                regular_showcase,
                bundle,
                pop_internal,
                key_prefix=f"sasrec_cold_regular_{show_n}",
            )
        submitted = st.form_submit_button("生成推荐", type="primary", use_container_width=True)
    selected = selected_prime + [iid for iid in selected_regular if iid not in set(selected_prime)]

    if not submitted:
        return

    if not selected:
        st.info("请先勾选感兴趣的影片。")
        return

    with st.spinner("正在生成推荐…"):
        recs = sasrec_rank_from_history(
            bundle,
            selected,
            top_k=top_k,
            exclude=set(selected),
        )
    grid_items = [
        {"item_id": iid, "rank": rank, "score": score, "kind": "冷启动"}
        for rank, (iid, score) in enumerate(recs, 1)
    ]
    st.markdown("#### 推荐影片")
    render_sasrec_product_grid(grid_items, bundle, pop_internal)


def render_sasrec_status_page(checkpoint_path: str, checkpoint_exists: bool, candidates: list[str]):
    st.subheader("📌 数据与模型状态")
    st.write(f"数据集：`datasets/{SASREC_DATASET}`")
    st.write(f"模型：`SASRec`")
    st.write(f"当前权重：`{checkpoint_path or '未解析到路径'}`")
    if checkpoint_exists:
        st.success("权重文件存在，可以进行在线推理。")
    else:
        st.error("权重文件不存在。请确认 checkpoint 已放在上方路径，或更新训练日志中的 `best_ckpt`。")
        with st.expander("尝试过的 checkpoint 路径"):
            for path in candidates:
                st.caption(path)

    metrics = sasrec_metric_summary()
    if metrics:
        st.markdown("#### 最近训练指标")
        cols = st.columns(4)
        test = metrics.get("test_result", {})
        full = metrics.get("test_full_sort_result", {})
        cols[0].metric("uni100 HR@10", f"{float(test.get('hr@10', 0)):.4f}")
        cols[1].metric("uni100 NDCG@10", f"{float(test.get('ndcg@10', 0)):.4f}")
        cols[2].metric("全库 HR@10", f"{float(full.get('hr@10', 0)):.4f}")
        cols[3].metric("全库 HR@50", f"{float(full.get('hr@50', 0)):.4f}")

    st.markdown("#### 展示 metadata 状态")
    item_meta, user_profiles = current_display_metadata()
    user_df = load_sasrec_user_sequences("test")
    pop_counts = load_popularity_by_token()
    item_overlap = len(set(item_meta.index.astype(str)) & set(pop_counts.head(5000).index.astype(str))) if not item_meta.empty else 0
    user_overlap = len(set(user_profiles.index.astype(str)) & set(user_df["user_id"].astype(str).head(5000))) if not user_profiles.empty and not user_df.empty else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("item metadata", len(item_meta))
    c2.metric("top item 可匹配", item_overlap)
    c3.metric("user profiles", len(user_profiles))
    c4.metric("示例用户可匹配", user_overlap)
    if len(item_meta) > 0 and item_overlap == 0:
        st.warning(
            "已读到 metadata，但 item_id 与 RecBole 数字 token 不匹配。"
            "需要提供 raw ASIN -> RecBole item_id 的映射后重新构建 display metadata。"
        )
    if len(user_profiles) > 0 and user_overlap == 0:
        st.warning(
            "已读到用户 profile，但 reviewerID 与 RecBole 数字 user_id 不匹配。"
            "需要提供 raw reviewerID -> RecBole user_id 的映射后重新构建 display metadata。"
        )


# ──────────────────────────────────────────────
# 主界面
# ──────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Movies & TV Store",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    inject_product_grid_css()
    st.markdown('<div class="store-topbar">Movies &amp; TV</div>', unsafe_allow_html=True)

    checkpoint_path, checkpoint_exists, candidates = resolve_sasrec_checkpoint()
    tab_existing, tab_new, tab_status = st.tabs(["我的首页", "新客发现", "状态"])

    if not checkpoint_exists:
        with tab_status:
            render_sasrec_status_page(checkpoint_path, checkpoint_exists, candidates)
        st.warning("SASRec 权重文件暂不可用，已先展示状态页。")
        return

    try:
        bundle = load_sasrec_bundle_cached(checkpoint_path)
    except Exception as exc:
        with tab_status:
            render_sasrec_status_page(checkpoint_path, checkpoint_exists, candidates)
            st.exception(exc)
        st.error("SASRec 模型加载失败，请先查看状态页中的异常信息。")
        return

    pop_counts = load_popularity_by_token()
    pop_internal = pop_counts_internal(bundle, pop_counts, limit=5000)

    with tab_existing:
        render_sasrec_existing_user_page(bundle, pop_internal)

    with tab_new:
        render_sasrec_new_user_page(bundle, pop_internal)

    with tab_status:
        render_sasrec_status_page(checkpoint_path, checkpoint_exists, candidates)


def _running_inside_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__":
    if not _running_inside_streamlit():
        # 直接 `python app.py` 时拉起 Streamlit 子进程
        import subprocess
        import sys

        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", __file__, "--server.headless", "true"],
            check=False,
        )
    else:
        # `streamlit run app.py` 由 Streamlit 执行脚本时真正渲染页面
        main()
