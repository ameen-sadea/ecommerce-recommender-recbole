#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
构建 Movies & TV 页面展示 metadata。

目标输出不参与 RecBole 训练，只供 Streamlit 展示使用:
  datasets/movies_tv/display/item_metadata.csv
  datasets/movies_tv/display/user_profiles.csv

支持输入:
  - Amazon reviews JSON/JSON.GZ: reviewerID, reviewerName, asin, overall
  - Amazon metadata JSON/JSON.GZ: asin, title, imUrl/imageURL/imageURLHighRes
  - 可选映射文件，把 raw reviewerID/asin 映射到 RecBole user_id/item_id

示例:
  python scripts/build_movies_tv_display_metadata.py ^
    --reviews D:/amazon/Movies_and_TV_5.json.gz ^
    --metadata D:/amazon/meta_Movies_and_TV.json.gz

如果当前 RecBole 数据的 user_id/item_id 已经是 raw reviewerID/asin，可不传映射。
如果是重新编码后的数字 token，请传:
  --user-map user_mapping.json --item-map item_mapping.json

映射文件支持 JSON(dict) 或 CSV 两列:
  raw_id, token_id
  reviewerID, user_id
  asin, item_id
"""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from typing import Any, Iterable

import pandas as pd

PLATFORM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT_DIR = os.path.join(PLATFORM_ROOT, "datasets", "movies_tv", "display")
DEFAULT_INTERACTIONS = os.path.join(PLATFORM_ROOT, "data", "Movies_and_TV", "cleaned.csv")


def open_text(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")


def iter_json_lines(path: str) -> Iterable[dict[str, Any]]:
    with open_text(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Older Amazon dumps are Python-literal-like rather than strict JSON.
                try:
                    row = ast.literal_eval(line)
                    if isinstance(row, dict):
                        yield row
                except (SyntaxError, ValueError):
                    continue


def load_mapping(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return {str(k): str(v) for k, v in data.items()}

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        if len(cols) < 2:
            raise ValueError(f"{path} 至少需要两列")
        raw_col = next((c for c in cols if c.lower() in {"raw_id", "reviewerid", "asin"}), cols[0])
        token_col = next((c for c in cols if c.lower() in {"token_id", "user_id", "item_id"}), cols[1])
        return {str(row[raw_col]): str(row[token_col]) for row in reader if row.get(raw_col)}


def mapped(mapping: dict[str, str], raw_id: Any) -> str:
    raw = str(raw_id)
    return mapping.get(raw, raw)


def _first_text(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, list):
        for item in value:
            text = _first_text(item)
            if text:
                return text
    return ""


def extract_title(row: dict[str, Any]) -> str:
    for key in ("title", "subtitle", "store"):
        text = _first_text(row.get(key))
        if text:
            return text
    for key in ("features", "description"):
        text = _first_text(row.get(key))
        if text:
            return text[:120]
    categories = row.get("categories")
    if isinstance(categories, list) and categories:
        flat = [str(x) for group in categories for x in (group if isinstance(group, list) else [group]) if str(x).strip()]
        if flat:
            return flat[-1]
    return ""


def image_url_score(url: str, size_key: str = "") -> int:
    if not url:
        return -10_000
    score = 0
    if "Default_Background_Art" in url:
        return -10_000
    if "/images/I/" in url:
        score += 1000
        if size_key in {"hi_res", "large"} or any(tag in url for tag in ("_SL1500_", "_SL1600_", "_SL1200_")):
            score += 200
    elif "pv-target-images" in url:
        score += 100
        if "_BR-" in url and "_AC_SX" in url:
            score += 900
        if "_RI_TTW" in url:
            score -= 700
        elif "_RI_" in url:
            score -= 350
        if "SX1080" in url or size_key == "1080w":
            score += 60
        elif "SX720" in url or size_key == "720w":
            score += 80
    if size_key == "thumb":
        score -= 120
    return score


def pick_cover_image(row: dict[str, Any]) -> str:
    candidates: list[tuple[int, str]] = []

    def add(url: Any, size_key: str = "") -> None:
        if isinstance(url, str) and url.strip().startswith(("http://", "https://")):
            candidates.append((image_url_score(url.strip(), size_key), url.strip()))

    for key in ("imUrl", "image", "imageURL", "imageURLHighRes"):
        val = row.get(key)
        if isinstance(val, str):
            add(val, key)
    for item in row.get("images") or []:
        if isinstance(item, str):
            add(item)
        elif isinstance(item, dict):
            for image_key in (
                "hi_res",
                "large",
                "1080w",
                "720w",
                "480w",
                "360w",
                "1440w",
                "1920w",
                "thumb",
            ):
                add(item.get(image_key), image_key)
    if not candidates:
        return ""
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    return candidates[0][1]


PRIME_TITLE_RE = re.compile(
    r"<title>\s*(?:Watch\s+)?(.+?)\s*\|\s*Prime Video\s*</title>",
    re.IGNORECASE,
)
PRIME_POSTER_RE = re.compile(
    r"https://m\.media-amazon\.com/images/S/pv-target-images/[^\"'\s>]+_BR-[^\"'\s>]+_AC_SX1080_FMjpg_\.jpg"
)


def fetch_prime_video_enrichment(asin: str, timeout: float = 12.0) -> dict[str, str]:
    url = f"https://www.amazon.com/gp/video/detail/{asin}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; RecBoleDemo/1.0)", "Accept-Language": "en-US,en;q=0.9"},
    )
    try:
        html = urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8", "ignore")
    except (urllib.error.URLError, TimeoutError, OSError):
        return {}
    title_match = PRIME_TITLE_RE.search(html)
    title = title_match.group(1).strip() if title_match else ""
    posters = PRIME_POSTER_RE.findall(html)
    im_url = posters[0] if posters else ""
    if not im_url:
        fallback = re.findall(
            r"https://m\.media-amazon\.com/images/S/pv-target-images/[^\"'\s>]+_BR-[^\"'\s>]+\.jpg",
            html,
        )
        im_url = fallback[0] if fallback else ""
    out: dict[str, str] = {}
    if title:
        out["title"] = title
    if im_url:
        out["imUrl"] = im_url
    return out


def load_enrichment_cache(path: str) -> dict[str, dict[str, str]]:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): {str(kk): str(vv) for kk, vv in v.items()} for k, v in data.items()}


def save_enrichment_cache(path: str, cache: dict[str, dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def popular_asins_from_interactions(interactions_path: str, item_map: dict[str, str], limit: int) -> list[str]:
    token_to_asin = {str(token): str(asin) for asin, token in item_map.items()}
    counts: Counter[str] = Counter()
    for chunk in pd.read_csv(interactions_path, usecols=["item_id"], chunksize=500_000):
        for item_id in chunk["item_id"].astype(str):
            counts[item_id] += 1
    ranked: list[str] = []
    for token, _ in counts.most_common():
        asin = token_to_asin.get(str(token))
        if asin:
            ranked.append(asin)
        if len(ranked) >= limit:
            break
    return ranked


def _needs_prime_title(title: Any) -> bool:
    text = str(title or "").strip()
    return not text or text.startswith("Prime Video ·")


def enrich_prime_video_rows(
    rows: list[dict[str, Any]],
    *,
    cache_path: str,
    interactions_path: str | None,
    item_map: dict[str, str],
    limit: int,
    sleep_s: float,
) -> int:
    cache = load_enrichment_cache(cache_path)
    asin_to_row = {str(row["asin"]): row for row in rows}
    targets: list[str] = []
    for row in rows:
        asin = str(row["asin"])
        needs_title = _needs_prime_title(row.get("title"))
        needs_poster = "_RI_" in str(row.get("imUrl") or "") or not str(row.get("imUrl") or "").strip()
        if row.get("main_category") == "Prime Video" and (needs_title or needs_poster):
            targets.append(asin)
    if interactions_path and os.path.exists(interactions_path):
        popular = popular_asins_from_interactions(interactions_path, item_map, limit)
        ordered = []
        seen: set[str] = set()
        for asin in popular + targets:
            if asin in asin_to_row and asin not in seen:
                ordered.append(asin)
                seen.add(asin)
            if len(ordered) >= limit:
                break
        targets = ordered
    else:
        targets = targets[:limit]

    updated = 0
    for idx, asin in enumerate(targets, 1):
        row = asin_to_row.get(asin)
        if not row:
            continue
        cached = cache.get(asin, {})
        needs_title = _needs_prime_title(row.get("title"))
        needs_poster = "_RI_" in str(row.get("imUrl") or "") or not str(row.get("imUrl") or "").strip()
        if cached.get("title") and needs_title:
            row["title"] = cached["title"]
            updated += 1
        if cached.get("imUrl") and needs_poster:
            row["imUrl"] = cached["imUrl"]
            updated += 1
        if (needs_title and row.get("title")) and (not needs_poster or row.get("imUrl")):
            continue
        if (not needs_title) and (not needs_poster):
            continue

        fetched = fetch_prime_video_enrichment(asin)
        if fetched:
            cache[asin] = {**cache.get(asin, {}), **fetched}
            if fetched.get("title") and needs_title:
                row["title"] = fetched["title"]
                updated += 1
            if fetched.get("imUrl") and needs_poster:
                row["imUrl"] = fetched["imUrl"]
                updated += 1
            if idx % 20 == 0:
                save_enrichment_cache(cache_path, cache)
                print(f"prime enrich progress: {idx}/{len(targets)}")
            if sleep_s > 0:
                time.sleep(sleep_s)

    save_enrichment_cache(cache_path, cache)
    return updated


def build_item_metadata(
    metadata_path: str,
    item_map: dict[str, str],
    out_dir: str,
    *,
    enrich_prime: bool = False,
    enrich_limit: int = 8000,
    enrich_sleep: float = 0.35,
    interactions_path: str | None = None,
) -> str:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in iter_json_lines(metadata_path):
        # Amazon Reviews'23 metadata uses parent_asin as the product key.
        # Older Amazon metadata used asin directly for the same role.
        asin = row.get("parent_asin") or row.get("asin")
        if not asin:
            continue
        item_id = mapped(item_map, asin)
        if item_id in seen:
            continue
        seen.add(item_id)
        categories = row.get("categories")
        if isinstance(categories, list):
            categories_text = " > ".join(str(x) for group in categories for x in (group if isinstance(group, list) else [group]))
        else:
            categories_text = str(categories or "")
        rows.append(
            {
                "item_id": item_id,
                "asin": str(asin),
                "title": extract_title(row),
                "imUrl": pick_cover_image(row),
                "brand": str(row.get("brand") or row.get("store") or ""),
                "price": row.get("price", ""),
                "categories": categories_text,
                "main_category": str(row.get("main_category") or ""),
                "details_starring": _join_starring(row.get("details")),
                "average_rating": row.get("average_rating", ""),
                "rating_number": row.get("rating_number", ""),
            }
        )

    cache_path = os.path.join(out_dir, "prime_video_enrichment.json")
    cached = load_enrichment_cache(cache_path)
    if cached:
        merged = 0
        for row in rows:
            extra = cached.get(str(row["asin"]), {})
            if extra.get("title") and _needs_prime_title(row.get("title")):
                row["title"] = extra["title"]
                merged += 1
            if extra.get("imUrl") and ("_RI_" in str(row.get("imUrl") or "") or not str(row.get("imUrl") or "").strip()):
                row["imUrl"] = extra["imUrl"]
                merged += 1
        if merged:
            print(f"merged cached prime enrichment: {merged:,} fields")

    if enrich_prime:
        print(f"enriching Prime Video display fields (limit={enrich_limit:,})…")
        updated = enrich_prime_video_rows(
            rows,
            cache_path=cache_path,
            interactions_path=interactions_path,
            item_map=item_map,
            limit=enrich_limit,
            sleep_s=enrich_sleep,
        )
        print(f"prime enrichment updated rows/fields: {updated:,}")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "item_metadata.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8")
    return out_path


def _join_starring(details: Any) -> str:
    if not isinstance(details, dict):
        return ""
    starring = details.get("Starring")
    if not isinstance(starring, list) or not starring:
        return ""
    names = ", ".join(str(name) for name in starring[:3] if str(name).strip())
    return names


def build_user_profiles(reviews_path: str, user_map: dict[str, str], item_map: dict[str, str], out_dir: str) -> str:
    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"rating_sum": 0.0, "rating_count": 0, "reviewer_names": defaultdict(int)}
    )
    for row in iter_json_lines(reviews_path):
        reviewer = row.get("reviewerID") or row.get("user_id")
        if not reviewer:
            continue
        user_id = mapped(user_map, reviewer)
        rating = row.get("overall", row.get("rating", 0))
        try:
            rating_float = float(rating)
        except (TypeError, ValueError):
            rating_float = 0.0
        stats[user_id]["rating_sum"] += rating_float
        stats[user_id]["rating_count"] += 1
        name = row.get("reviewerName")
        if name:
            stats[user_id]["reviewer_names"][str(name)] += 1

    rows: list[dict[str, Any]] = []
    for user_id, s in stats.items():
        names = s["reviewer_names"]
        display_name = max(names.items(), key=lambda x: x[1])[0] if names else f"User {user_id}"
        count = int(s["rating_count"])
        rows.append(
            {
                "user_id": user_id,
                "display_name": display_name,
                "avg_rating": (float(s["rating_sum"]) / count) if count else 0.0,
                "rating_count": count,
            }
        )
    rows.sort(key=lambda r: int(r["rating_count"]), reverse=True)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "user_profiles.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8")
    return out_path


def build_user_profiles_from_interactions(interactions_path: str, out_dir: str) -> str:
    """从本地已编码交互 CSV 生成 user profile，保证 user_id 与 RecBole token 对齐。"""
    if not os.path.exists(interactions_path):
        raise FileNotFoundError(interactions_path)
    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"rating_sum": 0.0, "rating_count": 0, "raw_user": ""}
    )
    for chunk in pd.read_csv(
        interactions_path,
        usecols=lambda c: c in {"reviewerID", "user_id", "overall", "rating"},
        chunksize=500_000,
    ):
        rating_col = "overall" if "overall" in chunk.columns else "rating"
        for row in chunk.itertuples(index=False):
            data = row._asdict()
            user_id = str(data.get("user_id"))
            if not user_id or user_id == "None":
                continue
            try:
                rating = float(data.get(rating_col, 0) or 0)
            except (TypeError, ValueError):
                rating = 0.0
            stats[user_id]["rating_sum"] += rating
            stats[user_id]["rating_count"] += 1
            raw_user = data.get("reviewerID")
            if raw_user and not stats[user_id]["raw_user"]:
                stats[user_id]["raw_user"] = str(raw_user)

    rows: list[dict[str, Any]] = []
    for user_id, s in stats.items():
        count = int(s["rating_count"])
        raw_user = s.get("raw_user") or ""
        rows.append(
            {
                "user_id": user_id,
                "display_name": f"Reviewer {raw_user[-8:]}" if raw_user else f"User {user_id}",
                "raw_user_id": raw_user,
                "avg_rating": (float(s["rating_sum"]) / count) if count else 0.0,
                "rating_count": count,
            }
        )
    rows.sort(key=lambda r: int(r["rating_count"]), reverse=True)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "user_profiles.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build display metadata for Movies & TV app.")
    parser.add_argument("--reviews", required=True, help="Movies & TV reviews JSON/JSON.GZ")
    parser.add_argument("--metadata", required=True, help="Movies & TV metadata JSON/JSON.GZ")
    parser.add_argument(
        "--interactions",
        default=DEFAULT_INTERACTIONS if os.path.exists(DEFAULT_INTERACTIONS) else None,
        help="Local encoded interactions CSV. If present, user_profiles.csv is built from this file so user_id matches RecBole.",
    )
    parser.add_argument("--user-map", default=None, help="Optional raw reviewerID -> RecBole user_id mapping")
    parser.add_argument("--item-map", default=None, help="Optional raw asin -> RecBole item_id mapping")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--enrich-prime",
        action="store_true",
        help="Fetch missing Prime Video titles/posters from amazon.com/gp/video/detail (slow; cached).",
    )
    parser.add_argument("--enrich-limit", type=int, default=8000, help="Max Prime Video ASINs to enrich online.")
    parser.add_argument("--enrich-sleep", type=float, default=0.35, help="Seconds between Prime page requests.")
    args = parser.parse_args()

    user_map = load_mapping(args.user_map)
    item_map = load_mapping(args.item_map)
    print(f"user map: {len(user_map):,} entries")
    print(f"item map: {len(item_map):,} entries")

    item_out = build_item_metadata(
        args.metadata,
        item_map,
        args.out_dir,
        enrich_prime=args.enrich_prime,
        enrich_limit=args.enrich_limit,
        enrich_sleep=args.enrich_sleep,
        interactions_path=args.interactions,
    )
    if args.interactions:
        print(f"user profiles source: {args.interactions}")
        user_out = build_user_profiles_from_interactions(args.interactions, args.out_dir)
    else:
        user_out = build_user_profiles(args.reviews, user_map, item_map, args.out_dir)
    print(f"item metadata: {item_out}")
    print(f"user profiles: {user_out}")


if __name__ == "__main__":
    main()
