# -*- coding: utf-8 -*-
# app1.py  — 搜索/Top5/对比 + 对比后 AI 评审（深色主题可读性 + pill 高亮稳定版）

import os
import re
import ast
from html import escape
import urllib.request

import faiss
import numpy as np
import pandas as pd

import streamlit as st
from streamlit.components.v1 import html as st_html

st.set_page_config(page_title="📱 Amazon Cell Phone Product Search", layout="wide")

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# ✅ 引入 AI 评审模块（与本文件同目录）
from crew_judge_module import judge_products, render_streamlit_card

import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyAGm6MeqjCbxoaBQPBLYwE0xQGpG-gGJ0w"

# =====================
# Utilities
# =====================
def as_list(x):
    """Robustly coerce x to a list (handles ndarray/Series/str/'[a,b]' etc)."""
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, (np.ndarray, pd.Series)):
        return [v for v in x.tolist() if pd.notna(v)]
    if isinstance(x, str) and x.strip():
        try:
            v = ast.literal_eval(x)
            if isinstance(v, (list, tuple)):
                return [u for u in v if pd.notna(u)]
        except Exception:
            pass
        return [x]
    return []


def as_scalar(x):
    """Pick the first scalar value from list/ndarray/Series; otherwise return x."""
    if isinstance(x, (list, tuple)):
        return x[0] if x else None
    if isinstance(x, (np.ndarray, pd.Series)):
        arr = x.tolist()
        return arr[0] if arr else None
    return x


def _fmt_price_str(p: dict) -> str:
    pf = as_scalar(p.get("price_float"))
    if isinstance(pf, (int, float, np.floating)) and not pd.isna(pf):
        return f"$ {float(pf):.2f}"
    pr = p.get("price") or p.get("display_price")
    if isinstance(pr, (int, float, np.floating)) and not pd.isna(pr):
        return f"$ {float(pr):.2f}"
    return str(pr) if pr not in (None, "", "None") else "N/A"


def _price_value(p: dict) -> float | None:
    pf = as_scalar(p.get("price_float"))
    if isinstance(pf, (int, float, np.floating)) and not pd.isna(pf):
        return float(pf)
    pr = p.get("price") or p.get("display_price")
    if isinstance(pr, (int, float, np.floating)) and not pd.isna(pr):
        return float(pr)
    if isinstance(pr, str):
        m = re.search(r"[-+]?\d*\.?\d+", pr)
        if m:
            try:
                return float(m.group())
            except Exception:
                return None
    return None


def product_card_html(p: dict) -> str:
    price = _fmt_price_str(p)
    rating = as_scalar(p.get("average_rating"))
    cnt = as_scalar(p.get("rating_number"))
    brand = p.get("store") or p.get("brand_clean") or p.get("brand") or ""
    cats = as_list(p.get("categories", []))[:3]
    feats = as_list(p.get("features", []))[:4]
    desc = str(p.get("description", ""))[:180]

    badges = [
        f"<span class='badge price'>💲 {escape(str(price))}</span>",
        f"<span class='badge rating'>⭐ {escape(str(rating if rating not in (None,'') else 'N/A'))}{f' ({escape(str(cnt))})' if cnt not in (None,'') else ''}</span>",
        f"<span class='badge brand'>🏷️ {escape(str(brand))}</span>" if brand else "",
    ]
    chips_html = "".join([f"<span class='chip'>{escape(str(c))}</span>" for c in [*cats, *feats]])

    return f"""
      <div class='prd-card'>
        <div class='prd-title'>{escape(str(p.get('title','(Untitled)')))}</div>
        <div class='badges'>{"".join(badges)}</div>
        <div class='chips'>{chips_html}</div>
        <div class='desc'>{escape(desc)}</div>
      </div>
    """


def inject_card_css():
    if st.session_state.get("_card_css_injected"):
        return
    st.markdown(
        """
        <style>
        .prd-card {border-radius:16px; padding:14px 16px; margin:10px 0;
                   border:1px solid rgba(148,163,184,0.25);
                   background: rgba(148,163,184,0.06);}
        .prd-card:hover {border-color:#7C9CFF; box-shadow:0 4px 14px rgba(124,156,255,0.15);}
        .prd-title {font-size:15px; line-height:1.35; margin:0 0 6px 0; font-weight:600;}
        .badges {display:flex; gap:8px; flex-wrap:wrap; margin:6px 0 2px 0;}
        .badge {padding:2px 8px; border-radius:999px; border:1px solid rgba(148,163,184,0.35); font-size:12px;}
        .badge.price  {background:#10b98122; color:#10b981; border-color:#10b98155;}
        .badge.rating {background:#f59e0b22; color:#f59e0b; border-color:#f59e0b55;}
        .badge.brand  {background:#60a5fa22; color:#60a5fa; border-color:#60a5fa55;}
        .chips {display:flex; gap:6px; flex-wrap:wrap; margin-top:6px;}
        .chip {font-size:12px; padding:2px 8px; border-radius:999px;
               border:1px solid rgba(148,163,184,0.35); background:rgba(148,163,184,0.12);}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.session_state._card_css_injected = True


def render_pretty_compare(left: dict, right: dict):
    """Comparison Table（iframe rendering + pill highlighting + dark theme clear fonts）"""
    def _fmt_rate(p):
        r = p.get("average_rating")
        c = p.get("rating_number")
        return f"{r} ({c})" if r not in (None, "") else "N/A"

    lp, rp = _price_value(left), _price_value(right)
    better_price = "A" if (lp is not None and rp is not None and lp < rp) else ("B" if (lp is not None and rp is not None and rp < lp) else None)
    lr, rr = left.get("average_rating"), right.get("average_rating")
    better_rate = "A" if (isinstance(lr, (int, float, np.floating)) and isinstance(rr, (int, float, np.floating)) and lr > rr) else ("B" if (isinstance(lr, (int, float, np.floating)) and isinstance(rr, (int, float, np.floating)) and rr > lr) else None)
    la, lb = left.get("rating_number"), right.get("rating_number")
    better_vol = "A" if (isinstance(la, (int, np.integer)) and isinstance(lb, (int, np.integer)) and la > lb) else ("B" if (isinstance(la, (int, np.integer)) and isinstance(lb, (int, np.integer)) and lb > la) else None)

    def row(label, aval, bval, winner=None):
        acontent = f"<span class='win-pill'>{escape(str(aval))}</span>" if winner == "A" else escape(str(aval))
        bcontent = f"<span class='win-pill'>{escape(str(bval))}</span>" if winner == "B" else escape(str(bval))
        return f"""
        <tr class="cmp-row">
          <td class="cmp-key">{escape(str(label))}</td>
          <td class="cmp-val">{acontent}</td>
          <td class="cmp-val">{bcontent}</td>
        </tr>"""

    a_brand = left.get("store") or left.get("brand_clean") or ""
    b_brand = right.get("store") or right.get("brand_clean") or ""
    a_cats = ", ".join(map(str, as_list(left.get("categories", []))[:3])) or "—"
    b_cats = ", ".join(map(str, as_list(right.get("categories", []))[:3])) or "—"
    a_feat = ", ".join(map(str, as_list(left.get("features", []))[:8])) or "—"
    b_feat = ", ".join(map(str, as_list(right.get("features", []))[:8])) or "—"
    a_desc = str(left.get("description", ""))[:400] or "—"
    b_desc = str(right.get("description", ""))[:400] or "—"

    html = f"""
    <table class="cmp-table">
      {row("price", _fmt_price_str(left), _fmt_price_str(right), better_price)}
      {row("average_rating", _fmt_rate(left), _fmt_rate(right), better_rate)}
      {row("rating_number", la or 'N/A', lb or 'N/A', better_vol)}
      {row("brand/store", a_brand or '—', b_brand or '—', None)}
      {row("categories", a_cats, b_cats, None)}
      {row("features", a_feat, b_feat, None)}
      {row("description", a_desc, b_desc, None)}
    </table>
    """

    # 用普通三引号字符串，避免 f-string 花括号解析导致 NameError
    css = """
<style>
  html, body { background: transparent; }

  .cmp-table { width:100%; border-collapse:separate; border-spacing:0 10px; }
  .cmp-row { background: transparent; border:1px solid rgba(148,163,184,0.18); }
  .cmp-row td { padding:10px 12px; }
  .cmp-row:hover { background: rgba(255,255,255,0.03); }

  .cmp-key { width:160px; color:#f1f5f9; font-weight:600; }
  .cmp-val { border-left:1px dashed rgba(148,163,184,0.25); color:#f8fafc; }

  .win-pill {
    display:inline-flex; align-items:center; gap:6px;
    padding:2px 8px; border-radius:999px;
    background:rgba(16,185,129,0.18);
    border:1px solid rgba(16,185,129,0.55);
    font-weight:600;
  }
  .win-pill::after { content:"✓"; font-size:12px; }
</style>
"""
    st_html(css + html, height=420, scrolling=True)


# =====================
# Model / Data Loading
# =====================
@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data(show_spinner=False)
def load_data():
    parquet_url = (
        "https://huggingface.co/GovinKin/MGTA415database/resolve/main/"
        "cellphones_clean_with_price000.parquet?download=1"
    )
    df = pd.read_parquet(parquet_url)

    index_url = (
        "https://huggingface.co/GovinKin/MGTA415database/resolve/main/"
        "cellphones_with_price000.faiss?download=1"
    )
    local_index_path = "cellphones_with_price000.faiss"
    if not os.path.exists(local_index_path):
        urllib.request.urlretrieve(index_url, local_index_path)
    index = faiss.read_index(local_index_path)
    return df, index


# =====================
# Search Helpers
# =====================
def search(query, model, df, index, top_k=10):
    qv = model.encode([query]).astype("float32")
    distances, indices = index.search(qv, k=top_k)
    results = df.iloc[indices[0]].copy()
    results["distance"] = distances[0]
    return results


def search_plus(query, model, df, index, top_k=30):
    results = search(query, model, df, index, top_k=top_k)

    price_upper = None
    price_lower = None
    lower_m = re.search(r"(above|over|more than|greater than|higher than)\s*\$?(\d+)", query.lower())
    if lower_m:
        price_lower = float(lower_m.group(2))
    upper_m = re.search(r"(under|below|less than|cheaper than)\s*\$?(\d+)", query.lower())
    if upper_m:
        price_upper = float(upper_m.group(2))

    if not results.empty and (price_lower is not None or price_upper is not None):
        if "price_float" in results.columns:
            if price_lower is not None:
                results = results[results["price_float"] > price_lower]
            if price_upper is not None:
                results = results[results["price_float"] < price_upper]

    stop_words = {"i","need","want","a","an","the","for","with","to","is","it","my","buy","on","of","and","in"}
    keywords = [kw for kw in query.lower().split() if kw not in stop_words and len(kw) > 2]
    if not results.empty and keywords:
        pattern = "|".join([re.escape(kw) for kw in keywords])
        results = results[results["title"].astype(str).str.lower().str.contains(pattern, na=False)]

    return results


def rerank_by_similarity(query, results, model, top_n=5):
    if results.empty:
        return results
    query_vec = model.encode([query], convert_to_tensor=True)
    titles = results["title"].astype(str).tolist()
    title_vecs = model.encode(titles, convert_to_tensor=True)
    similarities = cos_sim(query_vec, title_vecs)[0].cpu().numpy()
    results["similarity_score"] = similarities
    results = results.sort_values("similarity_score", ascending=False)
    return results.head(top_n)


# =====================
# Compare UI
# =====================
def render_top5_with_compare(top5_results, query=None):
    inject_card_css()

    if "compare_pool" not in st.session_state:
        st.session_state.compare_pool = []  # [(idx, asin)]

    def toggle_selection(idx, asin):
        pool = st.session_state.compare_pool
        key = (idx, asin)
        if key in pool:
            pool.remove(key)
        else:
            pool.append(key)
            if len(pool) > 2:
                pool = pool[-2:]
        st.session_state.compare_pool = pool

    st.markdown("## Top 5")

    for i, p in enumerate(top5_results):
        asin = p.get("parent_asin") or p.get("asin") or str(i)
        col_txt, col_chk = st.columns([7, 1])

        with col_txt:
            st.markdown(product_card_html(p), unsafe_allow_html=True)

        with col_chk:
            checked = (i, asin) in st.session_state.compare_pool
            if st.checkbox("", value=checked, key=f"compare_{i}_{asin}"):
                if not checked:
                    toggle_selection(i, asin)
            else:
                if checked:
                    toggle_selection(i, asin)

    sel = st.session_state.compare_pool
    if len(sel) == 2:
        left_raw, right_raw = [top5_results[s[0]] for s in sel]
        left, right = left_raw, right_raw

        st.markdown("---")
        st.subheader("🔍 Comparison Panel (2 items selected)")
        st.markdown(f"**A. {left.get('title','(Untitled)')}**")
        st.markdown(f"**B. {right.get('title','(Untitled)')}**")

        # 美观对比表（pill 高亮 + 深色清晰字体）
        render_pretty_compare(left, right)

        # === 在对比表后调用 AI 评审 ===
        ai_col1, ai_col2 = st.columns([1, 3])
        with ai_col1:
            use_ai = st.toggle("Enable AI Agent", value=True,
                               help="If disabled, only local interpretable scoring is used; if enabled, GOOGLE_API_KEY in .env is required")
        with ai_col2:
            model_name = st.selectbox("LLM（Gemini）", ["gemini-1.5-flash", "gemini-1.5-pro"], index=0)

        try:
            result = judge_products(
                query or "",
                left,
                right,
                use_crewai=use_ai,
                model=model_name
            )
            render_streamlit_card(result)
        except Exception as e:
            st.warning(f"There was a problem with the AI ​​agent and it has reverted to local scoring.：{e}")
            result = judge_products(query or "", left, right, use_crewai=False)
            render_streamlit_card(result)


# =====================
# UI
# =====================
st.title("🔍 Amazon Cell Phone Product Search")
st.caption("Dense retrieval + Top 5 Comparison Panel (cleaned data, no image dependency)")

model = load_model()
df_all, index = load_data()

query = st.text_input("Enter your product search query:", placeholder="e.g. slim iPhone X case / note 5 leather case")

if query:
    with st.spinner("Searching..."):
        results = search_plus(query, model, df_all, index, top_k=30)
        results = rerank_by_similarity(query, results, model, top_n=5)

    if results.empty:
        st.warning("❌ No results found. Try broader keywords.")
    else:
        tmp = results.copy()
        # 统一字段，避免空值导致显示问题
        tmp["price"] = tmp.get("display_price", "N/A")
        tmp["average_rating"] = tmp.get("average_rating", np.nan)
        tmp["rating_number"] = tmp.get("rating_number", np.nan)
        tmp["features"] = tmp.get("features", [])
        tmp["categories"] = tmp.get("categories", [])
        tmp["description"] = tmp.get("description", "")
        if "store" not in tmp.columns and "brand_clean" in tmp.columns:
            tmp["store"] = tmp["brand_clean"]

        top5_results = tmp.head(5).to_dict(orient="records")
        render_top5_with_compare(top5_results, query)
else:
    st.info("Please enter your search terms to begin searching.（For example：oneplus 7t case / note 5 leather case）")
