import os
import re
import urllib.request

import faiss
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

def as_list(x):
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, (np.ndarray, pd.Series)):
        return [v for v in x.tolist() if pd.notna(v)]
    if isinstance(x, str) and x.strip():
        # 兼容 "['a','b']" 这种字符串列表
        try:
            v = ast.literal_eval(x)
            if isinstance(v, (list, tuple)):
                return [u for u in v if pd.notna(u)]
        except Exception:
            pass
        return [x]
    return []

def as_scalar(x):
    if isinstance(x, (list, tuple)):
        return x[0] if x else None
    if isinstance(x, (np.ndarray, pd.Series)):
        arr = x.tolist()
        return arr[0] if arr else None
    return x

# =====================
# Page Configuration
# =====================
st.set_page_config(page_title="📱 Amazon Cell Phone Product Search", layout="wide")

# =====================
# Compare UI (image-free)
# =====================
def render_top5_with_compare(top5_results, query=None):
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

    st.markdown("## Top 5 结果（可勾选对比）")

    for i, p in enumerate(top5_results):
        asin = p.get("parent_asin") or p.get("asin") or str(i)
        col_txt, col_chk = st.columns([3, 0.6])
        with col_txt:
            st.markdown(f"**{p.get('title','(Untitled)')}**")
        cats = as_list(p.get("categories", []))
        if len(cats) > 0:
            st.caption(" / ".join(map(str, cats[:3])))
            price_str = p.get("price", "N/A")
            avg_r = p.get("average_rating", "N/A")
            cnt_r = p.get("rating_number", "")
            brand = p.get("store") or p.get("brand") or p.get("brand_clean") or ""
            st.write("  ".join([m for m in [price_str, f"{avg_r} · {cnt_r} reviews", brand] if m]))
            feats = as_list(p.get("features", []))
            if len(feats) > 0:
                st.caption(" • ".join(map(str, feats[:4])))
        with col_chk:
            checked = (i, asin) in st.session_state.compare_pool
            if st.checkbox("对比", value=checked, key=f"compare_{i}_{asin}"):
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
        st.subheader("🔍 对比面板 (已选择 2 项)")
        st.markdown(f"**A. {left.get('title','(Untitled)')}**")
        st.markdown(f"**B. {right.get('title','(Untitled)')}**")

        rows = [
    ("价格", left.get("price","N/A"), right.get("price","N/A")),
    ("评分", f"{as_scalar(left.get('average_rating'))} ({as_scalar(left.get('rating_number'))})",
             f"{as_scalar(right.get('average_rating'))} ({as_scalar(right.get('rating_number'))})"),
    ("品牌/店铺", left.get("store") or left.get("brand_clean") or "",
                 right.get("store") or right.get("brand_clean") or ""),
    ("品类",
     " / ".join(map(str, as_list(left.get("categories", []))[:3])),
     " / ".join(map(str, as_list(right.get("categories", []))[:3]))),
    ("特性",
     ", ".join(map(str, as_list(left.get("features", []))[:8])),
     ", ".join(map(str, as_list(right.get("features", []))[:8]))),
    ("描述(摘要)", str(left.get("description",""))[:400],
                 str(right.get("description",""))[:400]),
                ]
        _df = pd.DataFrame(rows, columns=["字段", "A", "B"])
        st.dataframe(_df, use_container_width=True, hide_index=True)
        st.info("提示：当前仅提供对比视图；CrewAI 裁判与打分将在下一步接入。")

# =====================
# Model / Data Loading (use your HF repo with cleaned data)
# =====================
@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data(show_spinner=False)
def load_data():
    # Cleaned, text-only parquet (already filtered to have price)
    parquet_url = (
        "https://huggingface.co/GovinKin/MGTA415database/resolve/main/"
        "cellphones_clean_with_price000.parquet?download=1"
    )
    df = pd.read_parquet(parquet_url)

    # Matching FAISS index built on the cleaned parquet
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

    # Optional price filter from query text, applied on price_float
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

    # simple keyword filter on title
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
# UI
# =====================
st.title("\U0001F50D Amazon Cell Phone Product Search")
st.caption("Dense retrieval + Top5 对比面板（清洗后、无图片依赖）")

model = load_model()
df_all, index = load_data()  # 已是清洗后数据，无需再次清洗

query = st.text_input("Enter your product search query:", placeholder="e.g. slim iPhone X case")

if query:
    with st.spinner("Searching..."):
        results = search_plus(query, model, df_all, index, top_k=30)
        results = rerank_by_similarity(query, results, model, top_n=5)

    if results.empty:
        st.warning("❌ No results found. Try broader keywords.")
    else:
        tmp = results.copy()
        # Map cleaned columns to UI schema expected by compare panel
        # （清洗表已包含这些列）
        tmp["price"] = tmp.get("display_price", "N/A")
        tmp["average_rating"] = tmp.get("average_rating", np.nan)
        tmp["rating_number"] = tmp.get("rating_number", np.nan)
        tmp["features"] = tmp.get("features", [])
        tmp["categories"] = tmp.get("categories", [])
        tmp["description"] = tmp.get("description", "")
        # store/brand_clean 任意其一
        if "store" not in tmp.columns and "brand_clean" in tmp.columns:
            tmp["store"] = tmp["brand_clean"]

        top5_results = tmp.head(5).to_dict(orient="records")
        render_top5_with_compare(top5_results, query)
else:
    st.info("请输入查询词开始搜索（例如：oneplus 7t case / note 5 leather case）")
