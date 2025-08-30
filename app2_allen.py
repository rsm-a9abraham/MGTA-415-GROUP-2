# -*- coding: utf-8 -*-
# app1.py — Amazon Cell Phone Product Search + Gemini-first Chat (robust)
# Notes:
# - Shows "✅ App loaded" immediately so you know the UI is rendering.
# - Safe fallbacks: older Streamlit (no st.chat_input), missing crew_judge_module, Gemini off, etc.
# - Errors during data/model load are shown in the page (no silent blank).
# Run:   streamlit run app1.py

import os, re, ast, json, sys, platform, urllib.request
from typing import Optional, Tuple, List, Dict
from html import escape

import numpy as np
import pandas as pd

import streamlit as st
from streamlit.components.v1 import html as st_html

# ─────────────────────────────────────────────────────────
# Page setup FIRST so UI renders even if later steps fail
# ─────────────────────────────────────────────────────────
st.set_page_config(page_title="📱 Amazon Cell Phone Product Search", layout="wide")
st.title("🔍 Amazon Cell Phone Product Search")
st.caption("Dense retrieval + Top 5 Comparison Panel + 💬 Gemini chat (grounded on your dataset)")
st.success("✅ App loaded (UI is running)")  # early indicator so you never see a blank page

# ─────────────────────────────────────────────────────────
# Debug info (minimal, collapsible)
# ─────────────────────────────────────────────────────────
with st.expander("Diagnostics"):
    st.write({
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "streamlit": st.__version__,
    })

# ─────────────────────────────────────────────────────────
# 🔑 HARD-CODE YOUR GEMINI KEY HERE
# ─────────────────────────────────────────────────────────
GOOGLE_API_KEY = "AIzaSyCcMoX7rNRmg6VWvN7o2gOi8-BkXvBRvwI"   # ← replace this
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Gemini: import non-fatal
try:
    import google.generativeai as genai
    genai.configure(api_key=GOOGLE_API_KEY)
    GEMINI_OK = True
except Exception as _ge:
    GEMINI_OK = False
    st.warning(f"Gemini not active: {_ge}")

# Try judge module, otherwise fallback
try:
    from crew_judge_module import judge_products, render_streamlit_card  # optional
except Exception:
    def judge_products(query, left, right, use_crewai=True, model="gemini-1.5-flash"):
        def val(p):
            try:
                pr = float(p.get("price_float") or np.inf)
                ra = float(p.get("average_rating") or 0.0)
                return ra / pr if pr > 0 else 0.0
            except Exception:
                return 0.0
        a, b = val(left), val(right)
        return {
            "title": "Local Verdict",
            "winner": "A" if a >= b else "B",
            "reasoning": f"Value scores — A:{a:.3f} vs B:{b:.3f} (higher is better)"
        }

    def render_streamlit_card(result: dict):
        st.success(f"**{result.get('title','Verdict')}** → Winner: **{result.get('winner','A/B')}**")
        if result.get("reasoning"):
            st.write(result["reasoning"])

# ─────────────────────────────────────────────────────────
# Styles
# ─────────────────────────────────────────────────────────
st.markdown("""
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

  .cmp-table { width:100%; border-collapse:separate; border-spacing:0 10px; }
  .cmp-row { background: transparent; border:1px solid rgba(148,163,184,0.18); }
  .cmp-row td { padding:10px 12px; }
  .cmp-row:hover { background: rgba(255,255,255,0.03); }
  .cmp-key { width:160px; color:#f1f5f9; font-weight:600; }
  .cmp-val { border-left:1px dashed rgba(148,163,184,0.25); color:#f8fafc; }

  .win-pill { display:inline-flex; align-items:center; gap:6px; padding:2px 8px; border-radius:999px;
              background:rgba(16,185,129,0.18); border:1px solid rgba(16,185,129,0.55); font-weight:600; }
  .win-pill::after { content:"✓"; font-size:12px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# Imports that can be heavy (wrapped so errors show on page)
# ─────────────────────────────────────────────────────────
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim
    HEAVY_OK = True
except Exception as e:
    HEAVY_OK = False
    st.error(f"Failed to import FAISS / SentenceTransformer: {e}")

# ─────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────
def as_list(x):
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

def _price_value(p: dict) -> Optional[float]:
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
    chips_html = "".join([f"<span class='chip'>{escape(str(c))}</span>" for c in [*cats,*feats]])
    return f"""
      <div class='prd-card'>
        <div class='prd-title'>{escape(str(p.get('title','(Untitled)')))}</div>
        <div class='badges'>{"".join(badges)}</div>
        <div class='chips'>{chips_html}</div>
        <div class='desc'>{escape(desc)}</div>
      </div>
    """

def render_pretty_compare(left: dict, right: dict):
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
    st_html(html, height=420, scrolling=True)

# ─────────────────────────────────────────────────────────
# Heavy resources (model & data) with clear error output
# ─────────────────────────────────────────────────────────
if HEAVY_OK:
    @st.cache_resource(show_spinner=True)
    def load_model():
        return SentenceTransformer("all-MiniLM-L6-v2")
else:
    def load_model():
        raise RuntimeError("SentenceTransformer import failed")

@st.cache_data(show_spinner=True)
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

# Attempt loads (with messages instead of blank page)
try:
    model = load_model()
    st.info("Encoder: all-MiniLM-L6-v2 ready")
except Exception as e:
    model = None
    st.error(f"❌ Failed to load encoder: {e}")

try:
    df_all, index = load_data()
    st.info(f"Dataset loaded: {len(df_all):,} rows")
except Exception as e:
    df_all, index = None, None
    st.error(f"❌ Failed to load dataset/index: {e}\n"
             f"Tips: ensure internet; `pip install pyarrow` for parquet; allow first-time downloads.")

# ─────────────────────────────────────────────────────────
# Retrieval helpers
# ─────────────────────────────────────────────────────────
def search(query, model, df, index, top_k=10):
    qv = model.encode([query]).astype("float32")
    distances, indices = index.search(qv, k=top_k)
    results = df.iloc[indices[0]].copy()
    results["distance"] = distances[0]
    return results

def search_plus(query, model, df, index, top_k=30):
    results = search(query, model, df, index, top_k=top_k)
    lower_m = re.search(r"(above|over|more than|greater than|higher than)\s*\$?(\d+)", query.lower())
    upper_m = re.search(r"(under|below|less than|cheaper than)\s*\$?(\d+)", query.lower())
    lo = float(lower_m.group(2)) if lower_m else None
    hi = float(upper_m.group(2)) if upper_m else None
    if not results.empty and (lo is not None or hi is not None):
        if "price_float" in results.columns:
            if lo is not None: results = results[results["price_float"] > lo]
            if hi is not None: results = results[results["price_float"] < hi]
    stop_words = {"i","need","want","a","an","the","for","with","to","is","it","my","buy","on","of","and","in"}
    kws = [kw for kw in query.lower().split() if kw not in stop_words and len(kw) > 2]
    if not results.empty and kws:
        pat = "|".join([re.escape(kw) for kw in kws])
        results = results[results["title"].astype(str).str.lower().str.contains(pat, na=False)]
    return results

from sentence_transformers.util import cos_sim as _cos_sim  # already imported but keep explicit
def rerank_by_similarity(query, results, model, top_n=5):
    if results.empty: return results
    query_vec = model.encode([query], convert_to_tensor=True)
    titles = results["title"].astype(str).tolist()
    title_vecs = model.encode(titles, convert_to_tensor=True)
    similarities = _cos_sim(query_vec, title_vecs)[0].cpu().numpy()
    results = results.copy()
    results["similarity_score"] = similarities
    return results.sort_values("similarity_score", ascending=False).head(top_n)

# ─────────────────────────────────────────────────────────
# Gemini helpers (safe)
# ─────────────────────────────────────────────────────────
def gemini_opening(user_msg: str, style="friendly & concise", emphasis="value over brand"):
    if not GEMINI_OK:
        return "Got it — I’ll help with that."
    try:
        m = genai.GenerativeModel(
            "gemini-1.5-flash",
            system_instruction=("You are a friendly, concise shopping assistant. "
                                "Reply in 1–3 sentences. Acknowledge the user and say what you did/will do. "
                                f"Tone: {style}. Emphasis: {emphasis}. No JSON.")
        )
        out = m.generate_content(user_msg)
        return (out.text or "").strip() or "Okay!"
    except Exception as e:
        return "Okay!"

def grounded_reply(user_msg: str, df_ctx: pd.DataFrame) -> str:
    if not GEMINI_OK or df_ctx.empty:
        return ""
    try:
        bullets=[]
        for _, r in df_ctx.head(6).iterrows():
            price = r.get("price_float")
            price_s = f"${price:.2f}" if isinstance(price,(int,float,np.floating)) and not pd.isna(price) else "N/A"
            rating = r.get("average_rating")
            rating_s = f"{float(rating):.2f}" if pd.notna(rating) else "N/A"
            reviews = r.get("rating_number")
            reviews_s = f"{int(reviews):,}" if pd.notna(reviews) else "N/A"
            bullets.append(f"- {str(r.get('title',''))[:90]} — {price_s}, ⭐ {rating_s} ({reviews_s})")
        sys = "Use ONLY these catalog bullets to answer concisely (≤6 sentences). If nothing fits, say so politely.\n" + "\n".join(bullets)
        m = genai.GenerativeModel("gemini-1.5-flash", system_instruction=sys)
        out = m.generate_content(user_msg)
        return (out.text or "").strip()
    except Exception:
        return ""

# ─────────────────────────────────────────────────────────
# Chat section (with fallback if old Streamlit)
# ─────────────────────────────────────────────────────────
LETTER_TO_INDEX = {c: i for i, c in enumerate(list("ABCDE"))}

def parse_compare_nums(text: str, max_len: int) -> Tuple[Optional[int], Optional[int]]:
    nums = re.findall(r"\b([1-9]|10)\b", text)
    if len(nums) >= 2:
        i, j = int(nums[0]) - 1, int(nums[1]) - 1
        if 0 <= i < max_len and 0 <= j < max_len and i != j:
            return i, j
    letters = re.findall(r"\b([A-Ea-e])\b", text)
    if len(letters) >= 2:
        i, j = LETTER_TO_INDEX[letters[0].upper()], LETTER_TO_INDEX[letters[1].upper()]
        if 0 <= i < max_len and 0 <= j < max_len and i != j:
            return i, j
    return None, None

def apply_text_action(user_msg: str, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    m = user_msg.lower()
    mt = re.search(r"between\s*\$?\s*(\d+(?:\.\d+)?)\s*(?:and|to|-)\s*\$?\s*(\d+(?:\.\d+)?)", m)
    if mt and "price_float" in df.columns:
        lo, hi = float(mt.group(1)), float(mt.group(2))
        if lo > hi: lo, hi = hi, lo
        return df[(df["price_float"] >= lo) & (df["price_float"] <= hi)].copy(), f"Filtered between ${lo:.0f} and ${hi:.0f}."
    if any(k in m for k in ["under","below","less than","cheaper than","<"]):
        mt = re.search(r"\$?\s*(\d+(?:\.\d+)?)", m)
        if mt and "price_float" in df.columns:
            cap = float(mt.group(1)); return df[df["price_float"] < cap].copy(), f"Filtered under ${cap:.0f}."
    if any(k in m for k in ["over","above","greater than","higher than",">"]):
        mt = re.search(r"\$?\s*(\d+(?:\.\d+)?)", m)
        if mt and "price_float" in df.columns:
            lo = float(mt.group(1)); return df[df["price_float"] > lo].copy(), f"Filtered over ${lo:.0f}."
    if "sort" in m:
        if "price" in m and "price_float" in df.columns:
            asc = "low" in m
            return df.sort_values("price_float", ascending=asc).copy(), f"Sorted by price ({'lowest→highest' if asc else 'highest→lowest'})."
        if "rating" in m and "average_rating" in df.columns:
            return df.sort_values("average_rating", ascending=False).copy(), "Sorted by rating (highest→lowest)."
        if "review" in m and "rating_number" in df.columns:
            return df.sort_values("rating_number", ascending=False).copy(), "Sorted by reviews (most→least)."
        if "value" in m and all(c in df.columns for c in ["average_rating","price_float"]):
            tmp = df.copy()
            tmp["value_score"] = tmp["average_rating"] / tmp["price_float"].replace(0,np.nan)
            return tmp.sort_values(["value_score","average_rating","rating_number"], ascending=[False,False,False]).copy(), "Sorted by value (rating/price)."
    if any(k in m for k in ["recommend","best","suggest","pick for me"]):
        cap = None
        mt = re.search(r"\$?\s*(\d+(?:\.\d+)?)", m)
        if mt: cap = float(mt.group(1))
        tmp = df.copy()
        if cap and "price_float" in tmp.columns:
            tmp = tmp[tmp["price_float"] < cap]
        if all(c in tmp.columns for c in ["average_rating","price_float"]):
            tmp["value_score"] = tmp["average_rating"] / tmp["price_float"].replace(0,np.nan)
            tmp = tmp.sort_values(["value_score","average_rating","rating_number"], ascending=[False,False,False]).head(3)
        lines=[]
        for _, r in tmp.iterrows():
            price = r.get("price_float")
            price_s = f"${price:.2f}" if isinstance(price,(int,float,np.floating)) and not pd.isna(price) else "N/A"
            rating = r.get("average_rating")
            rating_s = f"{float(rating):.2f}" if pd.notna(rating) else "N/A"
            reviews = r.get("rating_number")
            reviews_s = f"{int(reviews):,}" if pd.notna(reviews) else "N/A"
            lines.append(f"- **{str(r.get('title',''))[:90]}** — {price_s}, ⭐ {rating_s} ({reviews_s})")
        if lines:
            return df, "Top picks:\n" + "\n".join(lines)
    return df, None

def retrieval_for_chat(q: str, model, df_all, index, limit=12) -> pd.DataFrame:
    res = search_plus(q, model, df_all, index, top_k=40)
    if "price_float" not in res.columns:
        def _coerce(x):
            if isinstance(x,(int,float,np.floating)) and not pd.isna(x): return float(x)
            if isinstance(x,str):
                m = re.search(r"[-+]?\d*\.?\d+", x)
                if m:
                    try: return float(m.group())
                    except: return np.nan
            return np.nan
        res["price_float"] = res.get("display_price", np.nan).map(_coerce)
    return res.head(limit).copy()

def chat_input_safe(prompt: str):
    # Fallback for old Streamlit versions
    if hasattr(st, "chat_input"):
        return st.chat_input(prompt)
    else:
        return st.text_input(prompt, key="chat_input_fallback")

def render_chat_section(seed_query: str, model, df_all, index, top5_results):
    st.markdown("### 💬 Chat (Gemini)")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role":"assistant","content":"Hi! Ask me anything — budget picks, sorting, or try 'compare 2 & 3'."}
        ]
    for m in st.session_state.chat_history:
        with st.chat_message(m["role"]) if hasattr(st, "chat_message") else st.container():
            st.markdown(m["content"])

    user_msg = chat_input_safe("Ask naturally… e.g., 'best under $20', 'compare 2 & 3', 'sort by value'")
    if not user_msg:
        return
    st.session_state.chat_history.append({"role":"user","content":user_msg})
    with st.chat_message("user") if hasattr(st, "chat_message") else st.container():
        st.markdown(user_msg)

    opening = gemini_opening(user_msg)

    # Compare refs to Top 5 (e.g., "2 & 3" or "B vs D")
    if top5_results:
        def _parse_nums(text: str, max_len: int):
            nums = re.findall(r"\b([1-5])\b", text)
            if len(nums) >= 2:
                i, j = int(nums[0]) - 1, int(nums[1]) - 1
                if 0 <= i < max_len and 0 <= j < max_len and i != j: return i, j
            letters = re.findall(r"\b([A-Ea-e])\b", text)
            if len(letters) >= 2:
                mapL = {c:i for i,c in enumerate("ABCDE")}
                i, j = mapL[letters[0].upper()], mapL[letters[1].upper()]
                if 0 <= i < max_len and 0 <= j < max_len and i != j: return i, j
            return None, None

        i, j = _parse_nums(user_msg, len(top5_results))
        if i is not None and j is not None:
            left, right = top5_results[i], top5_results[j]
            reply = opening + f"\n\nComparing **#{i+1}** and **#{j+1}** (see panel)."
            st.session_state.chat_history.append({"role":"assistant","content":reply})
            with st.chat_message("assistant") if hasattr(st, "chat_message") else st.container():
                st.markdown(reply)
            st.markdown("---")
            st.subheader("🔍 Comparison (Chat)")
            st.markdown(f"**A. {left.get('title','(Untitled)')}**")
            st.markdown(f"**B. {right.get('title','(Untitled)')}**")
            render_pretty_compare(left, right)
            try:
                result = judge_products(seed_query or user_msg, left, right, use_crewai=True, model="gemini-1.5-flash")
                render_streamlit_card(result)
            except Exception as e:
                st.warning(f"AI agent issue; using local scoring: {e}")
                result = judge_products(seed_query or user_msg, left, right, use_crewai=False)
                render_streamlit_card(result)
            return

    # Retrieval + optional filter/sort/recommend
    if df_all is None or index is None or model is None:
        final = opening + "\n\n(I couldn't access the dataset/models right now — check errors above.)"
        st.session_state.chat_history.append({"role":"assistant","content":final})
        with st.chat_message("assistant") if hasattr(st, "chat_message") else st.container():
            st.markdown(final)
        return

    retrieved = retrieval_for_chat(user_msg, model, df_all, index, limit=12)
    acted_df, summary = apply_text_action(user_msg, retrieved)
    grounded = grounded_reply(user_msg, acted_df if not acted_df.empty else retrieved)

    final = opening
    if summary: final += "\n\n" + summary
    if grounded: final += ("\n\n" if summary else "\n\n") + grounded

    st.session_state.chat_history.append({"role":"assistant","content":final})
    with st.chat_message("assistant") if hasattr(st, "chat_message") else st.container():
        st.markdown(final)

# ─────────────────────────────────────────────────────────
# Main search bar & Top 5 + Compare
# ─────────────────────────────────────────────────────────
query = st.text_input("Enter your product search query:", placeholder="e.g. slim iPhone X case / note 5 leather case")

if model is not None and df_all is not None and index is not None:
    if query:
        with st.spinner("Searching..."):
            full_results = search_plus(query, model, df_all, index, top_k=30)
            if "price_float" not in full_results.columns:
                def _coerce(x):
                    if isinstance(x,(int,float,np.floating)) and not pd.isna(x): return float(x)
                    if isinstance(x,str):
                        m = re.search(r"[-+]?\d*\.?\d+", x)
                        if m:
                            try: return float(m.group())
                            except: return np.nan
                    return np.nan
                full_results["price_float"] = full_results.get("display_price", np.nan).map(_coerce)

            results = rerank_by_similarity(query, full_results, model, top_n=5)

        if results.empty:
            st.warning("❌ No results found. Try broader keywords.")
            render_chat_section(query, model, df_all, index, top5_results=[])
        else:
            tmp = results.copy()
            tmp["price"] = tmp.get("display_price", "N/A")
            tmp["average_rating"] = tmp.get("average_rating", np.nan)
            tmp["rating_number"] = tmp.get("rating_number", np.nan)
            tmp["features"] = tmp.get("features", [])
            tmp["categories"] = tmp.get("categories", [])
            tmp["description"] = tmp.get("description", "")
            if "store" not in tmp.columns and "brand_clean" in tmp.columns:
                tmp["store"] = tmp["brand_clean"]

            top5_results = tmp.head(5).to_dict(orient="records")

            # Render Top 5 cards + compare panel
            st.markdown("## Top 5")
            for i, p in enumerate(top5_results):
                col_txt, _ = st.columns([7, 1])
                with col_txt:
                    st.markdown(product_card_html(p), unsafe_allow_html=True)
                    st.caption(f"#{i+1}")

            # Compare checkbox panel (same as before)
            st.divider()
            st.subheader("Select any 2 above (A/B) using the old compare panel?")
            # Re-render using checkboxes version for familiarity
            # (kept minimal: just device the old function inline)
            # We'll reconstruct pool quickly:
            if "compare_pool" not in st.session_state:
                st.session_state.compare_pool = []
            def add_or_remove(idx):
                if idx in st.session_state.compare_pool:
                    st.session_state.compare_pool.remove(idx)
                else:
                    st.session_state.compare_pool.append(idx)
                    if len(st.session_state.compare_pool) > 2:
                        st.session_state.compare_pool = st.session_state.compare_pool[-2:]

            cols = st.columns(5)
            for i in range(len(top5_results)):
                with cols[i]:
                    if st.checkbox(f"Pick #{i+1}", key=f"pick_{i}"):
                        add_or_remove(i)

            if len(st.session_state.compare_pool) == 2:
                a_i, b_i = st.session_state.compare_pool
                left, right = top5_results[a_i], top5_results[b_i]
                st.markdown("---")
                st.subheader("🔍 Comparison Panel (2 items selected)")
                st.markdown(f"**A. {left.get('title','(Untitled)')}**")
                st.markdown(f"**B. {right.get('title','(Untitled)')}**")
                render_pretty_compare(left, right)
                try:
                    result = judge_products(query or "", left, right, use_crewai=True, model="gemini-1.5-flash")
                    render_streamlit_card(result)
                except Exception as e:
                    st.warning(f"AI agent issue; using local scoring: {e}")
                    result = judge_products(query or "", left, right, use_crewai=False)
                    render_streamlit_card(result)

            # Chat (Gemini-first)
            st.divider()
            render_chat_section(query, model, df_all, index, top5_results=top5_results)

    else:
        st.info("Type a query to begin — or just chat below, I’ll search when needed.")
        render_chat_section("", model, df_all, index, top5_results=[])
else:
    st.stop()  # we already showed error messages above
