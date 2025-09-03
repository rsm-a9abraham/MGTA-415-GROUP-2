# -*- coding: utf-8 -*-
# app2_allen.py — Gemini-first chat app (everything Gemini, grounded on your dataset)
# - Small talk: Gemini (explicitly instructed to avoid products unless asked)
# - Shopping: Gemini with dataset bullets (numbered, spaced), or local fallback if Gemini fails
# - HF shards (parquet + faiss)
# - Gemini key from allen_apikey.txt
# - Mini green/red badges, numbered bullets, comparison flow continues
# - Top-right Reload app (full clear), bottom-right Reset conversation (chat only)
# Run: PYTHONNOUSERSITE=1 streamlit run app2_allen.py --server.port 8501

import os, re, ast, sys, platform, urllib.request
from typing import Optional, Tuple, List
from html import escape

import numpy as np
import pandas as pd

import streamlit as st
from streamlit.components.v1 import html as st_html

st.set_page_config(page_title="📱 Bot", layout="wide")

# ---------------------- Styles ----------------------
st.markdown("""
<style>
.badge-row { display:flex; flex-wrap:wrap; gap:10px; margin:6px 0 0 0; }
.badge-mini { font-size:12px; padding:6px 10px; border-radius:10px; border:1px solid transparent; }
.badge-mini.ok { background: #10b98122; border-color:#10b98155; color:#10b981; }
.badge-mini.err{ background: #ef444422; border-color:#ef444480; color:#ef4444; }

/* Comparison table: stronger contrast */
.cmp-table { width:100%; border-collapse:separate; border-spacing:0 10px; }
.cmp-row { background: rgba(148,163,184,0.10); border:1px solid rgba(148,163,184,0.38); }
.cmp-row td { padding:12px 14px; }
.cmp-row:hover { background: rgba(255,255,255,0.08); }
.cmp-key { width:160px; color:#e5e7eb; font-weight:700; }
.cmp-val { border-left:1px dashed rgba(148,163,184,0.45); color:#f8fafc; }
.win-pill { display:inline-flex; align-items:center; gap:6px; padding:2px 10px; border-radius:999px;
            background:rgba(16,185,129,0.22); border:1px solid rgba(16,185,129,0.65); font-weight:700; }
.win-pill::after { content:"✓"; font-size:12px; }

/* Normalize LLM typography */
.stChatMessage, .stMarkdown p, .stMarkdown li, .stMarkdown {
  font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji" !important;
  font-size: 16px !important;
  line-height: 1.55 !important;
}
</style>
""", unsafe_allow_html=True)

col_title, col_btn = st.columns([0.84, 0.16])
with col_title:
    st.title("💬 Customer Intelligence Tool (Cellphones-2023 Dataset)")
    st.caption("Gemini-first assistant — small talk & shopping; shopping grounded on HF dataset")
with col_btn:
    def hard_reload():
        try:
            for k in list(st.session_state.keys()):
                del st.session_state[k]
        except Exception:
            pass
        try: st.cache_data.clear()
        except Exception: pass
        try: st.cache_resource.clear()
        except Exception: pass
        st.rerun()
    if st.button("🔁 Reload app", help="Fully clear caches and restart the app", use_container_width=True):
        hard_reload()

st.success("✅ App Working")

def badge(msg: str, ok: bool = True):
    c = "ok" if ok else "err"
    st.markdown(f"""<div class="badge-row"><div class="badge-mini {c}">{escape(msg)}</div></div>""",
                unsafe_allow_html=True)

# ---------------------- Key loading ----------------------
def load_api_key_from_file(path: str = "allen_apikey.txt") -> Optional[str]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                if s.upper().startswith("GOOGLE_API_KEY="):
                    return s.split("=", 1)[1].strip()
                return s
    except Exception:
        return None

GEMINI_OK = False
GEM_MODEL = "gemini-1.5-flash"
API_KEY = load_api_key_from_file()
if API_KEY:
    os.environ["GOOGLE_API_KEY"] = API_KEY
try:
    import google.generativeai as genai
    if API_KEY:
        genai.configure(api_key=API_KEY)
        GEMINI_OK = True
        badge("Gemini connected", ok=True)
    else:
        badge("Gemini key file not found (allen_apikey.txt).", ok=False)
except Exception as e:
    badge(f"Gemini not active: {e}", ok=False)

# ---------------------- Judge (optional) ----------------------
try:
    from crew_judge_module import judge_products, render_streamlit_card
except Exception:
    def judge_products(query, left, right, use_crewai=True, model=GEM_MODEL):
        def val(p):
            try:
                pr = float(p.get("price_float") or np.inf)
                ra = float(p.get("average_rating") or 0.0)
                return ra / pr if pr > 0 else 0.0
            except Exception:
                return 0.0
        a, b = val(left), val(right)
        return {"title": "Local Verdict", "winner": "A" if a >= b else "B",
                "reasoning": f"Value scores — A:{a:.3f} vs B:{b:.3f}"}
    def render_streamlit_card(result: dict):
        st.success(f"**{result.get('title','Verdict')}** → Winner: **{result.get('winner','A/B')}**")
        if result.get("reasoning"):
            st.caption(result["reasoning"])

# ---------------------- Heavy deps ----------------------
HEAVY_OK = True
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim as _cos_sim
except Exception as e:
    HEAVY_OK = False
    badge(f"Failed to import FAISS / SentenceTransformer: {e}", ok=False)

# ---------------------- Utils ----------------------
def price_str(v) -> str:
    try:
        if isinstance(v, (int, float, np.floating)) and not pd.isna(v):
            return f"${float(v):.2f}"
        if isinstance(v, str):
            m = re.search(r"[-+]?\d*\.?\d+", v)
            if m:
                return f"${float(m.group()):.2f}"
    except Exception:
        pass
    return "N/A"

def sanitize_md(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"</?[^>]+>", "", s)
    s = s.replace("`", "")
    s = re.sub(r"[_*~]+", "", s)
    s = s.replace("\u200b", "").replace("\u200c", "").replace("\ufeff", "")
    s = re.sub(r"[ \t]+\n", "\n", s)
    return s.strip()

def ensure_price_float(df: pd.DataFrame) -> pd.DataFrame:
    if "price_float" in df.columns:
        return df
    def _coerce(x):
        if isinstance(x, (int, float, np.floating)) and not pd.isna(x): return float(x)
        if isinstance(x, str):
            m = re.search(r"[-+]?\d*\.?\d+", x)
            if m:
                try: return float(m.group())
                except: return np.nan
        return np.nan
    df = df.copy()
    df["price_float"] = df.get("display_price", np.nan).map(_coerce)
    return df

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

# ---------------------- Compare rendering ----------------------
def render_pretty_compare(left: dict, right: dict):
    def _pval(p):
        v = p.get("price_float") if "price_float" in p else p.get("display_price")
        try:
            if isinstance(v, (int,float,np.floating)) and not pd.isna(v): return float(v)
            if isinstance(v, str):
                m = re.search(r"[-+]?\d*\.?\d+", v)
                if m: return float(m.group())
        except Exception: pass
        return None

    def row(label, aval, bval, winner=None):
        acontent = f"<span class='win-pill'>{escape(str(aval))}</span>" if winner == "A" else escape(str(aval))
        bcontent = f"<span class='win-pill'>{escape(str(bval))}</span>" if winner == "B" else escape(str(bval))
        return f"""
        <tr class="cmp-row">
          <td class="cmp-key">{escape(str(label))}</td>
          <td class="cmp-val">{acontent}</td>
          <td class="cmp-val">{bcontent}</td>
        </tr>"""

    lp, rp = _pval(left), _pval(right)
    better_price = "A" if (lp is not None and rp is not None and lp < rp) else ("B" if (lp is not None and rp is not None and rp < lp) else None)
    lr, rr = left.get("average_rating"), right.get("average_rating")
    better_rate = "A" if (isinstance(lr,(int,float,np.floating)) and isinstance(rr,(int,float,np.floating)) and lr > rr) else ("B" if (isinstance(lr,(int,float,np.floating)) and isinstance(rr,(int,float,np.floating)) and rr > lr) else None)
    la, lb = left.get("rating_number"), right.get("rating_number")
    better_vol = "A" if (isinstance(la,(int,np.integer)) and isinstance(lb,(int,np.integer)) and la > lb) else ("B" if (isinstance(la,(int,np.integer)) and isinstance(lb,(int,np.integer)) and lb > la) else None)

    a_brand = left.get("store") or left.get("brand_clean") or ""
    b_brand = right.get("store") or right.get("brand_clean") or ""
    a_feat = ", ".join(map(str, as_list(left.get("features", []))[:8])) or "—"
    b_feat = ", ".join(map(str, as_list(right.get("features", []))[:8])) or "—"
    a_desc = str(left.get("description", ""))[:400] or "—"
    b_desc = str(right.get("description", ""))[:400] or "—"

    html = f"""
    <table class="cmp-table">
      {row("price", price_str(lp), price_str(rp), better_price)}
      {row("average_rating", left.get("average_rating","N/A"), right.get("average_rating","N/A"), better_rate)}
      {row("rating_number", la or 'N/A', lb or 'N/A', better_vol)}
      {row("brand/store", a_brand or '—', b_brand or '—', None)}
      {row("features", a_feat, b_feat, None)}
      {row("description", a_desc, b_desc, None)}
    </table>
    """
    st_html(html, height=420, scrolling=True)

# ---------------------- Encoder + HF shards ----------------------
if HEAVY_OK:
    @st.cache_resource(show_spinner=True)
    def load_encoder():
        return SentenceTransformer("all-MiniLM-L6-v2")
    try:
        model = load_encoder()
        badge("Encoder: all-MiniLM-L6-v2 ready", ok=True)
    except Exception as e:
        model = None
        badge(f"❌ Failed to load encoder: {e}", ok=False)
else:
    model = None

HF_BASE = "https://huggingface.co/GovinKin/MGTA415database/resolve/main/"
PARQUETS = [
    "cellphones_clean_with_price000.parquet",
    "cellphones_clean_with_price-0001.parquet",
    "cellphones_clean_with_price-0002.parquet",
    "cellphones_clean_with_price-0003.parquet",
    "cellphones_clean_with_price-0004.parquet",
    "cellphones_clean_with_price-0005.parquet",
    "cellphones_clean_with_price-0006.parquet",
]
FAISS_FILES = [
    "cellphones_with_price000.faiss",
    "full-00001-of-00007.parquet.faiss",
    "full-00002-of-00007.parquet.faiss",
    "full-00003-of-00007.parquet.faiss",
    "full-00004-of-00007.parquet.faiss",
    "full-00005-of-00007.parquet.faiss",
    "full-00006-of-00007.parquet.faiss",
]

def _download_if_missing(remote_url: str, local_name: str):
    if os.path.exists(local_name):
        return local_name
    urllib.request.urlretrieve(remote_url + "?download=1", local_name)
    return local_name

@st.cache_data(show_spinner=True)
def load_shards():
    dfs, idxs, total = [], [], 0
    for pq, fx in zip(PARQUETS, FAISS_FILES):
        pq_path = _download_if_missing(HF_BASE + pq, pq)
        fx_path = _download_if_missing(HF_BASE + fx, fx)
        df = pd.read_parquet(pq_path)
        idx = faiss.read_index(fx_path)
        dfs.append(df); idxs.append(idx); total += len(df)
    return dfs, idxs, total

dfs, idxs, total_rows = [], [], 0
try:
    if HEAVY_OK:
        dfs, idxs, total_rows = load_shards()
        badge(f"Loaded {len(dfs)} shard pairs from HF ({total_rows:,} rows total).", ok=True)
        badge(f"Dataset loaded: {total_rows:,} rows", ok=True)
    else:
        badge("❌ Skipped loading shards (FAISS/ST not imported).", ok=False)
except Exception as e:
    badge(f"❌ Failed to load dataset/index bundle: {e}", ok=False)

# ---------------------- Retrieval helpers ----------------------
def search_all_shards(query: str, model, dfs: List[pd.DataFrame], idxs: List[faiss.Index],
                      top_k_per=10, final_top=30) -> pd.DataFrame:
    if model is None or not dfs or not idxs:
        return pd.DataFrame()
    qv = model.encode([query]).astype("float32")
    rows = []
    for df, index in zip(dfs, idxs):
        try:
            D, I = index.search(qv, k=top_k_per)
            sub = df.iloc[I[0]].copy()
            sub["distance"] = D[0]
            rows.append(sub)
        except Exception:
            continue
    if not rows: return pd.DataFrame()
    merged = pd.concat(rows, ignore_index=True)
    merged = merged.sort_values("distance", ascending=True).head(final_top)
    return merged

def rerank_by_similarity(query: str, results: pd.DataFrame, model, top_n=5) -> pd.DataFrame:
    if results.empty or model is None: return results
    query_vec = model.encode([query], convert_to_tensor=True)
    titles = results["title"].astype(str).tolist()
    title_vecs = model.encode(titles, convert_to_tensor=True)
    sims = _cos_sim(query_vec, title_vecs)[0].cpu().numpy()
    out = results.copy()
    out["similarity_score"] = sims
    return out.sort_values("similarity_score", ascending=False).head(top_n)

# ---------------------- Intent & text actions ----------------------
RE_COMPARE = re.compile(r"\b(?:compare|vs|versus)\b", re.I)
RE_NUMPAIR = re.compile(r"\b([1-9]|10)\s*(?:&|and|,|\s)\s*([1-9]|10)\b")
RE_LETTERPAIR = re.compile(r"\b([A-Ea-e])\s*(?:&|and|,|\s)\s*([A-Ea-e])\b")
RE_PRICE_RANGE = re.compile(r"between\s*\$?\s*(\d+(?:\.\d+)?)\s*(?:and|to|-)\s*\$?\s*(\d+(?:\.\d+)?)", re.I)
RE_UNDER = re.compile(r"(?:under|below|less than|cheaper than|<)\s*\$?\s*(\d+(?:\.\d+)?)", re.I)
RE_OVER  = re.compile(r"(?:over|above|greater than|higher than|>)\s*\$?\s*(\d+(?:\.\d+)?)", re.I)

def is_small_talk(msg: str) -> bool:
    m = msg.lower().strip()
    chat_keys = ["how are you", "how's it going", "hello", "hi", "hey", "what's up", "thank you", "thanks", "what is your name", "who are you"]
    prod_keys = ["case","phone","iphone","samsung","pixel","android","accessory","charger","protector","magnet","battery","screen","otterbox","spigen","clear","wallet","wireless"]
    return any(k in m for k in chat_keys) and not any(k in m for k in prod_keys) and not RE_COMPARE.search(m)

def has_shopping_intent(msg: str) -> bool:
    m = msg.lower()
    if RE_COMPARE.search(m): return True
    if any(k in m for k in ["best","recommend","suggest","option","pick","looking for","buy","budget","under","below","cheap","value","rating","reviews","protect","thin","slim","leather","mag-safe","magsafe","kickstand"]):
        return True
    if "$" in m or re.search(r"\b\d+\s?(?:stars?|reviews?)\b", m):
        return True
    if any(k in m for k in ["case","phone","iphone","samsung","pixel","android","accessory","charger","protector"]):
        return True
    return False

def parse_pair(text: str, top_len: int) -> Tuple[Optional[int], Optional[int]]:
    m = RE_NUMPAIR.search(text)
    if m:
        i, j = int(m.group(1))-1, int(m.group(2))-1
        if 0 <= i < top_len and 0 <= j < top_len and i != j: return i, j
    m = RE_LETTERPAIR.search(text)
    if m:
        mapL = {c:i for i,c in enumerate("ABCDE")}
        i, j = mapL[m.group(1).upper()], mapL[m.group(2).upper()]
        if 0 <= i < top_len and 0 <= j < top_len and i != j: return i, j
    return None, None

def apply_text_action(user_msg: str, df: pd.DataFrame):
    msg = user_msg.lower()
    df = ensure_price_float(df)
    r = RE_PRICE_RANGE.search(msg)
    if r:
        lo, hi = float(r.group(1)), float(r.group(2))
        if lo > hi: lo, hi = hi, lo
        flt = df[(df["price_float"] >= lo) & (df["price_float"] <= hi)].copy()
        return flt, f"Filtered between ${lo:.0f} and ${hi:.0f}."
    r = RE_UNDER.search(msg)
    if r:
        cap = float(r.group(1))
        flt = df[df["price_float"] < cap].copy()
        return flt, f"Filtered under ${cap:.0f}."
    r = RE_OVER.search(msg)
    if r:
        lo = float(r.group(1))
        flt = df[df["price_float"] > lo].copy()
        return flt, f"Filtered over ${lo:.0f}."
    if "sort" in msg:
        if "price" in msg:
            asc = "low" in msg or "asc" in msg
            return df.sort_values("price_float", ascending=asc).copy(), f"Sorted by price ({'low→high' if asc else 'high→low'})."
        if "rating" in msg:
            return df.sort_values("average_rating", ascending=False).copy(), "Sorted by rating (highest→lowest)."
        if "review" in msg:
            return df.sort_values("rating_number", ascending=False).copy(), "Sorted by reviews (most→least)."
        if "value" in msg and "average_rating" in df.columns:
            tmp = df.copy()
            tmp["value_score"] = tmp["average_rating"] / tmp["price_float"].replace(0,np.nan)
            return tmp.sort_values(["value_score","average_rating","rating_number"], ascending=[False,False,False]).copy(), "Sorted by value (rating/price)."
    return df, None

# ---------------------- Gemini helpers (everything via Gemini) ----------------------
def g_call(system_instruction: str, user_text: str) -> str:
    """Robust Gemini call with non-empty fallback string."""
    if not GEMINI_OK:
        return ""
    try:
        m = genai.GenerativeModel(GEM_MODEL, system_instruction=system_instruction)
        out = m.generate_content(user_text)
        txt = (out.text or "").strip()
        if not txt:
            # retry once with a nudge
            out2 = m.generate_content(user_text + "\n\nPlease answer naturally in 1–3 sentences.")
            txt = (out2.text or "").strip()
        return sanitize_md(txt) if txt else ""
    except Exception as e:
        badge(f"Gemini error: {e}", ok=False)
        return ""

def g_smalltalk(user_msg: str) -> str:
    sys = ("You are a warm, concise assistant. This is small talk only — "
           "do not suggest products unless the user asks about shopping. "
           "Respond in 1–3 sentences.")
    txt = g_call(sys, user_msg)
    if txt:
        return txt
    # final fallback (never 'Okay!')
    m = user_msg.lower()
    if "name" in m: return "I’m your shopping assistant."
    if any(x in m for x in ["how are you", "how's it going", "how are u"]):
        return "I’m doing great—thanks! What can I help you find today?"
    if any(x in m for x in ["hello", "hi", "hey", "what's up"]):
        return "Hi there! What are you shopping for today?"
    if any(x in m for x in ["thanks", "thank you", "thx"]):
        return "You’re welcome! Need anything else?"
    return "I’m here! Tell me what you’re looking for and I’ll help."

def g_grounded_list(user_msg: str, numbered_bullets: List[str]) -> str:
    """Ask Gemini to present numbered bullets with blank lines — using only provided bullets."""
    if not GEMINI_OK or not numbered_bullets:
        return ""
    sys = ("You are a helpful shopping assistant. Use ONLY the provided catalog bullets. "
           "Present at least 3 items as **numbered bullets** (1., 2., 3., …), "
           "with a blank line between bullets. Keep it concise.")
    txt = g_call(sys, user_msg + "\n\n" + "\n".join(numbered_bullets))
    if not txt:
        return ""
    # ensure spacing between numbered bullets
    txt = re.sub(r"\n(\d+\.)", r"\n\n\\1", txt).lstrip()
    return txt

def bullets_from_df(df: pd.DataFrame, limit=6) -> List[str]:
    out=[]
    df = ensure_price_float(df)
    for i, (_, r) in enumerate(df.head(limit).iterrows(), 1):
        p = price_str(r.get("price_float") if "price_float" in r else r.get("display_price"))
        rt = r.get("average_rating"); rt_s = f"{float(rt):.2f}" if pd.notna(rt) else "N/A"
        rv = r.get("rating_number"); rv_s = f"{int(rv):,}" if pd.notna(rv) else "N/A"
        out.append(f"{i}. **{str(r.get('title',''))[:90]}** — {p}, ⭐ {rt_s} ({rv_s})")
    return out

# ---------------------- Chat state ----------------------
if "chat" not in st.session_state:
    st.session_state.chat = []
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "last_top5" not in st.session_state:
    st.session_state.last_top5 = []

def reset_conversation():
    st.session_state.chat = []
    st.session_state.last_query = ""
    st.session_state.last_top5 = []

# History
for m in st.session_state.chat:
    role = m.get("role","assistant")
    with st.chat_message(role) if hasattr(st, "chat_message") else st.container():
        st.markdown(m.get("content",""))

# Welcome
if not st.session_state.chat:
    hello = "Hi! Ask me anything — e.g., **best under $20**, **thin iPhone 14 case**, **compare 2 & 5**, **sort by value**."
    st.session_state.chat.append({"role":"assistant","content":hello})
    with st.chat_message("assistant"): st.markdown(hello)

# Bottom-right Reset conversation button
with st.container():
    c1, c2, c3 = st.columns([0.70, 0.15, 0.15])
    with c3:
        if st.button("🗑️ Reset conversation", use_container_width=True, help="Clear chat (keep data/model cached)"):
            reset_conversation()
            st.rerun()

# ---------------------- Main chat loop ----------------------
user_msg = st.chat_input("Ask naturally… I’ll search when needed (e.g., “best under $20”, “compare 2 & 5”).")

def handle_compare_request(user_msg: str) -> bool:
    i, j = parse_pair(user_msg, len(st.session_state.last_top5))
    if i is None or j is None or not st.session_state.last_top5:
        return False
    left, right = st.session_state.last_top5[i], st.session_state.last_top5[j]
    bloc = f"Comparing **#{i+1}** and **#{j+1}** from your current list:"
    with st.chat_message("assistant"):
        st.markdown(bloc)
        render_pretty_compare(left, right)
        try:
            result = judge_products(st.session_state.last_query or user_msg, left, right, use_crewai=True, model=GEM_MODEL)
            render_streamlit_card(result)
        except Exception as e:
            st.caption(f"(AI judge fallback: {e})")
            render_streamlit_card(judge_products(st.session_state.last_query or user_msg, left, right, use_crewai=False))
        st.markdown("\n**Anything else I can help you with?**")
    st.session_state.chat.append({"role":"assistant","content":bloc + "\n\n**Anything else I can help you with?**"})
    return True

def run_retrieval_and_reply(user_msg: str):
    st.session_state.last_query = user_msg
    with st.spinner("Searching catalog…"):
        res = search_all_shards(user_msg, model, dfs, idxs, top_k_per=12, final_top=48)
    if res.empty:
        msg = g_call(
            "You are kind and concise. Explain there are no catalog matches and suggest trying different keywords.",
            user_msg
        ) or "I couldn’t find matches in the catalog. Try different keywords?"
        st.session_state.chat.append({"role":"assistant","content":msg})
        with st.chat_message("assistant"): st.markdown(msg)
        return

    res = ensure_price_float(res)
    top = rerank_by_similarity(user_msg, res, model, top_n=5)

    tmp = top.copy()
    tmp["price"] = tmp.get("display_price", "N/A")
    tmp["average_rating"] = tmp.get("average_rating", np.nan)
    tmp["rating_number"] = tmp.get("rating_number", np.nan)
    tmp["features"] = tmp.get("features", [])
    tmp["categories"] = tmp.get("categories", [])
    tmp["description"] = tmp.get("description", "")
    if "store" not in tmp.columns and "brand_clean" in tmp.columns:
        tmp["store"] = tmp["brand_clean"]
    st.session_state.last_top5 = tmp.to_dict(orient="records")

    acted_df, summary = apply_text_action(user_msg, res)
    bullets_src = acted_df if not acted_df.empty else res
    numbered = bullets_from_df(bullets_src, limit=max(5, min(6, len(bullets_src))))
    # ask Gemini to format the numbered list (at least 3, blank line spacing)
    grounded = g_grounded_list(user_msg, numbered)
    if grounded.strip():
        blines = grounded
    else:
        # local fallback
        fallback_list = bullets_from_df(tmp if len(tmp) >= 3 else res, limit=max(3, min(6, len(res))))
        blines = "\n\n".join(fallback_list)

    pre = g_call(
        "You are a concise, friendly shopping assistant. Acknowledge the request in 1 sentence.",
        user_msg
    ) or "Here are good options I found."
    if summary: pre += "\n\n" + summary

    tail = ("\n\nI can compare any two — e.g., **2 & 5**.\n\n"
            "**Anything else I can help you with?**")
    final = pre + ("\n\n" if blines else "") + blines + tail

    st.session_state.chat.append({"role":"assistant","content":final})
    with st.chat_message("assistant"): st.markdown(final)

try:
    if user_msg:
        st.session_state.chat.append({"role":"user","content":user_msg})
        with st.chat_message("user"): st.markdown(user_msg)

        # 1) If user references previous results to compare
        if handle_compare_request(user_msg):
            pass

        # 2) Shopping intent → retrieval + Gemini-grounded list
        elif has_shopping_intent(user_msg) and model is not None and dfs and idxs:
            run_retrieval_and_reply(user_msg)

        # 3) Small talk (Gemini-based, but no products)
        elif is_small_talk(user_msg):
            reply = g_smalltalk(user_msg)
            st.session_state.chat.append({"role":"assistant","content":reply})
            with st.chat_message("assistant"): st.markdown(reply)

        # 4) Fallback (Gemini reply; if no catalog available and not small talk, explain)
        else:
            if not (model and dfs and idxs):
                reply = g_call(
                    "You are helpful and concise. Explain you can chat, but catalog access is currently unavailable.",
                    user_msg
                ) or "I’m here to help — but I can’t access the catalog right now."
            else:
                reply = g_smalltalk(user_msg)
            st.session_state.chat.append({"role":"assistant","content":reply})
            with st.chat_message("assistant"): st.markdown(reply)

except Exception as e:
    st.error(f"Something went wrong handling your last message: {e}")
    st.info("You can keep chatting or tap **Reload app** (top-right) to fully restart.")

