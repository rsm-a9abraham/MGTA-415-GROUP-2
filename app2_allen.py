# -*- coding: utf-8 -*-
# app2_allen.py — Gemini-first Customer Intelligence Tool (single follow-up prompt)

import os, re, ast, sys, platform, urllib.request
from typing import Optional, Tuple, List
from html import escape

import numpy as np
import pandas as pd

import streamlit as st
from streamlit.components.v1 import html as st_html

st.set_page_config(page_title="📱 Customer Intelligence Tool", layout="wide")

# ---------------------- Styles ----------------------
st.markdown("""
<style>
.badge-row { display:flex; flex-wrap:wrap; gap:10px; margin:6px 0 0 0; }
.badge-mini { font-size:12px; padding:6px 10px; border-radius:10px; border:1px solid transparent; }
.badge-mini.ok { background: #10b98122; border-color:#10b98155; color:#10b981; }
.badge-mini.err{ background: #ef444422; border-color:#ef444480; color:#ef4444; }

/* Comparison table (page-level styles; iframe has its own) */
.cmp-table { width:100%; border-collapse:separate; border-spacing:0 10px; }
.cmp-row { background: rgba(148,163,184,0.10); border:1px solid rgba(148,163,184,0.38); }
.cmp-row td { padding:12px 14px; }
.cmp-row:hover { background: rgba(255,255,255,0.08); }
.cmp-key { width:180px; font-weight:700; }
.cmp-val { border-left:1px dashed rgba(148,163,184,0.45); }
.win-pill { display:inline-flex; align-items:center; gap:6px; padding:2px 10px; border-radius:999px;
            background:rgba(16,185,129,0.22); border:1px solid rgba(16,185,129,0.65); font-weight:700; color:#10b981; }
.win-pill::after { content:"✓"; font-size:12px; }

/* Normalize LLM typography */
.stChatMessage, .stMarkdown p, .stMarkdown li, .stMarkdown {
  font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji" !important;
  font-size: 16px !important;
  line-height: 1.55 !important;
}

/* Suggestion chips */
.suggest-row { display:flex; flex-wrap:wrap; gap:8px; margin:6px 0 0 0; }
.suggest { font-size:12px; padding:6px 10px; border-radius:999px; border:1px solid rgba(148,163,184,0.35);
           background:rgba(148,163,184,0.12); cursor: pointer; }
</style>
""", unsafe_allow_html=True)

# ---------------------- Header + Reload ----------------------
col_title, col_btn = st.columns([0.84, 0.16])
with col_title:
    st.title("💬 Customer Intelligence Tool (Cellphones & Accessories)")
    st.caption("Gemini-first assistant — small talk & shopping; shopping grounded on the HF dataset")
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

# ---------------------- Identity & single follow-up prompt ----------------------
BOT_IDENTITY = (
    "You are the **Customer Intelligence Tool** for **cellphones and accessories**. "
    "You help users find, compare, and recommend products using the curated catalog. "
    "Be warm, concise, and helpful. In small talk, do not suggest products unless asked."
)
ASSISTANT_PROMPT = "What can I help you with today?"

# Only non-compare chips at start; compare chips appear AFTER results exist
QUICK_CHIPS_BASE = [
    "best iPhone 14 case under $20",
    "thin MagSafe case",
    "Pixel 7 screen protector",
]

def get_quick_chips():
    chips = QUICK_CHIPS_BASE.copy()
    top = st.session_state.get("last_top5") or []
    n = len(top)
    if n >= 2:
        chips.append("compare 1 & 2")
        if n >= 3:
            chips.append("compare 1 & 3")
    return chips

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

def append_prompt_once(text: str) -> str:
    """Append ASSISTANT_PROMPT only if a similar line isn't already present."""
    norm = re.sub(r'[^a-z]+', ' ', (text or '').lower()).strip()
    already = [
        "what can i help you with today",
        "how can i help you today",
        "how may i help you today",
        "how can i help you",
        "how may i help you",
        "what can i do for you today",
    ]
    if any(p in norm for p in already):
        return text
    return (text or "").rstrip() + "\n\n_" + ASSISTANT_PROMPT + "_"

# ---------------------- Compare rendering (iframe with its own CSS) ----------------------
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

    a_brand = left.get("store") or left.get("brand_clean") or "—"
    b_brand = right.get("store") or right.get("brand_clean") or "—"
    a_feat = ", ".join(map(str, as_list(left.get("features", []))[:8])) or "—"
    b_feat = ", ".join(map(str, as_list(right.get("features", []))[:8])) or "—"
    a_desc = (str(left.get("description", ""))[:400] or "—")
    b_desc = (str(right.get("description", ""))[:400] or "—")

    html = f"""
    <style>
      /* Light, readable look inside the iframe so it's theme-agnostic */
      html, body {{
        background: #ffffff;
        color: #111827;
        font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
        margin: 0;
        padding: 0;
      }}
      .cmp-table {{ width:100%; border-collapse:separate; border-spacing:0 10px; }}
      .cmp-row {{ background:#ffffff; border:1px solid rgba(0,0,0,0.12); border-radius:10px; }}
      .cmp-row:hover {{ background:#f9fafb; }}
      .cmp-row td {{ padding:12px 14px; vertical-align:top; }}
      .cmp-key {{ width:180px; color:#374151; font-weight:700; }}
      .cmp-val {{ border-left:1px dashed rgba(0,0,0,0.16); color:#111827; }}
      .win-pill {{
        display:inline-flex; align-items:center; gap:6px; padding:2px 10px; border-radius:999px;
        background:rgba(16,185,129,0.12); border:1px solid rgba(16,185,129,0.45); font-weight:700; color:#047857;
      }}
      .win-pill::after {{ content:"✓"; font-size:12px; }}
    </style>
    <table class="cmp-table">
      {row("price", price_str(lp), price_str(rp), better_price)}
      {row("average_rating", left.get("average_rating","N/A"), right.get("average_rating","N/A"), better_rate)}
      {row("rating_number", la or 'N/A', lb or 'N/A', better_vol)}
      {row("brand/store", a_brand, b_brand, None)}
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

# ---------------------- Intent & actions ----------------------
RE_COMPARE = re.compile(r"\b(?:compare|vs|versus)\b", re.I)
RE_NUMPAIR = re.compile(r"\b([1-9]|10)\s*(?:&|and|,|\s)\s*([1-9]|10)\b")
RE_LETTERPAIR = re.compile(r"\b([A-Ea-e])\s*(?:&|and|,|\s)\s*([A-Ea-e])\b")
RE_PRICE_RANGE = re.compile(r"between\s*\$?\s*(\d+(?:\.\d+)?)\s*(?:and|to|-)\s*\$?\s*(\d+(?:\.\d+)?)", re.I)
RE_UNDER = re.compile(r"(?:under|below|less than|cheaper than|<)\s*\$?\s*(\d+(?:\.\d+)?)", re.I)
RE_OVER  = re.compile(r"(?:over|above|greater than|higher than|>)\s*\$?\s*(\d+(?:\.\d+)?)", re.I)

def is_small_talk(msg: str) -> bool:
    m = msg.lower().strip()
    chat_keys = [
        "how are you", "how's it going", "hello", "hi", "hey", "what's up",
        "thank you", "thanks", "what is your name", "who are you",
        "what is this tool", "what are you", "tell me about yourself"
    ]
    prod_keys = ["case","phone","iphone","samsung","pixel","android","accessory","charger",
                 "protector","magnet","battery","screen","otterbox","spigen","clear",
                 "wallet","wireless"]
    return any(k in m for k in chat_keys) and not any(k in m for k in prod_keys) and not RE_COMPARE.search(m)

def has_shopping_intent(msg: str) -> bool:
    m = msg.lower()
    if RE_COMPARE.search(m): return True
    if any(k in m for k in ["best","recommend","suggest","option","pick","looking for","buy",
                            "budget","under","below","cheap","value","rating","reviews","protect",
                            "thin","slim","leather","mag-safe","magsafe","kickstand"]):
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
    """Apply filter/sort actions inferred from text.
       NOTE: 'sort by value' is DISABLED by request.
    """
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
        if "value" in msg:
            # Disabled: keep dataset as-is and inform the user.
            return df.copy(), "Note: sorting by value is disabled."
    return df, None

# ---------------------- Gemini helpers ----------------------
def g_call(system_instruction: str, user_text: str) -> str:
    if not GEMINI_OK:
        return ""
    try:
        m = genai.GenerativeModel(GEM_MODEL, system_instruction=system_instruction)
        out = m.generate_content(user_text)
        txt = (out.text or "").strip()
        if not txt:
            out2 = m.generate_content(user_text + "\n\nPlease answer naturally in 1–3 sentences.")
            txt = (out2.text or "").strip()
        return sanitize_md(txt) if txt else ""
    except Exception as e:
        badge(f"Gemini error: {e}", ok=False)
        return ""

def g_smalltalk(user_msg: str) -> str:
    sys = (
        f"{BOT_IDENTITY} "
        "This is small talk only — do not suggest products unless the user asks about shopping. "
        "Respond in 1–3 sentences. Do not add closing prompts like “How can I help you today?”"
    )
    txt = g_call(sys, user_msg)
    if txt:
        return append_prompt_once(txt)

    m = user_msg.lower()
    if "name" in m or "what are you" in m or "what is this tool" in m:
        return append_prompt_once("I’m the Customer Intelligence Tool for cellphones and accessories.")
    if any(x in m for x in ["how are you", "how's it going", "how are u"]):
        return append_prompt_once("I’m doing great—thanks!")
    if any(x in m for x in ["hello", "hi", "hey", "what's up"]):
        return append_prompt_once("Hi there!")
    if any(x in m for x in ["thanks", "thank you", "thx"]):
        return append_prompt_once("You’re welcome!")
    return append_prompt_once("I’m here to help.")

def bullets_from_df(df: pd.DataFrame, limit=5) -> List[str]:
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

# Welcome (no compare / no sort examples)
if not st.session_state.chat:
    hello = (
        "Hi! I’m the **Customer Intelligence Tool** for **cellphones & accessories**. "
        "Ask naturally — e.g., **best under $20**, **thin iPhone 14 case**, **Pixel 7 screen protector**."
    )
    st.session_state.chat.append({"role":"assistant","content":hello})
    with st.chat_message("assistant"): st.markdown(hello)

# Quick prompt chips (compare chips appear only after results exist)
chips_to_show = get_quick_chips()
st.markdown('<div class="suggest-row">', unsafe_allow_html=True)
chip_cols = st.columns(len(chips_to_show))
chip_clicked_text = None
for i, label in enumerate(chips_to_show):
    if chip_cols[i].button(label, key=f"suggest_{i}"):
        chip_clicked_text = label
st.markdown('</div>', unsafe_allow_html=True)

# Bottom-right Reset conversation button
with st.container():
    c1, c2, c3 = st.columns([0.70, 0.15, 0.15])
    with c3:
        if st.button("🗑️ Reset conversation", use_container_width=True, help="Clear chat (keep data/model cached)"):
            reset_conversation()
            st.rerun()

# ---------------------- Main chat ----------------------
user_msg = chip_clicked_text or st.chat_input(
    "Ask naturally… I’ll search when needed (e.g., “best under $20”, “thin iPhone 14 case”)."
)

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
            f"{BOT_IDENTITY} Explain there are no catalog matches and suggest different keywords (1–2 sentences).",
            user_msg
        ) or "I couldn’t find matches in the catalog. Try different keywords?"
        st.session_state.chat.append({"role":"assistant","content":msg})
        with st.chat_message("assistant"): st.markdown(msg)
        return

    res = ensure_price_float(res)
    pool, summary = apply_text_action(user_msg, res)
    if pool is None or pool.empty:
        pool = res

    top = rerank_by_similarity(user_msg, pool, model, top_n=5)
    if len(top) < 3:
        extra = pool[~pool.index.isin(top.index)].head(3 - len(top))
        top = pd.concat([top, extra], axis=0)

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

    enumerated = bullets_from_df(tmp, limit=max(3, min(5, len(tmp))))

    pre = g_call(
        f"{BOT_IDENTITY} Acknowledge the user in 1 sentence.",
        user_msg
    ) or "Here are good options I found."
    if summary: pre += "\n\n" + summary

    tail = ("\n\nI can compare any two — e.g., **1 & 3**.\n\n"
            f"_{ASSISTANT_PROMPT}_")
    final = pre + "\n\n" + "\n\n".join(enumerated) + tail

    st.session_state.chat.append({"role":"assistant","content":final})
    with st.chat_message("assistant"): st.markdown(final)

try:
    if user_msg:
        st.session_state.chat.append({"role":"user","content":user_msg})
        with st.chat_message("user"): st.markdown(user_msg)

        if handle_compare_request(user_msg):
            pass
        elif has_shopping_intent(user_msg) and model is not None and dfs and idxs:
            run_retrieval_and_reply(user_msg)
        elif is_small_talk(user_msg):
            reply = g_smalltalk(user_msg)
            st.session_state.chat.append({"role":"assistant","content":reply})
            with st.chat_message("assistant"): st.markdown(reply)
        else:
            reply = g_call(
                f"{BOT_IDENTITY} If the user intent is unclear, kindly ask a brief follow-up. 1–2 sentences.",
                user_msg
            ) or "I’m here to help."
            reply = append_prompt_once(reply)  # ensure only one closing prompt
            st.session_state.chat.append({"role":"assistant","content":reply})
            with st.chat_message("assistant"): st.markdown(reply)

except Exception as e:
    st.error(f"Something went wrong handling your last message: {e}")
    st.info("You can keep chatting or tap **Reload app** (top-right) to fully restart.")
