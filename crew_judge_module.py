# crew_judge_module.py
# Enhanced AI judge module:
# - Loads GOOGLE_API_KEY from .env / api.env / Streamlit secrets (sanitized)
# - Local scoring (price/rating/review/keyword coverage)
# - Specs extraction from features/description/title/categories
# - Rich local reasons/evidence/risks/personas/actionables
# - Optional Gemini analysis with retry; final winner kept consistent with local

from __future__ import annotations

import os
import re
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
import streamlit as st

# --- GOOGLE API KEY bootstrap (module-level) ---
def _bootstrap_google_api_key_module() -> Optional[str]:
    key = os.getenv("GOOGLE_API_KEY")

    # try .env / api.env
    if not key:
        try:
            from dotenv import load_dotenv
            for name in (".env", "api.env"):
                if os.path.exists(name):
                    load_dotenv(name, override=False)
        except Exception:
            pass
        key = os.getenv("GOOGLE_API_KEY")

    # try Streamlit secrets
    if not key:
        try:
            if hasattr(st, "secrets"):
                sec = st.secrets.get("GOOGLE_API_KEY")
                if sec:
                    key = sec
        except Exception:
            pass

    # sanitize
    if key:
        key = key.strip().strip('"').strip("'")
        os.environ["GOOGLE_API_KEY"] = key

    # optional hardcode for local dev (do NOT commit real key)
    if not key:
        HARD_CODED = ""  # e.g. "AIza...YOUR_KEY"; leave empty in repo
        if HARD_CODED:
            os.environ["GOOGLE_API_KEY"] = HARD_CODED
            key = HARD_CODED
    return key

_BOOT_KEY = _bootstrap_google_api_key_module()


# ---------- Optional LLM (Gemini) ----------
_CREW_AVAILABLE = True
try:
    from langchain_google_genai import ChatGoogleGenerativeAI

    def _LLM(model: str):
        # Keep responses concise to reduce quota usage
        return ChatGoogleGenerativeAI(model=model, temperature=0.2)
except Exception:
    _CREW_AVAILABLE = False
    _LLM = None

# Optional retry (tenacity). If not installed, we will call once without retry.
try:
    from tenacity import retry, wait_exponential_jitter, stop_after_attempt

    def _retry_deco():
        return retry(
            wait=wait_exponential_jitter(initial=3, max=45),
            stop=stop_after_attempt(4),
            reraise=True,
        )
except Exception:
    def _retry_deco():
        # no-op decorator
        def decorator(fn):
            return fn
        return decorator


# ---------- Data model ----------
@dataclass
class Product:
    title: str = ""
    price: Optional[float] = None
    average_rating: Optional[float] = None
    rating_number: Optional[int] = None
    brand_clean: Optional[str] = None
    categories: Optional[str] = None
    features: Optional[str] = None
    description: Optional[str] = None

    @staticmethod
    def from_any(d: Dict[str, Any]) -> "Product":
        def _num(x):
            if x is None:
                return None
            try:
                return float(str(x).replace(",", "").strip())
            except Exception:
                return None

        def _int(x):
            v = _num(x)
            return int(v) if v is not None else None

        def _s(v):
            if v is None:
                return None
            try:
                import numpy as np, pandas as pd
                is_np = isinstance(v, np.ndarray)
                is_pd = isinstance(v, (pd.Series, pd.Index, pd.Array))
            except Exception:
                is_np = is_pd = False

            if is_np or is_pd or isinstance(v, (list, tuple, set)):
                if hasattr(v, "tolist"):
                    v = v.tolist()
                arr = [str(x).strip() for x in v if x not in (None, "", "None")]
                return ", ".join(arr) if arr else None

            s = str(v).strip()
            return None if s in ("", "None") else s

        return Product(
            title=str(d.get("title") or d.get("name") or "").strip(),
            price=_num(d.get("price") or d.get("price_float")),
            average_rating=_num(d.get("average_rating") or d.get("rating")),
            rating_number=_int(d.get("rating_number") or d.get("reviews")),
            brand_clean=(d.get("store") or d.get("brand_clean") or d.get("brand") or None),
            categories=_s(d.get("categories")),
            features=_s(d.get("features")),
            description=_s(d.get("description")),
        )


# ---------- Local scoring ----------
def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None

def _keyword_coverage(query: str, text: str) -> float:
    if not query or not text:
        return 0.0
    q = [w for w in re.split(r"[^\w]+", query.lower()) if w and len(w) > 2]
    if not q:
        return 0.0
    t = text.lower()
    hit = sum(1 for w in q if w in t)
    return hit / max(1, len(q))

def _fmt_money(x: Optional[float]) -> str:
    return f"${x:.2f}" if isinstance(x, (int, float, np.floating)) and x is not None else "N/A"

def _local_scores(query: str, A: Product, B: Product) -> Dict[str, Any]:
    priceA = _safe_float(A.price)
    priceB = _safe_float(B.price)
    ratingA = _safe_float(A.average_rating)
    ratingB = _safe_float(B.average_rating)
    volA = _safe_float(A.rating_number)
    volB = _safe_float(B.rating_number)

    textA = " ".join(filter(None, [A.categories, A.features, A.description, A.title]))
    textB = " ".join(filter(None, [B.categories, B.features, B.description, B.title]))
    kwA = _keyword_coverage(query, textA)
    kwB = _keyword_coverage(query, textB)

    def invert_price(p):
        if p is None:
            return 0.0
        return 1.0 / (1.0 + max(0.0, p))

    def zpair(a, b):
        vals = [v for v in (a, b) if v is not None]
        if len(vals) < 2:
            return (0.0, 0.0)
        mu = float(np.mean(vals))
        sd = float(np.std(vals)) or 1.0
        z = lambda v: (float(v) - mu) / sd if v is not None else -10.0
        return z(a), z(b)

    price_sA, price_sB = invert_price(priceA), invert_price(priceB)
    rating_sA, rating_sB = zpair(ratingA, ratingB)
    volume_sA, volume_sB = zpair(volA, volB)
    kw_sA, kw_sB = zpair(kwA, kwB)

    W = {"price": 0.30, "average_rating": 0.30, "rating_number": 0.25, "keyword_coverage": 0.15}
    dims = {
        "price": {"A": price_sA, "B": price_sB},
        "average_rating": {"A": rating_sA, "B": rating_sB},
        "rating_number": {"A": volume_sA, "B": volume_sB},
        "keyword_coverage": {"A": kw_sA, "B": kw_sB},
    }
    def win(d): return "A" if d["A"] > d["B"] else ("B" if d["B"] > d["A"] else "tie")
    for k in dims:
        dims[k]["winner"] = win(dims[k])

    totalA = sum(W[k] * dims[k]["A"] for k in W)
    totalB = sum(W[k] * dims[k]["B"] for k in W)
    total_winner = "A" if totalA > totalB else ("B" if totalB > totalA else "tie")

    return {
        "dimension_scores": dims,
        "weights": W,
        "total": {"A": totalA, "B": totalB, "winner": total_winner},
        "raw": {
            "price": {"A": priceA, "B": priceB},
            "average_rating": {"A": ratingA, "B": ratingB},
            "rating_number": {"A": volA, "B": volB},
            "keyword_coverage_raw": {"A": kwA, "B": kwB},
        }
    }


# ---------- Specs extraction from features/description ----------
def _normalize_text_for_specs(p: Product) -> str:
    return " ".join(filter(None, [p.title, p.categories, p.features, p.description])) or ""

def _extract_specs(text: str) -> Dict[str, Any]:
    t = (text or "").lower()
    specs: Dict[str, Any] = {
        "wireless_charging": bool(re.search(r"\bwireless charging\b", t)),
        "water_resistant": bool(re.search(r"\b(ip6[7-9]|water[- ]?resistan|water[- ]?proof)\b", t)),
        "noise_cancel": bool(re.search(r"\bnoise[- ]?cancel", t)),
        "unlocked": bool(re.search(r"\bunlocked\b", t)),
        "carrier_locked": bool(re.search(r"\blocked\b|only for (verizon|at&t|t-?mobile|sprint)", t)),
        "renewed": bool(re.search(r"\b(renewed|refurbished|pre[- ]?owned|used)\b", t)),
    }
    m = re.search(r"\b(\d+)\s*gb\b", t)  # storage
    if m:
        specs["storage_gb"] = int(m.group(1))
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:inch|in|")\b', t)  # display inches
    if m:
        specs["display_in"] = float(m.group(1))
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:h|hour|hours)\b', t)  # battery hours (very rough)
    if m:
        specs["battery_hours"] = float(m.group(1))
    return specs

def _bool2str(x: Optional[bool]) -> str:
    return "Yes" if x else ("No" if x is not None else "N/A")

def _spec_summary(spec: Dict[str, Any]) -> str:
    parts = []
    if spec.get("wireless_charging"): parts.append("wireless charging")
    if spec.get("water_resistant"):   parts.append("water resistance")
    if spec.get("noise_cancel"):      parts.append("noise cancelation")
    if "storage_gb" in spec:          parts.append(f'{spec["storage_gb"]}GB storage')
    if "display_in" in spec:          parts.append(f'{spec["display_in"]}" display')
    if "battery_hours" in spec:       parts.append(f'{spec["battery_hours"]}h battery note')
    if spec.get("unlocked"):          parts.append("unlocked")
    if spec.get("carrier_locked"):    parts.append("carrier-locked")
    if spec.get("renewed"):           parts.append("renewed/refurbished")
    return ", ".join(parts) if parts else "—"


# ---------- Risks ----------
def _contains(text: str, needles: List[str]) -> bool:
    if not text:
        return False
    t = text.lower()
    return any(n.lower() in t for n in needles)

def _risks_from_text(p: Product) -> List[str]:
    t = " ".join(filter(None, [p.title, p.features, p.description])) or ""
    risks = []
    if _contains(t, ["locked", "only for verizon", "at&t only", "carrier"]):
        risks.append("May be carrier-locked.")
    if _contains(t, ["renewed", "refurbished", "used", "pre-owned"]):
        risks.append("Renewed/refurbished/used item.")
    if _contains(t, ["no headphones", "no earphones"]):
        risks.append("Accessories may be incomplete (no headphones).")
    if _contains(t, ["no sim card"]):
        risks.append("SIM card not included.")
    if _contains(t, ["warranty", "guarantee"]) is False:
        risks.append("Warranty status unclear.")
    return risks


# ---------- Evidence & reasons (with specs) ----------
def _evidence_rows(local: Dict[str, Any],
                   specA: Optional[Dict[str, Any]] = None,
                   specB: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    raw = local.get("raw", {})
    rows = []
    rows.append({"point": "Average rating", "field": "average_rating",
                 "A": f'{raw.get("average_rating", {}).get("A")}' if raw.get("average_rating", {}).get("A") is not None else "N/A",
                 "B": f'{raw.get("average_rating", {}).get("B")}' if raw.get("average_rating", {}).get("B") is not None else "N/A"})
    rows.append({"point": "Number of ratings", "field": "rating_number",
                 "A": f'{raw.get("rating_number", {}).get("A")}' if raw.get("rating_number", {}).get("A") is not None else "N/A",
                 "B": f'{raw.get("rating_number", {}).get("B")}' if raw.get("rating_number", {}).get("B") is not None else "N/A"})
    rows.append({"point": "Price", "field": "price",
                 "A": _fmt_money(raw.get("price", {}).get("A")),
                 "B": _fmt_money(raw.get("price", {}).get("B"))})

    if specA is not None and specB is not None:
        rows.append({"point": "Unlocked", "field": "description/features",
                     "A": _bool2str(specA.get("unlocked")), "B": _bool2str(specB.get("unlocked"))})
        rows.append({"point": "Carrier-locked", "field": "description/features",
                     "A": _bool2str(specA.get("carrier_locked")), "B": _bool2str(specB.get("carrier_locked"))})
        rows.append({"point": "Wireless charging", "field": "description/features",
                     "A": _bool2str(specA.get("wireless_charging")), "B": _bool2str(specB.get("wireless_charging"))})
        rows.append({"point": "Water resistance", "field": "description/features",
                     "A": _bool2str(specA.get("water_resistant")), "B": _bool2str(specB.get("water_resistant"))})
        rows.append({"point": "Noise cancelation", "field": "description/features",
                     "A": _bool2str(specA.get("noise_cancel")), "B": _bool2str(specB.get("noise_cancel"))})
        rows.append({"point": "Storage (GB)", "field": "title/description",
                     "A": specA.get("storage_gb", "N/A"), "B": specB.get("storage_gb", "N/A")})
        rows.append({"point": 'Display (")', "field": "description",
                     "A": specA.get("display_in", "N/A"), "B": specB.get("display_in", "N/A")})
        rows.append({"point": "Battery hours (hint)", "field": "description",
                     "A": specA.get("battery_hours", "N/A"), "B": specB.get("battery_hours", "N/A")})
    return rows


def _reasons_from_local(query: str, A: Product, B: Product,
                        local: Dict[str, Any],
                        specA: Dict[str, Any],
                        specB: Dict[str, Any]) -> List[str]:
    R = []
    raw = local["raw"]

    # Price
    pa, pb = raw["price"]["A"], raw["price"]["B"]
    if pa is not None and pb is not None:
        if pa < pb:
            R.append(f"A is cheaper ({_fmt_money(pa)} vs {_fmt_money(pb)}).")
        elif pb < pa:
            R.append(f"B is cheaper ({_fmt_money(pb)} vs {_fmt_money(pa)}).")
    else:
        R.append("Both products lack reliable price info, so rating/feature evidence weighs more.")

    # Ratings
    ara, arb = raw["average_rating"]["A"], raw["average_rating"]["B"]
    if ara and arb and ara != arb:
        if ara > arb:
            R.append(f"A has higher average rating ({ara} vs {arb}).")
        else:
            R.append(f"B has higher average rating ({arb} vs {ara}).")

    # Volume
    rna, rnb = raw["rating_number"]["A"], raw["rating_number"]["B"]
    if rna and rnb and rna != rnb:
        if rna > rnb:
            R.append(f"A has more reviews ({rna} vs {rnb}).")
        else:
            R.append(f"B has more reviews ({rnb} vs {rna}).")

    # Specs-based reasons
    if specA.get("unlocked") and not specB.get("unlocked"):
        R.append("A is unlocked; B is not.")
    if specB.get("unlocked") and not specA.get("unlocked"):
        R.append("B is unlocked; A is not.")

    if specA.get("carrier_locked") and not specB.get("carrier_locked"):
        R.append("A may be carrier-locked; B is not.")
    if specB.get("carrier_locked") and not specA.get("carrier_locked"):
        R.append("B may be carrier-locked; A is not.")

    for k, label in [
        ("wireless_charging", "wireless charging"),
        ("water_resistant", "water resistance"),
        ("noise_cancel", "noise cancelation"),
    ]:
        a, b = bool(specA.get(k)), bool(specB.get(k))
        if a != b:
            R.append(f"{'A' if a else 'B'} offers {label} while the other does not.")

    for k, label in [
        ("storage_gb", "storage capacity"),
        ("display_in", "display size"),
        ("battery_hours", "battery-hour note"),
    ]:
        a, b = specA.get(k), specB.get(k)
        if a is not None and b is not None and a != b:
            better = "A" if (a > b) else "B"
            R.append(f"{better} has higher {label} ({a} vs {b}).")

    if not R:
        R.append("Local weighted score differentiates A and B based on rating volume and spec matches.")
    return R[:8]


# ---------- LLM prompt ----------
_PROMPT = """You are a product selection analyst. Based ONLY on the provided fields, return a single JSON object:
{
  "winner": "A|B|tie",
  "confidence": 0.xx,
  "reasons": ["..."],
  "evidence": [{"point":"...", "field":"rating_number|price|average_rating|description|features|categories|brand/store", "A":"...", "B":"..."}],
  "risks": ["..."],
  "personas": {
    "budget": {"pick":"A|B|tie", "why":"..."},
    "reputation": {"pick":"A|B|tie", "why":"..."},
    "feature": {"pick":"A|B|tie", "why":"..."}
  },
  "actionables": ["..."]
}
Use features/description/title/categories to infer specs (e.g., wireless charging, water resistance, noise cancelation,
unlocked vs carrier-locked, renewed/refurbished, storage GB, display inches, battery-hours hints). Include these in
reasons and evidence when relevant. Do NOT invent facts that are not present. Keep language concise and professional.
Final winner MUST match the provided local winner.
"""


def _build_ai_payload(query: str, A: Product, B: Product, local: Dict[str, Any],
                      specA: Dict[str, Any], specB: Dict[str, Any]) -> Dict[str, Any]:
    def to_dict(p: Product) -> Dict[str, Any]:
        return {
            "title": p.title,
            "price": p.price,
            "average_rating": p.average_rating,
            "rating_number": p.rating_number,
            "brand/store": p.brand_clean,
            "categories": p.categories,
            "features": p.features,
            "description": p.description,
        }
    return {"query": query, "A": to_dict(A), "B": to_dict(B), "local": local, "specA": specA, "specB": specB}


# ---------- Gemini call with retry ----------
class RateLimitOrQuota(Exception):
    pass

@_retry_deco()
def _run_gemini(payload: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    if not _CREW_AVAILABLE or _LLM is None:
        raise RuntimeError("Gemini / langchain-google-genai not available.")
    llm = _LLM(model_name)
    # try to limit tokens if supported
    try:
        llm.generation_config = getattr(llm, "generation_config", None) or {}
        llm.generation_config.update({"max_output_tokens": 512})
    except Exception:
        pass

    messages = [("system", _PROMPT), ("human", json.dumps(payload, ensure_ascii=False))]
    try:
        res = llm.invoke(messages)
        text = res.content if hasattr(res, "content") else (res if isinstance(res, str) else str(res))
    except Exception as e:
        s = str(e)
        if "429" in s or "quota" in s.lower() or "rate" in s.lower() or "RESOURCE_EXHAUSTED" in s:
            raise RateLimitOrQuota(s)
        raise

    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", text or "")
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {
        "winner": payload["local"]["total"]["winner"],
        "confidence": 0.55,
        "reasons": ["LLM returned non-JSON output; kept local decision."],
        "evidence": [],
        "risks": [],
        "personas": {
            "budget": {"pick": payload["local"]["total"]["winner"], "why": "Fallback to local scoring."},
            "reputation": {"pick": payload["local"]["total"]["winner"], "why": "Fallback to local scoring."},
            "feature": {"pick": payload["local"]["total"]["winner"], "why": "Fallback to local scoring."},
        },
        "actionables": [],
    }


# ---------- Public API ----------
def judge_products(
    query: str,
    left_raw: Dict[str, Any],
    right_raw: Dict[str, Any],
    use_crewai: bool = True,
    model: str = "gemini-1.5-flash",
) -> Dict[str, Any]:
    A = Product.from_any(left_raw)
    B = Product.from_any(right_raw)

    # Specs
    specA = _extract_specs(_normalize_text_for_specs(A))
    specB = _extract_specs(_normalize_text_for_specs(B))

    # Local scoring
    local = _local_scores(query, A, B)

    # Rich LOCAL analyst output
    reasons = _reasons_from_local(query, A, B, local, specA, specB)
    evidence = _evidence_rows(local, specA, specB)
    risksA = _risks_from_text(A)
    risksB = _risks_from_text(B)
    risks = [f"A: {r}" for r in risksA] + [f"B: {r}" for r in risksB]

    # personas
    dims = local["dimension_scores"]
    price_w = dims["price"]["winner"]
    rep_w = dims["rating_number"]["winner"]

    # feature persona: count positive feature flags
    def feature_score(sp: Dict[str, Any]) -> int:
        keys = ["wireless_charging", "water_resistant", "noise_cancel", "unlocked"]
        return sum(1 for k in keys if bool(sp.get(k)))

    fsA, fsB = feature_score(specA), feature_score(specB)
    feat_pick = "tie"
    if fsA > fsB:
        feat_pick = "A"
    elif fsB > fsA:
        feat_pick = "B"

    personas = {
        "budget": {"pick": "A" if price_w == "A" else ("B" if price_w == "B" else "tie"),
                   "why": "Lower effective price for this persona." if price_w in ("A", "B") else "Price parity or missing price."},
        "reputation": {"pick": "A" if rep_w == "A" else ("B" if rep_w == "B" else "tie"),
                       "why": "Higher review volume signals broader market trust." if rep_w in ("A", "B") else "Review counts are similar or missing."},
        "feature": {"pick": feat_pick,
                    "why": f"Feature flags suggest {'A' if fsA>=fsB else 'B'} has better match (A:{fsA} vs B:{fsB})." if fsA!=fsB else "Feature sets appear roughly comparable."},
    }

    actionables = ["Verify carrier lock, condition (renewed/refurbished), warranty and accessory completeness before purchase."]

    result = {
        "winner": local["total"]["winner"],
        "confidence": 0.65,
        "reasons": reasons,
        "evidence": evidence,
        "risks": risks,
        "personas": personas,
        "actionables": actionables,
        "local": local,
        "A": A.__dict__,
        "B": B.__dict__,
        "specA": specA,
        "specB": specB,
    }

    # LLM augmentation (if key & sdk are available)
    if use_crewai and _CREW_AVAILABLE and _LLM is not None and os.getenv("GOOGLE_API_KEY"):
        try:
            payload = _build_ai_payload(query, A, B, local, specA, specB)
            ai = _run_gemini(payload, model)
            ai["winner"] = local["total"]["winner"]  # keep consistent
            for k in ["confidence", "reasons", "evidence", "risks", "personas", "actionables"]:
                if ai.get(k):
                    result[k] = ai[k]
        except Exception as e:
            result["reasons"].append(f"AI analysis failed: {e}. Kept local decision.")
    else:
        if not os.getenv("GOOGLE_API_KEY"):
            result["reasons"].append("GOOGLE_API_KEY is not set; using local analyst report.")
        elif not _CREW_AVAILABLE:
            result["reasons"].append("langchain-google-genai not available; using local analyst report.")
        elif not use_crewai:
            result["reasons"].append("AI judge disabled; using local analyst report.")

    return result


# ---------- Streamlit rendering ----------
def _bullet_list(items):
    if not items:
        return ""
    return "\n".join([f"- {str(x)}" for x in items if str(x).strip()])

def render_streamlit_card(result: Dict[str, Any]) -> None:
    st.markdown("### 🧠 AI Judge")
    cols = st.columns([1, 3])
    with cols[0]:
        st.metric("Winner", result.get("winner", "tie").upper())
        st.caption(f"Confidence {result.get('confidence', 0):.2f}")
    with cols[1]:
        st.markdown("**Reasons**")
        st.markdown(_bullet_list(result.get("reasons", [])))

    ev = result.get("evidence") or []
    if ev:
        st.markdown("**Evidence alignment**")
        try:
            import pandas as pd
            df = pd.DataFrame(ev)
            st.dataframe(df, use_container_width=True, hide_index=True)
        except Exception:
            st.code(json.dumps(ev, indent=2))

    risks = result.get("risks") or []
    if risks:
        st.markdown("**Risks / To verify**")
        st.markdown(_bullet_list(risks))

    personas = result.get("personas") or {}
    if personas:
        st.markdown("**Persona-based recommendations**")
        for k in ["budget", "reputation", "feature"]:
            v = personas.get(k)
            if not v:
                continue
            st.write(f"- **{k.title()}** → pick **{v.get('pick','?')}**, {v.get('why','')}")

    acts = result.get("actionables") or []
    if acts:
        st.markdown("**Actionables**")
        st.markdown(_bullet_list(acts))
