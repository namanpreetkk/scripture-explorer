import re
import html
import unicodedata
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Scriptura", page_icon="🕉️", layout="wide")

# -------------------- CLEANING --------------------
ACCENTS = re.compile(r"[\u0951-\u0954\u1CD0-\u1CFF\uA8E0-\uA8FF\u1CEA\u00AC]")


def clean(t):
    t = unicodedata.normalize("NFC", str(t))
    t = ACCENTS.sub("", t)
    t = re.sub(r"\[[^\]]*\]", " ", t)
    t = re.sub(r"[0-9०-९]+", " ", t)
    t = re.sub(r"[।॥|,.;:!?\-–—'\"()\[\]]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


# -------------------- BROWSE INDEX (scripture -> chapter -> verse) --------------------
# A ref like "1.2.5" is split into chapter "1.2" and verse "5".
REF_SPLIT = r"^(.*?)[\s.:/,_\-]+([^\s.:/,_\-]+)\s*$"
FALLBACK_BLOCK = 200  # used only if a ref can't be split into chapter + verse


def natural_key(s):
    """Sort '1.2' < '1.10' < '2.1' properly."""
    return [(0, int(p), "") if p.isdigit() else (1, 0, p)
            for p in re.split(r"(\d+)", str(s)) if p != ""]


@st.cache_resource
def build_browse_index(_idx):
    refs = _idx["ref"].astype(str).str.strip()
    parts = refs.str.extract(REF_SPLIT)
    chapter = parts[0].fillna("").str.strip()
    verse_no = parts[1].fillna(refs)

    missing = chapter == ""
    if missing.any():
        pos = _idx.groupby("scripture").cumcount()
        blk = pos // FALLBACK_BLOCK
        fallback = ("Verses " + (blk * FALLBACK_BLOCK + 1).astype(str)
                    + "–" + ((blk + 1) * FALLBACK_BLOCK).astype(str))
        chapter = chapter.where(~missing, fallback)

    df = pd.DataFrame({
        "scripture": _idx["scripture"].astype(str).values,
        "chapter": chapter.values,
    }, index=_idx.index)

    tree = {}
    for (s, c), g in df.groupby(["scripture", "chapter"], sort=False):
        tree.setdefault(s, {})[c] = g.index.tolist()
    return tree, verse_no.to_numpy()


@st.cache_resource
def load_assets():
    clf = joblib.load("classifier.joblib")
    emb = np.load("embeddings.npy").astype(np.float32)
    idx = pd.read_csv("verses.csv")
    model = SentenceTransformer("intfloat/multilingual-e5-base")
    return clf, emb, idx, model


clf, emb, idx, model = load_assets()
SCRIPTURES = sorted(idx.scripture.unique())
BROWSE_TREE, VERSE_NO = build_browse_index(idx)


# -------------------- PREDICTION --------------------
def predict_scripture(verse):
    if not verse or not verse.strip():
        return {}
    p = clf.predict_proba([clean(verse)])[0]
    return {c: float(x) for c, x in zip(clf.classes_, p)}


# -------------------- SEMANTIC SEARCH --------------------
def search(query, top_k=5, scripture="All"):
    if not query or not query.strip():
        return pd.DataFrame()

    q = model.encode(["query: " + query], normalize_embeddings=True)[0]
    sims = emb @ q

    if scripture != "All":
        sims = np.where(idx.scripture.values == scripture, sims, -1)

    top = np.argsort(-sims)[:int(top_k)]
    out = idx.iloc[top].copy()
    out["raw"] = out.raw.str.replace(r"\s+", " ", regex=True)
    out["translation"] = out.translation.fillna("")
    out.insert(0, "score", sims[top].round(3))
    return out.rename(columns={"raw": "verse", "translation": "English (Ramayana only)"}).reset_index(drop=True)


# -------------------- EXAMPLES --------------------
QUERY_EXAMPLES = {
    "One should follow dharma (Hindi)": ("धर्म का पालन करना चाहिए", "All"),
    "Speak truth, follow dharma (Sanskrit)": ("सत्यं वद धर्मं चर", "All"),
    "Praise of the fire god (Hindi)": ("अग्नि देवता की स्तुति", "All"),
    "Rama and Sita's exile (Ramayana)": ("राम और सीता का वनवास", "Ramayana"),
    "Bravery in war (Mahabharata)": ("युद्ध में वीरता", "Mahabharata"),
    "Soma sacrifice (Yajurveda)": ("सोम यज्ञ", "Yajurveda"),
    "Duty of a warrior (English)": ("duty of a warrior", "All"),
}

PH_SCR = "Select a scripture"
PH_CH = "Select a chapter"
PH_V = -1  # placeholder for the verse selectbox (options are row numbers)


def verse_text(i):
    return re.sub(r"\s+", " ", str(idx["raw"].iat[i])).strip()


def verse_label(i):
    if i == PH_V:
        return "Select a verse"
    t = verse_text(i)
    return f"{VERSE_NO[i]} — {t[:38]}{'…' if len(t) > 38 else ''}"


# Callbacks keep the three linked dropdowns consistent.
def on_scripture_change():
    st.session_state["browse_chapter"] = PH_CH
    st.session_state["browse_verse"] = PH_V


def on_chapter_change():
    st.session_state["browse_verse"] = PH_V


def on_verse_change():
    i = st.session_state.get("browse_verse", PH_V)
    if i != PH_V:
        st.session_state["verse_input"] = verse_text(i)


def load_query_example():
    name = st.session_state.get("query_example", "Select an example")
    text, scripture = QUERY_EXAMPLES.get(name, ("", "All"))
    st.session_state["query_input"] = text
    st.session_state["scripture_filter"] = scripture


# -------------------- THEME / RESPONSIVE CSS --------------------
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;700&family=Cormorant+Garamond:ital,wght@0,500;0,600;0,700;1,500&family=Tiro+Devanagari+Sanskrit&display=swap');

:root {
    color-scheme: light;
    --ink: #3b1d0e;
    --ink-soft: #6b3a1e;
    --maroon: #4a0f0f;
    --maroon-2: #6e1b1b;
    --rust: #8a2a0e;
    --gold: #d4a017;
    --gold-soft: #f6d98a;
    --border: #c9a15a;
    --paper: #fffaf0;
    --paper-hover: #f6e3b8;
    --paper-selected: #efcf8a;
    --ui-font: 'Cormorant Garamond', 'Tiro Devanagari Sanskrit', Georgia, serif;
    --deva-font: 'Tiro Devanagari Sanskrit', 'Noto Sans Devanagari', 'Mangal', serif;
}

/* ---------- Base ---------- */
html, body { overflow-x: hidden; }
.stApp {
    background: linear-gradient(120deg, #f6e7c8, #f1d6a0, #f8ebd2, #eccb8c);
    background-attachment: fixed;
    color: var(--ink) !important;
}
.block-container {
    max-width: 1250px;
    padding: 1.5rem 2rem 2rem 2rem;
}
