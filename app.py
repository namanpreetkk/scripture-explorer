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
[data-testid="stHeader"] { background: transparent; }

.stApp, .stApp p, .stApp label, .stApp span, .stApp li,
[data-testid="stMarkdownContainer"] p,
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label {
    color: var(--ink);
}
[data-testid="stWidgetLabel"] p {
    font-family: var(--ui-font) !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    color: var(--maroon) !important;
}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
    color: var(--ink-soft) !important;
    font-family: var(--ui-font) !important;
    font-size: 1rem !important;
    font-style: italic;
}
h1, h2, h3, h4 { font-family: 'Cinzel', serif !important; color: var(--maroon) !important; }

/* ---------- Tabs ---------- */
div[data-baseweb="tab-list"] { gap: 6px; overflow-x: auto; }
button[data-baseweb="tab"] { color: var(--maroon) !important; font-weight: 600 !important; }
button[data-baseweb="tab"] p {
    color: var(--maroon) !important;
    font-family: var(--ui-font) !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
}
button[data-baseweb="tab"][aria-selected="true"] p { color: var(--rust) !important; }
div[data-baseweb="tab-highlight"] { background-color: var(--rust) !important; }
div[data-baseweb="tab-border"] { background-color: rgba(201,161,90,.55) !important; }

/* ---------- Text inputs ---------- */
div[data-testid="stTextArea"] textarea,
div[data-testid="stTextInput"] input {
    background: var(--paper) !important;
    color: var(--ink) !important;
    -webkit-text-fill-color: var(--ink) !important;
    border: 1px solid var(--border) !important;
    font-family: var(--deva-font) !important;
    font-size: 1.1rem !important;
    caret-color: var(--ink) !important;
}
div[data-testid="stTextArea"] > div > div,
div[data-testid="stTextInput"] > div > div {
    background: var(--paper) !important;
    border-radius: 8px;
}
div[data-testid="stTextArea"] textarea::placeholder,
div[data-testid="stTextInput"] input::placeholder {
    color: #76563a !important;
    -webkit-text-fill-color: #76563a !important;
    opacity: 1 !important;
}

/* ---------- Selectbox (closed state) ----------
   Streamlit paints the dark fill on an inner layer, so every layer is forced to parchment. */
div[data-testid="stSelectbox"] div[data-baseweb="select"],
div[data-testid="stSelectbox"] div[data-baseweb="select"] div,
div[data-testid="stSelectbox"] div[data-baseweb="select"] input {
    background-color: var(--paper) !important;
    background-image: none !important;
}
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    min-height: 46px;
}
div[data-testid="stSelectbox"] div[data-baseweb="select"] *,
div[data-testid="stSelectbox"] input {
    color: var(--ink) !important;
    -webkit-text-fill-color: var(--ink) !important;
    font-family: var(--ui-font) !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
}
div[data-testid="stSelectbox"] svg { fill: var(--rust) !important; color: var(--rust) !important; }
div[data-testid="stSelectbox"]:has(input[disabled]) { opacity: .55; }

/* ---------- Dropdown menu (rendered in a portal, outside .stApp) ---------- */
div[data-baseweb="popover"],
div[data-baseweb="popover"] > div,
div[data-baseweb="menu"],
ul[role="listbox"] {
    background-color: var(--paper) !important;
    color: var(--ink) !important;
    border-radius: 8px !important;
}
div[data-baseweb="popover"] {
    border: 1px solid var(--border) !important;
    box-shadow: 0 8px 24px rgba(74,15,15,.22) !important;
}
li[role="option"],
div[role="option"] {
    background-color: var(--paper) !important;
    color: var(--ink) !important;
    font-family: var(--ui-font) !important;
    font-size: 1.08rem !important;
    font-weight: 600 !important;
    white-space: normal !important;
    line-height: 1.3 !important;
}
li[role="option"] *, div[role="option"] * {
    color: var(--ink) !important;
    font-family: var(--ui-font) !important;
    font-size: 1.08rem !important;
    font-weight: 600 !important;
    white-space: normal !important;
    background-color: transparent !important;
}
li[role="option"]:hover, li[role="option"][data-highlighted="true"],
div[role="option"]:hover {
    background-color: var(--paper-hover) !important;
}
li[role="option"][aria-selected="true"], div[role="option"][aria-selected="true"] {
    background-color: var(--paper-selected) !important;
}

/* ---------- Buttons ---------- */
.stButton button, button[kind="primary"] {
    background: linear-gradient(135deg, var(--maroon-2), var(--maroon)) !important;
    color: var(--gold-soft) !important;
    border: 1px solid var(--gold) !important;
    border-radius: 7px !important;
    min-height: 46px;
}
.stButton button p, button[kind="primary"] p {
    color: var(--gold-soft) !important;
    font-family: 'Cinzel', serif !important;
    font-weight: 700 !important;
    letter-spacing: .12em;
}
.stButton button:hover { border-color: var(--rust) !important; filter: brightness(1.12); }

/* ---------- Slider ---------- */
div[data-testid="stSlider"] [role="slider"] {
    background-color: var(--maroon-2) !important;
    border-color: var(--maroon-2) !important;
}
div[data-testid="stSlider"] [data-testid="stThumbValue"],
div[data-testid="stSlider"] [data-testid="stTickBarMin"],
div[data-testid="stSlider"] [data-testid="stTickBarMax"] {
    color: var(--maroon) !important;
    font-family: var(--ui-font) !important;
    font-weight: 700 !important;
}

/* ---------- Progress bars ---------- */
div[data-testid="stProgress"] > div > div { background-color: #ead9bd !important; }
div[data-testid="stProgress"] [role="progressbar"] { background-color: #ead9bd !important; }
div[data-testid="stProgress"] [role="progressbar"] > div {
    background: linear-gradient(90deg, var(--rust), var(--gold)) !important;
}

/* ---------- Alerts ---------- */
[data-testid="stAlert"] {
    background: #fff3d6 !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}
[data-testid="stAlert"] p, [data-testid="stAlert"] div { color: #24364b !important; }

/* ---------- Result cards ---------- */
.verse-card {
    margin: 14px 0; padding: 18px 20px;
    border: 1px solid var(--border); border-left: 4px solid #b4541a;
    border-radius: 9px; background: rgba(255,250,240,.97);
    overflow-wrap: anywhere;
}
.verse-card .card-top {
    display: flex; justify-content: space-between; flex-wrap: wrap;
    gap: 8px 12px; align-items: center;
}
.verse-card .tag {
    background: var(--maroon-2); color: var(--gold-soft) !important;
    padding: 2px 10px; border-radius: 3px; font-size: .75rem; letter-spacing: .12em;
}
.verse-card .ref { color: var(--ink-soft); font-style: italic; font-family: var(--ui-font); font-size: 1.05rem; }
.verse-card .sim { color: #7a2e12; font-size: .8rem; }
.verse-card .verse-text {
    font-family: var(--deva-font); font-size: 1.35rem; line-height: 1.9; color: #2e1408;
}
.verse-card .translation {
    margin-top: 8px; font-style: italic; color: #5a3220;
    border-top: 1px dashed var(--border); padding-top: 8px;
    font-family: var(--ui-font); font-size: 1.1rem;
}
.verse-card.result-main { text-align: center; }
.verse-card .kicker { font-size: .85rem; letter-spacing: .18em; color: #7a2e12; }
.verse-card .big { font-size: clamp(1.4rem, 5vw, 2rem); color: var(--maroon); font-family: 'Cinzel', serif; }
.verse-card .conf { font-size: 1.15rem; color: var(--ink-soft); font-style: italic; font-family: var(--ui-font); }
.scorebar { height: 10px; background: #ead9bd; border-radius: 8px; overflow: hidden; margin: 8px 0 12px; }
.scorefill { height: 100%; background: linear-gradient(90deg,#8a2a0e,#d4a017); border-radius: 8px; }

/* ---------- Hero ---------- */
.scripture-hero {
    position: relative; overflow: hidden; text-align: center;
    padding: 44px 12px 28px; margin: 0 auto 22px;
    isolation: isolate;
}
.scripture-hero .mandala {
    position: absolute; z-index: -1; pointer-events: none;
    width: min(780px, 100%); aspect-ratio: 900 / 430;
    left: 50%; top: 48%; transform: translate(-50%, -50%);
    opacity: .26;
}
.scripture-hero .mandala svg { width: 100%; height: 100%; overflow: visible; }
.scripture-hero .orbit { transform-origin: 50% 50%; animation: mandala-spin 80s linear infinite; }
.scripture-hero .orbit-reverse { transform-origin: 50% 50%; animation: mandala-spin-reverse 110s linear infinite; }
.scripture-hero .pulse-ring { animation: ring-breathe 5s ease-in-out infinite; transform-origin: 50% 50%; }
.scripture-hero .hero-om {
    font-size: clamp(2.8rem, 10vw, 4.2rem); line-height: 1.15; color: var(--rust);
    font-family: var(--deva-font); animation: om-glow 4s ease-in-out infinite;
}
.scripture-hero h1 {
    position: relative;
    font-size: clamp(1.75rem, 7.2vw, 3.1rem) !important;
    letter-spacing: clamp(.06em, 1.8vw, .2em) !important;
    line-height: 1.2 !important;
    margin: 10px 0 16px !important;
    padding: 0 0 0 clamp(.06em, 1.8vw, .2em) !important;
    word-break: keep-all !important;
    overflow-wrap: normal !important;
    hyphens: none !important;
}
.scripture-hero .hero-subtitle {
    max-width: 820px; margin: 0 auto; color: var(--ink-soft) !important;
    font-family: var(--ui-font); font-weight: 600;
    font-size: clamp(1.05rem, 3.6vw, 1.4rem); font-style: italic;
    line-height: 1.5; position: relative;
}
.scripture-hero .hero-divider { width: 55%; border: 0; border-top: 1px solid #b4541a; margin: 22px auto 0; position: relative; }
.hero-speck { position: absolute; width: 6px; height: 6px; border-radius: 50%; background: #d9a13a; opacity: .75; animation: speck-drift 7s ease-in-out infinite; }
.hero-speck.s1 { left: 24%; top: 72%; animation-delay: -1s; }
.hero-speck.s2 { left: 74%; top: 60%; animation-delay: -3s; }
.hero-speck.s3 { left: 39%; top: 87%; animation-delay: -5s; }
.hero-speck.s4 { left: 86%; top: 39%; animation-delay: -2s; }
@keyframes mandala-spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
@keyframes mandala-spin-reverse { from { transform: rotate(360deg); } to { transform: rotate(0deg); } }
@keyframes ring-breathe { 0%,100% { opacity: .45; transform: scale(.98); } 50% { opacity: .9; transform: scale(1.025); } }
@keyframes om-glow { 0%,100% { text-shadow: 0 0 0 rgba(138,42,14,0); transform: translateY(0); } 50% { text-shadow: 0 0 18px rgba(180,84,26,.18); transform: translateY(-3px); } }
@keyframes speck-drift { 0%,100% { transform: translateY(0) translateX(0); opacity: .45; } 50% { transform: translateY(-12px) translateX(5px); opacity: .9; } }
@media (prefers-reduced-motion: reduce) { .scripture-hero * { animation: none !important; } }

.site-footer { text-align: center; color: var(--ink-soft); font-style: italic; padding: 20px 0; font-family: var(--ui-font); font-size: 1.05rem; }

/* ---------- Mobile ---------- */
@media (max-width: 640px) {
    .block-container { padding: 3.2rem 1rem 1.5rem 1rem !important; }
    .scripture-hero { padding: 26px 4px 20px; margin-bottom: 14px; }
    .scripture-hero .mandala { width: 120%; }
    .scripture-hero .hero-divider { width: 70%; margin-top: 18px; }
    .verse-card { padding: 14px 14px; }
    .verse-card .verse-text { font-size: 1.2rem; line-height: 1.8; }
    .verse-card .card-top { flex-direction: column; align-items: flex-start; }
    button[data-baseweb="tab"] { padding-left: 10px !important; padding-right: 10px !important; }
    button[data-baseweb="tab"] p { font-size: 1rem !important; }
    .stButton button { width: 100%; }
}
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

# -------------------- HEADER --------------------
HERO_HTML = (
    '<div class="scripture-hero">'
    '<div class="mandala" aria-hidden="true">'
    '<svg viewBox="0 0 900 430" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<g class="orbit" stroke="#b4541a" stroke-width="2.2" stroke-linecap="round">'
    '<circle class="pulse-ring" cx="450" cy="215" r="168" />'
    '<circle cx="450" cy="215" r="205" />'
    '<ellipse cx="450" cy="215" rx="78" ry="205" />'
    '<ellipse cx="450" cy="215" rx="145" ry="205" />'
    '<ellipse cx="450" cy="215" rx="205" ry="82" />'
    '<ellipse cx="450" cy="215" rx="205" ry="145" />'
    '<ellipse cx="450" cy="215" rx="205" ry="82" transform="rotate(30 450 215)" />'
    '<ellipse cx="450" cy="215" rx="205" ry="82" transform="rotate(60 450 215)" />'
    '<ellipse cx="450" cy="215" rx="205" ry="82" transform="rotate(90 450 215)" />'
    '<ellipse cx="450" cy="215" rx="205" ry="82" transform="rotate(120 450 215)" />'
    '<ellipse cx="450" cy="215" rx="205" ry="82" transform="rotate(150 450 215)" />'
    '</g>'
    '<g class="orbit-reverse" stroke="#c98c58" stroke-width="1.8" opacity=".75">'
    '<circle cx="450" cy="215" r="120" />'
    '<ellipse cx="450" cy="215" rx="120" ry="42" transform="rotate(45 450 215)" />'
    '<ellipse cx="450" cy="215" rx="120" ry="42" transform="rotate(-45 450 215)" />'
    '<ellipse cx="450" cy="215" rx="120" ry="42" transform="rotate(90 450 215)" />'
    '</g>'
    '</svg>'
    '</div>'
    '<span class="hero-speck s1"></span><span class="hero-speck s2"></span>'
    '<span class="hero-speck s3"></span><span class="hero-speck s4"></span>'
    '<div class="hero-om">ॐ</div>'
    '<h1>SCRIPTURE EXPLORER</h1>'
    '<p class="hero-subtitle">Identify the source of a verse, or search by meaning across the '
    'Mahabharata, the Ramayana, Rigveda and Yajurveda.</p>'
    '<hr class="hero-divider">'
    '</div>'
)
st.markdown(HERO_HTML, unsafe_allow_html=True)

# -------------------- TABS --------------------
tab1, tab2 = st.tabs(["Identify a Verse", "Search by Meaning"])

with tab1:
    # Browse the whole dataset: scripture -> chapter -> verse
    col_s, col_c, col_v = st.columns([1, 1, 2])
    with col_s:
        st.selectbox(
            "Scripture",
            [PH_SCR] + SCRIPTURES,
            key="browse_scripture",
            on_change=on_scripture_change,
        )
    scr = st.session_state.get("browse_scripture", PH_SCR)
    chapter_map = BROWSE_TREE.get(scr, {})
    chapters = sorted(chapter_map, key=natural_key)

    with col_c:
        st.selectbox(
            "Chapter",
            [PH_CH] + chapters,
            key="browse_chapter",
            on_change=on_chapter_change,
            disabled=not chapters,
        )
    ch = st.session_state.get("browse_chapter", PH_CH)
    verse_ids = chapter_map.get(ch, [])

    with col_v:
        st.selectbox(
            "Verse",
            [PH_V] + verse_ids,
            key="browse_verse",
            format_func=verse_label,
            on_change=on_verse_change,
            disabled=not verse_ids,
        )

    if chapters:
        total = sum(len(v) for v in chapter_map.values())
        st.caption(f"{scr}: {len(chapters):,} chapters, {total:,} verses"
                   + (f" · this chapter has {len(verse_ids):,}" if verse_ids else ""))

    verse = st.text_area(
        "Verse (Devanagari)",
        placeholder="Pick a verse above, or paste or type a Sanskrit verse here",
        height=140,
        key="verse_input",
    )

    if st.button("IDENTIFY", type="primary", key="identify_button"):
        if not verse.strip():
            st.info("Pick a verse from the dropdowns or paste one in.")
        else:
            ranked = sorted(predict_scripture(verse).items(), key=lambda kv: -kv[1])
            if not ranked:
                st.warning("Unable to identify this verse with the loaded classifier.")
            else:
                top, prob = ranked[0]
                st.markdown(
                    '<div class="verse-card result-main">'
                    '<div class="kicker">MOST LIKELY SOURCE</div>'
                    f'<div class="big">{html.escape(str(top))}</div>'
                    f'<div class="conf">{prob*100:.1f}% confidence</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )
                st.markdown("### Prediction Breakdown")
                for name, value in ranked:
                    st.markdown(f"**{html.escape(str(name))} — {value*100:.1f}%**")
                    st.progress(float(value))


with tab2:
    st.selectbox(
        "Choose an example query",
        ["Select an example"] + list(QUERY_EXAMPLES.keys()),
        key="query_example",
        on_change=load_query_example,
    )
    query = st.text_input(
        "Query (Hindi or Sanskrit works best)",
        placeholder="धर्म का पालन करना चाहिए",
        key="query_input",
    )
    scripture_options = ["All"] + SCRIPTURES
    st.selectbox("Restrict to", scripture_options, key="scripture_filter")
    k = st.slider("Number of results", min_value=1, max_value=10, value=5)

    if st.button("SEARCH", type="primary", key="search_button"):
        if not query.strip():
            st.info("Choose an example query or type one.")
        else:
            results = search(query, k, st.session_state.get("scripture_filter", "All"))
            if results.empty:
                st.info("No verses found.")
            else:
                for _, row in results.iterrows():
                    score = float(row["score"])
                    safe_scripture = html.escape(str(row["scripture"]))
                    safe_ref = html.escape(str(row["ref"]))
                    safe_verse = html.escape(str(row["verse"]))
                    translation = str(row.get("English (Ramayana only)", "") or "")
                    safe_translation = html.escape(translation)
                    width = max(0, min(100, score * 100))
                    trans_html = (
                        f'<div class="translation">{safe_translation}</div>'
                        if translation.strip() else ""
                    )
                    st.markdown(
                        '<div class="verse-card">'
                        '<div class="card-top">'
                        f'<span class="tag">{safe_scripture.upper()}</span>'
                        f'<span class="ref">{safe_ref}</span>'
                        f'<span class="sim">SIMILARITY {score:.3f}</span>'
                        '</div>'
                        f'<div class="scorebar"><div class="scorefill" style="width:{width:.1f}%;"></div></div>'
                        f'<div class="verse-text">{safe_verse}</div>'
                        f'{trans_html}'
                        '</div>',
                        unsafe_allow_html=True,
                    )

st.markdown(
    '<div class="site-footer">Data: DharmicData corpus. Verse-level Sanskrit text.</div>',
    unsafe_allow_html=True,
)
