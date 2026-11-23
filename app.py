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



@st.cache_resource

def load_assets():
    clf = joblib.load("classifier.joblib")
    emb = np.load("embeddings.npy").astype(np.float32)
    idx = pd.read_csv("verses.csv")
    model = SentenceTransformer("intfloat/multilingual-e5-base")
    return clf, emb, idx, model



clf, emb, idx, model = load_assets()
SCRIPTURES = sorted(idx.scripture.unique())


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
VERSE_EXAMPLES = {
    "Rigveda | Hymn to Agni (1.1.1)": "अग्निमीळे पुरोहितं यज्ञस्य देवमृत्विजम्",
    "Ramayana | Valmiki asks Narada": "तपस्स्वाध्याय निरतं तपस्वी वाग्विदां वरम् । नारदं परिपप्रच्छ वाल्मीकिर्मुनिपुंगवम्",
    "Mahabharata | Ugrashrava at Naimisha": "लॊमहर्षणपुत्र उग्रश्रवाः सूतः पौराणिकॊ नैमिषारण्ये शौनकस्य कुलपतेर दवादशवार्षिके सत्रे",
    "Yajurveda | Opening invocation": "इषे त्वोर्जे त्वा वायव स्थ देवो वः सविता प्रार्पयतु श्रेष्ठतमाय कर्मणे",
}

QUERY_EXAMPLES = {
    "One should follow dharma (Hindi)": ("धर्म का पालन करना चाहिए", "All"),
    "Speak truth, follow dharma (Sanskrit)": ("सत्यं वद धर्मं चर", "All"),
    "Praise of the fire god (Hindi)": ("अग्नि देवता की स्तुति", "All"),
    "Rama and Sita's exile (Ramayana)": ("राम और सीता का वनवास", "Ramayana"),
    "Bravery in war (Mahabharata)": ("युद्ध में वीरता", "Mahabharata"),
    "Soma sacrifice (Yajurveda)": ("सोम यज्ञ", "Yajurveda"),
    "Duty of a warrior (English)": ("duty of a warrior", "All"),
}


# Callbacks: examples reliably update the associated text input widgets.
def load_verse_example():
    name = st.session_state.get("verse_example", "Select an example")
    st.session_state["verse_input"] = VERSE_EXAMPLES.get(name, "")



def load_query_example():
    name = st.session_state.get("query_example", "Select an example")
    text, scripture = QUERY_EXAMPLES.get(name, ("", "All"))
    st.session_state["query_input"] = text
    st.session_state["scripture_filter"] = scripture


# -------------------- THEME / CONTRAST --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;700&family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400&family=Tiro+Devanagari+Sanskrit&display=swap');

.stApp {
    background: linear-gradient(120deg, #f6e7c8, #f1d6a0, #f8ebd2, #eccb8c);
    color: #3b1d0e !important;
}
.block-container { max-width: 1250px; padding-top: 1.5rem; padding-bottom: 2rem; }
[data-testid="stHeader"] { background: transparent; }

/* All standard text labels: dark ink on the light parchment background */
.stApp, .stApp p, .stApp label, .stApp span, .stApp li,
[data-testid="stMarkdownContainer"] p,
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label {
    color: #3b1d0e;
}
h1, h2, h3, h4 { font-family: 'Cinzel', serif !important; color: #4a0f0f !important; }

/* Tabs */
button[data-baseweb="tab"] { color: #4a0f0f !important; font-weight: 600 !important; }
button[data-baseweb="tab"] p { color: #4a0f0f !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: #8a2a0e !important; }

/* Input boxes, text areas, selectboxes */
div[data-testid="stTextArea"] textarea,
div[data-testid="stTextInput"] input,
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: #fffaf0 !important;
    color: #3b1d0e !important;
    border: 1px solid #c9a15a !important;
    font-family: 'Tiro Devanagari Sanskrit', serif !important;
    font-size: 1.08rem !important;
}
div[data-testid="stTextArea"] textarea { caret-color: #3b1d0e !important; }
div[data-testid="stTextArea"] textarea::placeholder,
div[data-testid="stTextInput"] input::placeholder,
div[data-testid="stTextArea"] textarea::-webkit-input-placeholder,
div[data-testid="stTextInput"] input::-webkit-input-placeholder {
    color: #76563a !important; opacity: 1 !important;
}
/* Dropdown menu options */
[role="listbox"], [role="option"], [data-baseweb="popover"] {
    background-color: #fffaf0 !important;
    color: #3b1d0e !important;
}
[role="option"] div, [role="option"] span, [role="option"] p { color: #3b1d0e !important; }

/* Buttons */
.stButton button, button[kind="primary"] {
    background: linear-gradient(135deg, #6e1b1b, #4a0f0f) !important;
    color: #f6d98a !important;
    border: 1px solid #d4a017 !important;
    border-radius: 7px !important;
}
.stButton button p, button[kind="primary"] p { color: #f6d98a !important; font-weight: 700 !important; }
.stButton button:hover { border-color: #8a2a0e !important; filter: brightness(1.08); }

/* Result cards */
.verse-card {
    margin: 14px 0; padding: 18px 20px;
    border: 1px solid #c9a15a; border-left: 4px solid #b4541a;
    border-radius: 9px; background: rgba(255,250,240,.97);
}
.scorebar { height: 10px; background: #ead9bd; border-radius: 8px; overflow: hidden; margin: 8px 0 12px; }
.scorefill { height: 100%; background: linear-gradient(90deg,#8a2a0e,#d4a017); border-radius: 8px; }
/* Ensure informational messages remain legible */
[data-testid="stAlert"] p { color: #24364b !important; }

/* Animated sacred-geometry backdrop behind the Om hero */
.scripture-hero {
    position: relative; overflow: hidden; text-align: center;
    padding: 50px 12px 28px; margin: 0 auto 22px;
    min-height: 300px; isolation: isolate;
}
.scripture-hero .mandala {
    position: absolute; z-index: -1; pointer-events: none;
    width: min(780px, 92vw); height: 390px;
    left: 50%; top: 48%; transform: translate(-50%, -50%);
    opacity: .28;
}
.scripture-hero .mandala svg { width: 100%; height: 100%; overflow: visible; }
.scripture-hero .orbit { transform-origin: 50% 50%; animation: mandala-spin 80s linear infinite; }
.scripture-hero .orbit-reverse { transform-origin: 50% 50%; animation: mandala-spin-reverse 110s linear infinite; }
.scripture-hero .pulse-ring { animation: ring-breathe 5s ease-in-out infinite; transform-origin: 50% 50%; }
.scripture-hero .hero-om { font-size: 4.2rem; line-height: 1.15; color: #8a2a0e;
    font-family: 'Tiro Devanagari Sanskrit', serif; animation: om-glow 4s ease-in-out infinite; }
.scripture-hero h1 { position: relative; letter-spacing: .22em; margin: 12px 0 18px; }
.scripture-hero .hero-subtitle { max-width: 900px; margin: 0 auto; color: #6b3a1e !important;
    font-size: 1.35rem; font-style: italic; line-height: 1.55; position: relative; }
.scripture-hero .hero-divider { width: 55%; border: 0; border-top: 1px solid #b4541a; margin: 25px auto 0; position: relative; }
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
</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown("""
<div class="scripture-hero">
  <div class="mandala" aria-hidden="true">
    <svg viewBox="0 0 900 430" fill="none" xmlns="http://www.w3.org/2000/svg">
      <g class="orbit" stroke="#b4541a" stroke-width="2.2" stroke-linecap="round">
        <circle class="pulse-ring" cx="450" cy="215" r="168" />
        <circle cx="450" cy="215" r="205" />
        <ellipse cx="450" cy="215" rx="78" ry="205" />
        <ellipse cx="450" cy="215" rx="145" ry="205" />
        <ellipse cx="450" cy="215" rx="205" ry="82" />
        <ellipse cx="450" cy="215" rx="205" ry="145" />
        <ellipse cx="450" cy="215" rx="205" ry="82" transform="rotate(30 450 215)" />
        <ellipse cx="450" cy="215" rx="205" ry="82" transform="rotate(60 450 215)" />
        <ellipse cx="450" cy="215" rx="205" ry="82" transform="rotate(90 450 215)" />
        <ellipse cx="450" cy="215" rx="205" ry="82" transform="rotate(120 450 215)" />
        <ellipse cx="450" cy="215" rx="205" ry="82" transform="rotate(150 450 215)" />
      </g>
      <g class="orbit-reverse" stroke="#c98c58" stroke-width="1.8" opacity=".75">
        <circle cx="450" cy="215" r="120" />
        <ellipse cx="450" cy="215" rx="120" ry="42" transform="rotate(45 450 215)" />
        <ellipse cx="450" cy="215" rx="120" ry="42" transform="rotate(-45 450 215)" />
        <ellipse cx="450" cy="215" rx="120" ry="42" transform="rotate(90 450 215)" />
      </g>
    </svg>
  </div>
  <span class="hero-speck s1"></span><span class="hero-speck s2"></span>
  <span class="hero-speck s3"></span><span class="hero-speck s4"></span>
  <div class="hero-om">ॐ</div>
  <h1>SCRIPTURE EXPLORER</h1>
  <p class="hero-subtitle">Identify the source of a verse, or search by meaning across the Mahabharata,<br>	he Ramayana, Rigveda and Yajurveda.</p>
  <hr class="hero-divider">
</div>
""", unsafe_allow_html=True)

# -------------------- TABS --------------------
tab1, tab2 = st.tabs(["Identify a Verse", "Search by Meaning"])


with tab1:
    st.selectbox(
        "Choose an example verse",
        ["Select an example"] + list(VERSE_EXAMPLES.keys()),
        key="verse_example",
        on_change=load_verse_example,
    )
    verse = st.text_area(
        "Verse (Devanagari)",
        placeholder="Paste or type a Sanskrit verse here",
        height=140,
        key="verse_input",
    )

    if st.button("IDENTIFY", type="primary", key="identify_button"):
        if not verse.strip():
            st.info("Choose an example verse or paste one in.")
        else:
            ranked = sorted(predict_scripture(verse).items(), key=lambda kv: -kv[1])
            if not ranked:
                st.warning("Unable to identify this verse with the loaded classifier.")
            else:
                top, prob = ranked[0]
                st.markdown(f"""
                <div class="verse-card" style="text-align:center;">
                    <div style="font-size:.85rem;letter-spacing:.18em;color:#7a2e12;">MOST LIKELY SOURCE</div>
                    <div style="font-size:2rem;color:#4a0f0f;font-family:'Cinzel',serif;">{html.escape(str(top))}</div>
                    <div style="font-size:1.1rem;color:#6b3a1e;font-style:italic;">{prob*100:.1f}% confidence</div>
                </div>
                """, unsafe_allow_html=True)
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
                    scripture_name = str(row["scripture"])
                    ref = str(row["ref"])
                    verse_text = str(row["verse"])
                    translation = str(row.get("English (Ramayana only)", "") or "")
                    safe_scripture = html.escape(scripture_name)
                    safe_ref = html.escape(ref)
                    safe_verse = html.escape(verse_text)
                    safe_translation = html.escape(translation)
                    st.markdown(f"""
                    <div class="verse-card">
                        <div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;align-items:center;">
                            <span style="background:#6e1b1b;color:#f6d98a;padding:2px 10px;border-radius:3px;font-size:.75rem;letter-spacing:.12em;">{safe_scripture.upper()}</span>
                            <span style="color:#6b3a1e;font-style:italic;">{safe_ref}</span>
                            <span style="color:#7a2e12;font-size:.8rem;">SIMILARITY {score:.3f}</span>
                        </div>
                        <div class="scorebar"><div class="scorefill" style="width:{max(0,min(100,score*100)):.1f}%;"></div></div>
                        <div style="font-family:'Tiro Devanagari Sanskrit',serif;font-size:1.35rem;line-height:1.9;color:#2e1408;">{safe_verse}</div>
                        {f'<div style="margin-top:8px;font-style:italic;color:#5a3220;border-top:1px dashed #c9a15a;padding-top:8px;">{safe_translation}</div>' if translation.strip() else ""}
                    </div>
                    """, unsafe_allow_html=True)

st.markdown("""\n<div style="text-align:center;color:#6b3a1e;font-style:italic;padding:20px 0;">
Data: DharmicData corpus. Verse-level Sanskrit text.
</div>
""", unsafe_allow_html=True)
