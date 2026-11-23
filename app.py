
import re
import html
import unicodedata
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="Scripture Explorer",
    page_icon="",
    layout="wide"
)

# CLEANING 

ACCENTS = re.compile(r"[\u0951-\u0954\u1CD0-\u1CFF\uA8E0-\uA8FF\u1CEA\u00AC]")

def clean(t):
    t = unicodedata.normalize("NFC", str(t))
    t = ACCENTS.sub("", t)
    t = re.sub(r"\[[^\]]*\]", " ", t)
    t = re.sub(r"[0-9०-९]+", " ", t)
    t = re.sub(r"[।॥|,.;:!?\-–—'\"()\[\]]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()

#  LOAD MODEL 
@st.cache_resource
def load_assets():
    clf = joblib.load("classifier.joblib")
    emb = np.load("embeddings.npy").astype(np.float32)
    idx = pd.read_csv("verses.csv")
    model = SentenceTransformer("intfloat/multilingual-e5-base")
    return clf, emb, idx, model

clf, emb, idx, model = load_assets()

SCRIPTURES = sorted(idx.scripture.unique())

#  PREDICTION 

def predict_scripture(verse):
    if not verse or not verse.strip():
        return {}

    p = clf.predict_proba([clean(verse)])[0]

    return {
        c: float(x)
        for c, x in zip(clf.classes_, p)
    }

# SEMANTIC SEARCH 

def search(query, top_k=5, scripture="All"):

    if not query or not query.strip():
        return pd.DataFrame()

    q = model.encode(
        ["query: " + query],
        normalize_embeddings=True
    )[0]

    sims = emb @ q

    if scripture != "All":
        sims = np.where(
            idx.scripture.values == scripture,
            sims,
            -1
        )

    top = np.argsort(-sims)[:int(top_k)]

    out = idx.iloc[top].copy()

    out["raw"] = out.raw.str.replace(
        r"\s+", " ", regex=True
    )

    out["translation"] = out.translation.fillna("")

    out.insert(0, "score", sims[top].round(3))

    return out.rename(
        columns={
            "raw": "verse",
            "translation": "English (Ramayana only)"
        }
    ).reset_index(drop=True)
# EXAMPLES 

VERSE_EXAMPLES = {
    "Rigveda | Hymn to Agni (1.1.1)":
        "अग्निमीळे पुरोहितं यज्ञस्य देवमृत्विजम्",

    "Ramayana | Valmiki asks Narada":
        "तपस्स्वाध्याय निरतं तपस्वी वाग्विदां वरम् । नारदं परिपप्रच्छ वाल्मीकिर्मुनिपुंगवम्",

    "Mahabharata | Ugrashrava at Naimisha":
        "लॊमहर्षणपुत्र उग्रश्रवाः सूतः पौराणिकॊ नैमिषारण्ये शौनकस्य कुलपतेर दवादशवार्षिके सत्रे",

    "Yajurveda | Opening invocation":
        "इषे त्वोर्जे त्वा वायव स्थ देवो वः सविता प्रार्पयतु श्रेष्ठतमाय कर्मणे",
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

#  THEME 

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;700&family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400&family=Tiro+Devanagari+Sanskrit&display=swap');

.stApp {
    background: linear-gradient(120deg,#f6e7c8,#f1d6a0,#f8ebd2,#eccb8c);
    color: #3b1d0e;
}

.block-container {
    max-width: 1050px;
    padding-top: 2rem;
}

h1, h2, h3 {
    font-family: 'Cinzel', serif !important;
    color: #4a0f0f !important;
}

div[data-testid="stTextArea"] textarea,
div[data-testid="stTextInput"] input {
    background: #fffaf0 !important;
    color: #3b1d0e !important;
    border: 1px solid #c9a15a !important;
    font-family: 'Tiro Devanagari Sanskrit', serif !important;
    font-size: 1.2rem !important;
}

button[kind="primary"] {
    background: linear-gradient(135deg,#6e1b1b,#4a0f0f) !important;
    color: #f6d98a !important;
    border: 1px solid #d4a017 !important;
}

.verse-card {
    margin: 14px 0;
    padding: 18px 20px;
    border: 1px solid #c9a15a;
    border-left: 4px solid #b4541a;
    border-radius: 6px;
    background: rgba(255,250,240,.95);
}

.scorebar {
    height: 10px;
    background: #ead9bd;
    border-radius: 8px;
    overflow: hidden;
    margin: 8px 0 12px;
}

.scorefill {
    height: 100%;
    background: linear-gradient(90deg,#8a2a0e,#d4a017);
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)



st.markdown("""
<div style="text-align:center;padding:28px 8px 10px;">
    <div style="font-size:3.2rem;color:#8a2a0e;
    font-family:'Tiro Devanagari Sanskrit',serif;">ॐ</div>

    <h1 style="letter-spacing:.15em;">SCRIPTURE EXPLORER</h1>

    <p style="font-size:1.3rem;color:#6b3a1e;font-style:italic;">
    Identify the source of a verse, or search by meaning across
    the Mahabharata, Ramayana, Rigveda and Yajurveda.
    </p>

    <hr style="width:55%;border-top:1px solid #b4541a;">
</div>
""", unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================

tab1, tab2 = st.tabs(["Identify a Verse", "Search by Meaning"])


with tab1:

    selected_verse = st.selectbox(
        "Choose an example verse",
        ["Select an example"] + list(VERSE_EXAMPLES.keys())
    )

    default_verse = VERSE_EXAMPLES.get(selected_verse, "")

    verse = st.text_area(
        "Verse (Devanagari)",
        value=default_verse,
        placeholder="Paste or type a Sanskrit verse here",
        height=140,
        key="verse_input"
    )

    if st.button("IDENTIFY", type="primary"):

        if not verse.strip():
            st.info("Choose an example verse or paste one in.")

        else:
            ranked = sorted(
                predict_scripture(verse).items(),
                key=lambda kv: -kv[1]
            )

            top, prob = ranked[0]

            st.markdown(f"""
            <div class="verse-card" style="text-align:center;">
                <div style="font-size:.85rem;letter-spacing:.18em;color:#7a2e12;">
                    MOST LIKELY SOURCE
                </div>
                <div style="font-size:2rem;color:#4a0f0f;font-family:'Cinzel',serif;">
                    {html.escape(str(top))}
                </div>
                <div style="font-size:1.1rem;color:#6b3a1e;font-style:italic;">
                    {prob*100:.1f}% confidence
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### Prediction Breakdown")

            for name, value in ranked:
                st.markdown(f"**{name} — {value*100:.1f}%**")
                st.progress(float(value))


with tab2:

    selected_query = st.selectbox(
        "Choose an example query",
        ["Select an example"] + list(QUERY_EXAMPLES.keys())
    )

    default_query, default_scripture = QUERY_EXAMPLES.get(
        selected_query, ("", "All")
    )

    query = st.text_input(
        "Query (Hindi or Sanskrit works best)",
        value=default_query,
        placeholder="धर्म का पालन करना चाहिए",
        key="query_input"
    )

    scripture_options = ["All"] + SCRIPTURES

    scripture = st.selectbox(
        "Restrict to",
        scripture_options,
        index=scripture_options.index(default_scripture)
        if default_scripture in scripture_options else 0
    )

    k = st.slider(
        "Number of results",
        min_value=1,
        max_value=10,
        value=5
    )

    if st.button("SEARCH", type="primary"):

        if not query.strip():
            st.info("Choose an example query or type one.")

        else:
            results = search(query, k, scripture)

            if results.empty:
                st.info("No verses found.")

            else:

                for _, row in results.iterrows():

                    score = float(row["score"])
                    scripture_name = str(row["scripture"])
                    ref = str(row["ref"])
                    verse_text = str(row["verse"])

                    translation = str(
                        row.get("English (Ramayana only)", "") or ""
                    )

                    safe_scripture = html.escape(scripture_name)
                    safe_ref = html.escape(ref)
                    safe_verse = html.escape(verse_text)
                    safe_translation = html.escape(translation)

                    st.markdown(f"""
                    <div class="verse-card">

                        <div style="display:flex;justify-content:space-between;
                        flex-wrap:wrap;gap:8px;">

                            <span style="background:#6e1b1b;color:#f6d98a;
                            padding:2px 10px;border-radius:3px;font-size:.75rem;
                            letter-spacing:.12em;">
                                {safe_scripture.upper()}
                            </span>

                            <span style="color:#6b3a1e;font-style:italic;">
                                {safe_ref}
                            </span>

                            <span style="color:#7a2e12;font-size:.8rem;">
                                SIMILARITY {score:.3f}
                            </span>
                        </div>

                        <div class="scorebar">
                            <div class="scorefill"
                            style="width:{max(0,min(100,score*100)):.1f}%;"></div>
                        </div>

                        <div style="font-family:'Tiro Devanagari Sanskrit',serif;
                        font-size:1.35rem;line-height:1.9;color:#2e1408;">
                            {safe_verse}
                        </div>

                        {f'<div style="margin-top:8px;font-style:italic;color:#5a3220;border-top:1px dashed #c9a15a;padding-top:8px;">{safe_translation}</div>' if translation.strip() else ""}

                    </div>
                    """, unsafe_allow_html=True)


st.markdown("""
<div style="text-align:center;color:#6b3a1e;font-style:italic;padding:20px 0;">
Data: DharmicData corpus. Verse-level Sanskrit text.
</div>
""", unsafe_allow_html=True)
