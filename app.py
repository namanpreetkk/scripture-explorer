import re
import html
import unicodedata
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Scripture Explorer", page_icon="🕉️", layout="wide")

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
