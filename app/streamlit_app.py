"""Streamlit UI: paste a Bangla news article -> fake probability + SHAP word highlights.

    MODEL_ID=your-hf-user/banglabert-fake-news streamlit run app/streamlit_app.py
"""
import json
import os

import numpy as np
import shap
import streamlit as st
import streamlit.components.v1 as components
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = os.environ.get("MODEL_ID", "model/hf")
st.set_page_config(page_title="বাংলা ফেক নিউজ ডিটেক্টর", page_icon="📰", layout="wide")


@st.cache_resource
def load():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()
    try:
        p = os.path.join(MODEL_ID, "threshold.json") if os.path.isdir(MODEL_ID) else hf_hub_download(MODEL_ID, "threshold.json")
        thr = json.load(open(p))["threshold"]
    except Exception:
        thr = 0.5
    return tok, model, thr


tok, model, THR = load()


@torch.no_grad()
def proba(texts):
    e = tok([str(t) for t in texts], truncation=True, max_length=384, padding=True, return_tensors="pt")
    p = torch.softmax(model(**e).logits, -1).numpy()
    return p


st.title("📰 বাংলা ফেক নিউজ ডিটেক্টর")
st.caption("BanglaBERT fine-tuned on BanFakeNews + BanFakeNews-2.0 · SHAP দেখায় কোন শব্দগুলো সিদ্ধান্তে প্রভাব ফেলেছে")
text = st.text_area("শিরোনাম + খবরের লেখা পেস্ট করুন", height=220)
col1, col2 = st.columns([1, 3])
explain = col1.checkbox("SHAP ব্যাখ্যা দেখাও (একটু সময় লাগে)", value=True)
if col1.button("যাচাই করো", type="primary") and text.strip():
    p = float(proba([text])[0, 1])
    verdict = "🚩 সম্ভবত **ভুয়া (FAKE)**" if p >= THR else "✅ সম্ভবত **সঠিক (AUTHENTIC)**"
    col2.metric("P(fake)", f"{p:.1%}", help=f"threshold {THR:.2f}")
    col2.markdown(verdict)
    if explain:
        short = " ".join(text.split()[:150])
        with st.spinner("SHAP হিসাব হচ্ছে..."):
            ex = shap.Explainer(proba, shap.maskers.Text(tok), output_names=["AUTHENTIC", "FAKE"])
            sv = ex([short], max_evals=300)
        st.markdown("লাল = FAKE-এর দিকে ঠেলে, নীল = AUTHENTIC-এর দিকে")
        components.html(shap.plots.text(sv[0, :, 1], display=False), height=420, scrolling=True)
st.info("এটি একটি গবেষণা মডেল — চূড়ান্ত সিদ্ধান্তের আগে নির্ভরযোগ্য সূত্রে যাচাই করুন।")
