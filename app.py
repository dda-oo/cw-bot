import streamlit as st
from sentence_transformers import SentenceTransformer
from helper import load_texts, build_knowledge_base
from sklearn.metrics.pairwise import cosine_similarity
import torch

st.set_page_config(page_title="Multilingual FAQ Bot", layout="wide")
st.title("🧠 Multilingual Internal FAQ Bot")

@st.cache_resource
def load_model_and_data():
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    texts = load_texts("data")
    chunks, embeddings, sources = build_knowledge_base(texts, model)
    return model, chunks, embeddings, sources

model, chunks, embeddings, sources = load_model_and_data()

question = st.text_input("Frage stellen (z. B. auf Deutsch, Englisch, etc.):")

if question:
    q_embed = model.encode([question], convert_to_tensor=True)
    sims = cosine_similarity(q_embed.cpu(), embeddings.cpu())[0]
    top_idx = sims.argmax()
    st.markdown(f"**Antwort:** {chunks[top_idx]}")
    st.caption(f"Quelle: {sources[top_idx]}")
