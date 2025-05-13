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
    # Encoding the question and converting it to a numpy array
    q_embed = model.encode([question], convert_to_tensor=False)  # Set convert_to_tensor=False to get a numpy array

    # Ensure that q_embed is 2D (it should have shape (1, embedding_dim))
    q_embed = q_embed.reshape(1, -1)  # Reshape to 2D if it's 1D

    # Convert embeddings to numpy arrays if they're in tensor form
    embeddings_np = embeddings.cpu().numpy() if isinstance(embeddings, torch.Tensor) else embeddings

    # Compute cosine similarity
    sims = cosine_similarity(q_embed, embeddings_np)[0]  # Ensure we're comparing numpy arrays
    top_idx = sims.argmax()  # Get the index of the most similar chunk

    # Display the answer
    st.markdown(f"**Antwort:** {chunks[top_idx]}")
    st.caption(f"Quelle: {sources[top_idx]}")
