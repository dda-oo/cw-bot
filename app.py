import streamlit as st
import requests
from io import BytesIO
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util

st.title("📚 Company Knowledge Bot")
st.write("Ask anything based on internal instructions.")

# Load model
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

model = load_model()

# Example OneDrive file (PDF)
onedrive_link = st.secrets["onedrive"]["public_folder_url"]

# Convert public OneDrive share link to direct download
def get_direct_link(share_link):
    if "onedrive.live.com" in share_link:
        return share_link.replace("redir?", "download?").replace("?", "&download=1")
    return share_link

# Load and extract text
def load_pdf_from_onedrive(link):
    response = requests.get(link)
    pdf = PdfReader(BytesIO(response.content))
    return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

# Embed docs
@st.cache_data
def load_docs():
    text = load_pdf_from_onedrive(get_direct_link(onedrive_link))
    chunks = text.split("\n\n")
    embeddings = model.encode(chunks, convert_to_tensor=True)
    return chunks, embeddings

chunks, embeddings = load_docs()

# Ask a question
question = st.text_input("Ask a question")

if question:
    question_embedding = model.encode(question, convert_to_tensor=True)
    scores = util.cos_sim(question_embedding, embeddings)[0]
    top_k = scores.topk(3)

    st.write("### Top Answers:")
    for score, idx in zip(top_k.values, top_k.indices):
        st.markdown(f"**Score**: {float(score):.2f}")
        st.write(chunks[idx])
