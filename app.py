import streamlit as st
from langdetect import detect
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import requests
from io import BytesIO
import fitz  # PyMuPDF
from docx import Document
from pptx import Presentation

# Load multilingual model
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

model = load_model()

# Language detection
def detect_language(text):
    return detect(text)

# Translator
def translate(text, target):
    return GoogleTranslator(source='auto', target=target).translate(text)

# Extract text from different formats
def extract_text_from_url(url):
    response = requests.get(url)
    if response.status_code != 200:
        return ""

    content_type = response.headers.get('Content-Type', '')
    if "pdf" in content_type:
        return extract_text_from_pdf(BytesIO(response.content))
    elif "officedocument.wordprocessingml" in content_type:
        return extract_text_from_docx(BytesIO(response.content))
    elif "presentationml.presentation" in content_type:
        return extract_text_from_pptx(BytesIO(response.content))
    else:
        return "Unsupported file format"

def extract_text_from_pdf(file_stream):
    doc = fitz.open(stream=file_stream, filetype="pdf")
    return "\n".join(page.get_text() for page in doc)

def extract_text_from_docx(file_stream):
    doc = Document(file_stream)
    return "\n".join(p.text for p in doc.paragraphs)

def extract_text_from_pptx(file_stream):
    ppt = Presentation(file_stream)
    text = []
    for slide in ppt.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text.append(shape.text)
    return "\n".join(text)

# Similarity search
def find_best_answer(question, texts):
    question_embedding = model.encode([question])
    text_embeddings = model.encode(texts)
    similarities = cosine_similarity(question_embedding, text_embeddings)
    best_idx = similarities.argmax()
    return texts[best_idx]

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("Multilingual Knowledge Bot 🤖🌍")

st.markdown("Supports **PDF**, **Word**, and **PowerPoint** from OneDrive. Ask in any European language.")

onedrive_links = st.text_area("📎 Paste OneDrive Direct File URLs (one per line):")

question = st.text_input("💬 Ask your question:")
submit = st.button("Ask")

if submit and question:
    language = detect_language(question)
    st.write(f"🔤 Detected language: `{language}`")

    all_texts = []

    for link in onedrive_links.strip().splitlines():
        st.write(f"🔗 Loading: {link}")
        text = extract_text_from_url(link.strip())
        if text:
            chunks = [text[i:i+500] for i in range(0, len(text), 500)]
            all_texts.extend(chunks)

    if not all_texts:
        st.warning("No content extracted.")
    else:
        answer = find_best_answer(question, all_texts)
        translated_answer = translate(answer, language)
        st.success(f"🧠 Answer: {translated_answer}")
