import streamlit as st
import os
import glob
import re
import numpy as np
from langdetect import detect
from sentence_transformers import SentenceTransformer
import pdfplumber
from docx import Document

# ----------------- Page Configuration -----------------
st.set_page_config(page_title="HR Assistant Bot", layout="centered")

# ----------------- Load Sentence Transformer -----------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# ----------------- Language Detection -----------------
def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

# ----------------- Load and Parse Documents -----------------
def load_documents():
    documents = []
    file_paths = glob.glob("data/*.pdf") + glob.glob("data/*.docx") + glob.glob("data/*.txt")
    st.write(f"Found files: {file_paths}")

    for file_path in file_paths:
        try:
            ext = os.path.splitext(file_path)[1].lower()
            content = ""

            if ext == ".pdf":
                with pdfplumber.open(file_path) as pdf:
                    content = "\n\n".join(page.extract_text() or "" for page in pdf.pages)

            elif ext == ".docx":
                doc = Document(file_path)
                content = "\n\n".join(para.text for para in doc.paragraphs)

            elif ext == ".txt":
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()

            # Split into paragraphs
            paragraphs = content.split('\n\n')
            for paragraph in paragraphs:
                clean_paragraph = re.sub(r'\s+', ' ', paragraph).strip()
                if len(clean_paragraph) > 20:
                    lang = detect_language(clean_paragraph)
                    documents.append({
                        "text": clean_paragraph,
                        "source": os.path.basename(file_path),
                        "language": lang
                    })

        except Exception as e:
            st.error(f"Error loading {file_path}: {str(e)}")

    return documents

# ----------------- Search Using Sentence Embeddings -----------------
def search_documents(query, documents, embedding_model, top_k=3):
    if not documents:
        return []

    query_embedding = embedding_model.encode(query, convert_to_tensor=True).cpu().numpy()
    results = []

    for doc in documents:
        try:
            doc_embedding = embedding_model.encode(doc["text"], convert_to_tensor=True).cpu().numpy()
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            results.append((doc, similarity))
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")

    results.sort(key=lambda x: x[1], reverse=True)
    return [item[0] for item in results[:top_k]]

# ----------------- Handle Multilingual Display -----------------
def handle_language_difference(doc_text, doc_lang, query_lang):
    if doc_lang != query_lang:
        if doc_lang == "de" and query_lang == "en":
            return f"{doc_text} [Translated from German]"
        elif doc_lang == "en" and query_lang == "de":
            return f"{doc_text} [Übersetzt aus dem Englischen]"
        else:
            return f"{doc_text} [Original in {doc_lang}]"
    return doc_text

# ----------------- Streamlit Main App -----------------
def main():
    st.title("🧑‍💼 HR Assistant Bot")

    # Load model
    with st.spinner("Loading language model..."):
        embedding_model = load_embedding_model()

    # Load documents
    documents = load_documents()

    if not documents:
        st.warning("No documents found in the data folder. Please upload PDF, DOCX, or TXT files to the /data folder.")
        return
    else:
        st.success(f"Loaded {len(documents)} relevant paragraphs.")
        lang_counts = {}
        for doc in documents:
            lang = doc["language"]
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        st.write("Detected languages: " + ", ".join(f"{lang} ({count})" for lang, count in lang_counts.items()))

    # Chat interface
    st.subheader("Ask your HR-related question:")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    query = st.chat_input("What do you want to know?")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            query_lang = detect_language(query)
            relevant_docs = search_documents(query, documents, embedding_model)

            if relevant_docs:
                intro = "Here's what I found:\n\n" if query_lang == "en" else "Hier ist, was ich gefunden habe:\n\n"
                response = intro

                for doc in relevant_docs:
                    processed_text = handle_language_difference(doc["text"], doc["language"], query_lang)
                    response += f"- {processed_text}\n\n"

                sources = ", ".join(set(doc["source"] for doc in relevant_docs))
                response += f"\nSources: {sources}" if query_lang == "en" else f"\nQuellen: {sources}"
            else:
                response = "No relevant information found." if query_lang == "en" else "Keine relevanten Informationen gefunden."

            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# ----------------- Run App -----------------
if __name__ == "__main__":
    main()
