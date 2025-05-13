import streamlit as st
import os
import glob
import re
import numpy as np
from langdetect import detect, LangDetectException
from sentence_transformers import SentenceTransformer
import pdfplumber
from docx import Document
from sklearn.metrics.pairwise import cosine_similarity
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="MAAP Assistant Bot",
    page_icon="🧑‍💼",
    layout="centered",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_embedding_model():
    """Load and cache the multilingual sentence transformer model"""
    with st.spinner("Loading language model (first run may take a minute)..."):
        return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def detect_language(text):
    """Detect the language of the given text with error handling"""
    try:
        if not text or len(text.strip()) < 10:
            return "en"  
        return detect(text)
    except LangDetectException:
        return "en"  

def extract_text_from_pdf(file_path):
    """Extract text from PDF files with error handling"""
    try:
        with pdfplumber.open(file_path) as pdf:
            return "\n\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception as e:
        logger.error(f"Error extracting text from PDF {file_path}: {str(e)}")
        return ""

def extract_text_from_docx(file_path):
    """Extract text from DOCX files with error handling"""
    try:
        doc = Document(file_path)
        return "\n\n".join(para.text for para in doc.paragraphs)
    except Exception as e:
        logger.error(f"Error extracting text from DOCX {file_path}: {str(e)}")
        return ""

def extract_text_from_txt(file_path):
    """Extract text from TXT files with error handling"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:

        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error reading TXT {file_path}: {str(e)}")
            return ""
    except Exception as e:
        logger.error(f"Error reading TXT {file_path}: {str(e)}")
        return ""

def load_documents():
    """Load all documents from the data folder and split into chunks"""
    documents = []
    file_paths = glob.glob("data/*.pdf") + glob.glob("data/*.docx") + glob.glob("data/*.txt")
    
    if not file_paths:
        logger.warning("No documents found in the data folder")
        return []
    
    logger.info(f"Found {len(file_paths)} files to process")
    
    for file_path in file_paths:
        try:
            ext = os.path.splitext(file_path)[1].lower()
            content = ""
            
            if ext == ".pdf":
                content = extract_text_from_pdf(file_path)
            elif ext == ".docx":
                content = extract_text_from_docx(file_path)
            elif ext == ".txt":
                content = extract_text_from_txt(file_path)
                
            if not content.strip():
                logger.warning(f"No content extracted from {file_path}")
                continue
                
            chunks = split_into_chunks(content)
            
            for chunk in chunks:
                if len(chunk) >= 20:  
                    lang = detect_language(chunk)
                    documents.append({
                        "text": chunk,
                        "source": os.path.basename(file_path),
                        "language": lang
                    })
                    
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
    
    return documents

def split_into_chunks(text, max_length=500):
    """Split text into manageable chunks for better search results"""

    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:

        if len(para) > max_length:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sentence in sentences:
                if len(sentence.strip()) > 0:
                    if len(current_chunk) + len(sentence) <= max_length:
                        current_chunk += " " + sentence if current_chunk else sentence
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence
        else:
            if len(current_chunk) + len(para) <= max_length:
                current_chunk += " " + para if current_chunk else para
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def search_documents(query, documents, embedding_model, top_k=3):
    """Search for relevant documents using semantic similarity"""
    if not documents:
        return []
    
    try:

        query_embedding = embedding_model.encode([query], convert_to_tensor=True).cpu().numpy()
        
        doc_texts = [doc["text"] for doc in documents]
        
        doc_embeddings = embedding_model.encode(doc_texts, convert_to_tensor=True).cpu().numpy()
        
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        
        results = [(documents[i], similarities[i]) for i in range(len(documents))]
        
        results.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Top result similarity: {results[0][1] if results else 'No results'}")
        
        filtered_results = [(doc, score) for doc, score in results if score > 0.3]
        
        return [item[0] for item in filtered_results[:top_k]]
    
    except Exception as e:
        logger.error(f"Error during document search: {str(e)}")
        return []

def handle_language_difference(doc_text, doc_lang, query_lang):
    """Add language information if document and query languages differ"""
    if doc_lang != query_lang:
        language_names = {
            "en": {"en": "English", "de": "Englisch"},
            "de": {"en": "German", "de": "Deutsch"},
            "fr": {"en": "French", "de": "Französisch"},
            "es": {"en": "Spanish", "de": "Spanisch"},
            "it": {"en": "Italian", "de": "Italienisch"}
        }
        
        if query_lang in ["en", "de"] and doc_lang in language_names:
            lang_name = language_names[doc_lang].get(query_lang, doc_lang)
            
            if query_lang == "en":
                return f"{doc_text}\n\n[Original in {lang_name}]"
            elif query_lang == "de":
                return f"{doc_text}\n\n[Original in {lang_name}]"
        
        return f"{doc_text}\n\n[Original in {doc_lang}]"
    
    return doc_text

def generate_response(query, relevant_docs, query_lang):
    """Generate a coherent response based on the retrieved documents"""
    if not relevant_docs:
        responses = {
            "en": "I couldn't find any relevant information in the HR documents. Could you rephrase your question?",
            "de": "Ich konnte keine relevanten Informationen in den HR-Dokumenten finden. Könnten Sie Ihre Frage umformulieren?",
        }
        return responses.get(query_lang, responses["en"])
    
    intros = {
        "en": "Based on our HR documents, I found this information:",
        "de": "Basierend auf unseren HR-Dokumenten habe ich folgende Informationen gefunden:",
    }
    
    sources = {
        "en": "Sources:",
        "de": "Quellen:",
    }
    
    response = intros.get(query_lang, intros["en"]) + "\n\n"
    
    for i, doc in enumerate(relevant_docs, 1):
        processed_text = handle_language_difference(doc["text"], doc["language"], query_lang)
        response += f"**{i}.** {processed_text}\n\n"
    
    source_text = sources.get(query_lang, sources["en"])
    unique_sources = set(doc["source"] for doc in relevant_docs)
    response += f"**{source_text}** {', '.join(unique_sources)}"
    
    return response

def sidebar():
    """Create a sidebar with information about the bot"""
    with st.sidebar:
        st.title("🧑‍💼 MAAP Bot Info")
        st.markdown("""
        ### How to use:
        1. Ask questions in English or German
        2. The bot will search through the documents and find the most relevant information
        
        ### Supported file types:
        - PDF (.pdf)
        - Word (.docx)
        - Text (.txt)
        """)
        
        if "documents" in st.session_state and st.session_state.documents:
            st.subheader("Document Statistics")
            docs = st.session_state.documents
            total_chunks = len(docs)
            lang_counts = {}
            file_counts = {}
            
            for doc in docs:
                lang = doc["language"]
                source = doc["source"]
                lang_counts[lang] = lang_counts.get(lang, 0) + 1
                file_counts[source] = file_counts.get(source, 0) + 1
            
            st.markdown(f"**Total text chunks:** {total_chunks}")
            
            st.markdown("**Languages detected:**")
            for lang, count in lang_counts.items():
                st.markdown(f"- {lang}: {count} chunks")
            
            st.markdown("**Source files:**")
            for source, count in file_counts.items():
                st.markdown(f"- {source}: {count} chunks")
            
            if st.button("Clear Chat History"):
                st.session_state.messages = []
                st.experimental_rerun()

def main():
    st.title("🧑‍💼 MAAP Assistant Bot")
    
    sidebar()
    
    embedding_model = load_embedding_model()
    
    if "documents" not in st.session_state:
        with st.spinner("Loading and processing HR documents..."):
            st.session_state.documents = load_documents()
    
    if not st.session_state.documents:
        st.warning("⚠️ No documents found in the data folder. Please upload PDF, DOCX, or TXT files to the /data folder.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    query = st.chat_input("Ask your HR-related question...")
    
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        
        with st.chat_message("user"):
            st.markdown(query)
        
        with st.chat_message("assistant"):
            with st.spinner("Searching HR documents..."):
                query_lang = detect_language(query)
                
                relevant_docs = search_documents(
                    query, 
                    st.session_state.documents, 
                    embedding_model, 
                    top_k=3
                )
                
                response = generate_response(query, relevant_docs, query_lang)
                
                st.markdown(response)
                
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
