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
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------- Page Configuration -----------------
st.set_page_config(
    page_title="HR Assistant Bot",
    page_icon="🧑‍💼",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Add debugging messages in UI
DEBUG_MODE = True

# ----------------- Load Sentence Transformer -----------------
@st.cache_resource
def load_embedding_model():
    """Load and cache the multilingual sentence transformer model"""
    # Use a smaller, faster model for better performance
    start_time = time.time()
    model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    load_time = time.time() - start_time
    if DEBUG_MODE:
        st.sidebar.info(f"Model loaded in {load_time:.2f} seconds")
    return model

# ----------------- Language Detection -----------------
def detect_language(text):
    """Detect the language of the given text with error handling"""
    try:
        if not text or len(text.strip()) < 10:
            return "en"  # Default to English for short text
        return detect(text)
    except LangDetectException:
        return "en"  # Default to English on error

# ----------------- Document Loading -----------------
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
        # Try with different encoding if UTF-8 fails
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error reading TXT {file_path}: {str(e)}")
            return ""
    except Exception as e:
        logger.error(f"Error reading TXT {file_path}: {str(e)}")
        return ""

# ----------------- Load and Parse Documents -----------------
def load_documents():
    """Load all documents from the data folder and split into chunks"""
    documents = []
    file_paths = glob.glob("data/*.pdf") + glob.glob("data/*.docx") + glob.glob("data/*.txt")
    
    if not file_paths:
        logger.warning("No documents found in the data folder")
        if DEBUG_MODE:
            st.sidebar.warning("No documents found in the data folder")
        return []
    
    if DEBUG_MODE:
        st.sidebar.info(f"Found {len(file_paths)} files to process")
    
    total_chunks = 0
    for file_path in file_paths:
        try:
            if DEBUG_MODE:
                st.sidebar.info(f"Processing: {file_path}")
                
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
                if DEBUG_MODE:
                    st.sidebar.warning(f"No content extracted from {file_path}")
                continue
                
            # Split into chunks - paragraphs or sentences for better search results
            chunks = split_into_chunks(content)
            
            file_chunks = 0
            for chunk in chunks:
                if len(chunk) >= 20:  # Only include meaningful chunks
                    lang = detect_language(chunk)
                    documents.append({
                        "text": chunk,
                        "source": os.path.basename(file_path),
                        "language": lang
                    })
                    file_chunks += 1
            
            total_chunks += file_chunks
            if DEBUG_MODE:
                st.sidebar.success(f"Added {file_chunks} chunks from {os.path.basename(file_path)}")
                    
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            if DEBUG_MODE:
                st.sidebar.error(f"Error processing {file_path}: {str(e)}")
    
    if DEBUG_MODE:
        st.sidebar.success(f"Total chunks processed: {total_chunks}")
    
    return documents

def split_into_chunks(text, max_length=300):  # Reduced from 500 to 300 for better performance
    """Split text into manageable chunks for better search results"""
    # First try to split by paragraphs
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        # If paragraph is very long, split it into sentences
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

# ----------------- Search Using Sentence Embeddings -----------------
def search_documents(query, documents, embedding_model, top_k=3):
    """Search for relevant documents using semantic similarity"""
    if not documents:
        return []
    
    try:
        start_time = time.time()
        
        # Limit the search to optimize performance
        max_docs_to_search = min(100, len(documents))  # Only search through first 100 documents
        docs_to_search = documents[:max_docs_to_search]
        
        # Convert query to embedding vector
        query_embedding = embedding_model.encode(query, convert_to_tensor=False)
        
        # Get all document texts
        doc_texts = [doc["text"] for doc in docs_to_search]
        
        # Convert all documents to embedding vectors in one batch (more efficient)
        doc_embeddings = embedding_model.encode(doc_texts, convert_to_tensor=False, show_progress_bar=False)
        
        # Calculate similarities
        similarities = []
        for i, doc_embedding in enumerate(doc_embeddings):
            # Calculate cosine similarity manually to avoid memory issues
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            similarities.append((docs_to_search[i], similarity))
        
        # Sort by similarity score (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Filter results with low relevance
        filtered_results = [(doc, score) for doc, score in similarities if score > 0.3]
        
        search_time = time.time() - start_time
        if DEBUG_MODE:
            st.sidebar.info(f"Search completed in {search_time:.2f} seconds")
            if filtered_results:
                st.sidebar.info(f"Top result similarity: {filtered_results[0][1]:.4f}")
        
        # Return top_k results
        return [item[0] for item in filtered_results[:top_k]]
    
    except Exception as e:
        logger.error(f"Error during document search: {str(e)}")
        if DEBUG_MODE:
            st.sidebar.error(f"Search error: {str(e)}")
        return []

# ----------------- Handle Multilingual Display -----------------
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
        
        # Default case
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
    
    # Prepare introduction based on language
    intros = {
        "en": "Based on our HR documents, I found this information:",
        "de": "Basierend auf unseren HR-Dokumenten habe ich folgende Informationen gefunden:",
    }
    
    sources = {
        "en": "Sources:",
        "de": "Quellen:",
    }
    
    response = intros.get(query_lang, intros["en"]) + "\n\n"
    
    # Add each relevant document with proper formatting
    for i, doc in enumerate(relevant_docs, 1):
        processed_text = handle_language_difference(doc["text"], doc["language"], query_lang)
        response += f"**{i}.** {processed_text}\n\n"
    
    # Add sources
    source_text = sources.get(query_lang, sources["en"])
    unique_sources = set(doc["source"] for doc in relevant_docs)
    response += f"**{source_text}** {', '.join(unique_sources)}"
    
    return response

# ----------------- File Uploader -----------------
def upload_files():
    """Allow user to upload files directly from the UI"""
    uploaded_files = st.file_uploader("Upload HR documents", 
                                    type=["pdf", "docx", "txt"], 
                                    accept_multiple_files=True)
    
    if uploaded_files:
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        for uploaded_file in uploaded_files:
            # Save uploaded file to data directory
            with open(os.path.join("data", uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        st.success(f"Successfully uploaded {len(uploaded_files)} files")
        # Clear session state to reload documents
        if "documents" in st.session_state:
            del st.session_state.documents
        st.experimental_rerun()

# ----------------- Streamlit Sidebar -----------------
def sidebar():
    """Create a sidebar with information about the bot"""
    with st.sidebar:
        st.title("🧑‍💼 HR Bot Info")
        
        # File uploader
        st.subheader("Upload Documents")
        upload_files()
        
        st.markdown("""
        ### How to use:
        1. Upload your HR documents using the uploader above
        2. Ask questions in English or German
        3. The bot will search through the documents and find the most relevant information
        """)
        
        # Debug section
        if DEBUG_MODE:
            st.subheader("Debug Information")
            st.markdown("This section shows processing details")
        
        # Show document stats
        if "documents" in st.session_state and st.session_state.documents:
            st.subheader("Document Statistics")
            docs = st.session_state.documents
            total_chunks = len(docs)
            
            if total_chunks > 0:
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

# ----------------- Streamlit Main App -----------------
def main():
    st.title("🧑‍💼 HR Assistant Bot")
    
    # Create sidebar
    sidebar()
    
    # Load model
    with st.spinner("Loading language model..."):
        embedding_model = load_embedding_model()
    
    # Load documents (only once when app starts)
    if "documents" not in st.session_state:
        with st.spinner("Loading and processing HR documents..."):
            st.session_state.documents = load_documents()
    
    # Check if documents were loaded
    if not st.session_state.documents:
        st.info("📄 Please upload your HR documents using the uploader in the sidebar to get started.")
    else:
        st.success(f"✅ {len(st.session_state.documents)} text chunks loaded and ready for searching.")
    
    # Initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display message history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Get user query
    query = st.chat_input("Ask your HR-related question...")
    
    if query:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(query)
        
        # Process query and generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching HR documents..."):
                # Detect query language
                query_lang = detect_language(query)
                
                # Search for relevant documents
                relevant_docs = search_documents(
                    query, 
                    st.session_state.documents, 
                    embedding_model, 
                    top_k=3
                )
                
                # Generate response
                response = generate_response(query, relevant_docs, query_lang)
                
                # Display response
                st.markdown(response)
                
                # Add assistant message to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})

# ----------------- Run App -----------------
if __name__ == "__main__":
    main()
