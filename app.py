import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from helper import load_texts, build_knowledge_base
import torch

# Page configuration
st.set_page_config(
    page_title="Multilingual FAQ Bot",
    layout="wide",
    page_icon="🧠"
)

st.title("🧠 Multilingual Internal FAQ Bot")
st.markdown("Ask questions in any language and get answers from your documents.")

# Initialize session state for language if not already set
if 'language' not in st.session_state:
    st.session_state.language = 'en'

# Language selector
languages = {
    'en': 'English',
    'de': 'Deutsch',
    'fr': 'Français',
    'es': 'Español',
    'zh': '中文',
}
selected_lang = st.sidebar.selectbox(
    "Interface Language",
    options=list(languages.keys()),
    format_func=lambda x: languages[x],
    index=list(languages.keys()).index(st.session_state.language)
)
st.session_state.language = selected_lang

# UI text based on selected language
ui_text = {
    'en': {
        'question_placeholder': 'Ask a question in any language...',
        'search_button': 'Search',
        'answer_label': 'Answer:',
        'source_label': 'Source:',
        'no_answer': 'No relevant information found. Please try rephrasing your question.',
        'loading': 'Loading model and processing documents...'
    },
    'de': {
        'question_placeholder': 'Stellen Sie eine Frage in beliebiger Sprache...',
        'search_button': 'Suchen',
        'answer_label': 'Antwort:',
        'source_label': 'Quelle:',
        'no_answer': 'Keine relevanten Informationen gefunden. Bitte versuchen Sie, Ihre Frage umzuformulieren.',
        'loading': 'Modell wird geladen und Dokumente werden verarbeitet...'
    },
    'fr': {
        'question_placeholder': 'Posez une question dans n\'importe quelle langue...',
        'search_button': 'Rechercher',
        'answer_label': 'Réponse:',
        'source_label': 'Source:',
        'no_answer': 'Aucune information pertinente trouvée. Veuillez essayer de reformuler votre question.',
        'loading': 'Chargement du modèle et traitement des documents...'
    },
    'es': {
        'question_placeholder': 'Haga una pregunta en cualquier idioma...',
        'search_button': 'Buscar',
        'answer_label': 'Respuesta:',
        'source_label': 'Fuente:',
        'no_answer': 'No se encontró información relevante. Intente reformular su pregunta.',
        'loading': 'Cargando modelo y procesando documentos...'
    },
    'zh': {
        'question_placeholder': '用任何语言提问...',
        'search_button': '搜索',
        'answer_label': '回答：',
        'source_label': '来源：',
        'no_answer': '未找到相关信息。请尝试重新表述您的问题。',
        'loading': '正在加载模型和处理文档...'
    }
}

texts = ui_text.get(selected_lang, ui_text['en'])

@st.cache_resource
def load_model_and_data():
    """Load the model and document data."""
    with st.spinner(texts['loading']):
        try:
            # Using multilingual model that supports 50+ languages
            model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            docs = load_texts("data")
            
            # Check if any documents were loaded
            if not docs:
                st.warning("No documents found in the data directory. Please add PDF or DOCX files.")
                # Return a model and dummy data to avoid crashes
                dummy_chunks = ["No documents found"]
                dummy_embeddings = np.zeros((1, 384))  # 384 is the embedding size for this model
                dummy_sources = ["None"]
                return model, dummy_chunks, dummy_embeddings, dummy_sources, True
                
            chunks, embeddings, sources = build_knowledge_base(docs, model)
            
            # Debug information
            st.sidebar.expander("Debug Info", expanded=False).write(f"Loaded {len(chunks)} chunks from {len(docs)} documents")
            
            return model, chunks, embeddings, sources, True
        except Exception as e:
            st.error(f"Error loading model or data: {str(e)}")
            return None, None, None, None, False

# Load model and data
model, chunks, embeddings, sources, loaded = load_model_and_data()

# Error handling for model loading
if not loaded:
    st.warning("Unable to load the model or documents. Please check your data directory and try again.")
    st.stop()

# Search functionality
col1, col2 = st.columns([4, 1])
with col1:
    question = st.text_input(
        "",
        placeholder=texts['question_placeholder'],
        key="question_input"
    )
with col2:
    search_button = st.button(texts['search_button'], use_container_width=True)

if question and (search_button or st.session_state.get('search_triggered', False)):
    st.session_state.search_triggered = False  # Reset trigger
    
    with st.spinner("Searching..."):
        try:
            # Check if embeddings are valid
            if len(embeddings) == 0 or (isinstance(embeddings, np.ndarray) and embeddings.size == 0):
                st.warning("No document embeddings available. Please check that your data directory contains valid documents.")
                st.stop()
                
            # Encode the question using the model
            q_embed = model.encode([question], convert_to_tensor=True)
            
            # Convert embeddings to numpy arrays for cosine similarity calculation
            q_embed_np = q_embed.cpu().numpy()
            embeddings_np = embeddings.cpu().numpy() if isinstance(embeddings, torch.Tensor) else embeddings
            
            # Debug shape information
            st.sidebar.expander("Debug Info", expanded=False).write(f"Query shape: {q_embed_np.shape}, Embeddings shape: {embeddings_np.shape}")
            
            # Safety check for dimensions
            if q_embed_np.shape[1] != embeddings_np.shape[1]:
                st.error(f"Dimension mismatch: Query dim {q_embed_np.shape[1]} != Embeddings dim {embeddings_np.shape[1]}")
                st.stop()
            
            # Calculate cosine similarity
            similarities = np.dot(q_embed_np, embeddings_np.T)[0] / (
                np.linalg.norm(q_embed_np) * np.linalg.norm(embeddings_np, axis=1) + 1e-8  # Add small epsilon to avoid division by zero
            )
            
            # Get top 3 results (or fewer if not enough chunks)
            num_results = min(3, len(chunks))
            if num_results == 0:
                st.warning("No document chunks available to search.")
                st.stop()
                
            top_indices = np.argsort(similarities)[::-1][:num_results]
            
            # Display results if similarity is above threshold
            if similarities[top_indices[0]] > 0.3:  # Minimum similarity threshold
                for i, idx in enumerate(top_indices):
                    similarity_score = similarities[idx]
                    if similarity_score > 0.3:  # Only show reasonably similar results
                        st.markdown(f"### {texts['answer_label']} {i+1}")
                        st.markdown(chunks[idx])
                        st.caption(f"{texts['source_label']} {sources[idx]} (Relevance: {similarity_score:.2f})")
            else:
                st.info(texts['no_answer'])
        except Exception as e:
            st.error(f"An error occurred during search: {str(e)}")
            st.error("Try changing your question or uploading different documents.")
            # Add detailed debug information in expandable section
            with st.expander("Technical Details", expanded=False):
                st.write(f"Error type: {type(e).__name__}")
                st.write(f"Model info: {type(model).__name__}")
                if chunks:
                    st.write(f"Number of chunks: {len(chunks)}")
                if embeddings is not None:
                    if isinstance(embeddings, np.ndarray):
                        st.write(f"Embeddings shape: {embeddings.shape}")
                    else:
                        st.write(f"Embeddings type: {type(embeddings)}")
                if sources:
                    st.write(f"Number of sources: {len(sources)}")
                # Don't expose full traceback to users

# Add sidebar with instructions and info
with st.sidebar:
    st.subheader("About")
    st.markdown("""
    This bot answers questions based on your internal documents.
    It can understand questions in multiple languages and find relevant information.
    """)
    
    st.subheader("How to use")
    st.markdown("""
    1. Type your question in any language
    2. Click the Search button
    3. View the most relevant answers from your documents
    """)
    
    # Add expandable technical details
    with st.expander("Technical Details"):
        st.markdown("""
        - Using SentenceTransformer model: paraphrase-multilingual-MiniLM-L12-v2
        - Supports 50+ languages
        - Documents are processed using semantic search
        - Answers are ranked by relevance score
        """)
