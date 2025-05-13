import streamlit as st
import os
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from langdetect import detect
import glob
import re

# Page configuration
st.set_page_config(page_title="Simple HR Assistant", layout="centered")

# Cache the loading of models to improve performance
@st.cache_resource
def load_models():
    # Load pre-trained models from Hugging Face
    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    # Load translation models
    en_de_translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-de")
    de_en_translator = pipeline("translation", model="Helsinki-NLP/opus-mt-de-en")
    
    return embedding_model, en_de_translator, de_en_translator

# Function to detect language
def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"  # Default to English if detection fails

# Load documents from data folder
def load_documents():
    documents = []
    
    # Get all .txt, .md files from the data directory
    file_paths = glob.glob("data/*.txt") + glob.glob("data/*.md")
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
                # Split into paragraphs
                paragraphs = content.split('\n\n')
                for paragraph in paragraphs:
                    if len(paragraph.strip()) > 20:  # Only include meaningful paragraphs
                        documents.append({
                            "text": paragraph.strip(),
                            "source": os.path.basename(file_path),
                            "language": detect_language(paragraph)
                        })
        except Exception as e:
            st.error(f"Error loading {file_path}: {str(e)}")
    
    return documents

# Simple search function using sentence embeddings
def search_documents(query, documents, embedding_model, top_k=3):
    if not documents:
        return []
    
    # Create embeddings
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    
    # Calculate similarities and find top matches
    results = []
    for doc in documents:
        doc_embedding = embedding_model.encode(doc["text"], convert_to_tensor=True)
        similarity = query_embedding @ doc_embedding.T  # Cosine similarity
        results.append((doc, similarity.item()))
    
    # Sort by similarity (highest first)
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Return top_k results
    return [item[0] for item in results[:top_k]]

# Translate text based on source and target language
def translate_text(text, source_lang, target_lang, en_de_translator, de_en_translator):
    if source_lang == target_lang:
        return text
    
    try:
        # Currently handling German-English translation pairs
        if source_lang == "de" and target_lang == "en":
            result = de_en_translator(text, max_length=512)
            return result[0]['translation_text']
        elif source_lang == "en" and target_lang == "de":
            result = en_de_translator(text, max_length=512)
            return result[0]['translation_text']
        else:
            # For unsupported language pairs, just return the original
            return text
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return text

# Main application
def main():
    st.title("Simple HR Assistant")
    
    # Load models
    with st.spinner("Loading models from Hugging Face..."):
        embedding_model, en_de_translator, de_en_translator = load_models()
    
    # Load documents from data folder
    documents = load_documents()
    
    if not documents:
        st.warning("No documents found in the data folder. Please add HR documents to the data folder in your GitHub repository.")
    else:
        st.success(f"Loaded {len(documents)} paragraphs from HR documents.")
    
    # Chat interface
    st.subheader("Ask your HR question")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Get user input
    query = st.chat_input("What would you like to know about HR policies?")
    
    if query:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Display user message
        with st.chat_message("user"):
            st.write(query)
        
        # Process query
        with st.chat_message("assistant"):
            # Detect language of the query
            query_lang = detect_language(query)
            
            # Find relevant documents
            relevant_docs = search_documents(query, documents, embedding_model)
            
            if relevant_docs:
                response = f"Here's what I found in the HR documents:\n\n"
                
                # Process each relevant document
                for doc in relevant_docs:
                    doc_lang = doc["language"]
                    
                    # If document is in a different language, translate it
                    if doc_lang != query_lang:
                        translated_text = translate_text(
                            doc["text"], 
                            doc_lang, 
                            query_lang,
                            en_de_translator,
                            de_en_translator
                        )
                        response += f"- {translated_text}\n\n"
                    else:
                        response += f"- {doc['text']}\n\n"
                
                response += f"Sources: {', '.join(set(doc['source'] for doc in relevant_docs))}"
            else:
                response = "I couldn't find specific information about this in the HR documents."
            
            st.write(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
