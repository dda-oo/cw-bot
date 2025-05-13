import streamlit as st
import os
from langdetect import detect
from sentence_transformers import SentenceTransformer
import glob
import re
import numpy as np

# Page configuration
st.set_page_config(page_title="HR Assistant Bot", layout="centered")

# Cache the loading of models to improve performance
@st.cache_resource
def load_embedding_model():
    # Load only the embedding model which is more reliable
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

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
    file_paths = glob.glob("data/*.*")
    
    st.write(f"Found files: {file_paths}")
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
                # Split into paragraphs
                paragraphs = content.split('\n\n')
                
                # Process each paragraph
                for paragraph in paragraphs:
                    # Only include meaningful paragraphs
                    if len(paragraph.strip()) > 20:
                        # Clean text
                        clean_paragraph = re.sub(r'\s+', ' ', paragraph).strip()
                        
                        # Detect language
                        lang = detect_language(clean_paragraph)
                        
                        documents.append({
                            "text": clean_paragraph,
                            "source": os.path.basename(file_path),
                            "language": lang
                        })
        except Exception as e:
            st.error(f"Error loading {file_path}: {str(e)}")
    
    return documents

# Simple search function using sentence embeddings
def search_documents(query, documents, embedding_model, top_k=3):
    if not documents:
        return []
    
    # Create embeddings for query
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    query_embedding = query_embedding.cpu().numpy()
    
    # Calculate similarities and find top matches
    results = []
    for doc in documents:
        try:
            doc_embedding = embedding_model.encode(doc["text"], convert_to_tensor=True)
            doc_embedding = doc_embedding.cpu().numpy()
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            
            results.append((doc, similarity))
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
    
    # Sort by similarity (highest first)
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Return top_k results
    return [item[0] for item in results[:top_k]]

# Simple function to handle cross-language scenarios
def handle_language_difference(doc_text, doc_lang, query_lang):
    # If document language doesn't match query language, add a note
    if doc_lang != query_lang:
        if doc_lang == "de" and query_lang == "en":
            return f"{doc_text} [Translated from German]"
        elif doc_lang == "en" and query_lang == "de":
            return f"{doc_text} [Übersetzt aus dem Englischen]"
        else:
            return f"{doc_text} [Original in {doc_lang}]"
    return doc_text

# Main application
def main():
    st.title("HR Assistant Bot")
    
    # Load embedding model
    with st.spinner("Loading language model..."):
        embedding_model = load_embedding_model()
    
    # Load documents from data folder
    documents = load_documents()
    
    if not documents:
        st.warning("No documents found in the data folder. Please add HR documents to the data folder in your GitHub repository.")
    else:
        st.success(f"Loaded {len(documents)} paragraphs from HR documents.")
        
        # Display language distribution
        languages = [doc["language"] for doc in documents]
        language_counts = {}
        for lang in languages:
            if lang in language_counts:
                language_counts[lang] += 1
            else:
                language_counts[lang] = 1
        
        lang_display = ", ".join([f"{lang} ({count})" for lang, count in language_counts.items()])
        st.write(f"Document languages: {lang_display}")
    
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
                # Adapt response based on query language
                if query_lang == "de":
                    response = f"Hier ist, was ich in den HR-Dokumenten gefunden habe:\n\n"
                else:
                    response = f"Here's what I found in the HR documents:\n\n"
                
                # Process each relevant document
                for doc in relevant_docs:
                    doc_lang = doc["language"]
                    
                    # Display document with language note if needed
                    processed_text = handle_language_difference(doc["text"], doc_lang, query_lang)
                    response += f"- {processed_text}\n\n"
                
                # Add sources
                if query_lang == "de":
                    response += f"Quellen: {', '.join(set(doc['source'] for doc in relevant_docs))}"
                else:
                    response += f"Sources: {', '.join(set(doc['source'] for doc in relevant_docs))}"
            else:
                # No results found
                if query_lang == "de":
                    response = "Ich konnte keine spezifischen Informationen darüber in den HR-Dokumenten finden."
                else:
                    response = "I couldn't find specific information about this in the HR documents."
            
            st.write(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
