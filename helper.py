from sentence_transformers import SentenceTransformer
import numpy as np
import os
import PyPDF2
import docx
import streamlit as st
import re

def load_texts(data_path="data"):
    """
    Load text content from PDF and DOCX files in the specified directory.
    
    Args:
        data_path (str): Path to the directory containing documents
        
    Returns:
        list: List of dictionaries with source filename and extracted text
    """
    texts = []
    
    # Check if directory exists
    if not os.path.exists(data_path):
        st.error(f"Data directory '{data_path}' not found. Please create this directory and add your documents.")
        return texts
    
    # Get list of files
    try:
        files = os.listdir(data_path)
    except Exception as e:
        st.error(f"Error accessing data directory: {str(e)}")
        return texts
    
    # Process each file
    for filename in files:
        path = os.path.join(data_path, filename)
        
        try:
            # Skip directories and hidden files
            if os.path.isdir(path) or filename.startswith('.'):
                continue
                
            # Process PDF files
            if filename.lower().endswith(".pdf"):
                with open(path, "rb") as f:
                    try:
                        reader = PyPDF2.PdfReader(f)
                        text = "\n".join(page.extract_text() or "" for page in reader.pages)
                    except Exception as e:
                        st.warning(f"Error reading PDF file '{filename}': {str(e)}")
                        continue
            
            # Process DOCX files
            elif filename.lower().endswith(".docx"):
                try:
                    doc = docx.Document(path)
                    text = "\n".join([para.text for para in doc.paragraphs])
                except Exception as e:
                    st.warning(f"Error reading DOCX file '{filename}': {str(e)}")
                    continue
            
            # Skip other file types
            else:
                continue
            
            # Only add if we extracted meaningful text
            if text.strip():
                texts.append({"source": filename, "text": text})
                
        except Exception as e:
            st.warning(f"Error processing file '{filename}': {str(e)}")
    
    return texts

def chunk_text(text, min_chunk_size=100, max_chunk_size=500):
    """
    Split text into meaningful chunks based on paragraphs and sentence boundaries.
    
    Args:
        text (str): Text to chunk
        min_chunk_size (int): Minimum characters per chunk
        max_chunk_size (int): Maximum characters per chunk
        
    Returns:
        list: List of text chunks
    """
    # Split by paragraphs first
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        # If paragraph is very long, split it further by sentences
        if len(para) > max_chunk_size:
            # Simple sentence splitting (handles most common sentence endings)
            sentences = re.split(r'(?<=[.!?])\s+', para)
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= max_chunk_size:
                    current_chunk += " " + sentence if current_chunk else sentence
                else:
                    if current_chunk and len(current_chunk) >= min_chunk_size:
                        chunks.append(current_chunk)
                    current_chunk = sentence
        else:
            # For shorter paragraphs, check if adding would exceed max size
            if len(current_chunk) + len(para) <= max_chunk_size:
                current_chunk += " " + para if current_chunk else para
            else:
                if current_chunk and len(current_chunk) >= min_chunk_size:
                    chunks.append(current_chunk)
                current_chunk = para
    
    # Don't forget the last chunk
    if current_chunk and len(current_chunk) >= min_chunk_size:
        chunks.append(current_chunk)
    
    return chunks

def build_knowledge_base(texts, model):
    """
    Process texts into chunks and create embeddings.
    
    Args:
        texts (list): List of dictionaries with source and text
        model: SentenceTransformer model
        
    Returns:
        tuple: (chunks, embeddings, sources)
    """
    all_chunks = []
    sources = []
    
    # Process each document
    for item in texts:
        # Create meaningful chunks from the text
        doc_chunks = chunk_text(item["text"])
        
        # Add each chunk with its source
        for chunk in doc_chunks:
            if len(chunk.strip()) >= 50:  # Only add substantial chunks
                all_chunks.append(chunk)
                sources.append(item["source"])
    
    # Return empty results if no chunks were created
    if not all_chunks:
        return [], np.array([]), []
    
    # Create embeddings
    try:
        # Convert to numpy array immediately to avoid tensor conversion issues
        embeddings = model.encode(all_chunks, convert_to_numpy=True)
        return all_chunks, embeddings, sources
    except Exception as e:
        st.error(f"Error creating embeddings: {str(e)}")
        return [], np.array([]), []
