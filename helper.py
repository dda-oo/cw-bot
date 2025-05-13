import os
import pdfplumber
from docx import Document
import glob

# Function to load all document texts from a directory
def load_texts(data_dir):
    texts = []
    files = glob.glob(f"{data_dir}/*")  # Get all files in the directory
    for file in files:
        if file.endswith(".pdf"):
            texts.append(load_pdf(file))
        elif file.endswith(".docx"):
            texts.append(load_docx(file))
        # Add other file types (PPTX, TXT) if needed
    return texts

# Function to read text from PDF files
def load_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
    return text

# Function to read text from DOCX files
def load_docx(docx_path):
    text = ""
    try:
        doc = Document(docx_path)
        for para in doc.paragraphs:
            text += para.text
    except Exception as e:
        print(f"Error reading DOCX {docx_path}: {e}")
    return text

# Function to build a knowledge base from the loaded texts
def build_knowledge_base(texts, model):
    chunks = []
    embeddings = []
    sources = []

    for i, text in enumerate(texts):
        # Split the text into smaller chunks (you can adjust this part)
        chunk_size = 500  # Define chunk size
        text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        for chunk in text_chunks:
            chunks.append(chunk)
            embeddings.append(model.encode([chunk], convert_to_tensor=True))
            sources.append(f"Document {i+1}")

    # Convert embeddings list to tensor
    embeddings_tensor = torch.stack(embeddings)
    return chunks, embeddings_tensor, sources
