import requests
from io import BytesIO
from PyPDF2 import PdfReader
import docx
import langid
from transformers import pipeline
import streamlit as st

# Function to load PDF file from GitHub
def load_pdf_from_github(file_url):
    response = requests.get(file_url)
    if response.status_code == 200:
        pdf = PdfReader(BytesIO(response.content))
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
        return text
    else:
        return None

# Function to load DOCX file from GitHub
def load_docx_from_github(file_url):
    response = requests.get(file_url)
    if response.status_code == 200:
        doc = docx.Document(BytesIO(response.content))
        text = ""
        for para in doc.paragraphs:
            text += para.text
        return text
    else:
        return None

# Function to load all documents from GitHub
def load_docs():
    # Define the GitHub raw URLs for your files (change these to your actual file URLs)
    doc_urls = [
        "https://github.com/yourusername/yourrepo/raw/main/docs/document1.pdf",
        "https://github.com/yourusername/yourrepo/raw/main/docs/document2.docx",
        # Add more documents as needed
    ]
    
    # Extract text from each document (PDF, DOCX, etc.)
    all_text = ""
    for url in doc_urls:
        if url.endswith(".pdf"):
            all_text += load_pdf_from_github(url)
        elif url.endswith(".docx"):
            all_text += load_docx_from_github(url)
        # Add other formats like PPTX as needed.
    
    return all_text

# Function to detect language of a given text
def detect_language(text):
    lang, _ = langid.classify(text)
    return lang

# Set up a multilingual QA model (from Hugging Face)
qa_pipeline = pipeline("question-answering", model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Example function to answer a question based on the documents
def answer_question(question, context):
    lang = detect_language(question)  # Detect language of the question
    print(f"Detected language: {lang}")
    
    # Here, you can modify your model or use a translation API if needed.
    answer = qa_pipeline(question=question, context=context)
    return answer['answer']

# Streamlit UI for user input
st.title("Document-based QA System")
st.write("This app answers questions based on the documents in your GitHub repo.")

# Load the documents and set them as context
context = load_docs()  # Load the content from the documents

# User input for the question
question = st.text_input("Ask a question:")

if question:
    answer = answer_question(question, context)
    st.write(f"Answer: {answer}")
