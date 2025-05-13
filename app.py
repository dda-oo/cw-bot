import requests
import pdfplumber
from io import BytesIO
import docx
import os
import langid
from transformers import pipeline
import streamlit as st

# Function to load PDF files from a GitHub repository
def load_pdf_from_github(file_url):
    try:
        response = requests.get(file_url)
        if response.status_code == 200:
            with pdfplumber.open(BytesIO(response.content)) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text()
            return text
        else:
            return None
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

# Function to load DOCX files from a GitHub repository
def load_docx_from_github(file_url):
    try:
        response = requests.get(file_url)
        if response.status_code == 200:
            doc = docx.Document(BytesIO(response.content))
            text = ""
            for para in doc.paragraphs:
                text += para.text
            return text
        else:
            return None
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
        return None

# Function to get all files from a GitHub repository folder
def get_github_files(repo_url, folder_path):
    # Construct the raw GitHub API URL for the folder content
    api_url = f"https://api.github.com/repos/{repo_url}/contents/{folder_path}"
    response = requests.get(api_url)
    files = []
    
    if response.status_code == 200:
        content = response.json()
        for file in content:
            if file['type'] == 'file':  # Only consider files (not directories)
                files.append(file['download_url'])
    return files

# Function to load all documents (PDF, DOCX, etc.) from GitHub
def load_docs_from_github():
    repo_url = "dda-oo/cw-bot"  # Replace with your GitHub repo
    folder_path = "docs"  # Replace with the folder where your docs are stored
    
    # Get all files from the folder on GitHub
    file_urls = get_github_files(repo_url, folder_path)
    
    all_text = ""
    for url in file_urls:
        if url.endswith(".pdf"):
            all_text += load_pdf_from_github(url)
        elif url.endswith(".docx"):
            all_text += load_docx_from_github(url)
        # Add other formats as needed (e.g., PPTX, TXT)
    return all_text

# Function to detect language of a given text
def detect_language(text):
    lang, _ = langid.classify(text)
    return lang

# Set up a multilingual QA model (from Hugging Face)
qa_pipeline = pipeline("question-answering", model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Function to answer a question based on the documents
def answer_question(question, context):
    lang = detect_language(question)  # Detect language of the question
    print(f"Detected language: {lang}")
    
    # Translate if necessary or adjust the model for the detected language
    answer = qa_pipeline(question=question, context=context)
    return answer['answer']

# Streamlit UI for user input
st.title("Document-based QA System")
st.write("This app answers questions based on the documents in your GitHub repo.")

# Load the documents and set them as context
context = load_docs_from_github()  # Load the content from the documents

# User input for the question
question = st.text_input("Ask a question:")

if question:
    answer = answer_question(question, context)
    st.write(f"Answer: {answer}")
