import requests
from io import BytesIO
from PyPDF2 import PdfReader
import docx
import langid
from transformers import pipeline
import streamlit as st

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

def load_docs():
    doc_urls = [
        "https://github.com/dda-oo/cw-bot/blob/84bc6ad19a0223f6790de5fce68d34053bfa934c/docs/QS%20Ordentliche%20virtuelle%20Hauptversammlung.docx",
        "https://github.com/dda-oo/cw-bot/blob/84bc6ad19a0223f6790de5fce68d34053bfa934c/docs/MAAP%202024_Planbedingungen.pdf",
          ]
    
    all_text = ""
    for url in doc_urls:
        if url.endswith(".pdf"):
            all_text += load_pdf_from_github(url)
        elif url.endswith(".docx"):
            all_text += load_docx_from_github(url)
    
    return all_text

def detect_language(text):
    lang, _ = langid.classify(text)
    return lang

qa_pipeline = pipeline("question-answering", model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def answer_question(question, context):
    lang = detect_language(question)  
    print(f"Detected language: {lang}")
    
    answer = qa_pipeline(question=question, context=context)
    return answer['answer']

st.title("Document-based QA System")
st.write("This app answers questions based on the documents in your GitHub repo.")

context = load_docs()  

question = st.text_input("Ask a question:")

if question:
    answer = answer_question(question, context)
    st.write(f"Answer: {answer}")
