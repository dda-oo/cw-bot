import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import os
from utils import extract_text_from_pdf, extract_text_from_word, process_text, get_answer

tokenizer = AutoTokenizer.from_pretrained("path_to_your_trained_model")
model = AutoModelForQuestionAnswering.from_pretrained("path_to_your_trained_model")

st.title("Internal Knowledge Bot")

uploaded_file = st.file_uploader("Upload a PDF or Word file", type=["pdf", "docx"])
if uploaded_file:
    # Extract text based on file type
    if uploaded_file.type == "application/pdf":
        text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = extract_text_from_word(uploaded_file)

    processed_text = process_text(text)

    st.write("Text processed successfully!")

    question = st.text_input("Ask a question:")

    if question:
        answer = get_answer(question, processed_text, tokenizer, model)
        st.write(f"Answer: {answer}")
