import streamlit as st
from src.qa_utils import read_text_from_files, build_knowledge_base, answer_question

st.set_page_config(page_title="Multilingual HR Bot", layout="centered")
st.title("🤖 HR Question Answering Bot")

with st.spinner("Loading knowledge base..."):
    docs = read_text_from_files()
    sentences, embeddings, metadata = build_knowledge_base(docs)

st.success("Bot is ready! Ask your HR questions below 👇")

user_question = st.text_input("💬 Ask a question about HR documents (in any language):")

if user_question:
    with st.spinner("Thinking..."):
        answers = answer_question(user_question, sentences, embeddings, metadata)
        for ans in answers:
            st.write(ans)
