import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langdetect import detect
from googletrans import Translator
import os

# Set up once
st.set_page_config(page_title="HR Bot", layout="wide")
st.title("🤖 HR Q&A Bot")

# Load API Key (Set in Streamlit Secrets or Environment)
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HUGGINGFACEHUB_API_TOKEN:
    st.error("Please set the Hugging Face API key in your environment.")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("📁 Upload HR Files")
    uploaded_files = st.file_uploader("Upload documents (PDF, TXT, DOCX)", type=["pdf", "txt", "docx"], accept_multiple_files=True)
    build_knowledge = st.button("🔧 Build Knowledge Base")

# Temporary directory to save uploads
os.makedirs("temp_docs", exist_ok=True)

# Save and load files
def save_and_load(files):
    documents = []
    for file in files:
        path = f"temp_docs/{file.name}"
        with open(path, "wb") as f:
            f.write(file.getbuffer())

        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif file.name.endswith(".txt"):
            loader = TextLoader(path)
        elif file.name.endswith(".docx"):
            loader = Docx2txtLoader(path)
        else:
            continue

        documents.extend(loader.load())
    return documents

# Build vector database
@st.cache_resource
def build_vector_store(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(texts, embeddings)
    return db

# Translate utility
translator = Translator()

# Load or build DB
vector_store = None
if uploaded_files and build_knowledge:
    with st.spinner("Processing documents..."):
        docs = save_and_load(uploaded_files)
        vector_store = build_vector_store(docs)
    st.success("Knowledge base built!")

# Ask the bot
st.markdown("---")
st.subheader("💬 Ask your HR Bot")

user_query = st.text_input("Enter your question (any language)", key="query")

if user_query and vector_store:
    with st.spinner("Generating answer..."):
        detected_lang = detect(user_query)

        # Create retriever chain
        qa = RetrievalQA.from_chain_type(
            llm=HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature": 0.3, "max_length": 256}),
            retriever=vector_store.as_retriever(),
            return_source_documents=False
        )

        # Ask in English internally if needed
        internal_query = user_query
        if detected_lang != "en":
            internal_query = translator.translate(user_query, src=detected_lang, dest="en").text

        answer = qa.run(internal_query)

        # Translate back if needed
        if detected_lang != "en":
            answer = translator.translate(answer, src="en", dest=detected_lang).text

    st.success(f"🤖 Answer: {answer}")
elif user_query:
    st.warning("⚠️ Please upload and build your knowledge base first.")
