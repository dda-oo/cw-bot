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

os.environ["LANGCHAIN_TRACING_V2"] = "false"

st.set_page_config(page_title="HR Bot", layout="wide")
st.title("🤖 HR Q&A Bot")

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HUGGINGFACEHUB_API_TOKEN:
    st.error("Please set the Hugging Face API key in your environment.")
    st.stop()

with st.sidebar:
    st.header("📁 Upload HR Files")
    uploaded_files = st.file_uploader("Upload documents (PDF, TXT, DOCX)", type=["pdf", "txt", "docx"], accept_multiple_files=True)
    build_knowledge = st.button("🔧 Build Knowledge Base")

os.makedirs("temp_docs", exist_ok=True)

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

@st.cache_resource
def build_vector_store(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(texts, embeddings)
    return db

translator = Translator()

vector_store = None
if uploaded_files and build_knowledge:
    with st.spinner("Processing documents..."):
        docs = save_and_load(uploaded_files)
        vector_store = build_vector_store(docs)
    st.success("Knowledge base built!")

st.markdown("---")
st.subheader("💬 Ask your HR Bot")

user_query = st.text_input("Enter your question (any language)", key="query")

if user_query and vector_store:
    with st.spinner("Generating answer..."):
        detected_lang = detect(user_query)

        qa = RetrievalQA.from_chain_type(
            llm=HuggingFaceHub(repo_id="distilbert-base-uncased", model_kwargs={"temperature": 0.3, "max_length": 256}),
            retriever=vector_store.as_retriever(),
            return_source_documents=False
        )

        internal_query = user_query
        if detected_lang != "en":
            internal_query = translator.translate(user_query, src=detected_lang, dest="en").text

        answer = qa.run(internal_query)

        if detected_lang != "en":
            answer = translator.translate(answer, src="en", dest=detected_lang).text

    st.success(f"🤖 Answer: {answer}")
elif user_query:
    st.warning("⚠️ Please upload and build your knowledge base first.")
