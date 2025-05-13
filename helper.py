from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os, docx, PyPDF2

def load_texts(data_path="data"):
    texts = []
    for filename in os.listdir(data_path):
        path = os.path.join(data_path, filename)
        if filename.endswith(".pdf"):
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
        elif filename.endswith(".docx"):
            doc = docx.Document(path)
            text = "\n".join([para.text for para in doc.paragraphs])
        else:
            continue
        texts.append({"source": filename, "text": text})
    return texts

def build_knowledge_base(texts, model):
    all_chunks = []
    sources = []
    for item in texts:
        lines = item["text"].split("\n")
        for line in lines:
            line = line.strip()
            if len(line) > 20:
                all_chunks.append(line)
                sources.append(item["source"])
    embeddings = model.encode(all_chunks, convert_to_tensor=True)
    return all_chunks, embeddings, sources
