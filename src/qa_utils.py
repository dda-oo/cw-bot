import os
from langdetect import detect
from googletrans import Translator
from sentence_transformers import SentenceTransformer, util
import torch
import fitz  # PyMuPDF
from docx import Document

model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
embedder = SentenceTransformer(model_name)
translator = Translator()

def read_text_from_files(data_folder='data'):
    texts = []
    for filename in os.listdir(data_folder):
        filepath = os.path.join(data_folder, filename)
        if filename.endswith('.pdf'):
            with fitz.open(filepath) as doc:
                text = " ".join([page.get_text() for page in doc])
        elif filename.endswith('.docx'):
            doc = Document(filepath)
            text = "\n".join([para.text for para in doc.paragraphs])
        elif filename.endswith('.txt'):
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            continue
        texts.append({'text': text, 'filename': filename})
    return texts

def get_language(text):
    return detect(text)

def translate(text, dest='en'):
    return translator.translate(text, dest=dest).text

def build_knowledge_base(docs):
    sentences = []
    metadata = []
    for doc in docs:
        for sent in doc['text'].split('.'):
            cleaned = sent.strip()
            if cleaned:
                sentences.append(cleaned)
                metadata.append(doc['filename'])
    embeddings = embedder.encode(sentences, convert_to_tensor=True)
    return sentences, embeddings, metadata

def answer_question(question, sentences, embeddings, metadata):
    question_lang = get_language(question)
    question_en = translate(question, dest='en') if question_lang != 'en' else question
    question_embedding = embedder.encode(question_en, convert_to_tensor=True)
    top_k = 3
    hits = util.semantic_search(question_embedding, embeddings, top_k=top_k)[0]
    answers = []
    for hit in hits:
        answer_text = sentences[hit['corpus_id']]
        if question_lang != 'en':
            answer_text = translate(answer_text, dest=question_lang)
        answers.append(f"📝 {answer_text} (from {metadata[hit['corpus_id']]})")
    return answers
