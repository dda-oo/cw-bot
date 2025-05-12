from docx import Document

def extract_text_from_word(word_file):
    doc = Document(word_file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text
