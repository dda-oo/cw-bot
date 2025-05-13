from docx import Document
import pdfplumber

def load_documents():
    documents = []
    file_paths = glob.glob("data/*.pdf") + glob.glob("data/*.docx")

    st.write(f"Found files: {file_paths}")

    for file_path in file_paths:
        try:
            if file_path.endswith(".pdf"):
                with pdfplumber.open(file_path) as pdf:
                    full_text = "\n\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
            elif file_path.endswith(".docx"):
                doc = Document(file_path)
                full_text = "\n\n".join(para.text for para in doc.paragraphs if para.text.strip())
            else:
                continue

            paragraphs = full_text.split('\n\n')

            for paragraph in paragraphs:
                if len(paragraph.strip()) > 20:
                    clean_paragraph = re.sub(r'\s+', ' ', paragraph).strip()
                    lang = detect_language(clean_paragraph)

                    documents.append({
                        "text": clean_paragraph,
                        "source": os.path.basename(file_path),
                        "language": lang
                    })

        except Exception as e:
            st.error(f"Error loading {file_path}: {str(e)}")

    return documents
