import streamlit as st
import os
import base64
import PyPDF2
import docx

st.set_page_config(page_title="Document Uploader", layout="wide")

st.title("📁 Document Uploader for Multilingual FAQ Bot")

st.markdown("""
This utility helps you upload documents to the FAQ bot's data directory. 
Upload PDF, DOCX, or TXT files that contain the knowledge your bot should use to answer questions.
""")

# Create data directory if it doesn't exist
if not os.path.exists("data"):
    os.makedirs("data")
    st.success("Created 'data' directory for document storage")

# File uploader
uploaded_files = st.file_uploader(
    "Upload documents (PDF, DOCX, TXT)", 
    type=["pdf", "docx", "txt"], 
    accept_multiple_files=True
)

# Handle uploaded files
if uploaded_files:
    for uploaded_file in uploaded_files:
        # Save the file to the data directory
        file_path = os.path.join("data", uploaded_file.name)
        
        try:
            # Write the file
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Validate the file by trying to open it
            if uploaded_file.name.lower().endswith(".pdf"):
                with open(file_path, "rb") as f:
                    try:
                        reader = PyPDF2.PdfReader(f)
                        pages = len(reader.pages)
                        st.success(f"✅ Saved and validated '{uploaded_file.name}' ({pages} pages)")
                    except Exception as e:
                        st.error(f"❌ File '{uploaded_file.name}' was saved but may be corrupted: {str(e)}")
            
            elif uploaded_file.name.lower().endswith(".docx"):
                try:
                    doc = docx.Document(file_path)
                    paragraphs = len(doc.paragraphs)
                    st.success(f"✅ Saved and validated '{uploaded_file.name}' ({paragraphs} paragraphs)")
                except Exception as e:
                    st.error(f"❌ File '{uploaded_file.name}' was saved but may be corrupted: {str(e)}")
            
            else:  # TXT files
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = len(f.readlines())
                    st.success(f"✅ Saved and validated '{uploaded_file.name}' ({lines} lines)")
                except UnicodeDecodeError:
                    # Try with different encoding
                    try:
                        with open(file_path, 'r', encoding='latin-1') as f:
                            lines = len(f.readlines())
                        st.success(f"✅ Saved '{uploaded_file.name}' with Latin-1 encoding ({lines} lines)")
                    except Exception as e:
                        st.error(f"❌ File '{uploaded_file.name}' was saved but has encoding issues: {str(e)}")
                except Exception as e:
                    st.error(f"❌ File '{uploaded_file.name}' was saved but may be corrupted: {str(e)}")
                    
        except Exception as e:
            st.error(f"❌ Error saving '{uploaded_file.name}': {str(e)}")

# List existing files
st.subheader("Currently Available Documents")
try:
    files = os.listdir("data")
    files = [f for f in files if f.lower().endswith(('.pdf', '.docx', '.txt')) and not f.startswith('.')]
    
    if not files:
        st.info("No documents found. Please upload some files.")
    else:
        for file in files:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"📄 {file}")
            with col2:
                if st.button("Delete", key=f"del_{file}"):
                    try:
                        os.remove(os.path.join("data", file))
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error deleting {file}: {str(e)}")
except Exception as e:
    st.error(f"Error listing files: {str(e)}")

# Instructions
with st.expander("How to use the Multilingual FAQ Bot"):
    st.markdown("""
    ### Setting up your FAQ Bot
    
    1. **Upload documents**: Use this utility to upload PDF, DOCX, or TXT files containing information your bot should use.
    
    2. **Run the main app**: Go to the main app to start asking questions in any language.
    
    3. **Important notes**:
       - The bot works best with well-structured documents
       - Upload documents in any language (German, English, etc.)
       - You can ask questions in any language, regardless of the document language
       - The more specific your questions, the better the answers
    """)
