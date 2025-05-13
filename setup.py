# Debugging/Setup Script for Multilingual Bot
# Run this file once to check your data directory and create a test document if needed

import os
import sys
import streamlit as st

st.set_page_config(page_title="Setup Debug", layout="wide")
st.title("🛠️ Setup and Debug")

# Check Python version
st.write(f"Python version: {sys.version}")

# Check if data directory exists
data_dir = "data"
if os.path.exists(data_dir):
    st.success(f"✅ Data directory exists at: {os.path.abspath(data_dir)}")
    
    # List files in data directory
    files = os.listdir(data_dir)
    st.write(f"Files in data directory ({len(files)} total):")
    for file in files:
        file_path = os.path.join(data_dir, file)
        size = os.path.getsize(file_path)
        st.write(f"- {file} ({size} bytes)")
        
    if len(files) == 0:
        st.warning("No files found in data directory.")
        
        # Create a test file
        if st.button("Create Test File"):
            test_file_path = os.path.join(data_dir, "test_document.txt")
            with open(test_file_path, "w", encoding="utf-8") as f:
                f.write("""This is a test document.

It contains some information about multilingual document processing.

You can ask questions about this test document, and the bot should be able to find this text.

Dies ist ein deutsches Beispiel. Der Bot kann in mehreren Sprachen antworten.

Ceci est un exemple en français. Le bot peut répondre en plusieurs langues.
""")
            st.success(f"Created test file at {test_file_path}")
else:
    st.error(f"❌ Data directory not found at: {os.path.abspath(data_dir)}")
    
    # Create the directory
    if st.button("Create Data Directory"):
        try:
            os.makedirs(data_dir)
            st.success(f"Created data directory at {os.path.abspath(data_dir)}")
            
            # Create a test file
            test_file_path = os.path.join(data_dir, "test_document.txt")
            with open(test_file_path, "w", encoding="utf-8") as f:
                f.write("""This is a test document.

It contains some information about multilingual document processing.

You can ask questions about this test document, and the bot should be able to find this text.

Dies ist ein deutsches Beispiel. Der Bot kann in mehreren Sprachen antworten.

Ceci est un exemple en français. Le bot peut répondre en plusieurs langues.
""")
            st.success(f"Created test file at {test_file_path}")
        except Exception as e:
            st.error(f"Error creating directory: {str(e)}")

# Check permissions
st.subheader("Directory Permissions")
try:
    if os.path.exists(data_dir):
        # Check read permission
        if os.access(data_dir, os.R_OK):
            st.success("✅ Read permission: Yes")
        else:
            st.error("❌ Read permission: No")
            
        # Check write permission
        if os.access(data_dir, os.W_OK):
            st.success("✅ Write permission: Yes")
        else:
            st.error("❌ Write permission: No")
            
        # Check execute permission
        if os.access(data_dir, os.X_OK):
            st.success("✅ Execute permission: Yes")
        else:
            st.error("❌ Execute permission: No")
except Exception as e:
    st.error(f"Error checking permissions: {str(e)}")

# Library versions
st.subheader("Required Libraries")
try:
    import PyPDF2
    st.write(f"PyPDF2 version: {PyPDF2.__version__}")
except ImportError:
    st.error("PyPDF2 not installed")
except Exception as e:
    st.error(f"Error checking PyPDF2: {str(e)}")

try:
    import docx
    st.write(f"python-docx: {docx.__version__}")
except ImportError:
    st.error("python-docx not installed")
except Exception as e:
    st.error(f"Error checking python-docx: {str(e)}")

try:
    from sentence_transformers import SentenceTransformer
    st.write("SentenceTransformer is installed")
except ImportError:
    st.error("SentenceTransformer not installed")
except Exception as e:
    st.error(f"Error checking SentenceTransformer: {str(e)}")

st.subheader("Next Steps")
st.markdown("""
1. If the data directory doesn't exist, click "Create Data Directory"
2. If no files are found, click "Create Test File" to add a sample document
3. Run the main app (app.py) after confirming your setup is correct
""")

# Current working directory
st.subheader("Environment Information")
st.write(f"Current working directory: {os.getcwd()}")
st.write(f"Absolute path to data directory: {os.path.abspath(data_dir)}")
