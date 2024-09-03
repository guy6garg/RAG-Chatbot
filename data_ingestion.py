import os, glob
import shutil
# document loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
    Docx2txtLoader,
    BSHTMLLoader
)
# text_splitter
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)

TMP_DIR = "./data/tmp"


def langchain_document_loader():
    """
    Create document loaders for PDF, TXT, DOCX and HTML files.
    """

    documents = []
    txt_loader = DirectoryLoader(
        TMP_DIR, glob="**/*.txt", loader_cls=TextLoader, show_progress=True
    )
    documents.extend(txt_loader.load())

    pdf_loader = DirectoryLoader(
        TMP_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True
    )
    documents.extend(pdf_loader.load())

    doc_loader = DirectoryLoader(
        TMP_DIR,
        glob="**/*.docx",
        loader_cls=Docx2txtLoader,
        show_progress=True,
    )
    documents.extend(doc_loader.load())
    html_loader = DirectoryLoader(
        TMP_DIR, glob="**/*.html", loader_cls=BSHTMLLoader, show_progress=True
    )
    documents.extend(html_loader.load())
    htm_loader = DirectoryLoader(
        TMP_DIR, glob="**/*.htm", loader_cls=BSHTMLLoader, show_progress=True
    )
    documents.extend(htm_loader.load())
    return documents

def split_documents_to_chunks(documents):
    """Split documents to chunks using RecursiveCharacterTextSplitter."""

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    return chunks

def delete_temp_files(directory):
    """Delete files from a directory"""
    # Check if the directory exists
    if os.path.exists(directory):
        # Iterate over all the files and subdirectories in the directory
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                # If it's a file, remove it
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                # If it's a directory, remove it and all its contents
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        print(f"The directory {directory} does not exist.")
