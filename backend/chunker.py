from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

import re

def split_pages(extracted_text):
    """
    Parses the full extracted text into a list of (page_number, text) tuples.
    Assumes pages are separated by lines like "--- Page X ---"
    """
    pattern = r"--- Page (\d+) ---"
    splits = re.split(pattern, extracted_text)
    
    # Resulting list: ["", "1", "Text of Page 1", "2", "Text of Page 2", ...]
    pages = []
    for i in range(1, len(splits), 2):
        page_number = int(splits[i])
        page_text = splits[i+1].strip()
        pages.append((page_number, page_text))
    
    return pages

def chunk_pdf_text(pages, chunk_size=800, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )

    chunks = []

    for page_number, text in pages:
        if not text:
            continue
        split_chunks = splitter.split_text(text)
        for chunk in split_chunks:
            chunks.append({
                "text": chunk.strip(),
                "metadata": {"page": page_number}
            })

    return chunks

def build_vector_store(chunks, persist_path="vector_db"):
    docs = [Document(page_content=chunk["text"], metadata=chunk["metadata"]) for chunk in chunks]
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(persist_path)
    print(f"✅ Vector store saved at: {persist_path}")