"""
LearnIQ — ingest.py
Run ONCE to build ChromaDB from your PDFs.

Usage:
  python ingest.py            # first time
  python ingest.py --rebuild  # when PDFs change
"""

import os
import sys
import shutil
from pathlib import Path

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

CONTENT_DIR   = "./LearnIQ_Content"
CHROMA_DIR    = "./chroma_db"
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50


def load_pdfs():
    print(f"\n📂 Scanning '{CONTENT_DIR}' for PDFs...")
    if not Path(CONTENT_DIR).exists():
        raise FileNotFoundError(f"Folder '{CONTENT_DIR}' not found.")
    files = list(Path(CONTENT_DIR).rglob("*.pdf"))
    if not files:
        raise ValueError(f"No PDFs found in '{CONTENT_DIR}'.")
    for f in files:
        print(f"   • {f}")
    loader = PyPDFDirectoryLoader(CONTENT_DIR, recursive=True, silent_errors=True)
    pages  = loader.load()
    print(f"   Loaded {len(pages)} pages")
    return pages


def chunk_pages(pages):
    print(f"\n✂️  Splitting into {CHUNK_SIZE}-char chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    for chunk in chunks:
        src  = chunk.metadata.get("source", "")
        page = chunk.metadata.get("page", 0) + 1
        if "worksheet" in src.lower():
            chunk.metadata["source_label"] = f"Worksheet · {Path(src).stem} · p{page}"
        elif "case" in src.lower():
            chunk.metadata["source_label"] = f"Case Study · {Path(src).stem} · p{page}"
        else:
            chunk.metadata["source_label"] = f"Textbook · Page {page}"
    print(f"   Created {len(chunks)} chunks")
    return chunks


def embed_and_save(chunks, force=False):
    print(f"\n💾 Saving to ChromaDB at '{CHROMA_DIR}'...")
    if force and Path(CHROMA_DIR).exists():
        shutil.rmtree(CHROMA_DIR)
        print("   Wiped existing index.")
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set. Add it to your .env file.")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key,
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    vectorstore.persist()
    print(f"   Saved {len(chunks)} vectors to '{CHROMA_DIR}/'")


if __name__ == "__main__":
    force = "--rebuild" in sys.argv
    print("=" * 50)
    print("  LearnIQ — Ingest Pipeline")
    print("=" * 50)
    try:
        pages  = load_pdfs()
        chunks = chunk_pages(pages)
        embed_and_save(chunks, force=force)
        print("\n✅ Done! Now run:  streamlit run app.py\n")
    except (FileNotFoundError, ValueError, EnvironmentError) as e:
        print(f"\n❌ {e}\n")
        sys.exit(1)
