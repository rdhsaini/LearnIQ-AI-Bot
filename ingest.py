"""
LearnIQ — ingest.py  (Pinecone edition)
Run ONCE locally to push vectors to Pinecone cloud.
Streamlit Cloud will then read from Pinecone — no chroma_db/ needed.

Usage:
  python ingest.py            # first time
  python ingest.py --rebuild  # wipe Pinecone index and rebuild
"""

import os
import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

CONTENT_DIR    = "./LearnIQ_Content"
CHUNK_SIZE     = 500
CHUNK_OVERLAP  = 50
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "learniq")


def check_env():
    missing = []
    if not os.getenv("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if not os.getenv("PINECONE_API_KEY"):
        missing.append("PINECONE_API_KEY")
    if missing:
        raise EnvironmentError(
            f"Missing in .env: {', '.join(missing)}\n"
            "Add them and try again."
        )


def load_pdfs():
    print(f"\n📂 Scanning '{CONTENT_DIR}' for PDFs...")
    if not Path(CONTENT_DIR).exists():
        raise FileNotFoundError(f"Folder '{CONTENT_DIR}' not found.")
    files = list(Path(CONTENT_DIR).rglob("*.pdf"))
    if not files:
        raise ValueError(f"No PDFs found in '{CONTENT_DIR}'.")
    for f in files:
        print(f"   • {f.name}")
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


def setup_pinecone(force=False):
    print(f"\n🌲 Connecting to Pinecone index '{PINECONE_INDEX}'...")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    existing = [i.name for i in pc.list_indexes()]

    if force and PINECONE_INDEX in existing:
        print("   Deleting existing index...")
        pc.delete_index(PINECONE_INDEX)
        existing = []

    if PINECONE_INDEX not in existing:
        print("   Creating new index (1536 dims, cosine)...")
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print("   Index created.")
    else:
        print("   Index already exists — upserting vectors.")


def embed_and_push(chunks):
    print(f"\n🚀 Embedding and pushing {len(chunks)} chunks to Pinecone...")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX,
    )
    print(f"   ✅ {len(chunks)} vectors pushed to Pinecone index '{PINECONE_INDEX}'")


if __name__ == "__main__":
    force = "--rebuild" in sys.argv
    print("=" * 50)
    print("  LearnIQ — Ingest Pipeline (Pinecone)")
    print("=" * 50)
    try:
        check_env()
        pages  = load_pdfs()
        chunks = chunk_pages(pages)
        setup_pinecone(force=force)
        embed_and_push(chunks)
        print("\n✅ Done! Vectors are in Pinecone cloud.")
        print("   Now push to GitHub and Streamlit Cloud will work.\n")
    except (FileNotFoundError, ValueError, EnvironmentError) as e:
        print(f"\n❌ {e}\n")
        sys.exit(1)
