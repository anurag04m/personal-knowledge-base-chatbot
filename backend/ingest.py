"""
ingest.py  (enhanced)
──────────────────────
PDF ingestion pipeline for the Personal Knowledge Base Chatbot.

Changes over original
──────────────────────
• After loading documents and BEFORE splitting, we run `extract_and_save`
  from module_extractor.py to build a module→topics sidecar JSON.
• The rest of the pipeline (splitting, embedding, FAISS) is unchanged.
• A --use-llm flag enables optional LLM-assisted topic extraction for
  modules where regex finds nothing.
"""

import argparse
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ── NEW: structured knowledge layer ─────────────────────────────────────────
from module_extractor import extract_and_save

DATA_PATH = "data"
VECTOR_DB_PATH = "vectorstore"
MODULE_TOPICS_PATH = "module_topics.json"   # sidecar JSON


# ─────────────────────────────────────────────────────────────────────────────
# Existing pipeline steps (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def load_documents():
    documents = []
    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            path = os.path.join(DATA_PATH, file)
            loader = PyPDFLoader(path)
            docs = loader.load()
            print(f"Loaded {len(docs)} pages from {file}")
            documents.extend(docs)
    return documents


def split_documents(documents):
    for doc in documents:
        doc.page_content = doc.page_content.replace("\n", " ")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1100,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
    return chunks


def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"batch_size": 32},
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(use_llm: bool = False):
    print("Loading documents...")
    documents = load_documents()

    # ── NEW STEP: structured extraction (runs on raw pages, before chunking) ─
    print("Extracting module-wise topics...")
    module_topics = extract_and_save(
        documents=documents,
        output_path=MODULE_TOPICS_PATH,
        use_llm=use_llm,
    )

    if module_topics:
        print("\n── Extracted knowledge structure ───────────────────────────")
        for module, topics in module_topics.items():
            topic_preview = topics[:3]
            more = len(topics) - 3
            print(f"  {module}: {topic_preview}" + (f" … +{more} more" if more > 0 else ""))
        print("────────────────────────────────────────────────────────────\n")
    else:
        print(
            "[ingest] Warning: no module headings detected. "
            "The chatbot will rely entirely on RAG for module questions."
        )

    # ── Existing pipeline ────────────────────────────────────────────────────
    print("Splitting documents...")
    chunks = split_documents(documents)

    print("Creating embeddings and FAISS index...")
    vectorstore = create_vectorstore(chunks)

    print("Saving vector database...")
    vectorstore.save_local(VECTOR_DB_PATH)

    print("Ingestion complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDFs into the knowledge base.")
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLaMA (via Ollama) to fill in modules where regex finds no topics.",
    )
    args = parser.parse_args()
    main(use_llm=args.use_llm)