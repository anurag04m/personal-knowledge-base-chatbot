
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import os

# Path to PDF folder
DATA_PATH = "backend/data"

# Path to save vector database
VECTOR_DB_PATH = "backend/vectorstore"


def load_documents():
    documents = []

    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
            documents.extend(loader.load())

    return documents


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80
    )

    return text_splitter.split_documents(documents)


def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore


def main():
    print("Loading documents...")
    documents = load_documents()

    print("Splitting documents...")
    chunks = split_documents(documents)

    print("Creating embeddings and FAISS index...")
    vectorstore = create_vectorstore(chunks)

    print("Saving vector database...")
    vectorstore.save_local(VECTOR_DB_PATH)

    print("Ingestion complete!")


if __name__ == "__main__":
    main()
	