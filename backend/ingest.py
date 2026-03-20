from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

DATA_PATH = "data"
VECTOR_DB_PATH = "vectorstore"


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
        length_function=len
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")

    return chunks


def create_vectorstore(chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"batch_size": 32}
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