from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
import ollama
import re

VECTOR_DB_PATH = "vectorstore"


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    tokens = text.split()
    return tokens


def build_bm25_index(vectorstore):

    docs = list(vectorstore.docstore._dict.values())

    corpus = [
        doc.page_content
        for doc in docs
        if len(doc.page_content.strip()) > 40
    ]

    tokenized_corpus = [tokenize(doc) for doc in corpus]

    bm25 = BM25Okapi(tokenized_corpus)

    return bm25, docs


def retrieve_context(vectorstore, bm25, all_docs, question, k=6):

    queries = generate_queries(question)

    scored_docs = []

    # Vector retrieval
    for q in queries:
        results = vectorstore.similarity_search_with_score(q, k=k)
        scored_docs.extend(results)

    scored_docs.sort(key=lambda x: x[1])

    vector_docs = [doc for doc, score in scored_docs[:k]]

    # Keyword retrieval
    tokenized_query = tokenize(question)
    tokenized_query += ["module", "chapter", "section", "lesson"]
    bm25_scores = bm25.get_scores(tokenized_query)

    bm25_docs = [all_docs[i] for i in sorted(range(len(bm25_scores)),
                    key=lambda i: bm25_scores[i], reverse=True)[:k*2]]

    # Merge results
    combined_docs = vector_docs + bm25_docs

    seen = set()
    unique_docs = []

    for doc in combined_docs:
        if doc.page_content not in seen:
            unique_docs.append(doc)
            seen.add(doc.page_content)

    top_docs = unique_docs[:k]

    expanded_docs = []

    for doc in top_docs:
        expanded_docs.append(doc)

        # include neighboring chunks
        idx = all_docs.index(doc)

        if idx > 0:
            expanded_docs.append(all_docs[idx - 1])

        if idx < len(all_docs) - 1:
            expanded_docs.append(all_docs[idx + 1])

    # remove duplicates again
    expanded_unique = {doc.page_content: doc for doc in expanded_docs}

    top_docs = list(expanded_unique.values())[:k]

    context = ""

    for i, doc in enumerate(top_docs):
        context += f"Chunk {i+1} (Page {doc.metadata.get('page')}):\n{doc.page_content}\n\n"

    return context


def generate_queries(question):

    prompt = f"""
Generate 2 alternative search queries that could help retrieve
relevant information for answering the question.

Question:
{question}

Return each query on a new line.
"""

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    queries = response["message"]["content"].split("\n")

    queries = [q.strip("- ").strip() for q in queries if q.strip()]

    queries.append(question)

    return queries


def ask_llama(context, question):

    prompt = f"""
You are a helpful assistant answering questions about a document.

Use ONLY the information provided in the context below.
Do NOT use outside knowledge.

If the answer cannot be found in the context, respond with:
"I cannot find the answer in the provided document."

Explain the concept clearly using the context.
Use bullet points when appropriate.
Do not simply copy sentences — summarize and explain in your own words.

Context:
----------------------
{context}
----------------------

Question:
{question}

Answer:
"""

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


def main():
    print("Loading vector database...")
    vectorstore = load_vectorstore()

    bm25, all_docs = build_bm25_index(vectorstore)

    print("Chatbot ready! Type 'exit' to quit.\n")

    while True:
        try:
            question = input("Ask a question: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chatbot.")
            break

        if question.lower() == "exit":
            break

        context = retrieve_context(vectorstore, bm25, all_docs, question)

        print("\nRetrieved chunks:")
        for i, chunk in enumerate(context.split("Chunk")[1:], 1):
            print(f"\nChunk {i}:")
            print(chunk[:200], "...\n")

        answer = ask_llama(context, question)

        print("\nAnswer:\n", answer)
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()