from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import ollama

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


def retrieve_context(vectorstore, question, k=8):

    queries = generate_queries(question)

    scored_docs = []

    for q in queries:
        results = vectorstore.similarity_search_with_score(q, k=k)
        scored_docs.extend(results)

    # sort by similarity score (lower score = better match)
    scored_docs.sort(key=lambda x: x[1])

    docs = [doc for doc, score in scored_docs]

    unique_docs = {doc.page_content: doc for doc in docs}

    top_docs = list(unique_docs.values())[:6]

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

    print("Chatbot ready! Type 'exit' to quit.\n")

    while True:
        try:
            question = input("Ask a question: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chatbot.")
            break

        if question.lower() == "exit":
            break

        context = retrieve_context(vectorstore, question)

        print("\nRetrieved chunks:")
        for i, chunk in enumerate(context.split("Chunk")[1:], 1):
            print(f"\nChunk {i}:")
            print(chunk[:200], "...\n")

        answer = ask_llama(context, question)

        print("\nAnswer:\n", answer)
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()

	
