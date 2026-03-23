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


def retrieve_context(vectorstore, query, k=6):
    docs = vectorstore.similarity_search(query, k=k)

    context = "\n\n".join([doc.page_content for doc in docs])

    return context


def ask_llama(context, question):

    prompt = f"""
You are a question-answering assistant.

Use ONLY the information provided in the context below to answer the question.

If the answer cannot be found in the context, say:
"I cannot find the answer in the provided document."

Be concise and clear.

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
        question = input("Ask a question: ")

        if question.lower() == "exit":
            break

        context = retrieve_context(vectorstore, question)

        print("\nRetrieved context:\n", context[:500])

        answer = ask_llama(context, question)

        print("\nAnswer:\n", answer)
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()
	
