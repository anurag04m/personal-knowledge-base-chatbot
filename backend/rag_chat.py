chat_history = []

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
import ollama
import re
import time

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

    # Dynamic k adjustment
    question_lower = question.lower()

    is_overview = any(word in question_lower for word in [
        "module", "topics", "list", "overview", "sections", "covered"
    ])

    if any(word in question.lower() for word in ["module", "topics", "list", "overview", "sections", "covered"]):
        k = 20  # increase significantly

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
        try:
            idx = all_docs.index(doc)
        except ValueError:
            continue

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

    pages = list(set(doc.metadata.get('page') for doc in top_docs if doc.metadata.get('page') is not None))

    return context, pages


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

    rewritten_query = rewrite_question(question)

    queries = response["message"]["content"].split("\n")
    queries = [q.strip("- ").strip() for q in queries if q.strip()]

    queries.append(rewritten_query)
    queries.append(question)

    return queries


def ask_llama(context, question):

    prompt = f"""
You are a helpful assistant answering questions about a document.

Use ONLY the information provided in the context.
Do NOT use outside knowledge.

If the question asks about modules or topics:

1. If topics are explicitly listed:
   - Extract and present them clearly using bullet points.

2. If topics are NOT explicitly listed:
   - Clearly state: "Topics are not explicitly listed for this module."
   - Then provide a GENERAL IDEA of the module based ONLY on nearby context.
   - This general idea should be a short summary (1–2 lines max).
   - Do NOT invent or add topics that are not supported by the text.

3. If there is no relevant information at all:
   - Say: "I cannot find the answer in the provided document."

Guidelines:
- Prefer extraction over inference.
- If you infer, keep it minimal and clearly implied by the context.
- Do NOT hallucinate specific topic names.
- Use bullet points when appropriate.
- Keep answers clear and structured.

Context:
----------------------
{context}
----------------------

Question:
{question}

Answer:
"""

    # Build message list (THIS is where memory is added)
    messages = []

    # Add previous conversation (last 3 exchanges = 6 messages)
    messages.extend(chat_history[-6:])

    # Add current prompt
    messages.append({"role": "user", "content": prompt})

    response = ollama.chat(
        model="llama3",
        messages=messages
    )

    answer = response["message"]["content"]

    # Save conversation
    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": answer})

    return answer


def get_topic_question():
    skip_words = [
        "rewrite", "bullet", "elaborate", "summarize",
        "explain more", "rephrase", "convert", "format"
    ]

    for msg in reversed(chat_history):
        if msg["role"] == "user":
            q = msg["content"].lower()

            if not any(word in q for word in skip_words):
                return msg["content"]

    return None


def rewrite_question(question):
    if not chat_history:
        return question

    last_user_q = get_topic_question()

    if not last_user_q:
        return question

    prompt = f"""
Rewrite the follow-up question into a complete standalone question
optimized for document retrieval.

- Include important keywords from the previous question
- Make it specific and explicit
- Prefer noun phrases and keywords over vague wording

Previous question:
{last_user_q}

Follow-up question:
{question}

Rewritten search query:
"""

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    rewritten_query = response["message"]["content"].strip()

    return rewritten_query


def main():
    print("Loading vector database...")
    vectorstore = load_vectorstore()

    bm25, all_docs = build_bm25_index(vectorstore)

    print("Chatbot ready! Type 'exit' to quit.\n")

    while True:
        try:
            question = input("\n🧠 Ask me something: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chatbot.")
            break

        if question.lower() == "exit":
            break

        if question.lower() == "clear":
            chat_history.clear()
            print("🧹 Conversation cleared.")
            continue

        # Use previous question for context if it's a follow-up
        combined_question = question  # default

        if chat_history:
            last_user_q = None

            for msg in reversed(chat_history):
                if msg["role"] == "user":
                    last_user_q = msg["content"]
                    break

            followup_words = ["it", "this", "they", "them", "how", "why"]

            if last_user_q and any(word in question.lower() for word in followup_words):
                combined_question = rewrite_question(question)

        context, pages = retrieve_context(vectorstore, bm25, all_docs, combined_question)

        print("\nRetrieved chunks:")
        for i, chunk in enumerate(context.split("Chunk")[1:], 1):
            print(f"\nChunk {i}:")
            print(chunk[:200], "...\n")

        print("\n🤔 Thinking...\n")

        answer = ask_llama(context, question)

        if pages:
            answer += f"\n\n📄 Sources: {', '.join(str(p) for p in pages)}"

        print("\n🤖 Answer:\n")

        for char in answer:
            print(char, end="", flush=True)
            time.sleep(0.005)

        print("\n\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()