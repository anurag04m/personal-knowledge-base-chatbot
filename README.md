# 🧠 BrainBox AI — Personal Knowledge Base Chatbot

A local RAG (Retrieval-Augmented Generation) chatbot that lets you upload a PDF and ask questions about it. Runs entirely on your machine — no API keys, no cloud, no data leaving your system.

---

## How It Works

```
PDF Upload
   │
   ▼
ingest.py
   ├── Extracts module/topic structure  →  module_topics.json
   ├── Splits text into chunks
   └── Generates embeddings            →  vectorstore/ (FAISS)

User Query
   │
   ├── Query Router  (instant, no LLM)
   │     ├── "How many modules?"    →  answered from JSON directly
   │     ├── "Topics in Module II?" →  answered from JSON directly
   │     └── everything else        →  falls through to RAG
   │
   └── RAG Pipeline
         ├── Parallel: query rewrite + query expansion (LLaMA)
         ├── Hybrid retrieval: FAISS (semantic) + BM25 (keyword)
         ├── Neighbour chunk expansion
         └── Answer generation (LLaMA via Ollama)
```

---

## Project Structure

```
brainbox-ai/
│
├── backend/
│   ├── ingest.py            # PDF ingestion pipeline
│   ├── rag_chat.py          # RAG retrieval + LLM answer generation
│   ├── query_router.py      # Structured query interceptor (zero-latency)
│   ├── module_extractor.py  # Module/topic extraction from PDF text
│   └── server.py            # Flask API server
│
├── frontend/
│   ├── index.html           # Single-page UI
│   ├── styles.css           # Styles
│   └── app.js               # Upload, processing, and chat logic
│
├── data/                    # PDFs go here (auto-created on first upload)
├── vectorstore/             # FAISS index (auto-created after ingestion)
├── module_topics.json       # Structured topic index (auto-created)
├── requirements.txt
└── README.md
```

---

## Prerequisites

### 1. Python 3.10+

Verify with:
```bash
python --version
```

### 2. Ollama + LLaMA 3

Install Ollama from [ollama.com](https://ollama.com), then pull the model:
```bash
ollama pull llama3
```

Verify Ollama is running before starting the server:
```bash
ollama serve
```

---

## Installation

```bash
# 1. Clone or download the project
git clone https://github.com/yourname/brainbox-ai.git
cd brainbox-ai

# 2. Create and activate a virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note on PyTorch:** `pip install -r requirements.txt` installs the CPU version of PyTorch by default. If you have an NVIDIA GPU and want faster embeddings, install the CUDA build manually from [pytorch.org](https://pytorch.org/get-started/locally/) before running the above command.

---

## Running the App

### Step 1 — Start Ollama
```bash
ollama serve
```
Keep this terminal open.

### Step 2 — Start the Flask backend
```bash
# From the project root
python backend/server.py
```
The server starts on `http://localhost:5000`.

### Step 3 — Open the frontend

Open `frontend/index.html` directly in your browser. No separate web server needed.

---

## Usage

1. **Upload** — Drag and drop a PDF (or click "Browse Files"). The ingestion pipeline runs automatically. This takes ~1–3 minutes the first time as it generates embeddings.
2. **Chat** — Ask any question about the document.
3. **Clear** — Click "Clear Chat" in the header to upload a new document.

### Example questions

| Query | How it's handled |
|---|---|
| `"How many modules are there?"` | Query router → instant answer from JSON |
| `"What topics are in Module II?"` | Query router → instant answer from JSON |
| `"Explain virtual memory."` | Full RAG pipeline (~45–60s on CPU) |
| `"How is it implemented?"` *(follow-up)* | Cached context reused, skips retrieval (~45s) |

---

## Running Ingestion Manually

If you want to re-ingest PDFs placed directly in the `data/` folder:

```bash
# Standard (regex-only topic extraction)
python backend/ingest.py

# With LLM-assisted topic extraction for modules where regex finds nothing
python backend/ingest.py --use-llm
```

---

## Configuration

Key constants you can tune in the source files:

| File | Variable | Default | Description |
|---|---|---|---|
| `rag_chat.py` | `k` in `retrieve_context()` | `6` | Number of chunks retrieved per query |
| `ingest.py` | `chunk_size` | `1100` | Characters per chunk |
| `ingest.py` | `chunk_overlap` | `200` | Overlap between chunks |
| `module_extractor.py` | `TOPIC_MIN_WORDS` | `2` | Min words for a line to be a topic |
| `module_extractor.py` | `TOPIC_MAX_WORDS` | `8` | Max words for a line to be a topic |

---

## Performance Notes

- **First query** after ingestion: ~45–60s (CPU-bound LLM inference via Ollama)
- **Follow-up questions**: context is cached and reused, so only the final LLM call runs
- **Module/count queries**: answered in <1s from the JSON index, no LLM involved
- **GPU acceleration**: if your machine has a CUDA GPU and Ollama detects it, response times drop to ~3–5s

---

## Troubleshooting

**`Error: Knowledge base not initialized`**
The vectorstore hasn't been built yet. Upload a PDF through the UI or run `ingest.py` manually.

**`Connection to server failed`**
Make sure `server.py` is running on port 5000 and Ollama is running (`ollama serve`).

**Module topics not detected**
Your PDF may use non-standard heading formats. Try running ingestion with `--use-llm` to use LLaMA for topic extraction instead of regex.

**Very slow responses**
This is expected on CPU-only machines. Each Ollama inference call takes ~45–60s. For faster responses, use a machine with an NVIDIA GPU.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | LLaMA 3 via Ollama |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | FAISS (CPU) |
| Keyword retrieval | BM25 (rank-bm25) |
| PDF loading | LangChain + PyPDF |
| Backend API | Flask |
| Frontend | Vanilla HTML / CSS / JS |