import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Import ingest's main method so we can trigger the processing programmatically.
# Using a try/except in case we are running server.py from outside backend/
try:
    from backend.ingest import main as ingest_main
    import backend.rag_chat as rc
    from backend.query_router import route_query, is_followup
    from backend.module_extractor import load_module_topics
except ImportError:
    from ingest import main as ingest_main
    import rag_chat as rc
    from query_router import route_query, is_followup
    from module_extractor import load_module_topics

app = Flask(__name__)
CORS(app)

# Global variables for RAG models
rag_models = {
    "initialized": False,
    "vectorstore": None,
    "bm25": None,
    "all_docs": None,
    "module_topics": None
}

def init_rag_models():
    """Initialize RAG models if not already loaded"""
    if not rag_models["initialized"]:
        try:
            rag_models["vectorstore"] = rc.load_vectorstore()
            rag_models["bm25"], rag_models["all_docs"] = rc.build_bm25_index(rag_models["vectorstore"])
            rag_models["module_topics"] = load_module_topics(rc.MODULE_TOPICS_PATH)
            rag_models["initialized"] = True
            return True
        except Exception as e:
            print(f"Error initializing RAG models: {e}")
            return False
    return True

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    if not data or 'query' not in data:
        return jsonify({"error": "No query provided"}), 400
    
    question = data['query']
    
    if not init_rag_models():
        return jsonify({"error": "Knowledge base not initialized. Please upload a PDF first."}), 503
        
    try:
        # Step 1: structured router
        routed_answer, was_routed = route_query(question, rag_models["module_topics"])
        
        if was_routed:
            rc.chat_history.append({"role": "user", "content": question})
            rc.chat_history.append({"role": "assistant", "content": routed_answer})
            rc._cache["context"] = None   # structured answers don't warm the RAG cache
            rc._cache["pages"] = None
            return jsonify({"answer": routed_answer, "sources": []})
            
        # Step 2: follow-up cache check
        use_cached = is_followup(question) and rc._cache["context"] is not None
        
        if use_cached:
            context = rc._cache["context"]
            pages = rc._cache["pages"]
        else:
            # Step 3: hybrid retrieval
            context, pages = rc.retrieve_context(
                rag_models["vectorstore"], 
                rag_models["bm25"], 
                rag_models["all_docs"], 
                question
            )
            rc._cache["context"] = context
            rc._cache["pages"] = pages
            
        # Step 4: LLM answer
        answer = rc.ask_llama(context, question)
        
        if routed_answer:
            answer = routed_answer + "\n\n---\n\n" + answer
            
        source_pages = list(pages) if pages else []
        return jsonify({"answer": answer, "sources": source_pages}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400
    
    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Trigger processing (re-ingests the data folder)
        try:
            ingest_main(use_llm=False)
            return jsonify({"status": "Success", "message": f"File {filename} uploaded and processed."}), 200
        except Exception as e:
            return jsonify({"error": f"Error during processing: {str(e)}"}), 500
    else:
        return jsonify({"error": "Invalid file format, please upload a PDF."}), 400

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)
