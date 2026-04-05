import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Import ingest's main method so we can trigger the processing programmatically.
# Using a try/except in case we are running server.py from outside backend/
try:
    from backend.ingest import main as ingest_main
except ImportError:
    from ingest import main as ingest_main

app = Flask(__name__)
CORS(app)

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
