# 🚛 Logistics AI Document Processor
A high-performance, lightweight RAG (Retrieval-Augmented Generation) system optimized for document extraction and QA on resource-constrained hardware.

## 🚀 Quick Start
### Prerequisites
* **Python 3.9+**
* **Ollama** installed.
* **Model:** `ollama pull qwen2.5:1.5b`

### Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt  

   Run the Backend: python backend/main.py

Run the Frontend: streamlit run frontend/app.py

🏗️ Architecture
The system uses a Bimodal Backend (FastAPI) and a Reactive Frontend (Streamlit).

Vector Engine: FAISS (Facebook AI Similarity Search) using all-MiniLM-L6-v2 embeddings.

LLM Engine: Qwen2.5-1.5B (Local via Ollama).

Optimization: Specifically tuned for 3.1GB RAM environments by bypassing heavy JSON validation libraries in favor of direct regex-based JSON extraction.Due to the target environment's 3.1GB RAM limitation, this project implements 'Lean Inference'—using a 1.5B parameter model and manual JSON parsing to avoid the memory overhead of Pydantic-based extraction tools.

✂️ Chunking Strategy
Method: Fixed-size character splitting.

Size: 500 characters per chunk.

Overlap: 100 characters.

Why: This ensures semantic context is preserved at boundaries while keeping the retrieval window small enough to stay within RAM limits during LLM inference.

🔍 Retrieval Method
Type: Semantic Similarity Search.

Metric: Inner Product (IP) similarity.

Top-K: 2 (We retrieve the top 2 most relevant chunks).

Process: Queries are embedded using SentenceTransformers and matched against the FAISS index to provide grounded context for the "Ask" feature.

🛡️ Guardrails Approach
Context Anchoring: System prompts force the model to answer "ONLY from context" to eliminate hallucinations.

Input Truncation: For extraction, the model focuses on the first 1000 characters (the document header) to ensure high-density data capture without crashing local memory.

Output Cleaning: A regex-based post-processor cleans LLM output to ensure the UI only renders valid JSON objects.

📈 Confidence Scoring
Heuristic Scoring: * 0.90: Assigned when a valid entity is successfully parsed and returned from the document context.

0.10: Assigned if the model returns a "Not found" or safety response.

⚠️ Failure Cases & Future Ideas
Failure Cases: Scanned/Image-based PDFs (requires OCR) and multi-page documents where rates are buried in the footer (requires larger context windows).

Improvements: * Integrate PaddleOCR for handwritten documents.

Implement Hybrid Search (Keyword + Vector) for better Shipment ID lookup.

Add Metadata Filtering to search by Date or Carrier.