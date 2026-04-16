from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn, uuid, os, pdfplumber, docx, faiss, json, re
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

app = FastAPI()

a = 9
b = 8
print(f"Debug: a={a}, b={b}, a+b={a+b}")    



# --- Setup LLM (Direct Ollama API) ---
# We use the standard OpenAI client to save RAM by avoiding extra libraries
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# --- Memory & Vector Storage ---
model = SentenceTransformer("all-MiniLM-L6-v2")
doc_store = {} 

class AskRequest(BaseModel):
    doc_id: str
    question: str

# --- Helper: Extract Text ---
def get_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".pdf":
            with pdfplumber.open(file_path) as pdf:
                return "\n".join(p.extract_text() or "" for p in pdf.pages)
        elif ext == ".docx":
            doc = docx.Document(file_path)
            return "\n".join([p.text for p in doc.paragraphs])
        return open(file_path, 'r', encoding='utf-8').read()
    except Exception as e:
        print(f"Text Extraction Error: {e}")
        return ""
    
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    doc_id = str(uuid.uuid4())
    temp_path = f"temp_{file.filename}"
    
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    
    # 1. Get and Print Raw Text
    text = get_text(temp_path)
    print("\n--- [DEBUG] FULL TEXT ---")
    print(text[:1000])  # Printing first 1000 characters to avoid cluttering terminal
    
    # 2. Chunk and Print Chunks
    chunks = [text[i:i+500] for i in range(0, len(text), 400)]
    print(f"\n--- [DEBUG] CHUNKS (Total: {len(chunks)}) ---")
    for i, chunk in enumerate(chunks[:3]): # Print first 3 chunks to see the overlap
        print(f"Chunk {i}: {chunk}\n")
    
    # 3. Create Embeddings and Index
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))
    
    # 4. Print Index Info
    print("\n--- [DEBUG] FAISS INDEX ---")
    print(f"Vectors stored in index: {index.ntotal}")
    print(f"Dimensions per vector: {index.d}")
    
    doc_store[doc_id] = {"text": text, "chunks": chunks, "index": index}
    
    os.remove(temp_path)
    return {"doc_id": doc_id}
    
'''
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    doc_id = str(uuid.uuid4())
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    
    text = get_text(temp_path)
    chunks = [text[i:i+500] for i in range(0, len(text), 400)]
    
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))
    
    doc_store[doc_id] = {"text": text, "chunks": chunks, "index": index}
    os.remove(temp_path)
    return {"doc_id": doc_id}
'''
@app.post("/ask")
async def ask(req: AskRequest):
    data = doc_store.get(req.doc_id)
    if not data: raise HTTPException(404, "Doc not found")
    
    q_emb = model.encode([req.question])
    D, I = data["index"].search(np.array(q_emb).astype("float32"), k=2)
    context = "\n".join([data["chunks"][i] for i in I[0]])

    try:
        response = client.chat.completions.create(
            model="qwen2.5:1.5b",
            messages=[
                {"role": "system", "content": "Answer ONLY from context. If missing, say 'Not found'."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {req.question}"}
            ]
        )
        answer = response.choices[0].message.content
        return {"answer": answer, "sources": context, "confidence": 0.9}
    except Exception as e:
        return {"answer": f"Error: {e}", "sources": "", "confidence": 0.0}

@app.post("/extract")
async def extract(req: AskRequest):
    data = doc_store.get(req.doc_id)
    if not data: raise HTTPException(404, "Doc not found")
    
    # We use a very small snippet to ensure it fits in 3GB RAM
    input_text = data["text"][:1000] 

    prompt = f"""
    Extract structured shipment data from the text below. 
    Return ONLY a JSON object with these keys:
    shipment_id, shipper, consignee, pickup_datetime, delivery_datetime, equipment_type, mode, rate, currency, weight, carrier_name.
    
    Use null if missing. No conversational text, only JSON.

    Text: {input_text}
    """

    try:
        response = client.chat.completions.create(
            model="qwen2.5:1.5b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0  # Keep it consistent
        )
        
        # Clean the output to find the JSON block
        raw_content = response.choices[0].message.content
        json_match = re.search(r"\{.*\}", raw_content, re.DOTALL)
        
        if json_match:
            return json.loads(json_match.group())
        else:
            raise ValueError("No JSON found in response")
            
    except Exception as e:
        print(f"Extraction Error: {e}")
        return {
            "shipment_id": None, "shipper": None, "consignee": None, 
            "pickup_datetime": None, "delivery_datetime": None, 
            "equipment_type": None, "mode": None, "rate": None, 
            "currency": "USD", "weight": None, "carrier_name": None,
            "error": "Model busy or RAM full"
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)