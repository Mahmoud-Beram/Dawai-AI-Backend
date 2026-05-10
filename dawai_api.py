"""
Dawai API - Medicine OCR Recognition Server (V10 - RapidOCR)
=============================================================
Flutter sends an image → we return medicine info as JSON.

Run:  python dawai_api.py
Test: http://localhost:8000/docs (Swagger UI)
"""

import sys
import os
import time

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# FastAPI
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

# Import V10 engine (RapidOCR + RapidFuzz)
from dawai_master import process_image, get_ocr_engines, get_matcher
from dawai_chatbot import get_chat_response

#####################################################################
# GLOBAL: Load models ONCE at startup
#####################################################################

print("=" * 50, flush=True)
print(" DAWAI API V10 - Starting up...", flush=True)
print("=" * 50, flush=True)

print("[1/2] Loading RapidOCR models (EN+AR)...", flush=True)
t0 = time.time()
get_ocr_engines()
print(f"      Done in {time.time()-t0:.1f}s", flush=True)

print("[2/2] Loading Medicine Database...", flush=True)
matcher = get_matcher()

print(f"      {len(matcher.parsed_db)} medicines loaded.", flush=True)

print("\n>>> SERVER READY! <<<\n", flush=True)

#####################################################################
# FastAPI App
#####################################################################

app = FastAPI(title="Dawai API", description="Medicine OCR Recognition (V10 - RapidOCR)")

# Allow Flutter to connect from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def health_check():
    return {"status": "Dawai API is running!"}


@app.post("/scan")
async def scan_medicine(file: UploadFile = File(...)):
    """
    Receive a medicine image from Flutter, return medicine info.
    """
    start_time = time.time()
    
    temp_dir = os.path.join(os.path.dirname(__file__), 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"scan_{int(time.time())}.jpg")
    
    try:
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Run OCR using V10 (RapidOCR + RapidFuzz)
        result_dict = process_image(temp_path)
        
        elapsed = time.time() - start_time
        
        if not result_dict or not result_dict.get("found"):
            return {
                "found": False,
                "medicine_id": 0,
                "confidence": 0,
                "time_seconds": round(elapsed, 1)
            }
            
        return {
            "found": True,
            "medicine_id": int(result_dict.get('medicine_id', 0)),
            "confidence": result_dict.get('confidence', 0),
            "time_seconds": result_dict.get('time_total', round(elapsed, 1))
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)




class ChatRequest(BaseModel):
    message: str
    medicine_id: Optional[int] = None

@app.post("/chat")
async def chat_with_bot(payload: ChatRequest):
    """
    NLP Chatbot endpoint.
    Payload: {"message": "عندي مغص", "medicine_id": 15}
    """
    user_msg = payload.message
    
    # تحويل الـ ID لاسم الدواء عشان الشات بوت يفهمه
    context_med = None
    if payload.medicine_id:
        mask = matcher.db['id'] == payload.medicine_id
        if not matcher.db[mask].empty:
            context_med = matcher.db[mask].iloc[0]['Name_EN']
            
    if not user_msg:
        return {"response": "عذراً، رسالتك فارغة!"}
        
    print(f"[API] POST /chat - Query: '{user_msg}'", flush=True)
    result = get_chat_response(user_msg, context_med)
    
    if isinstance(result, str):
        print(f"[CHATBOT] Responded with text: {result[:50]}...", flush=True)
        return {"response": result, "medicine_id": None}
        
    returned_med_id = None
    if result.get("context_medicine"):
        # تحويل اسم الدواء اللي الشات بوت اقترحه إلى ID للباك إند
        mask = matcher.db['Name_EN'] == result["context_medicine"]
        if not matcher.db[mask].empty:
            returned_med_id = int(matcher.db[mask].iloc[0]['id'])
            
    print(f"[CHATBOT] Responded with dict: {result['text'][:50]}... | MedID: {returned_med_id}", flush=True)
    return {
        "response": result["text"],
        "medicine_id": returned_med_id
    }

@app.get("/health")
def health_check():
    """Check if server is running"""
    return {"status": "ok", "medicines_loaded": len(matcher.parsed_db)}


if __name__ == "__main__":
    print("\n" + "=" * 50, flush=True)
    print(" Starting Dawai API on http://localhost:8000", flush=True)
    print(" Swagger Docs: http://localhost:8000/docs", flush=True)
    print("=" * 50 + "\n", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)
