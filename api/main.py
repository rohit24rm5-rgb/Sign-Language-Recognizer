from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import base64
import sys
import os
import json

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ai_pipeline import SignDetectorPipeline

app = FastAPI(title="Hand Sign Recognition API", description="API for Agentic AI Hand Sign Detection")
pipeline = SignDetectorPipeline()

class SignResponse(BaseModel):
    final_sign: str
    confidence: float
    drafter_guess: str
    critic_feedback: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Hand Sign Recognition API"}

@app.post("/detect", response_model=SignResponse)
async def detect_sign(file: UploadFile = File(...), landmarks: str = Form(None)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
    
    contents = await file.read()
    base64_image = base64.b64encode(contents).decode('utf-8')
    
    try:
        lm_list_parsed = json.loads(landmarks) if landmarks else None
        result = await pipeline.detect_sign(base64_image, lmList=lm_list_parsed)
        return SignResponse(**result)
    except Exception as e:
        import traceback
        err = traceback.format_exc()
        with open("server_error.log", "a") as f:
            f.write(err + "\n")
        print("API ERROR:", err)
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}\n\n{err}")

# To run this API:
# uvicorn api.main:app --reload
