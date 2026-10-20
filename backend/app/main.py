from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import FRONTEND_ORIGIN
from app.database.history import initialize_database, add_history, list_history, clear_history
from app.schemas.models import PredictionRequest, PredictionResponse, TextStatistics, ModelInfo, DISCLAIMER
from app.services.preprocessing import text_statistics
from app.services.model_service import predict, model_available, read_metrics, CONFIG_PATH

@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_database()
    yield

app = FastAPI(title="Fake News & Misinformation Classifier API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=[FRONTEND_ORIGIN], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/api/health")
def health():
    return {"status": "ok", "model_available": model_available()}

@app.get("/api/model/info", response_model=ModelInfo)
def model_info():
    if not model_available():
        return ModelInfo(available=False, model_type="BiLSTM (3-class)", message="No trained model is available. Follow the training instructions in the README.")
    import json
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return ModelInfo(available=True, model_type=config.get("model_type", "BiLSTM (3-class)"), message="Trained model is available.", max_sequence_length=config["max_length"], vocabulary_size=config["vocab_size"])

@app.get("/api/model/metrics")
def metrics():
    return read_metrics()

@app.post("/api/predict", response_model=PredictionResponse)
def prediction(request: PredictionRequest):
    fake, real, verification, _ = predict(request.text)
    scores = {"Likely Real": real, "Needs Verification": verification, "Likely Fake": fake}
    label = max(scores, key=scores.get)
    confidence = scores[label]
    add_history(request.text, label, confidence)
    return PredictionResponse(prediction=label, fake_probability=fake, real_probability=real, verification_probability=verification, confidence=confidence, model_type="BiLSTM (3-class)", disclaimer=DISCLAIMER, text_statistics=TextStatistics(**text_statistics(request.text)))

@app.get("/api/history")
def history():
    return list_history()

@app.delete("/api/history")
def delete_history():
    clear_history()
    return {"message": "Analysis history cleared."}
