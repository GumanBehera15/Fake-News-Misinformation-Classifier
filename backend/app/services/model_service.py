import json
import pickle
from functools import lru_cache
from pathlib import Path
import numpy as np
from fastapi import HTTPException
from app.core.config import MODEL_DIR
from app.services.preprocessing import normalize_text

MODEL_PATH = MODEL_DIR / "bilstm_model.keras"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.pkl"
CONFIG_PATH = MODEL_DIR / "model_config.json"
METRICS_PATH = MODEL_DIR / "metrics.json"

def model_available() -> bool:
    return MODEL_PATH.exists() and TOKENIZER_PATH.exists() and CONFIG_PATH.exists()

@lru_cache(maxsize=1)
def load_artifacts():
    if not model_available():
        raise HTTPException(status_code=503, detail="Model is not trained. Prepare a dataset and run the training pipeline.")
    try:
        import tensorflow as tf
        with TOKENIZER_PATH.open("rb") as file:
            tokenizer = pickle.load(file)
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        return tf.keras.models.load_model(MODEL_PATH), tokenizer, config
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=503, detail="Saved model could not be loaded safely.") from error

def predict(text: str) -> tuple[float, float, float, dict]:
    model, tokenizer, config = load_artifacts()
    import tensorflow as tf
    normalized = normalize_text(text)
    sequence = tokenizer.texts_to_sequences([normalized])
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=config["max_length"], padding="post", truncating="post")
    probabilities = np.asarray(model.predict(padded, verbose=0)[0], dtype=float)
    if probabilities.shape != (3,):
        raise HTTPException(status_code=503, detail="Saved model uses an old binary architecture. Retrain with the three-class pipeline.")
    probabilities = np.clip(probabilities, 0, 1)
    probabilities /= probabilities.sum()
    real, verification, fake = (float(value) for value in probabilities)
    return fake, real, verification, config

def read_metrics() -> dict:
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="Metrics are unavailable until a model has been evaluated.")
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=500, detail="Metrics file is invalid.") from error

