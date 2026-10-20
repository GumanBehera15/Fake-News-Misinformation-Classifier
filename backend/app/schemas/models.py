from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from app.core.config import MAX_INPUT_CHARS

DISCLAIMER = "This is a model estimate, not a factual verification."

class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_INPUT_CHARS)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("Please enter text to analyze.")
        return cleaned

class TextStatistics(BaseModel):
    characters: int
    words: int
    normalized_characters: int

class PredictionResponse(BaseModel):
    prediction: str
    fake_probability: float = Field(ge=0, le=1)
    real_probability: float = Field(ge=0, le=1)
    verification_probability: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    model_type: str
    disclaimer: str = DISCLAIMER
    text_statistics: TextStatistics

class HistoryItem(BaseModel):
    id: int
    preview: str
    prediction: str
    confidence: float
    created_at: datetime

class ModelInfo(BaseModel):
    available: bool
    model_type: str
    message: str
    max_sequence_length: int | None = None
    vocabulary_size: int | None = None

