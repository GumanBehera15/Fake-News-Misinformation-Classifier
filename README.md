# Fake News & Misinformation Classifier

A college microproject that uses a bidirectional LSTM (BiLSTM) to classify the language patterns of pasted news content as **Likely Real** or **Likely Fake**. It supports a complete FastAPI API, a React dashboard, locally persisted history, and a reproducible TensorFlow training pipeline.

> The output is an ML estimate, not factual verification. It cannot establish whether an event happened or a source is trustworthy.

## Architecture

```mermaid
flowchart LR
  U[User] --> R[React Frontend] --> F[FastAPI Backend] --> P[Text Preprocessing]
  P --> T[Tokenizer and Padding] --> M[Trained BiLSTM Model] --> O[Prediction Probabilities] --> UI[Results UI]
```

```mermaid
flowchart LR
  D[Dataset] --> C[Validation and Cleaning] --> S[Train / Validation / Test Split]
  S --> T[Tokenization] --> L[LSTM Training] --> E[Evaluation] --> A[Saved Model and Metrics]
```

## Prerequisites

- Python 3.11 (TensorFlow compatibility is platform dependent)
- Node.js 20+ and npm
- A licensed, labeled news-text CSV for meaningful training

## Setup

### Backend

Windows PowerShell:

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Linux/macOS:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

The API will start before a model is trained. `GET /api/health` shows `model_available: false`; prediction requests return HTTP 503 until the artifacts are available.

### Frontend

```powershell
cd frontend
Copy-Item .env.example .env
npm install
npm run dev
```

On Linux/macOS use `cp .env.example .env`. Set `VITE_API_URL` if the backend does not run at `http://127.0.0.1:8000`.

The frontend uses React, TypeScript, Tailwind's build pipeline, Lucide icons, and Recharts. Component-level CSS supplies the editorial dashboard treatment while Tailwind remains available for utility styling and production processing.

## Dataset and training

Put a dataset CSV in `backend/data/`; it is intentionally ignored by Git. The source needs `label` and either `text` or `title` + `text`. The 3-class pipeline normalizes `REAL`/`TRUE`/`MOSTLY-TRUE` to Likely Real, `HALF-TRUE`/`BARELY-TRUE`/`UNVERIFIED` to Needs Verification, and `FAKE`/`0` to Likely Fake. Adjust the `--real-labels`, `--verification-labels`, and `--fake-labels` options to match your dataset.

```powershell
cd backend
python ml/prepare_dataset.py data/source.csv --output data/processed.csv
python ml/train_model.py --dataset data/processed.csv --epochs 8
python ml/evaluate_model.py --dataset data/processed.csv
```

The pipeline removes blank and duplicate text, splits stratified train/validation/test sets, fits the tokenizer only on training text, applies class weights, uses early stopping, and saves model/tokenizer/configuration/real held-out metrics to `backend/models/`. The built-in demonstration pathway is deliberately absent: this project never claims a score or produces an inference until a real model is trained.

For a larger workflow-only synthetic dataset, run `python ml/generate_demo_dataset.py`, then prepare `data/demo_news_large.csv`. Its generated labels are deliberately easy to learn, so its test metrics are not evidence of real-world performance. To improve actual accuracy, replace it with a licensed, diverse, externally labeled news dataset; preserve a held-out test set that has never influenced training or tuning.

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/api/health` | API and trained-model availability |
| GET | `/api/model/info` | Architecture and saved configuration |
| GET | `/api/model/metrics` | Actual saved evaluation metrics |
| POST | `/api/predict` | Validate and classify input |
| GET | `/api/history` | Recent local analyses |
| DELETE | `/api/history` | Clear local history |

Example request:

```json
{ "text": "A city council published the minutes of its public meeting." }
```

Successful prediction response (values depend on the trained model):

```json
{ "prediction": "Likely Fake", "fake_probability": 0.82, "real_probability": 0.18, "confidence": 0.82, "model_type": "BiLSTM", "disclaimer": "This is a model estimate, not a factual verification." }
```

## Tests and verification

```powershell
cd backend
pytest
cd ../frontend
npm run build
```

## Academic notes

**Problem statement:** misleading content can erode public trust and decision-making. **Objectives:** classify linguistic patterns, show calibrated class probabilities, and communicate uncertainty responsibly. **Methodology:** preprocessing → train-only tokenization → padded sequences → embedding → BiLSTM → sigmoid probability; evaluate on a held-out test set using accuracy, precision, recall, F1, and a confusion matrix.

**Results:** none are included until you run a real evaluation; the Model page reads `metrics.json` rather than displaying invented metrics. **Limitations:** dataset coverage, time sensitivity, label quality, demographic and publisher bias, and adversarial wording may affect results. The classifier does not fact-check claims.

**Future work:** source/retrieval-based verification, multilingual datasets, calibration analysis, human review workflows, and bias audits. This supports **CO3** through model implementation and evaluation, **CO4** through deployment/responsible analysis, and **SDG 16** by encouraging careful engagement with information.

## Troubleshooting

- **503 Model is not trained:** run the preparation and training commands from `backend`.
- **CORS/network error:** make sure the backend is running and `VITE_API_URL` matches it.
- **TensorFlow installation problem:** use supported Python/OS builds or a compatible TensorFlow release; the UI and non-model endpoints can still start.
- **Metrics unavailable:** this is expected until training finishes and writes `backend/models/metrics.json`.
