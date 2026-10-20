import argparse, json
from pathlib import Path
import sys
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Support direct execution: `python ml/evaluate_model.py ...` from backend/.
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
from app.services.model_service import load_artifacts
from app.services.preprocessing import normalize_text

def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved BiLSTM against a labeled CSV.")
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    model, tokenizer, config = load_artifacts()
    df = pd.read_csv(args.dataset).dropna()
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("Dataset needs text and label columns.")
    import tensorflow as tf
    sequences = tokenizer.texts_to_sequences(df.text.map(normalize_text))
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=config["max_length"], padding="post", truncating="post")
    probabilities = model.predict(padded, verbose=0)
    predictions = probabilities.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(df.label, predictions, average="macro", zero_division=0)
    print(json.dumps({"accuracy": float(accuracy_score(df.label, predictions)), "precision": float(precision), "recall": float(recall), "f1": float(f1), "metric_average": "macro", "confusion_matrix": confusion_matrix(df.label, predictions, labels=[0,1,2]).tolist(), "labels": ["Likely Real", "Needs Verification", "Likely Fake"]}, indent=2))

if __name__ == "__main__":
    main()

