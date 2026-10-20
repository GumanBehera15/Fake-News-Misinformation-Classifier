"""Train and evaluate a reproducible bidirectional LSTM; never uses demo data implicitly."""
import argparse, json, pickle, random
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

# Support direct execution: `python ml/train_model.py ...` from backend/.
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
from app.services.preprocessing import normalize_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--vocab-size", type=int, default=20000)
    parser.add_argument("--max-length", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); keras.utils.set_random_seed(args.seed)
    df = pd.read_csv(args.dataset)
    if not {"text", "label"}.issubset(df.columns): raise ValueError("Dataset requires text and label columns. Run prepare_dataset.py first.")
    df = df.dropna().drop_duplicates("text"); df["text"] = df.text.map(normalize_text); df = df[df.text != ""]
    if set(df.label.unique()) != {0, 1, 2}: raise ValueError("Training needs all three labels: Real=0, Needs Verification=1, Fake=2.")
    train, temp = train_test_split(df, test_size=.3, stratify=df.label, random_state=args.seed)
    valid, test = train_test_split(temp, test_size=.5, stratify=temp.label, random_state=args.seed)
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=args.vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(train.text)
    def sequences(values):
        return keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(values), maxlen=args.max_length, padding="post", truncating="post")
    x_train, x_valid, x_test = sequences(train.text), sequences(valid.text), sequences(test.text)
    model = keras.Sequential([keras.layers.Input(shape=(args.max_length,)), keras.layers.Embedding(args.vocab_size, 128, mask_zero=True), keras.layers.Bidirectional(keras.layers.LSTM(64, dropout=.3, recurrent_dropout=0)), keras.layers.Dropout(.4), keras.layers.Dense(32, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)), keras.layers.Dense(3, activation="softmax")])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    weights = compute_class_weight("balanced", classes=np.array([0,1,2]), y=train.label); class_weight={index: float(value) for index, value in enumerate(weights)}
    callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)]
    history = model.fit(x_train, train.label, validation_data=(x_valid, valid.label), epochs=args.epochs, batch_size=args.batch_size, class_weight=class_weight, callbacks=callbacks)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    model.save(args.model_dir / "bilstm_model.keras")
    with (args.model_dir / "tokenizer.pkl").open("wb") as f: pickle.dump(tokenizer, f)
    (args.model_dir / "model_config.json").write_text(json.dumps({"vocab_size": args.vocab_size, "max_length": args.max_length, "model_type": "BiLSTM (3-class)", "class_names": ["Likely Real", "Needs Verification", "Likely Fake"]}, indent=2))
    (args.model_dir / "training_history.json").write_text(json.dumps({k: [float(x) for x in v] for k,v in history.history.items()}, indent=2))
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    probs = model.predict(x_test, verbose=0); preds=probs.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(test.label, preds, average="macro", zero_division=0)
    metrics={"accuracy": float(accuracy_score(test.label,preds)), "precision": float(precision), "recall": float(recall), "f1": float(f1), "metric_average": "macro", "confusion_matrix": confusion_matrix(test.label,preds,labels=[0,1,2]).tolist(), "test_samples": int(len(test)), "labels": ["Likely Real", "Needs Verification", "Likely Fake"], "dataset_warning": "Synthetic demonstration data must not be used to claim real-world performance."}
    (args.model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__": main()

