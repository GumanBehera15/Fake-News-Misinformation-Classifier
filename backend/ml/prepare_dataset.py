"""Clean a local CSV and normalize labels to Real=0, Needs Verification=1, Fake=2."""
import argparse
from pathlib import Path
import sys
import pandas as pd

# Support direct execution: `python ml/prepare_dataset.py ...` from backend/.
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
from app.services.preprocessing import normalize_text

def to_label(value: object, fake_labels: set[str], verification_labels: set[str], real_labels: set[str]) -> int | None:
    normalized = str(value).strip().upper()
    if normalized in fake_labels:
        return 2
    if normalized in verification_labels:
        return 1
    if normalized in real_labels:
        return 0
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/processed.csv"))
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--title-column", default="title")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--fake-labels", default="FAKE,0")
    parser.add_argument("--verification-labels", default="HALF-TRUE,BARELY-TRUE,MIXED,UNVERIFIED,UNCERTAIN")
    parser.add_argument("--real-labels", default="REAL,TRUE,1,MOSTLY-TRUE")
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    if args.label_column not in df:
        raise ValueError(f"Missing label column: {args.label_column}")
    body = df[args.text_column].fillna("") if args.text_column in df else pd.Series("", index=df.index)
    title = df[args.title_column].fillna("") if args.title_column in df else pd.Series("", index=df.index)
    if args.text_column not in df and args.title_column not in df:
        raise ValueError("Provide at least a text or title column.")
    fake_labels = {item.strip().upper() for item in args.fake_labels.split(",")}
    verification_labels = {item.strip().upper() for item in args.verification_labels.split(",")}
    real_labels = {item.strip().upper() for item in args.real_labels.split(",")}
    cleaned = pd.DataFrame({"text": (title.astype(str) + " " + body.astype(str)).map(normalize_text), "label": df[args.label_column].map(lambda x: to_label(x, fake_labels, verification_labels, real_labels))})
    cleaned = cleaned.dropna().query("text != ''").drop_duplicates(subset=["text"]).astype({"label": "int64"})
    if cleaned.label.nunique() != 3:
        raise ValueError("The cleaned data must contain REAL, a verification/uncertain class, and FAKE labels.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(args.output, index=False)
    print(f"Saved {len(cleaned)} cleaned records to {args.output}")

if __name__ == "__main__":
    main()

