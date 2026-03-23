"""
Q1 - Step 1: Data Preprocessing
JoshTalks AI Researcher Intern Assessment

Downloads transcription JSONs from GCS, extracts (audio_url, text) pairs,
normalizes text, and saves as a HuggingFace-compatible dataset.
"""

import json
import re
import os
import requests  # type: ignore
import pandas as pd  # type: ignore
from pathlib import Path
from datasets import Dataset, DatasetDict, Audio  # type: ignore
from tqdm import tqdm  # type: ignore

# ─── CONFIG ───────────────────────────────────────────────────────────────────
GCS_BASE = "https://storage.googleapis.com/upload_goai"
DATA_CSV  = "dataset_index.csv"      # export the Google Sheet to CSV here
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── TEXT NORMALIZATION ────────────────────────────────────────────────────────
HINDI_PUNCT = re.compile(r'[।!?,;:()\[\]"\'-]')
MULTI_SPACE = re.compile(r'\s+')

def normalize_text(text: str) -> str:
    """
    Normalize Hindi ASR transcription text for training.
    Steps:
      1. Strip leading/trailing whitespace
      2. Remove Hindi punctuation (danda, commas, etc.)
      3. Collapse multiple spaces
      4. Lowercase (Devanagari is caseless, but handles any Latin chars)
    """
    text = text.strip()
    text = HINDI_PUNCT.sub(' ', text)
    text = MULTI_SPACE.sub(' ', text)
    text = text.strip()
    return text


# ─── DOWNLOAD A SINGLE TRANSCRIPTION JSON ─────────────────────────────────────
def fetch_transcription(url: str) -> list[dict]: # type: ignore
    """
    Download transcription JSON for a given URL.
    Returns list of utterance dicts.
    """
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  [WARN] Failed to fetch {url}: {e}")
        return []


# ─── LOAD DATASET INDEX ───────────────────────────────────────────────────────
def load_index(csv_path: str) -> pd.DataFrame:
    """
    Load dataset index CSV (exported from Google Sheet).
    Expected columns: user_id, recording_id, language, duration,
                      rec_url_gcp, transcription_url, metadata_url
    """
    df = pd.read_csv(csv_path)
    # Filter Hindi only
    if 'language' in df.columns:
        df = df[df['language'].str.lower().isin(['hi', 'hindi'])]
    print(f"Loaded {len(df)} Hindi recordings from index.")
    return df


# ─── BUILD UTTERANCE DATASET ──────────────────────────────────────────────────
def build_dataset(df: pd.DataFrame, max_samples: int = None) -> list[dict]:  # type: ignore
    """
    For each recording in the index, fetch transcription JSON, extract utterances,
    and pair with the audio GCS URL.
    """
    records = []
    rows = df.iterrows()

    for i, row in tqdm(rows, total=len(df), desc="Downloading transcriptions"):
        if max_samples and len(records) >= max_samples:
            break

        uid   = int(row['user_id'])
        rid   = int(row['recording_id'])
        audio = row.get('rec_url_gcp', '')

        # Use the actual transcription URL from CSV, replacing the prefix with upload_goai
        raw_url = str(row.get('transcription_url_gcp', ''))
        fixed_url = raw_url.replace("joshtalks-data-collection/hq_data/hi", "upload_goai")

        utterances = fetch_transcription(fixed_url)
        for utt in utterances:
            text = utt.get('text', '').strip()
            if not text:
                continue

            norm_text = normalize_text(text)
            if not norm_text:
                continue

            records.append({
                "user_id":      uid,
                "recording_id": rid,
                "audio_url":    audio,
                "start":        utt.get("start", 0.0),
                "end":          utt.get("end",   0.0),
                "duration":     utt.get("end", 0.0) - utt.get("start", 0.0),
                "raw_text":     text,
                "text":         norm_text,   # ← training target
                "language":     "hi",
            })

    print(f"Extracted {len(records)} utterances total.")
    return records


# ─── FILTER OUTLIERS ──────────────────────────────────────────────────────────
def filter_utterances(records: list[dict],
                       min_dur: float = 1.0,
                       max_dur: float = 30.0,
                       min_words: int = 1) -> list[dict]:
    """
    Filter out:
      - Too short (< 1s): likely noise or silence
      - Too long  (> 30s): Whisper handles at most ~30s without chunking
      - Empty after normalization
    """
    filtered = [
        r for r in records
        if min_dur <= r["duration"] <= max_dur
        and len(r["text"].split()) >= min_words
    ]
    print(f"After filtering: {len(filtered)} utterances "
          f"(removed {len(records)-len(filtered)})")
    return filtered


# ─── SAVE AS HUGGINGFACE DATASET ──────────────────────────────────────────────
def save_dataset(records: list[dict], split_ratio: float = 0.9):
    """
    Save as HuggingFace DatasetDict with train/validation splits.
    """
    df = pd.DataFrame(records)

    # Shuffle deterministically
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    n_train = int(len(df) * split_ratio)
    train_df = df[:n_train]  # type: ignore
    val_df   = df[n_train:]  # type: ignore

    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_ds   = Dataset.from_pandas(val_df.reset_index(drop=True))

    ds = DatasetDict({"train": train_ds, "validation": val_ds})
    ds.save_to_disk(str(OUTPUT_DIR / "hindi_asr_dataset"))

    # Also save as CSV for inspection
    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUTPUT_DIR   / "val.csv",   index=False)

    print(f"\nDataset saved to {OUTPUT_DIR}/hindi_asr_dataset")
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Val  : {len(val_ds)} samples")
    return ds


# ─── DEMO MODE (uses sample transcription JSON) ───────────────────────────────
def demo_from_sample_json():
    """Run preprocessing on the provided sample transcription JSON."""
    print("Running in DEMO mode with sample recording 825780...")
    demo_url = "https://storage.googleapis.com/upload_goai/967179/825780_transcription.json"
    sample = fetch_transcription(demo_url)
    if not sample:
        print("Could not fetch sample. Check connectivity.")
        return

    records = []
    for utt in sample:
        text = utt.get("text", "").strip()
        if not text:
            continue
        norm = normalize_text(text)
        records.append({
            "user_id":      967179,
            "recording_id": 825780,
            "audio_url":    "",
            "start":        utt.get("start", 0.0),
            "end":          utt.get("end",   0.0),
            "duration":     utt.get("end", 0.0) - utt.get("start", 0.0),
            "raw_text":     text,
            "text":         norm,
            "language":     "hi",
        })

    records = filter_utterances(records)

    print("\nSample preprocessed utterances:")
    for r in records[:5]:  # type: ignore
        print(f"  [{r['start']:.1f}s–{r['end']:.1f}s] "
              f"RAW: {r['raw_text'][:60]}")  # type: ignore
        print(f"   →  NORM: {r['text'][:60]}")  # type: ignore
        print()

    # Save sample CSV
    pd.DataFrame(records).to_csv(OUTPUT_DIR / "sample_preprocessed.csv",
                                 index=False, encoding="utf-8")
    print(f"Saved to {OUTPUT_DIR / 'sample_preprocessed.csv'}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo",       action="store_true",
                        help="Run on sample JSON (no sheet needed)")
    parser.add_argument("--index",      default=DATA_CSV,
                        help="Path to dataset index CSV")
    parser.add_argument("--max",        type=int, default=None,
                        help="Max samples to download")
    args = parser.parse_args()

    if args.demo or not os.path.exists(args.index):
        demo_from_sample_json()
    else:
        df      = load_index(args.index)
        records = build_dataset(df, max_samples=args.max)
        records = filter_utterances(records)
        save_dataset(records)
