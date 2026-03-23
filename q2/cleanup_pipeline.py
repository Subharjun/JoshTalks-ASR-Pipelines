"""
Q2 - ASR Cleanup Pipeline
JoshTalks AI Researcher Intern Assessment

Combines:
  1. Number Normalization (Hindi words → digits)
  2. English Word Detection and tagging ([EN]...[/EN])

Run: python cleanup_pipeline.py --test
     python cleanup_pipeline.py --input transcription.json
"""

import json
import argparse
import requests  # type: ignore
from pathlib import Path
from number_normalizer import normalize_numbers  # type: ignore
from english_detector import tag_english_words  # type: ignore

# ─── PIPELINE ─────────────────────────────────────────────────────────────────
def run_pipeline(text: str) -> dict:  # type: ignore
    """
    Full cleanup pipeline for a single Hindi ASR utterance.
    Steps:
      1. Number normalization
      2. English word detection & tagging
    Returns dict with all intermediate outputs.
    """
    # Step 1: Normalize numbers
    after_numbers, number_changes = normalize_numbers(text)

    # Step 2: Tag English words
    after_english, english_words = tag_english_words(after_numbers)

    return {
        "original":       text,
        "after_numbers":  after_numbers,
        "final":          after_english,
        "number_changes": number_changes,
        "english_words":  english_words,
    }


# ─── BEFORE/AFTER REPORT ──────────────────────────────────────────────────────
def print_example(result: dict, label: str = ""):
    if label:
        print(f"\n▶ {label}")
    print(f"  BEFORE : {result['original']}")
    if result['after_numbers'] != result['original']:
        print(f"  +NUMS  : {result['after_numbers']}")
    print(f"  AFTER  : {result['final']}")
    if result['number_changes']:
        for c in result['number_changes']:
            note = "PRESERVED (idiom)" if c['skipped'] else f"'{c['original']}' → '{c['converted']}'"
            print(f"           Numbers: {note}")
    if result['english_words']:
        print(f"           English: {result['english_words']}")


# ─── DEMO ON REAL DATA ────────────────────────────────────────────────────────
def demo_on_sample(url: str = "https://storage.googleapis.com/upload_goai/967179/825780_transcription.json"):
    """Download the sample transcription and run the full pipeline."""
    print("[INFO] Fetching sample transcription from GCS...")
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        utterances = resp.json()
    except Exception as e:
        print(f"[WARN] Could not fetch data: {e}")
        utterances = []

    results = [run_pipeline(u["text"]) for u in utterances if u.get("text")]

    # Print first 5 that have changes
    changed = [r for r in results if
               r["after_numbers"] != r["original"] or r["english_words"]]

    print(f"\nFound {len(changed)} utterances with changes out of {len(results)} total.")

    print("\n" + "=" * 70)
    print("   BEFORE / AFTER EXAMPLES FROM REAL DATA")
    print("=" * 70)
    for i, r in enumerate(changed[:5], 1):  # type: ignore
        print_example(r, label=f"Example {i}")


# ─── BUILT-IN TEST CASES ──────────────────────────────────────────────────────
BUILT_IN_TESTS = [
    # ── CORRECT CONVERSIONS ──────────────────────────────────────────────────
    {
        "label":    "Simple number",
        "text":     "मुझे दो किलो आटा चाहिए",
        "type":     "number",
    },
    {
        "label":    "Compound number",
        "text":     "वहाँ तीन सौ चौवन लोग थे",
        "type":     "number",
    },
    {
        "label":    "Large number",
        "text":     "यह घर एक हज़ार स्क्वेयर फुट का है",
        "type":     "number",
    },
    {
        "label":    "Sentence with year",
        "text":     "मैं उन्नीस सौ पचानवे में पैदा हुआ",
        "type":     "number",
    },
    {
        "label":    "English words in Latin script",
        "text":     "मैंने Google और Amazon में job apply किया है",
        "type":     "english",
    },
    # ── EDGE CASES ───────────────────────────────────────────────────────────
    {
        "label":    "EDGE: Idiomatic 'दो-चार बातें' — should NOT convert",
        "text":     "उससे दो-चार बातें कर लो",
        "type":     "edge",
    },
    {
        "label":    "EDGE: 'एक-दो' as vague quantifier — should NOT convert",
        "text":     "एक-दो दिन और रुक जाओ",
        "type":     "edge",
    },
    {
        "label":    "EDGE: Devanagari English word — should NOT tag",
        "text":     "मुझे कंप्यूटर पर काम करना है",
        "type":     "edge",
    },
    {
        "label":    "EDGE: Mixed — numbers + English",
        "text":     "मुझे पच्चीस salary मिलती है",
        "type":     "edge",
    },
]


def run_tests():
    print("\n" + "=" * 70)
    print("   PIPELINE TEST CASES")
    print("=" * 70)

    for i, tc in enumerate(BUILT_IN_TESTS, 1):
        result = run_pipeline(tc["text"])
        print_example(result, label=f"[{tc['type'].upper()}] {tc['label']}")

    print("\n" + "=" * 70)
    print("All tests complete.")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",   action="store_true",
                        help="Run built-in test cases")
    parser.add_argument("--demo",   action="store_true",
                        help="Demo on real sample transcription JSON")
    parser.add_argument("--input",  type=str,
                        help="Path to a local transcription JSON file")
    parser.add_argument("--text",   type=str,
                        help="Process a single input string")
    args = parser.parse_args()

    if args.text:
        result = run_pipeline(args.text)
        print_example(result)

    elif args.test:
        run_tests()

    elif args.input:
        with open(args.input, encoding="utf-8") as f:
            utterances = json.load(f)
        for utt in utterances:
            result = run_pipeline(utt.get("text", ""))
            print_example(result)

    else:
        # Default: run tests + demo
        run_tests()
        demo_on_sample()


if __name__ == "__main__":
    main()
