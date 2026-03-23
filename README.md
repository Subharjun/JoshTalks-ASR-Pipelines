# Josh Talks ASR Project :studio_microphone:

This repository contains completely natively executable Python data pipelines that solve 4 advanced Automatic Speech Recognition (ASR) challenges: Preprocessing, Data Cleanup, Spelling Classification, and Lattice-Based Word Error Rate (WER) Evaluation.

## 🚀 Features

*   **Q1: Audio & Transcription Preprocessing:** Connects to Google Cloud buckets to parse massive JSON transcription payloads, filters Hindi phonetic errors, aligns timestamps, and builds a clean Hugging Face `Dataset` object.
*   **Q2: ASR Cleanup Pipeline:** Dynamically normalizes complex Hindi conversational numbers (e.g. "तीन सौ चौवन" → "354") while preserving idiomatic phrases (e.g. "दो-चार बातें"), and tags English Code-Mix words (`[EN]salary[/EN]`) in Devanagari text.
*   **Q3: Multi-Signal Spelling Classifier:** Checks over 1.77L words against a massive custom dictionary, applying Regex Phonotactic rules and N-Gram Language Modeling to statistically calculate a Confidence Score (High/Medium/Low) for misspelled Hindi words.
*   **Q4: Dynamic Programming Lattice WER:** Executes Sequence Alignment to prove the mathematical superiority of Lattice-Based WER over Standard Edit-Distance WER by resolving multi-model synonymous ASR outputs across N-Best paths.

## 🛠 Usage

This project has built-in testing commands that run fully native in your terminal out-of-the-box! No IDE debugging required!

**Q1 Data Preprocessing:**
```bash
python3 q1/preprocess.py --demo
```

**Q2 Number Normalizer & Code-Mix Pipeline:**
```bash
python3 q2/cleanup_pipeline.py --text "मुझे छब्बीस हज़ार salary मिली"
```

**Q3 Spelling Classifier N-Gram Math:**
```bash
python3 q3/spelling_classifier.py --word अच्चा
```

**Q4 Lattice WER Matrix Output:**
```bash
python3 q4/lattice_builder.py --demo
```

## 📦 Dependencies
- `pandas`, `transformers`, `datasets`, `jiwer`, `torch`, `torchaudio`

*Designed and engineered natively in Python 3.14.*
