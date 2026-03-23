"""
Q1 - Step 4: Error Analysis
JoshTalks AI Researcher Intern Assessment

Systematically samples 25 error utterances from the fine-tuned model's output,
builds an error taxonomy, proposes fixes.
Run: python error_analysis.py
"""

import json
import re
import math
import jiwer  # type: ignore
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass

# ─── CONFIG ───────────────────────────────────────────────────────────────────
PREDICTIONS_FILE = Path("results/predictions_Whisper-small_(fine-tuned_on_Hindi).jsonl")
OUTPUT_DIR       = Path("results")

# jiwer transform for WER computation
WER_TRANSFORM = jiwer.Compose([  # type: ignore
    jiwer.ToLowerCase(),  # type: ignore
    jiwer.RemovePunctuation(),  # type: ignore
    jiwer.Strip(),  # type: ignore
    jiwer.ReduceToListOfListOfWords(),  # type: ignore
])


# ─── DATA CLASS ───────────────────────────────────────────────────────────────
@dataclass
class ErrorSample:
    idx:        int
    reference:  str
    prediction: str
    wer:        float
    error_type: str = ""
    reasoning:  str = ""

    def short_ref(self) -> str:  # type: ignore
        return self.reference[:60]  # type: ignore

    def short_pred(self) -> str:  # type: ignore
        return self.prediction[:60]  # type: ignore


# ─── LOAD PREDICTIONS ─────────────────────────────────────────────────────────
def load_predictions(path):  # type: ignore
    if not path.exists():
        alt = path.parent / "predictions_Whisper-small_(pretrained_baseline).jsonl"
        if alt.exists():
            print(f"[INFO] Fine-tuned predictions not found. Using baseline: {alt}")
            path = alt
        else:
            raise FileNotFoundError(
                f"No prediction files found in {path.parent}. Run evaluate.py first."
            )
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    print(f"Loaded {len(samples)} prediction samples.")
    return samples


# ─── COMPUTE PER-UTTERANCE WER ────────────────────────────────────────────────
def utterance_wer(reference, prediction):  # type: ignore
    try:
        return jiwer.wer(  # type: ignore
            reference, prediction,
            truth_transform=WER_TRANSFORM,
            hypothesis_transform=WER_TRANSFORM,
        )
    except Exception:
        return 1.0


# ─── CLASSIFY ERROR TYPE ──────────────────────────────────────────────────────
DEVANAGARI = re.compile(r'[\u0900-\u097F]')
LATIN       = re.compile(r'[a-zA-Z]')


def classify_error(ref, pred):  # type: ignore
    """
    Heuristically classify what kind of ASR error this is.
    Returns (error_type, reasoning).

    Categories:
      1. DELETION   — model outputs much fewer words than reference
      2. INSERTION  — model outputs many extra words
      3. SUBSTITUTION_OOV  — word in reference not likely in training vocab
      4. CODE_MIX   — reference or prediction contains non-Hindi (English/mixed)
      5. REPETITION — model repeats words/phrases (hallucination)
      6. NOISE_DISFLUENCY — reference is a filler/noise ("हाँ", "हूं" etc.)
    """
    ref_words  = ref.strip().split()
    pred_words = pred.strip().split()

    ref_len: int  = max(len(ref_words),  1)
    pred_len: int = max(len(pred_words), 1)

    ratio = pred_len / ref_len  # type: ignore

    if LATIN.search(ref) or LATIN.search(pred):
        return ("CODE_MIX",
                "Reference or prediction contains English/Latin-script words. "
                "Whisper may transcribe English words in Devanagari or vice versa.")

    if ref_len <= 2:
        return ("NOISE_DISFLUENCY",
                "Reference is a very short filler/disfluency token (e.g., हाँ, हूं). "
                "Model may expand or mishear these.")

    if ratio < 0.5:
        return ("DELETION",
                f"Model produced {pred_len} words vs {ref_len} in reference "
                f"(ratio {ratio:.2f}). Likely skipped a segment of speech.")

    if ratio > 2.0:
        return ("INSERTION",
                f"Model produced {pred_len} words vs {ref_len} in reference "
                f"(ratio {ratio:.2f}). Possible hallucination or repeated phrases.")

    if pred_len > 3:
        pred_counter = Counter(pred_words)
        most_common_count = pred_counter.most_common(1)[0][1]  # type: ignore
        if most_common_count > pred_len * 0.4:  # type: ignore
            return ("REPETITION",
                    "Model repeated the same word multiple times — hallucination pattern.")

    return ("SUBSTITUTION_OOV",
            "Model substituted words, possibly due to OOV pronunciation, "
            "regional accent variation, or dialect mismatch.")


# ─── SAMPLE 25 ERRORS SYSTEMATICALLY ─────────────────────────────────────────
def sample_errors(samples, n=25):  # type: ignore
    """
    Sampling strategy:
    1. Compute WER for every utterance.
    2. Keep only utterances with WER > 0 (actual errors).
    3. Sort by WER descending.
    4. Stratify: pick from low-WER (0–0.3), mid-WER (0.3–0.7), high-WER (0.7+).
    5. Within each stratum, take every Nth sample to avoid cherry-picking.
    """
    scored = []
    for i, s in enumerate(samples):
        w = utterance_wer(s["reference"], s["prediction"])  # type: ignore
        if w > 0:
            scored.append((i, s["reference"], s["prediction"], w))  # type: ignore

    print(f"\nTotal utterances with errors: {len(scored)} / {len(samples)}")

    low   = [x for x in scored if x[3] <= 0.30]   # type: ignore
    mid   = [x for x in scored if 0.30 < x[3] <= 0.70]  # type: ignore
    high  = [x for x in scored if x[3]  > 0.70]   # type: ignore

    print(f"  Low WER  (0–30%): {len(low)}")
    print(f"  Mid WER (30–70%): {len(mid)}")
    print(f"  High WER  (>70%): {len(high)}")

    def every_nth(lst, target):  # type: ignore
        if not lst:
            return []
        step = max(1, len(lst) // target)  # type: ignore
        return lst[::step][:target]  # type: ignore

    selected = (
        every_nth(low,  5)  +  # type: ignore
        every_nth(mid, 10)  +  # type: ignore
        every_nth(high, 10)  # type: ignore
    )[:n]  # type: ignore

    errors = []
    for idx, ref, pred, wer_score in selected:  # type: ignore
        etype, reason = classify_error(ref, pred)
        errors.append(ErrorSample(
            idx=idx, reference=ref, prediction=pred,
            wer=wer_score, error_type=etype, reasoning=reason,
        ))

    return errors


# ─── BUILD ERROR TAXONOMY ─────────────────────────────────────────────────────
TAXONOMY_FIXES = {
    "DELETION": {
        "description": "Model skips words or large segments of speech.",
        "cause":        "Long audio segments, fast speech, or overlapping voices cause the model to miss chunks.",
        "examples":     [],
        "fix": (
            "1. Segment audio at natural sentence boundaries (apply VAD) before feeding to Whisper.\n"
            "2. Use Whisper's chunked long-form transcription with `chunk_length_s=15`.\n"
            "3. Add more examples of fast/dense Hindi speech to training data."
        ),
    },
    "INSERTION": {
        "description": "Model generates extra words not in the reference (hallucination).",
        "cause":        "Background noise, silence, or very short utterances can trigger Whisper hallucinations.",
        "examples":     [],
        "fix": (
            "1. Apply silence/noise filter before inference (webrtcvad or Silero VAD).\n"
            "2. Use `no_speech_threshold` parameter in Whisper pipeline to skip near-silent segments.\n"
            "3. Fine-tune with negative examples (silence segments labeled as empty string)."
        ),
    },
    "SUBSTITUTION_OOV": {
        "description": "Model replaces words with phonetically similar but wrong words.",
        "cause":        "Out-of-vocabulary words, regional accents, or dialect variations not seen during training.",
        "examples":     [],
        "fix": (
            "1. Augment training data with diverse speaker accents (collect from multiple Indian states).\n"
            "2. Add a Hindi language model (n-gram or neural LM) for beam search rescoring.\n"
            "3. Use SpecAugment during training for better accent robustness."
        ),
    },
    "CODE_MIX": {
        "description": "English words transcribed incorrectly (in wrong script or wrong word).",
        "cause":        "Whisper trained predominantly on English may transcribe Hindi-English code-mix inconsistently.",
        "examples":     [],
        "fix": (
            "1. Fine-tune on code-mixed Hindi-English data with consistent English → Devanagari transliteration.\n"
            "2. Post-process: detect Latin-script outputs and transliterate to Devanagari using indic-trans.\n"
            "3. Use a code-mix–aware language model for rescoring."
        ),
    },
    "REPETITION": {
        "description": "Model repeats the same phrase multiple times (hallucination loop).",
        "cause":        "Whisper decoder can get stuck in a repetition loop on long or low-confidence audio.",
        "examples":     [],
        "fix": (
            "1. Set `no_repeat_ngram_size=3` during generation to block repeated n-grams.\n"
            "2. Apply compression ratio thresholding (discard transcription if too repetitive).\n"
            "3. Reduce `temperature` during generation or use temperature fallback logic."
        ),
    },
    "NOISE_DISFLUENCY": {
        "description": "Short filler words (हाँ, हूं) misrecognized or expanded.",
        "cause":        "Model may not handle ultra-short audio segments or conversational fillers well.",
        "examples":     [],
        "fix": (
            "1. Apply minimum duration filter (skip < 1.5s segments).\n"
            "2. Fine-tune on conversational filler data.\n"
            "3. Post-process: map common model outputs to known fillers using a small lookup table."
        ),
    },
}


# ─── PRINT REPORT ─────────────────────────────────────────────────────────────
def print_taxonomy_report(errors):  # type: ignore
    grouped = defaultdict(list)
    for e in errors:
        grouped[e.error_type].append(e)

    for etype, ex_list in grouped.items():
        TAXONOMY_FIXES[etype]["examples"] = ex_list[:5]  # type: ignore

    sorted_types = sorted(grouped.items(), key=lambda x: -len(x[1]))  # type: ignore

    print("\n" + "=" * 70)
    print("  ERROR TAXONOMY — Top Sampled Error Categories")
    print("=" * 70)

    for rank, (etype, ex_list) in enumerate(sorted_types, 1):
        info = TAXONOMY_FIXES[etype]
        pct  = len(ex_list) / len(errors) * 100  # type: ignore

        print(f"\n{'─'*70}")
        print(f"  #{rank}  {etype}  ({len(ex_list)} / {len(errors)} samples, {pct:.0f}%)")
        print(f"{'─'*70}")
        print(f"  Description: {info['description']}")
        print(f"  Root Cause:  {info['cause']}")

        print(f"\n  Examples:")
        for i, ex in enumerate(info["examples"][:3], 1):  # type: ignore
            print(f"\n    [{i}] WER: {ex.wer:.0%}")
            print(f"        REF : {ex.short_ref()}")
            print(f"        PRED: {ex.short_pred()}")
            print(f"        WHY : {ex.reasoning}")

        if rank <= 3:
            print(f"\n  ✅ Proposed Fix (implemented/actionable):")
            for line in info["fix"].split("\n"):  # type: ignore
                print(f"     {line}")

    print("\n" + "=" * 70)


# ─── SAVE REPORT ──────────────────────────────────────────────────────────────
def save_report(errors):  # type: ignore
    report = []
    for e in errors:
        report.append({
            "idx":        e.idx,
            "reference":  e.reference,
            "prediction": e.prediction,
            "wer":        round(e.wer, 4),
            "error_type": e.error_type,
            "reasoning":  e.reasoning,
        })
    path = OUTPUT_DIR / "error_analysis_25samples.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nError samples saved to {path}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():  # type: ignore
    samples = load_predictions(PREDICTIONS_FILE)
    errors  = sample_errors(samples, n=25)
    print_taxonomy_report(errors)
    save_report(errors)


if __name__ == "__main__":
    main()
