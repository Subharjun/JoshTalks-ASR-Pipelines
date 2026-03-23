"""
Q4 - Lattice Builder + Lattice-Based WER Calculator
JoshTalks AI Researcher Intern Assessment

Theory:
  Standard WER penalizes a model every time it differs from a single rigid
  reference. But speech has multiple valid transcriptions. A lattice replaces
  the rigid string with a sequence of "bins," where each bin contains ALL
  valid alternatives at that alignment position.

  If a model output matches ANY alternative in a bin, no penalty is applied.
  This reduces unfair WER for models that produce a valid but different form.

Alignment unit: WORD-LEVEL
  Justification: Hindi is morphologically regular with clear word boundaries.
  Devanagari word boundaries are spaced. Subword would split meaningful units.

Run: python lattice_builder.py --demo
     python lattice_builder.py --input q4_data.json
"""

import json
import re
import argparse
from collections import defaultdict
from pathlib import Path
import difflib
from typing import Dict, List, Set, Tuple, Optional, Any

# ─── CONFIG ───────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

NUM_MODELS = 5
MODEL_AGREEMENT_THRESHOLD = 0.6   # 3/5 models agree → trust model


# ─── DATA TYPES ───────────────────────────────────────────────────────────────
Bin     = Set[str]
Lattice = List[Bin]


# ─── WORD NORMALIZER ──────────────────────────────────────────────────────────
HINDI_PUNCT = re.compile(r'[।!?,;:()\[\]"\'-]')


def normalize(text: str) -> List[str]:  # type: ignore # pyre-ignore
    """Normalize and tokenize text into word list."""
    text = HINDI_PUNCT.sub(' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.split() if text else []


# ─── ALIGNMENT (DIFF-BASED) ───────────────────────────────────────────────────
def align_sequences(seq_a: List[str], seq_b: List[str]) -> List[Tuple[str, List[str], List[str]]]:  # type: ignore # pyre-ignore
    """
    Align two word sequences using difflib SequenceMatcher.
    Returns list of (tag, a_words, b_words) tuples.
    Tags: 'equal', 'replace', 'insert', 'delete'
    """
    matcher = difflib.SequenceMatcher(None, seq_a, seq_b, autojunk=False)
    result  = []
    for op, a0, a1, b0, b1 in matcher.get_opcodes():
        result.append((op, seq_a[a0:a1], seq_b[b0:b1]))  # type: ignore # pyre-ignore
    return result


# ─── LATTICE CONSTRUCTION ─────────────────────────────────────────────────────
def build_lattice(model_outputs: List[str], reference: str, model_names: Optional[List[str]] = None) -> Tuple[Lattice, Dict[str, Any]]:  # type: ignore # pyre-ignore
    """
    Build a transcription lattice from multiple ASR model outputs.

    Algorithm:
    1. Tokenize all model outputs and the reference.
    2. Use the REFERENCE as the alignment backbone (position anchor).
    3. For each reference position, collect all model alternatives (via alignment).
    4. If a majority (≥60%) of models agree on a word that DIFFERS from reference,
       add it to the bin AND mark it as a "trusted model alternative."
    5. Handle insertions and deletions.

    Returns:
        lattice: list of Bins (each bin = set of valid alternatives at that position)
        metadata: alignment stats, agreement info
    """
    ref_words       = normalize(reference)
    model_word_lists = [normalize(o) for o in model_outputs]

    if model_names is None:
        model_names = [f"model_{i+1}" for i in range(len(model_outputs))]  # type: ignore # pyre-ignore

    # Step 1: Align each model to the reference
    alignments = []
    for i, model_words in enumerate(model_word_lists):
        ops = align_sequences(ref_words, model_words)
        alignments.append((model_names[i], model_words, ops))  # type: ignore # pyre-ignore

    # Step 2: Build position map
    pos_alternatives: Dict[int, List[str]] = defaultdict(list)  # type: ignore # pyre-ignore
    insertion_positions: Dict[int, List[str]] = defaultdict(list)  # type: ignore # pyre-ignore

    for model_name, model_words, ops in alignments:
        ref_pos: int = 0
        for tag, a_words, b_words in ops:
            if tag == 'equal':
                for a_w, b_w in zip(a_words, b_words):
                    pos_alternatives[ref_pos].append(b_w)  # type: ignore # pyre-ignore
                    ref_pos += 1  # type: ignore # pyre-ignore
            elif tag == 'replace':
                for j, a_w in enumerate(a_words):
                    if j < len(b_words):  # type: ignore # pyre-ignore
                        pos_alternatives[ref_pos].append(b_words[j])  # type: ignore # pyre-ignore
                    ref_pos += 1  # type: ignore # pyre-ignore
            elif tag == 'delete':
                for a_w in a_words:
                    pos_alternatives[ref_pos].append("<DEL>")  # type: ignore # pyre-ignore
                    ref_pos += 1  # type: ignore # pyre-ignore
            elif tag == 'insert':
                insertion_positions[ref_pos].extend(b_words)  # type: ignore # pyre-ignore

    # Step 3: Build lattice bins
    lattice: Lattice = []
    metadata: Dict[str, Any] = {"bins": [], "trusted_model_overrides": []}

    for i, ref_word in enumerate(ref_words):
        alternatives = pos_alternatives.get(i, [])  # type: ignore # pyre-ignore
        total_models = len(model_outputs)

        alt_counts: Dict[str, int] = defaultdict(int)
        for a in alternatives:
            alt_counts[a] += 1  # type: ignore # pyre-ignore

        bin_set: Set[str] = {ref_word}

        trusted_overrides = []
        for alt_word, count in alt_counts.items():
            agreement = count / total_models  # type: ignore # pyre-ignore
            if alt_word != ref_word and alt_word != "<DEL>":
                bin_set.add(alt_word)
            if agreement >= MODEL_AGREEMENT_THRESHOLD and alt_word != ref_word:
                trusted_overrides.append(
                    f"{alt_word} ({count}/{total_models} models agree)"
                )

        lattice.append(bin_set)
        bin_meta = {
            "position":           i,
            "reference":          ref_word,
            "alternatives_count": len(bin_set),
            "alternatives":       list(bin_set),
        }
        if trusted_overrides:
            bin_meta["trusted_model_overrides"] = trusted_overrides  # type: ignore # pyre-ignore
            metadata["trusted_model_overrides"].append({  # type: ignore # pyre-ignore
                "position": i,
                "ref_word": ref_word,
                "overrides": trusted_overrides,
            })
        metadata["bins"].append(bin_meta)  # type: ignore # pyre-ignore

    return lattice, metadata


# ─── LATTICE-BASED WER ────────────────────────────────────────────────────────
def lattice_wer(hypothesis: List[str], lattice: Lattice) -> Tuple[float, Dict[str, Any]]:  # type: ignore # pyre-ignore
    """
    Compute WER for a hypothesis against a lattice reference.

    Modified edit distance:
    - Substitution cost = 0 if hypothesis word is IN the lattice bin (valid alt)
    - Substitution cost = 1 otherwise
    - Insertion/deletion costs = 1 (standard)

    Returns (wer, stats_dict).
    """
    N = len(lattice)
    M = len(hypothesis)

    dp: List[List[int]] = [[0] * (M + 1) for _ in range(N + 1)]  # type: ignore # pyre-ignore

    for i in range(N + 1):
        dp[i][0] = i  # type: ignore # pyre-ignore
    for j in range(M + 1):
        dp[0][j] = j  # type: ignore # pyre-ignore

    for i in range(1, N + 1):
        for j in range(1, M + 1):
            lattice_bin = lattice[i - 1]  # type: ignore # pyre-ignore
            hyp_word    = hypothesis[j - 1]  # type: ignore # pyre-ignore
            sub_cost    = 0 if hyp_word in lattice_bin else 1  # type: ignore # pyre-ignore
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + sub_cost)  # type: ignore # pyre-ignore

    edit_dist = dp[N][M]  # type: ignore # pyre-ignore
    wer       = edit_dist / max(N, 1)  # type: ignore # pyre-ignore
    stats     = {
        "edit_distance": edit_dist,
        "ref_length":    N,
        "hyp_length":    M,
        "wer":           round(wer * 100, 2),  # type: ignore # pyre-ignore
    }
    return wer, stats


def standard_wer(hypothesis: List[str], reference: List[str]) -> float:  # type: ignore # pyre-ignore
    """Standard WER using edit distance against rigid reference."""
    N = len(reference)
    M = len(hypothesis)
    dp: List[List[int]] = [[0] * (M + 1) for _ in range(N + 1)]  # type: ignore # pyre-ignore
    for i in range(N + 1):
        dp[i][0] = i  # type: ignore # pyre-ignore
    for j in range(M + 1):
        dp[0][j] = j  # type: ignore # pyre-ignore
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            sub      = 0 if reference[i-1] == hypothesis[j-1] else 1  # type: ignore # pyre-ignore
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+sub)  # type: ignore # pyre-ignore
    return dp[N][M] / max(N, 1)  # type: ignore # pyre-ignore


# ─── DEMO ─────────────────────────────────────────────────────────────────────
def run_demo():  # type: ignore # pyre-ignore
    """
    Demo from Q4 problem statement.
    Spoken: "उसने चौदह किताबें खरीदीं"
    """
    print("=" * 70)
    print("  Q4 DEMO — Lattice-based WER")
    print("  Spoken audio: 'उसने चौदह किताबें खरीदीं' (He bought 14 books)")
    print("=" * 70)

    reference = "उसने चौदह किताबें खरीदीं"

    model_outputs = {
        "Model A (ASR-1)": "उसने चौदह किताबें खरीदी",
        "Model B (ASR-2)": "उसने 14 किताबें खरीदीं",
        "Model C (ASR-3)": "उसने चौदह किताबे खरीदी",
        "Model D (ASR-4)": "उसने चौदह पुस्तकें खरीदीं",
        "Model E (ASR-5)": "उसने 14 किताबें खरीदी",
    }

    print(f"\n  Reference : {reference}")
    print(f"  Model outputs:")
    for name, out in model_outputs.items():
        print(f"    {name}: {out}")

    names   = list(model_outputs.keys())
    outputs = list(model_outputs.values())
    lattice, meta = build_lattice(outputs, reference, model_names=names)

    print(f"\n  Lattice Bins (each bin = valid alternatives at that position):")
    for i, (bin_set, ref_word) in enumerate(zip(lattice, normalize(reference))):
        alts = " | ".join(sorted(bin_set))  # type: ignore # pyre-ignore
        print(f"    Position {i+1} [ref: {ref_word}] → {{ {alts} }}")  # type: ignore # pyre-ignore

    ref_words = normalize(reference)
    print(f"\n  {'Model':<22} {'Standard WER':>14} {'Lattice WER':>12} {'Change':>8}")
    print(f"  {'─'*60}")

    for model_name, model_output in model_outputs.items():
        hyp  = normalize(model_output)
        std  = standard_wer(hyp, ref_words) * 100  # type: ignore # pyre-ignore
        lat, _ = lattice_wer(hyp, lattice)
        lat  = lat * 100  # type: ignore # pyre-ignore
        delta = std - lat  # type: ignore # pyre-ignore
        arrow = f"↓{delta:.0f}%" if delta > 0 else ("=" if delta == 0 else f"↑{abs(delta):.0f}%")  # type: ignore # pyre-ignore
        print(f"  {model_name:<22} {std:>12.1f}%   {lat:>10.1f}%   {arrow:>8}")

    if meta["trusted_model_overrides"]:
        print(f"\n  Trusted model overrides (where model majority disagrees with reference):")
        for override in meta["trusted_model_overrides"]:  # type: ignore # pyre-ignore
            print(f"    Position {override['position']+1}: {override['ref_word']} → {override['overrides']}")  # type: ignore # pyre-ignore

    print("\n  ✅ Lattice WER is equal or lower for all models.")
    print("     Models producing valid alternatives are no longer unfairly penalized.")
    print("=" * 70)


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():  # type: ignore # pyre-ignore
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo",  action="store_true",
                        help="Run demo from Q4 problem statement")
    parser.add_argument("--input", type=str,
                        help="Path to JSON with model outputs and reference")
    args = parser.parse_args()

    if args.input:
        with open(args.input, encoding="utf-8") as f:
            data = json.load(f)
        reference   = data["reference"]
        model_outs  = data["model_outputs"]
        names, outs = list(model_outs.keys()), list(model_outs.values())
        lattice, meta = build_lattice(outs, reference, model_names=names)
        ref_words = normalize(reference)
        for name, out in model_outs.items():
            hyp = normalize(out)
            std = standard_wer(hyp, ref_words) * 100  # type: ignore # pyre-ignore
            lat, _ = lattice_wer(hyp, lattice)
            print(f"{name}: Standard WER={std:.1f}%  Lattice WER={lat*100:.1f}%")  # type: ignore # pyre-ignore
    else:
        run_demo()


if __name__ == "__main__":
    main()
