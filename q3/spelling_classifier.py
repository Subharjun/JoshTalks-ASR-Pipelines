"""
Q3 - Spelling Classifier for Hindi Words
JoshTalks AI Researcher Intern Assessment

Classifies ~1,77,000 unique Hindi words from the dataset as:
  - correct_spelling / incorrect_spelling
With confidence level: high / medium / low

Strategy (multi-signal ensemble):
  1. Dictionary lookup (indic-nlp / custom wordlist)
  2. Character n-gram language model (trained on clean Hindi text)
  3. Rule-based Devanagari phonotactics
  4. English words in Devanagari script → always CORRECT (per guidelines)

Run: python spelling_classifier.py --input wordlist.txt --output output/results.csv
     python spelling_classifier.py --sample 100   (demo on sample words)
"""

import re
import csv
import json
import math
import argparse
from pathlib import Path
from collections import Counter
from dataclasses import dataclass

# ─── CONFIG ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

DEVANAGARI = re.compile(r'^[\u0900-\u097F\s]+$')
LATIN      = re.compile(r'[A-Za-z]')
DIGIT      = re.compile(r'\d')

# ─── DATA CLASS ───────────────────────────────────────────────────────────────
@dataclass
class WordResult:
    word:       str
    label:      str   # "correct_spelling" | "incorrect_spelling"
    confidence: str   # "high" | "medium" | "low"
    signals:    dict  # breakdown of each signal


# ─── SIGNAL 1: DEVANAGARI PHONOTACTICS ────────────────────────────────────────
VALID_DEVANAGARI = re.compile(
    r'^[\u0900-\u0963\u0966-\u096F\u0970-\u097F]+$'
)
INVALID_PATTERNS = [
    re.compile(r'[\u093E][\u093E]'),
    re.compile(r'[\u093F][\u093F]'),
    re.compile(r'[\u0940][\u0940]'),
    re.compile(r'[\u094B][\u094B]'),
    re.compile(r'[\u094C][\u094C]'),
    re.compile(r'[\u0902][\u0902]'),
    re.compile(r'[\u0964][\u0964]'),
]
HALANT = '\u094D'


def phonotactic_check(word):  # type: ignore
    """
    Check if word has valid Devanagari phonotactic structure.
    Returns (is_valid, reason).
    """
    if not word:
        return False, "empty"

    if not VALID_DEVANAGARI.match(word):
        return False, "contains invalid characters"

    for pat in INVALID_PATTERNS:
        if pat.search(word):
            return False, f"invalid character combination: {pat.pattern}"

    halant_count = word.count(HALANT)
    if halant_count > 3 and len(word) < 8:  # type: ignore
        return False, "too many virama in short word"

    if len(word) == 1:
        return True, "single character (likely filler)"

    return True, "phonotactically valid"


# ─── SIGNAL 2: CHARACTER N-GRAM LANGUAGE MODEL ────────────────────────────────
class CharNgramLM:
    """
    Character-level bigram/trigram language model for Devanagari.
    Trained on a reference clean word list.
    """
    def __init__(self, n=3):  # type: ignore
        self.n = n
        self.counts = Counter()  # type: ignore
        self.context_counts = Counter()  # type: ignore
        self.vocab = set()  # type: ignore

    def train(self, words):  # type: ignore
        for word in words:
            padded = "^" * (self.n - 1) + word + "$"  # type: ignore
            for i in range(len(padded) - self.n + 1):  # type: ignore
                ngram   = padded[i : i + self.n]  # type: ignore
                context = ngram[:-1]  # type: ignore
                self.counts[ngram]          += 1  # type: ignore
                self.context_counts[context] += 1  # type: ignore
                self.vocab.update(ngram)

    def score(self, word):  # type: ignore
        """Return average log-probability per character (higher = more natural)."""
        if not self.counts:
            return 0.0

        padded    = "^" * (self.n - 1) + word + "$"  # type: ignore
        log_prob: float = 0.0
        steps: int      = 0
        V: int          = len(self.vocab)  # type: ignore

        for i in range(len(padded) - self.n + 1):  # type: ignore
            ngram   = padded[i : i + self.n]  # type: ignore
            context = ngram[:-1]  # type: ignore
            count   = self.counts.get(ngram, 0) + 1  # type: ignore
            total   = self.context_counts.get(context, 0) + V  # type: ignore
            log_prob += math.log(count / total)  # type: ignore
            steps    += 1  # type: ignore

        return log_prob / steps if steps > 0 else -10.0  # type: ignore


# ─── SIGNAL 3: COMMON HINDI WORDLIST ─────────────────────────────────────────
COMMON_HINDI_WORDS = {
    # Common verbs
    "है", "हैं", "था", "थी", "थे", "हो", "हूँ", "हूं", "होता", "होती",
    "होते", "करना", "करता", "करती", "करते", "जाना", "जाता", "जाती",
    "आना", "आता", "आती", "देना", "लेना", "बोलना", "कहना", "सुनना",
    "देखना", "जानना", "मानना", "पाना", "रहना", "रहता", "रहती", "रखना",
    # Common nouns
    "दिन", "रात", "समय", "काम", "जगह", "बात", "लोग", "आदमी", "औरत",
    "बच्चा", "घर", "देश", "शहर", "गाँव", "पानी", "खाना", "पैसा",
    "बाजार", "स्कूल", "कॉलेज", "सड़क", "गाड़ी", "किताब", "कमरा",
    # Common adjectives / adverbs
    "अच्छा", "बुरा", "बड़ा", "छोटा", "नया", "पुराना", "सही", "गलत",
    "बहुत", "थोड़ा", "जल्दी", "धीरे", "ऐसा", "वैसा", "कभी", "हमेशा",
    # Pronouns / particles
    "मैं", "तुम", "आप", "वो", "हम", "यह", "वह", "इस", "उस", "जो",
    "तो", "भी", "ही", "न", "नहीं", "हाँ", "और", "लेकिन", "क्योंकि",
    # Numbers (spoken)
    "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ", "दस",
}


def dictionary_check(word):  # type: ignore
    """Check if word is in known-correct Hindi dictionary."""
    if word in COMMON_HINDI_WORDS:
        return True, "found in dictionary"
    return None, "not in dictionary"


# ─── SIGNAL 4: DEVANAGARI ENGLISH WORDS ──────────────────────────────────────
DEVANAGARI_ENGLISH_HINTS = [
    re.compile(r'कंप्यूटर|कम्प्यूटर'),
    re.compile(r'इंटरव्यू|इंटरव्यु'),
    re.compile(r'मोबाइल|मोबाईल'),
    re.compile(r'इंटरनेट|इन्टरनेट'),
    re.compile(r'सर्विस|सर्विस'),
    re.compile(r'ऑफिस|ऑफ़िस'),
    re.compile(r'ऑनलाइन'),
    re.compile(r'एप्लिकेशन|अप्लिकेशन'),
    re.compile(r'[ऑॉ]'),
]


def is_devanagari_english(word):  # type: ignore
    """Check if word is an English word written in Devanagari (always correct)."""
    for pat in DEVANAGARI_ENGLISH_HINTS:
        if pat.search(word):
            return True
    return False


# ─── MAIN CLASSIFIER ─────────────────────────────────────────────────────────
class SpellingClassifier:
    def __init__(self):  # type: ignore
        self.lm = CharNgramLM(n=3)
        self._train_lm()
        self.lm_threshold_high   = -1.8
        self.lm_threshold_medium = -2.5

    def _train_lm(self):  # type: ignore
        """Train n-gram LM on known correct words."""
        self.lm.train(list(COMMON_HINDI_WORDS))

    def classify(self, word):  # type: ignore
        word = word.strip()
        signals = {}  # type: ignore

        # ── Devanagari English loanwords (always correct) ─────────────────
        if is_devanagari_english(word):
            signals["devanagari_english"] = True  # type: ignore
            return WordResult(
                word=word,
                label="correct_spelling",
                confidence="high",
                signals=signals,
            )

        # ── Phonotactic check ─────────────────────────────────────────────
        phono_ok, phono_reason = phonotactic_check(word)
        signals["phonotactic"] = {"ok": phono_ok, "reason": phono_reason}  # type: ignore

        if not phono_ok:
            return WordResult(
                word=word,
                label="incorrect_spelling",
                confidence="high",
                signals=signals,
            )

        # ── Dictionary check ──────────────────────────────────────────────
        dict_result, dict_reason = dictionary_check(word)
        signals["dictionary"] = {"found": dict_result, "reason": dict_reason}  # type: ignore

        if dict_result is True:
            return WordResult(
                word=word,
                label="correct_spelling",
                confidence="high",
                signals=signals,
            )

        # ── LM score ──────────────────────────────────────────────────────
        lm_score = self.lm.score(word)
        signals["lm_score"] = round(lm_score, 4)  # type: ignore

        if lm_score >= self.lm_threshold_high:
            return WordResult(
                word=word,
                label="correct_spelling",
                confidence="high",
                signals=signals,
            )
        elif lm_score >= self.lm_threshold_medium:
            return WordResult(
                word=word,
                label="correct_spelling",
                confidence="medium",
                signals=signals,
            )
        else:
            return WordResult(
                word=word,
                label="incorrect_spelling",
                confidence="low",
                signals=signals,
            )

    def classify_batch(self, words):  # type: ignore
        return [self.classify(w) for w in words]


# ─── SAVE TO CSV ──────────────────────────────────────────────────────────────
def save_csv(results, path):  # type: ignore
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "word", "spelling_label", "confidence", "signals"
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "word":           r.word,
                "spelling_label": r.label,
                "confidence":     r.confidence,
                "signals":        json.dumps(r.signals, ensure_ascii=False),
            })
    print(f"Saved {len(results)} results to {path}")


# ─── STATS PRINTER ────────────────────────────────────────────────────────────
def print_stats(results):  # type: ignore
    total:  int = len(results)
    correct: int = sum(1 for r in results if r.label == "correct_spelling")  # type: ignore
    wrong: int   = total - correct  # type: ignore
    high: int    = sum(1 for r in results if r.confidence == "high")  # type: ignore
    medium: int  = sum(1 for r in results if r.confidence == "medium")  # type: ignore
    low: int     = sum(1 for r in results if r.confidence == "low")  # type: ignore

    print(f"\n{'='*50}")
    print(f"  SPELLING CLASSIFICATION SUMMARY")
    print(f"{'='*50}")
    print(f"  Total words   : {total:,}")
    print(f"  Correct       : {correct:,}  ({correct/total*100:.1f}%)")  # type: ignore
    print(f"  Incorrect     : {wrong:,}  ({wrong/total*100:.1f}%)")  # type: ignore
    print(f"  Conf HIGH     : {high:,}  ({high/total*100:.1f}%)")  # type: ignore
    print(f"  Conf MEDIUM   : {medium:,}  ({medium/total*100:.1f}%)")  # type: ignore
    print(f"  Conf LOW      : {low:,}  ({low/total*100:.1f}%)")  # type: ignore
    print(f"{'='*50}\n")

    low_words = [r for r in results if r.confidence == "low"][:20]  # type: ignore
    if low_words:
        print("  Sample LOW-confidence words:")
        for r in low_words:
            print(f"    {r.word:<25} {r.label:<22} lm={r.signals.get('lm_score', 'N/A')}")  # type: ignore


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():  # type: ignore
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  type=str,
                        help="Path to word list (one word per line)")
    parser.add_argument("--output", type=str,
                        default=str(OUTPUT_DIR / "spelling_results.csv"))
    parser.add_argument("--sample", type=int,
                        help="Classify N sample words from built-in list")
    parser.add_argument("--word", type=str,
                        help="Check spelling for a single word")
    args = parser.parse_args()

    clf = SpellingClassifier()

    if args.word:
        words = [args.word]
        print(f"Checking single word: '{args.word}'")
    elif args.input:
        with open(args.input, encoding="utf-8") as f:
            words = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(words):,} words from {args.input}")
    else:
        words = list(COMMON_HINDI_WORDS) + [
            "अच्चा", "बहूत", "हमेशाा", "करनाा", "तीीन",
            "अभिलाषा", "पराक्रम", "आकांक्षा",
            "कंप्यूटर", "ऑनलाइन", "मोबाइल",
        ]
        if args.sample:
            test_cases = words[-11:] # type: ignore # pyre-ignore
            words = words[:args.sample - len(test_cases)] + test_cases # type: ignore # pyre-ignore
        print(f"Using {len(words)} sample words (built-in + misspelling test cases)")

    results = clf.classify_batch(words)
    print_stats(results)
    save_csv(results, Path(args.output))


if __name__ == "__main__":
    main()
