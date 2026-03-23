"""
Q2 - English Word Detector
JoshTalks AI Researcher Intern Assessment

Detects English words in Hindi transcripts and wraps them with [EN]...[/EN] tags.
Important guideline: English words transcribed in Devanagari script (e.g., कंप्यूटर)
are considered CORRECT per JoshTalks transcription guidelines — do NOT tag them.
Only tag Latin-script English words embedded in Hindi text.
"""

import re
import unicodedata
from pathlib import Path

# ─── UNICODE RANGES ───────────────────────────────────────────────────────────
DEVANAGARI_RANGE = re.compile(r'[\u0900-\u097F]')    # Hindi script
LATIN_RANGE      = re.compile(r'[A-Za-z]')            # English script
DIGIT_RANGE      = re.compile(r'\d')

# A token is considered "English" if it:
#   1. Contains at least one Latin character, AND
#   2. Is not a common punctuation or number
LATIN_TOKEN      = re.compile(r'^[A-Za-z][A-Za-z0-9\'-]*$')

# ─── COMMON HINDI SHORT WORDS THAT LOOK LATIN (false positives) ─────────────
# These are romanized Hindi words sometimes used in mixed-script messages
# but NOT actual English words — we exclude them from tagging
FALSE_POSITIVE_TOKENS = {
    "ji", "hai", "ho", "na", "nhi", "haan", "thik",
    "ok", "hmm", "ah",   # conversational markers
}

# ─── COMMON ENGLISH STOPWORDS (always English, always tag) ────────────────────
DEFINITE_ENGLISH = {
    "the", "is", "are", "was", "were", "i", "you", "he", "she",
    "we", "they", "this", "that", "it", "and", "or", "but",
    "not", "have", "has", "had", "will", "would", "can", "could",
    "should", "may", "might", "do", "does", "did", "a", "an",
    "in", "on", "at", "by", "for", "from", "to", "of", "with",
}


# ─── PER-TOKEN CLASSIFIER ─────────────────────────────────────────────────────
def is_english_word(token: str) -> bool:
    """
    Determine if a token is an English word that should be tagged.

    Rules:
    1. Must contain Latin characters.
    2. Must NOT be a false-positive (ji, hai, ok, hmm…).
    3. Definite English words → always tag.
    4. Mixed-script or pure Devanagari → never tag.
    """
    # Strip punctuation from edges for classification
    clean = token.strip(".,!?;:\"'()[]{}")

    if not clean:
        return False

    # Must contain at least one Latin char
    if not LATIN_RANGE.search(clean):
        return False

    # If it also contains Devanagari it's mixed-script → don't tag
    if DEVANAGARI_RANGE.search(clean):
        return False

    # Must match a valid token pattern
    if not LATIN_TOKEN.match(clean):
        return False

    lower = clean.lower()

    # Definite English
    if lower in DEFINITE_ENGLISH:
        return True

    # Known false positives
    if lower in FALSE_POSITIVE_TOKENS:
        return False

    # Default: tag it (Latin-only token in Hindi text = English word)
    return True


# ─── SENTENCE-LEVEL TAGGER ────────────────────────────────────────────────────
def tag_english_words(text: str) -> tuple[str, list[str]]:
    """
    Process a Hindi sentence and tag English words with [EN]...[/EN].

    Per guidelines:
    - Devanagari-script English (कंप्यूटर, इंटरव्यू) → NOT tagged (correct as-is)
    - Latin-script English embedded in Hindi sentence → TAGGED

    Returns:
        tagged_text: string with [EN]...[/EN] markers
        detected_words: list of English words detected
    """
    # Tokenize while preserving whitespace
    tokens    = re.split(r'(\s+)', text)
    output    = []
    detected  = []

    for token in tokens:
        if re.match(r'\s+', token):
            output.append(token)
            continue

        if is_english_word(token):
            # Preserve surrounding punctuation
            match = re.match(r'^([,।!?;:"\'\(\)\[\]]*)(.*?)([,।!?;:"\'\(\)\[\]]*)$', token)
            if match:
                pre, word, post = match.groups()
                output.append(f"{pre}[EN]{word}[/EN]{post}")
            else:
                output.append(f"[EN]{token}[/EN]")
            detected.append(token.strip(".,!?;:"))
        else:
            output.append(token)

    return "".join(output), detected


# ─── BATCH PROCESSOR ──────────────────────────────────────────────────────────
def process_transcript(utterances: list[dict]) -> list[dict]:
    """
    Process a list of utterance dicts (with 'text' key).
    Returns each utterance with added 'tagged_text' and 'english_words' fields.
    """
    results = []
    for utt in utterances:
        text   = utt.get("text", "")
        tagged, eng_words = tag_english_words(text)
        results.append({
            **utt,
            "tagged_text":   tagged,
            "english_words": eng_words,
        })
    return results


# ─── DEMO / TEST ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        # From the question
        "मेरा इंटरव्यू बहुत अच्छा गया और मुझे जॉब मिल गई",
        # English words in Latin script
        "मेरा interview बहुत अच्छा गया और मुझे job मिल गई",
        # Code-mixed with English proper nouns
        "मैंने Google और Amazon में apply किया है",
        # Devanagari-script English (should NOT be tagged)
        "मुझे कंप्यूटर पर work करना अच्छा लगता है",
        # Mostly Hindi
        "तुम कहाँ से हो और क्या करते हो",
        # Numbers don't get tagged
        "मुझे 25 साल हो गए हैं",
        # Mixed example
        "यह problem solve नहीं हो रहा था",
    ]

    print("English Word Detector — Test Cases")
    print("=" * 70)
    for text in test_cases:
        tagged, words = tag_english_words(text)
        print(f"IN : {text}")
        print(f"OUT: {tagged}")
        if words:
            print(f"EN words detected: {words}")
        else:
            print("EN words detected: (none)")
        print()
