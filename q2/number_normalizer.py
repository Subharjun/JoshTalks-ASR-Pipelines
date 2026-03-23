"""
Q2 - Number Normalizer
JoshTalks AI Researcher Intern Assessment

Converts Hindi spoken number words into digits.
Handles simple, compound, large numbers, and edge cases (idioms).
"""

import re

# ─── WORD MAPS ────────────────────────────────────────────────────────────────
ONES = {
    "शून्य": 0, "एक": 1, "दो": 2, "तीन": 3, "चार": 4,
    "पाँच": 5, "पांच": 5, "छह": 6, "छः": 6, "सात": 7,
    "आठ": 8, "नौ": 9, "दस": 10, "ग्यारह": 11, "बारह": 12,
    "तेरह": 13, "चौदह": 14, "पंद्रह": 15, "सोलह": 16,
    "सत्रह": 17, "अठारह": 18, "उन्नीस": 19,
}
TENS = {
    "बीस": 20, "इक्कीस": 21, "बाईस": 22, "तेईस": 23,
    "चौबीस": 24, "पच्चीस": 25, "छब्बीस": 26, "सत्ताईस": 27,
    "अट्ठाईस": 28, "उनतीस": 29, "तीस": 30, "इकतीस": 31,
    "बत्तीस": 32, "तैंतीस": 33, "चौंतीस": 34, "पैंतीस": 35,
    "छत्तीस": 36, "सैंतीस": 37, "अड़तीस": 38, "उनचालीस": 39,
    "चालीस": 40, "इकतालीस": 41, "बयालीस": 42, "तैंतालीस": 43,
    "चवालीस": 44, "पैंतालीस": 45, "छियालीस": 46, "सैंतालीस": 47,
    "अड़तालीस": 48, "उनचास": 49, "पचास": 50, "इक्यावन": 51,
    "बावन": 52, "तिरपन": 53, "चौवन": 54, "पचपन": 55,
    "छप्पन": 56, "सत्तावन": 57, "अट्ठावन": 58, "उनसठ": 59,
    "साठ": 60, "इकसठ": 61, "बासठ": 62, "तिरसठ": 63,
    "चौंसठ": 64, "पैंसठ": 65, "छियासठ": 66, "सड़सठ": 67,
    "अड़सठ": 68, "उनहत्तर": 69, "सत्तर": 70, "इकहत्तर": 71,
    "बहत्तर": 72, "तिहत्तर": 73, "चौहत्तर": 74, "पचहत्तर": 75,
    "छिहत्तर": 76, "सतहत्तर": 77, "अठहत्तर": 78, "उनासी": 79,
    "अस्सी": 80, "इक्यासी": 81, "बयासी": 82, "तिरासी": 83,
    "चौरासी": 84, "पचासी": 85, "छियासी": 86, "सत्तासी": 87,
    "अट्ठासी": 88, "नवासी": 89, "नब्बे": 90, "इक्यानवे": 91,
    "बानवे": 92, "तिरानवे": 93, "चौरानवे": 94, "पचानवे": 95,
    "छियानवे": 96, "सत्तानवे": 97, "अट्ठानवे": 98, "निन्यानवे": 99,
}
MULTIPLIERS = {
    "सौ":    100,
    "हज़ार":  1_000,
    "हजार":  1_000,
    "लाख":   1_00_000,
    "करोड़":  1_00_00_000,
}

ALL_NUMBER_WORDS = {**ONES, **TENS}

# ─── IDIOMATIC PATTERNS (DO NOT CONVERT) ─────────────────────────────────────
IDIOMATIC_PATTERNS = [
    re.compile(r'दो[‐\-]चार\s+बात'),
    re.compile(r'दो[‐\-]चार'),
    re.compile(r'एक[‐\-]दो'),
    re.compile(r'तीन[‐\-]चार'),
    re.compile(r'चार[‐\-]पाँच'),
    re.compile(r'आठ[‐\-]दस'),
    re.compile(r'दस[‐\-]बीस'),
    re.compile(r'एक न एक'),
    re.compile(r'एक ही'),
]


def is_idiomatic(text, start, end):  # type: ignore
    """Check if the matched number span is part of an idiom."""
    window = text[max(0, start - 10): end + 15]  # type: ignore
    for pattern in IDIOMATIC_PATTERNS:
        if pattern.search(window):
            return True
    return False


# ─── CORE NUMBER PARSER ───────────────────────────────────────────────────────
def words_to_number(tokens):  # type: ignore
    """
    Convert a list of Hindi number tokens to an integer.
    E.g., ["तीन", "सौ", "चौवन"] → 354
    """
    result: int = 0
    current: int = 0

    for token in tokens:
        if token in ONES:
            current += ONES[token]  # type: ignore
        elif token in TENS:
            current += TENS[token]  # type: ignore
        elif token in MULTIPLIERS:
            mult = MULTIPLIERS[token]
            if mult == 100:
                current = (current if current > 0 else 1) * 100  # type: ignore
            else:
                if current == 0:
                    current = 1
                result += current * mult  # type: ignore
                current = 0
        else:
            return None

    result += current  # type: ignore
    return result if result > 0 else None


# ─── MAIN NORMALIZER ──────────────────────────────────────────────────────────
_ALL_WORDS_PATTERN = re.compile(
    r'\b(' +
    '|'.join(
        re.escape(w)
        for w in sorted(
            list(ALL_NUMBER_WORDS.keys()) + list(MULTIPLIERS.keys()),
            key=len, reverse=True
        )
    ) +
    r')(\s+(' +
    '|'.join(
        re.escape(w)
        for w in sorted(
            list(ALL_NUMBER_WORDS.keys()) + list(MULTIPLIERS.keys()),
            key=len, reverse=True
        )
    ) +
    r'))*\b'
)


def normalize_numbers(text):  # type: ignore
    """
    Replace Hindi number words with digits in text.
    Returns (normalized_text, list of change records).
    Skips idiomatic expressions.
    """
    changes = []
    offset: int = 0
    result = text

    for m in _ALL_WORDS_PATTERN.finditer(text):
        span_text  = m.group(0)
        tokens     = span_text.split()
        start, end = m.start(), m.end()

        if is_idiomatic(text, start, end):
            changes.append({
                "original":  span_text,
                "converted": span_text,
                "reason":    "Idiomatic expression — preserved",
                "skipped":   True,
            })
            continue

        number = words_to_number(tokens)
        if number is None:
            continue

        digit_str = str(number)
        adj_start = start + offset  # type: ignore
        adj_end   = end   + offset  # type: ignore

        result = result[:adj_start] + digit_str + result[adj_end:]  # type: ignore
        offset += len(digit_str) - len(span_text)  # type: ignore

        changes.append({
            "original":  span_text,
            "converted": digit_str,
            "reason":    f"Standard number conversion: {span_text} → {digit_str}",
            "skipped":   False,
        })

    return result, changes


# ─── DEMO / TEST ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        ("दो",              "2",   False),
        ("दस",              "10",  False),
        ("सौ",              "100", False),
        ("तीन सौ चौवन",     "354", False),
        ("पच्चीस",          "25",  False),
        ("एक हज़ार",         "1000", False),
        ("दो-चार बातें करना",   None,  True),
        ("एक-दो काम बाकी हैं",  None,  True),
        ("मैं तीस साल का हूँ",  None,  False),
    ]

    print("Number Normalizer — Test Cases")
    print("=" * 60)
    for original, expected, is_idiom in test_cases:
        normalized, changes = normalize_numbers(original)
        skipped = any(c["skipped"] for c in changes)
        status  = "✅" if (skipped == is_idiom) else "⚠️"
        print(f"{status}  IN : {original}")
        print(f"    OUT: {normalized}")
        if changes:
            for c in changes:
                note = "PRESERVED (idiom)" if c["skipped"] else f"→ {c['converted']}"
                print(f"    [{note}] {c['reason']}")
        print()
