import unicodedata
from typing import Iterable, Iterator, Set

# ---------------------------------------------------------------------------
# Word counting
# ---------------------------------------------------------------------------
# Every transcript word floor in the backend counts words with
# ``transcript_word_count`` (the streaming Dream floor applies the same
# per-character rule through ``is_cjk_letter``). Han, kana and the other
# unspaced CJK scripts put no spaces between words, so ``len(text.split())``
# counts a whole sentence as one token and a floor that reads it never opens for
# Japanese or Chinese speech (#7065). #7104 introduced the approximation for the
# relevance floor: two CJK characters make one word. Here it is applied per
# character: the count is the whitespace tokens of everything that is not a
# CJK letter plus half the CJK letters. Text with no wide (East Asian W/F/H)
# characters counts exactly ``len(text.split())``; wide punctuation, symbols and
# emoji are separators, never words; Hangul and fullwidth Latin are letters of
# spaced scripts and count by whitespace; and appending text never lowers the
# count.
_WIDE_EAST_ASIAN_WIDTHS = frozenset({'W', 'F', 'H'})
# Wide letters whose scripts put spaces between words: Hangul (incl. halfwidth)
# and the fullwidth ASCII digits and letters. Token material, not CJK letters.
_SPACED_WIDE_LETTER_RANGES = (
    (0x1100, 0x11FF),
    (0x3130, 0x318F),
    (0xA960, 0xA97F),
    (0xAC00, 0xD7FF),
    (0xFFA0, 0xFFDC),
    (0xFF10, 0xFF19),
    (0xFF21, 0xFF3A),
    (0xFF41, 0xFF5A),
)
CJK_LETTERS_PER_WORD = 2
_SEPARATOR, _TOKEN, _CJK_LETTER = 0, 1, 2


def _char_class(char: str) -> int:
    if char.isspace():
        return _SEPARATOR
    if unicodedata.east_asian_width(char) not in _WIDE_EAST_ASIAN_WIDTHS:
        return _TOKEN
    if unicodedata.category(char)[0] not in 'LN':
        return _SEPARATOR  # wide punctuation, symbols, emoji, ideographic space
    code = ord(char)
    if any(low <= code <= high for low, high in _SPACED_WIDE_LETTER_RANGES):
        return _TOKEN
    return _CJK_LETTER


def is_cjk_letter(char: str) -> bool:
    """A letter or digit of an unspaced CJK script (Han, kana, bopomofo, ...)."""
    return _char_class(char) == _CJK_LETTER


def _running_word_counts(chars: Iterable[str]) -> Iterator[int]:
    """Yield the word count so far after each character."""
    tokens, cjk_letters, in_token = 0, 0, False
    for char in chars:
        kind = _char_class(char)
        if kind == _CJK_LETTER:
            cjk_letters += 1
            in_token = False
        elif kind == _SEPARATOR:
            in_token = False
        elif not in_token:
            tokens += 1
            in_token = True
        yield tokens + cjk_letters // CJK_LETTERS_PER_WORD


def transcript_word_count(text: str) -> int:
    """Word count shared by every transcript word floor.

    Whitespace tokens of the non-CJK text plus one word per ``CJK_LETTERS_PER_WORD``
    CJK letters. Text with no wide characters counts exactly ``len(text.split())``.
    """
    if not text:
        return 0
    if text.isascii():
        return len(text.split())
    count = 0
    for count in _running_word_counts(text):
        pass
    return count


def last_words(text: str, count: int) -> str:
    """The suffix of ``text`` holding its last ``count`` words under the same rule.

    Text without CJK letters keeps the classic ``' '.join(text.split()[-count:])``.
    Otherwise the shortest suffix that counts ``count`` words is kept, so a trim
    bounds a CJK transcript instead of returning all of it.
    """
    if count <= 0 or not text:
        return ''
    if text.isascii() or not any(is_cjk_letter(char) for char in text):
        return ' '.join(text.split()[-count:])
    for offset, seen in enumerate(_running_word_counts(reversed(text))):
        if seen >= count:
            index = len(text) - 1 - offset
            # A whitespace token is counted at its last character; keep all of it.
            while index > 0 and _char_class(text[index]) == _TOKEN and _char_class(text[index - 1]) == _TOKEN:
                index -= 1
            return text[index:]
    return text


def _normalize_text(text: str) -> str:
    """Normalize text: canonical NFC composition, lowercase, and collapse whitespace."""
    return ' '.join(unicodedata.normalize('NFC', text).lower().split())


def _get_trigrams(text: str) -> Set[str]:
    """Get character trigrams from normalized text."""
    text = _normalize_text(text)
    if len(text) < 3:
        return {text} if text else set()
    return {text[i : i + 3] for i in range(len(text) - 2)}


def compute_text_similarity(text1: str, text2: str) -> float:
    """
    Compute text similarity using character trigram Jaccard.
    Language-agnostic: works for all languages including CJK.

    Returns:
        Similarity score 0.0 to 1.0 (1.0 = identical)
    """
    trigrams1 = _get_trigrams(text1)
    trigrams2 = _get_trigrams(text2)

    if not trigrams1 or not trigrams2:
        return 0.0

    return len(trigrams1 & trigrams2) / len(trigrams1 | trigrams2)


def compute_text_containment(transcript: str, expected: str) -> float:
    """
    Compute containment of transcript trigrams within expected text.
    Language-agnostic: works for all languages including CJK.

    Args:
        transcript: Transcript text to check for containment
        expected: Expected text that should contain the transcript

    Returns:
        Containment score 0.0 to 1.0 (1.0 = fully contained)
    """
    transcript_norm = _normalize_text(transcript)
    expected_norm = _normalize_text(expected)

    if not transcript_norm:
        return 0.0
    if len(transcript_norm) < 3:
        return 1.0 if transcript_norm in expected_norm else 0.0

    trigrams_transcript = _get_trigrams(transcript)
    trigrams_expected = _get_trigrams(expected)

    if not trigrams_transcript:
        return 0.0

    return len(trigrams_transcript & trigrams_expected) / len(trigrams_transcript)
