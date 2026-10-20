import re
import unicodedata
from functools import lru_cache
from typing import Iterator, Set

# ---------------------------------------------------------------------------
# Word counting
# ---------------------------------------------------------------------------
# Every transcript word floor in the backend counts words with
# ``transcript_word_count`` (the Dream admission floor keeps its own
# alphanumeric-run count for other scripts and applies the CJK rule below
# through ``is_cjk_letter``). Han, kana and the other unspaced CJK scripts put
# no spaces between words, so ``len(text.split())`` counts a whole sentence as
# one token and a floor that reads it never opens for Japanese or Chinese
# speech (#7065). #7104 introduced the approximation for the relevance floor:
# two CJK characters make one word. Here it is applied per character:
#
#     whitespace tokens of the non-CJK text
#     + max(CJK letters // 2, whitespace chunks holding a CJK letter)
#
# Text with no wide (East Asian W/F/H) characters counts exactly
# ``len(text.split())``; wide punctuation, symbols and emoji are separators,
# never words; narrow punctuation trailing a CJK letter (``はい.``) belongs to
# that word; Hangul and fullwidth Latin are letters of spaced scripts and count
# by whitespace; space-segmented Han (``我 想 去 北京``) never counts below its
# token count; and appending text never lowers the count. Thai and the other
# unspaced narrow scripts are not covered.
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
# Halfwidth katakana voiced sound marks modify the letter before them.
_WIDE_MARK_CODES = frozenset({0xFF9E, 0xFF9F})
CJK_LETTERS_PER_WORD = 2
_SEPARATOR, _TOKEN, _CJK_LETTER, _PENDING = 0, 1, 2, 3
# Every assigned W/F/H code point lies in one of these ranges (pinned by a
# test), so text that matches nothing is counted by ``str.split`` alone.
_MAYBE_WIDE_RE = re.compile(
    '[\u1100-\u115f\u20a9\u2300-\u27ff\u2b00-\u2bff\u2e80-\ua4cf\ua960-\ua97f'
    '\uac00-\ud7ff\uf900-\ufaff\ufe10-\ufe6f\uff00-\uffef\U00010000-\U0010ffff]'
)


@lru_cache(maxsize=4096)
def _char_class(char: str) -> int:
    if char.isspace():
        return _SEPARATOR
    if unicodedata.east_asian_width(char) not in _WIDE_EAST_ASIAN_WIDTHS:
        return _TOKEN
    if unicodedata.category(char)[0] not in 'LN':
        return _SEPARATOR  # wide punctuation, symbols, emoji, ideographic space
    code = ord(char)
    if code in _WIDE_MARK_CODES:
        return _SEPARATOR
    if any(low <= code <= high for low, high in _SPACED_WIDE_LETTER_RANGES):
        return _TOKEN
    return _CJK_LETTER


def is_cjk_letter(char: str) -> bool:
    """A letter or digit of an unspaced CJK script (Han, kana, bopomofo, ...)."""
    return _char_class(char) == _CJK_LETTER


def _counted_units(text: str) -> Iterator[tuple[int, int, int]]:
    """Yield ``(index, kind, chunk)`` for each counted unit of ``text``.

    A unit is the first character of a whitespace token (``_TOKEN``) or one CJK
    letter (``_CJK_LETTER``); ``chunk`` numbers the whitespace-delimited run the
    unit sits in. A narrow run that follows a CJK letter or wide punctuation
    without whitespace counts as a token only once it shows a letter or digit;
    until then it is punctuation belonging to the word before it.
    """
    state, chunk = _SEPARATOR, 0
    for index, char in enumerate(text):
        if char.isspace():
            state = _SEPARATOR
            chunk += 1
            continue
        kind = _char_class(char)
        if kind == _CJK_LETTER:
            state = _CJK_LETTER
            yield index, _CJK_LETTER, chunk
        elif kind == _SEPARATOR:
            state = _PENDING
        elif state == _TOKEN:
            continue
        elif state != _SEPARATOR and not char.isalnum():
            state = _PENDING
        else:
            state = _TOKEN
            yield index, _TOKEN, chunk


def _running_word_counts(text: str) -> Iterator[tuple[int, int]]:
    """Yield ``(index, words so far)`` at each counted unit, in text order."""
    tokens, letters, chunks, last_chunk = 0, 0, 0, -1
    for index, kind, chunk in _counted_units(text):
        if kind == _TOKEN:
            tokens += 1
        else:
            letters += 1
            if chunk != last_chunk:
                chunks += 1
                last_chunk = chunk
        yield index, tokens + max(letters // CJK_LETTERS_PER_WORD, chunks)


def transcript_word_count(text: str) -> int:
    """Word count shared by every transcript word floor.

    Whitespace tokens of the non-CJK text plus one word per ``CJK_LETTERS_PER_WORD``
    CJK letters (never fewer than the whitespace chunks that hold a CJK letter).
    Text with no wide characters counts exactly ``len(text.split())``.
    """
    if not text:
        return 0
    if text.isascii() or _MAYBE_WIDE_RE.search(text) is None:
        return len(text.split())
    count = 0
    for _index, count in _running_word_counts(text):
        pass
    return count


def has_at_least_words(text: str, minimum: int) -> bool:
    """``transcript_word_count(text) >= minimum`` without counting past ``minimum``."""
    if minimum <= 0:
        return True
    if not text:
        return False
    if text.isascii() or _MAYBE_WIDE_RE.search(text) is None:
        return len(text.split()) >= minimum
    return any(count >= minimum for _index, count in _running_word_counts(text))


def last_words(text: str, count: int) -> str:
    """The suffix of ``text`` holding its last ``count`` words under the same rule.

    Text without wide characters keeps the classic ``' '.join(text.split()[-count:])``.
    Otherwise the shortest suffix starting at a counted unit that counts ``count``
    words is kept, so a trim bounds a CJK transcript instead of returning all of it.
    """
    if count <= 0 or not text:
        return ''
    if text.isascii() or _MAYBE_WIDE_RE.search(text) is None:
        return ' '.join(text.split()[-count:])
    units = list(_counted_units(text))
    tokens, letters, chunks, last_chunk = 0, 0, 0, -1
    for position in range(len(units) - 1, -1, -1):
        index, kind, chunk = units[position]
        if kind == _TOKEN:
            tokens += 1
        else:
            letters += 1
            if chunk != last_chunk:
                chunks += 1
                last_chunk = chunk
        if tokens + letters // CJK_LETTERS_PER_WORD >= count:
            return text[index:]
        # Reached only through the chunk floor: do not cut inside that chunk.
        opens_chunk = position == 0 or units[position - 1][2] != chunk
        if opens_chunk and tokens + chunks >= count:
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
