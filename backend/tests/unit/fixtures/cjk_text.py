"""Invented Japanese speech for the CJK word-floor tests (#7065).

``JA_SENTENCE`` holds twenty-eight CJK letters and two full stops: fourteen
words by the shared floor, one by ``str.split``. ``ja_text`` builds
punctuation-free text with an exact word count under that rule (two CJK
letters per word).
"""

from utils.text_utils import CJK_LETTERS_PER_WORD

JA_SENTENCE = '明日の会議は十時から始まります。資料は田中さんが準備します。'
JA_LETTERS = JA_SENTENCE.replace('。', '')
JA_SENTENCE_WORDS = len(JA_LETTERS) // CJK_LETTERS_PER_WORD


def ja_text(words: int) -> str:
    """Japanese text the shared floor counts as exactly ``words`` words."""
    letters = words * CJK_LETTERS_PER_WORD
    return (JA_LETTERS * (letters // len(JA_LETTERS) + 1))[:letters]
