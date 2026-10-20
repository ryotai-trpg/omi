"""CJK transcripts must open the same word floors as English (#7065).

Han and kana put no spaces between words, so ``len(text.split())`` counts a
whole sentence as one token and every whitespace word floor stayed shut for
Japanese and Chinese speech. The floors now share
``utils.text_utils.transcript_word_count``: the whitespace tokens of the
non-CJK text plus one word per two CJK letters, the approximation #7104
introduced for the relevance floor applied per character. No threshold
changes; text without CJK letters counts exactly ``len(text.split())``, and
Hangul keeps its spaces.

Every utterance below is invented.
"""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from config import episodes as episodes_config
from config.episode_writer import EpisodeWriterSettings
from models.agent_tools import HarnessEdit
from models.transcript_segment import TranscriptSegment
from utils.agent_capabilities import guards
from utils.conversations.episode_tiers import episode_tier
from utils.episodes import incremental as episodes_incremental
from utils.episodes import moment_render
from utils.episodes.live import MomentTracker
from utils.conversations.recovery import (
    RECOVERY_QUIET_MINIMUM_MAX_WORDS,
    recovery_transcript_below_content_floor,
    recovery_transcript_word_count,
)
from utils.llm import followup
from utils.stt import pre_recorded
from utils import translation_coordinator
from utils import onboarding as onboarding_mod
from utils.agent_capabilities import cleanup_edits
from utils.conversations import relevance_jev
from utils.llm import conversation_processing
from utils import text_utils
from utils.text_utils import has_at_least_words, last_words, transcript_word_count
from tests.unit.fixtures.cjk_text import JA_SENTENCE as JA, JA_SENTENCE_WORDS, ja_text

EN = 'The meeting tomorrow starts at ten. Tanaka will prepare the materials.'  # 11 words
ZH = '明天下午三点开会。'
KO = '회의는 내일입니다.'


# ── the shared counter ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    'text',
    ['', '   ', EN, "don't stop — it's fine...", 'word ' * 40, 'Không có tiêu đề', 'café  crème\n brûlée'],
)
def test_text_without_wide_characters_counts_whitespace_tokens_exactly(text):
    assert transcript_word_count(text) == len(text.split())


def test_wide_punctuation_and_symbols_separate_tokens_in_any_script():
    # The one way a text without CJK letters differs from ``str.split``: wide
    # punctuation, symbols and emoji are separators, never words.
    assert transcript_word_count('hello👋world') == 2
    assert transcript_word_count('hello，world') == 2
    assert transcript_word_count('wave 👋 bye') == 2


def test_spaced_wide_letters_count_by_whitespace():
    assert transcript_word_count('ＡＢＣＤ') == 1
    assert transcript_word_count('ｍｅｅｔｉｎｇ ｔｏｍｏｒｒｏｗ') == 2
    assert transcript_word_count('ﾡﾢﾣﾤ') == 1  # halfwidth Hangul


@pytest.mark.parametrize('text,words', [(JA, 14), (ZH, 4), ('User: ' + JA, 15)])
def test_cjk_text_counts_two_letters_per_word(text, words):
    assert transcript_word_count(text) == words
    assert len(text.split()) <= 2  # what every floor used to see


def test_japanese_sentence_counts_like_its_english_translation():
    assert transcript_word_count(JA) == JA_SENTENCE_WORDS == 14
    assert transcript_word_count(EN) == 11


KO_12_WORDS = '오늘 회의는 열 시에 시작합니다 자료는 김 씨가 준비합니다 감사합니다 네 알겠습니다'


def test_hangul_keeps_its_whitespace_words():
    assert transcript_word_count(KO) == len(KO.split()) == 2
    assert transcript_word_count('네 알겠습니다 감사합니다') == 3
    assert transcript_word_count(KO_12_WORDS) == len(KO_12_WORDS.split()) == 12


def test_korean_floors_move_from_the_syllable_count_to_the_whitespace_count():
    # #7104 counted Korean at two syllables per word (KO_12_WORDS was 17); the
    # relevance and recovery floors now see the same whitespace count as every
    # other floor always did for Korean.
    assert recovery_transcript_below_content_floor(_conversation(KO_12_WORDS))
    assert not recovery_transcript_below_content_floor(_conversation(KO_12_WORDS + ' 끝'))


def test_cjk_punctuation_and_symbols_are_separators_not_words():
    assert transcript_word_count('はい！！！') == 1
    assert transcript_word_count('はい。はい。') == 2
    assert transcript_word_count('。。。') == 0
    assert transcript_word_count('great 🎉 party') == 2


def test_mixed_script_text_counts_both_scripts():
    assert transcript_word_count('明日のmeetingはten o clockから始まる') == 8
    assert transcript_word_count('Speaker 0: はい') == 3


@pytest.mark.parametrize('head', ['a b c d e f g h', '明日の会議', 'yes - ', ''])
@pytest.mark.parametrize('tail', ['明日の会議は十時', ' more words', '。', '회의는', 'x'])
def test_appending_text_never_lowers_the_count(head, tail):
    assert transcript_word_count(head + tail) >= transcript_word_count(head)


def test_ja_text_helper_hits_exact_counts():
    assert [transcript_word_count(ja_text(n)) for n in (1, 10, 25, 40, 120, 1500)] == [1, 10, 25, 40, 120, 1500]


@pytest.mark.parametrize(
    ('text', 'words'),
    [
        ('是的.是的.是的.', 3),  # like 是的。是的。是的。: the ASCII full stops belong to the words
        ('はい.', 1),
        ('会議(meeting)', 2),
        ('明日の会議は10:00から', 5),
        ('葛\U000e0100城', 1),  # a variation selector is not a word
        ('ｶﾞｷﾞｸﾞ', 1),  # halfwidth voiced marks modify their letters, like ガギグ
    ],
)
def test_narrow_punctuation_after_a_cjk_letter_belongs_to_that_word(text, words):
    assert transcript_word_count(text) == words


def test_space_segmented_han_never_counts_below_its_tokens():
    # Per-character and word-segmented Chinese rows keep their whitespace count.
    assert transcript_word_count('我 想 去 北 京') == 5
    assert transcript_word_count('我 想 去 北京') == 4
    assert transcript_word_count('我想去北京') == 2
    assert transcript_word_count('え') == transcript_word_count('うん') == 1


def test_fast_path_regex_covers_every_assigned_wide_code_point():
    # Text that matches nothing is counted by str.split alone, so the class must be a superset.
    import sys
    import unicodedata

    missed = [
        code
        for code in range(sys.maxunicode + 1)
        if unicodedata.category(chr(code)) != 'Cn'
        and unicodedata.east_asian_width(chr(code)) in ('W', 'F', 'H')
        and text_utils._MAYBE_WIDE_RE.search(chr(code)) is None
    ]
    assert missed == []
    assert text_utils._MAYBE_WIDE_RE.search('don\u2019t stop caf\u00e9') is None


@pytest.mark.parametrize('text', [EN, JA, KO_12_WORDS, '明日のmeetingはten o clockから始まる', ''])
def test_has_at_least_words_matches_the_count(text):
    count = transcript_word_count(text)
    assert has_at_least_words(text, count)
    assert not has_at_least_words(text, count + 1)
    assert has_at_least_words(text, 0)


# ── relevance floor (KEEP_WORD_COUNT) ────────────────────────────────────────


def test_keep_word_count_opens_for_japanese_without_a_model_call():
    long_ja = ja_text(conversation_processing.KEEP_WORD_COUNT + 1)
    with patch.object(conversation_processing, 'get_llm', side_effect=AssertionError('no model call')):
        assert conversation_processing.should_discard_conversation(long_ja) is False
    assert not relevance_jev.jev_tier_applies(long_ja)
    assert relevance_jev.jev_tier_applies(ja_text(conversation_processing.KEEP_WORD_COUNT))


# ── recovery quiet minimum ────────────────────────────────────────────────────


def _conversation(*texts):
    return {'transcript_segments': [{'text': text} for text in texts]}


def test_recovery_counts_cjk_segments_with_the_shared_floor():
    assert recovery_transcript_word_count(_conversation(JA, JA)) == transcript_word_count(JA + ' ' + JA) == 28
    assert recovery_transcript_word_count(_conversation(EN, ' ', 'word')) == 12
    assert recovery_transcript_word_count(_conversation()) == 0


def test_recovery_content_floor_sees_cjk_speech():
    assert recovery_transcript_below_content_floor(_conversation('はい、明日。'))
    assert recovery_transcript_below_content_floor(_conversation(ja_text(RECOVERY_QUIET_MINIMUM_MAX_WORDS)))
    assert not recovery_transcript_below_content_floor(_conversation(ja_text(RECOVERY_QUIET_MINIMUM_MAX_WORDS + 1)))


# ── episode tier volume ───────────────────────────────────────────────────────


def _item(kind, content=''):
    return SimpleNamespace(source_kind=kind, content=content)


def test_episode_tier_volume_counts_japanese_speech():
    settings = EpisodeWriterSettings(tiered=True)
    assert settings.tier_min_words == 1500
    roster = _item('roster', 'name')
    assert episode_tier([_item('speech', ja_text(1500)), roster], settings)[1:] == ('C6', 'speech_and_source_volume')
    assert episode_tier([_item('speech', ja_text(1499)), roster], settings)[1] == 'C7'
    assert episode_tier([_item('speech', 'word ' * 1500), roster], settings)[1] == 'C6'
    assert episode_tier([_item('speech', 'word ' * 1499), roster], settings)[1] == 'C7'


# ── Dream summary admission ───────────────────────────────────────────────────


def test_summary_floor_opens_for_japanese_speech():
    minimum = guards.MIN_SUMMARY_TRANSCRIPT_WORDS
    assert minimum == 40
    assert guards.has_summary_speech(ja_text(minimum), minimum)
    assert not guards.has_summary_speech(ja_text(minimum - 1), minimum)
    assert not guards.has_summary_speech(JA, minimum)
    assert guards.has_summary_speech(JA, 0)


def test_summary_floor_is_unchanged_for_non_cjk_speech():
    minimum = guards.MIN_SUMMARY_TRANSCRIPT_WORDS
    assert guards.has_summary_speech('word ' * minimum, minimum)
    assert not guards.has_summary_speech('word ' * (minimum - 1), minimum)
    # Alphanumeric runs still decide: forty whitespace tokens of "yes -" are twenty words.
    assert not guards.has_summary_speech('yes - ' * 20, minimum)
    assert guards.has_summary_speech('crème brûlée ' * 20, minimum)


@pytest.mark.parametrize('kind', ['title', 'overview'])
def test_summary_field_admission_reads_japanese_transcripts(kind):
    row = {
        'structured': {'title': '', 'overview': ''},
        'transcript_segments': [{'text': ja_text(20)}, {'text': ja_text(20)}],
    }
    assert guards.summary_field_rejection(kind, row) is None
    row['transcript_segments'] = [{'text': JA}]
    assert guards.summary_field_rejection(kind, row) == 'insufficient_speech'


# ── follow-up question floor ──────────────────────────────────────────────────


def _segments(text):
    return [TranscriptSegment(text=text, speaker='SPEAKER_00', is_user=True, start=0.0, end=1.0)]


def _followup(text):
    llm = MagicMock()
    llm.invoke.return_value = SimpleNamespace(content='What time should we meet?')
    with patch.object(followup, 'get_llm', return_value=llm), patch.object(
        followup, 'track_usage', return_value=nullcontext()
    ):
        result = followup.followup_question_prompt('uid-1', _segments(text))
    return result, llm.invoke


def test_followup_question_floor_opens_for_japanese_speech():
    result, invoke = _followup(JA)
    assert result == 'What time should we meet?'
    assert JA in invoke.call_args.args[0]
    result, invoke = _followup('はい、明日。')
    assert result == '' and not invoke.called


def test_followup_question_floor_is_unchanged_for_english():
    # The rendered transcript starts with the "User:" speaker label, which the
    # whitespace count has always included as a token.
    result, invoke = _followup('word ' * 8)
    assert result == '' and not invoke.called
    result, invoke = _followup('word ' * 9)
    assert result == 'What time should we meet?'


def test_followup_trim_bounds_a_long_japanese_transcript():
    opening = '最初の部分です。'
    result, invoke = _followup(opening + JA * 20)
    assert result == 'What time should we meet?'
    prompt = invoke.call_args.args[0]
    assert opening not in prompt
    assert JA[-10:] in prompt


# ── the shared tail ───────────────────────────────────────────────────────────


def test_last_words_keeps_the_classic_whitespace_trim_for_non_cjk():
    text = 'w0 ' * 150 + "don't  stop\n now"
    assert last_words(text, 100) == ' '.join(text.split()[-100:])
    assert last_words('a b c', 10) == 'a b c'
    assert last_words('a b c', 0) == ''
    assert last_words('', 5) == ''


def test_last_words_keeps_whole_latin_tokens_inside_cjk_text():
    assert last_words('これは apple です', 2) == 'apple です'
    assert last_words('今日は apple banana です', 3) == 'apple banana です'
    assert last_words('今日は apple banana です', 1) == 'です'


def test_last_words_bounds_cjk_text_by_the_same_approximation():
    text = '最初の部分です。' + JA * 20
    tail = last_words(text, 100)
    assert text.endswith(tail)
    assert transcript_word_count(tail) == 100
    assert '最初の部分' not in tail
    assert last_words(JA, 100) == JA


# ── Dream summary admission on long and mixed speech ──────────────────────────


def test_summary_floor_counts_long_and_mixed_japanese_speech():
    long_ja = ja_text(1500)
    assert guards.has_summary_speech(long_ja, guards.MIN_SUMMARY_TRANSCRIPT_WORDS)
    assert guards.has_summary_speech('word ' * 20 + ja_text(20), 40)
    assert not guards.has_summary_speech('word ' * 20 + ja_text(19), 40)
    assert guards.has_summary_speech(long_ja, 1500)
    assert not guards.has_summary_speech(long_ja, 1501)


# ── Dream cleanup: duplicate-capture merge and language floors ────────────────


def _capture(*texts, start='2026-10-20T10:00:00+00:00', end='2026-10-20T10:10:00+00:00'):
    return {
        'status': 'completed',
        'started_at': start,
        'finished_at': end,
        'language': '',
        'transcript_segments': [{'text': text} for text in texts],
    }


def test_duplicate_capture_merge_floor_opens_for_japanese_speech():
    cleanup_edits.require_duplicate_capture(_capture(JA, JA), _capture(JA, JA))
    with pytest.raises(ValueError, match='agent_capture_speech_insufficient'):
        cleanup_edits.require_duplicate_capture(_capture('はい、明日。'), _capture('はい、明日。'))
    cleanup_edits.require_duplicate_capture(_capture('word ' * 12), _capture('word ' * 12))
    with pytest.raises(ValueError, match='agent_capture_speech_insufficient'):
        cleanup_edits.require_duplicate_capture(_capture('word ' * 11), _capture('word ' * 11))


def _language_edit(after='ja'):
    return HarnessEdit(
        kind='set_language',
        target='conversations/c1',
        before='',
        after=after,
        reason='Invented',
        evidence=['conversations/c1'],
    )


def test_language_edit_floor_opens_for_japanese_speech():
    assert cleanup_edits.validate_cleanup(_language_edit(), {'conversations/c1': _capture(JA, JA)}) is None
    with pytest.raises(ValueError, match='agent_language_speech_insufficient'):
        cleanup_edits.validate_cleanup(_language_edit(), {'conversations/c1': _capture(JA)})
    assert cleanup_edits.validate_cleanup(_language_edit('en'), {'conversations/c1': _capture('word ' * 20)}) is None
    with pytest.raises(ValueError, match='agent_language_speech_insufficient'):
        cleanup_edits.validate_cleanup(_language_edit('en'), {'conversations/c1': _capture('word ' * 19)})


# ── onboarding answer fallback ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_onboarding_fallback_accepts_a_japanese_answer_when_the_model_fails():
    async def send_message(event):
        pass

    handler = onboarding_mod.OnboardingHandler('user-1', send_message)
    with patch.object(onboarding_mod, 'generated_compute_configured', return_value=False), patch.object(
        onboarding_mod, 'get_llm', side_effect=RuntimeError('provider down')
    ), patch.object(onboarding_mod, 'track_usage', return_value=nullcontext()):
        assert await handler._ai_check_answer('Where do you live?', '東京に住んでいます') is True
        assert await handler._ai_check_answer('Where do you live?', 'はい') is False
        assert await handler._ai_check_answer('Where do you live?', 'in Austin') is True
        assert await handler._ai_check_answer('Where do you live?', 'Austin') is False


# ── pre-recorded language votes ───────────────────────────────────────────────


def test_language_votes_count_japanese_utterances(monkeypatch):
    monkeypatch.setenv('MODULATE_API_KEY', 'test-key')
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {
        'utterances': [
            {'text': 'okay let us start', 'start_ms': 0, 'duration_ms': 1500, 'speaker': 1, 'language': 'en'},
            {'text': JA, 'start_ms': 2000, 'duration_ms': 6000, 'speaker': 2, 'language': 'ja'},
            {'text': JA, 'start_ms': 9000, 'duration_ms': 6000, 'speaker': 2, 'language': 'ja'},
        ]
    }
    client = MagicMock()
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    client.post.return_value = response
    with patch.object(pre_recorded.httpx, 'Client', return_value=client):
        words, language = pre_recorded.modulate_prerecorded_from_bytes(b'\x00' * 100, 16000, return_language=True)
    # One four-token English line used to silence every Japanese vote (one token each).
    assert language == 'ja'
    assert len(words) == 3


# ── live translation soft boundary ────────────────────────────────────────────


def test_translation_soft_boundary_counts_japanese_words():
    def signals(text):
        return translation_coordinator._compute_stability_signals(text, 0.0, 0.0, None, None)

    boundary = {translation_coordinator.STABILITY_SOFT_BOUNDARY}
    assert signals(ja_text(translation_coordinator.SOFT_BOUNDARY_TOKEN_COUNT)) == boundary
    assert signals(ja_text(translation_coordinator.SOFT_BOUNDARY_TOKEN_COUNT - 1)) == set()
    assert signals('w ' * translation_coordinator.SOFT_BOUNDARY_TOKEN_COUNT) == boundary
    assert signals('w ' * (translation_coordinator.SOFT_BOUNDARY_TOKEN_COUNT - 1)) == set()


# ── live episode moments ──────────────────────────────────────────────────────


def _first_moment(text):
    """Four batches of ``text`` ten seconds apart; returns the buffer's words before the fourth and the moments it ended."""
    tracker = MomentTracker()
    for wall in (0, 10, 20):
        assert tracker.observe('u', [{'text': text}], 'c1', 'omi', now=wall) == []
    buffered = tracker._buffers['u'].words
    return buffered, tracker.observe('u', [{'text': text}], 'c1', 'omi', now=30)


def test_first_live_moment_opens_for_japanese_speech_at_the_english_threshold():
    per_batch = episodes_config.FIRST_MOMENT_MIN_WORDS // 3
    en_words, en_ended = _first_moment(' '.join(['word'] * per_batch))
    ja_words, ja_ended = _first_moment(ja_text(per_batch))
    assert en_words == ja_words == 3 * per_batch
    assert [m.trigger for m in en_ended] == [m.trigger for m in ja_ended] == ['interval']
    # Each Japanese batch used to count one word, so no first moment came within the window.
    assert len(ja_text(per_batch).split()) == 1


def test_live_moment_rollover_counts_japanese_words():
    tracker = MomentTracker()
    tracker.observe('u', [{'text': ja_text(episodes_config.MOMENT_MAX_WORDS - 1)}], 'c1', 'omi', now=0)
    ended = tracker.observe('u', [{'text': ja_text(1)}], 'c1', 'omi', now=10)
    assert [m.trigger for m in ended] == ['rollover']


def test_rendered_moment_counts_japanese_words_with_the_shared_floor(monkeypatch):
    segments = [{'text': JA, 'is_user': True}, {'text': EN, 'speaker_id': 1}]
    expected = JA_SENTENCE_WORDS + len(EN.split())
    assert episodes_incremental.render_moment(segments).words == expected
    assert episodes_incremental.render_moment([{'text': EN, 'is_user': True}]).words == len(EN.split())
    # The NOTES_PACKET_V2_MODE serve renderer counts the same way.
    assert moment_render.render_segments(segments)[1] == expected
    monkeypatch.setattr(episodes_incremental, 'shadow_or_serve', lambda flag: True)
    monkeypatch.setattr(episodes_incremental, 'serving_for', lambda flag, uid=None: True)
    monkeypatch.setattr(episodes_incremental, 'get_evidence', lambda uid, conversation_id: None)
    monkeypatch.setattr(episodes_incremental, 'shadow', lambda *args, **kwargs: None)
    assert episodes_incremental.render_moment(segments, uid='u', conversation_id='c').words == expected
