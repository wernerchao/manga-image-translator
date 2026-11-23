"""DeepSeek must mark lines it could not get an answer for, never return them as empty strings.

Empty strings pass for a translation: the caller never falls back, and the page ships with its
speech bubbles erased and nothing written in them.
"""
import asyncio
from unittest.mock import AsyncMock

import httpx
import openai
import pytest

from manga_translator.translators.deepseek import DeepseekTranslator

FAILED = '__FAILED_TO_TRANSLATE__: '
QUERIES = ['こんにちは', 'ありがとう', 'どこへ行くの？', '待って！']


def _status_error(cls, status):
    request = httpx.Request('POST', 'https://api.deepseek.com/chat/completions')
    return cls('rejected', response=httpx.Response(status, request=request), body=None)


@pytest.fixture
def translator(monkeypatch):
    monkeypatch.setattr(openai, 'api_key', 'test-key')
    return DeepseekTranslator()


def _translate(translator, queries=QUERIES):
    return asyncio.run(translator._translate('JPN', 'FRA', queries))


def test_rejected_key_marks_every_line_after_one_request(translator):
    translator._request_translation = AsyncMock(
        side_effect=_status_error(openai.AuthenticationError, 401))

    assert _translate(translator) == [FAILED + q for q in QUERIES]
    # Retrying or splitting the batch can't fix a rejected key.
    assert translator._request_translation.await_count == 1


def test_request_errors_mark_every_line_after_the_retries(translator):
    translator._request_translation = AsyncMock(
        side_effect=_status_error(openai.InternalServerError, 503))

    assert _translate(translator) == [FAILED + q for q in QUERIES]
    assert translator._request_translation.await_count == translator._RETRY_ATTEMPTS


def test_one_failed_request_then_an_answer_translates(translator):
    answer = '<|1|>Bonjour\n<|2|>Merci\n<|3|>Où vas-tu ?\n<|4|>Attends !'
    translator._request_translation = AsyncMock(
        side_effect=[_status_error(openai.InternalServerError, 503), answer])

    assert _translate(translator) == ['Bonjour', 'Merci', 'Où vas-tu ?', 'Attends !']


def test_an_answer_with_an_empty_line_is_not_marked_failed(translator):
    # DeepSeek answered: an empty line can be a real translation of noise, so it stays empty.
    translator._request_translation = AsyncMock(return_value='<|1|>')

    assert _translate(translator, ['…']) == ['']
