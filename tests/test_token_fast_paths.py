"""Equivalence tests for transcript and summary token fast paths."""

from __future__ import annotations

import random
import string
import unittest

import summarize
import transcript_search


def reference_content_words(sentence: str) -> list[str]:
    """Return tokens using the pre-fast-path summarizer algorithm."""

    words: list[str] = []
    for raw in sentence.split():
        token = summarize._TOKEN_STRIP_RE.sub("", raw).lower()
        if token and token not in summarize._STOPWORDS:
            words.append(token)
    return words


def reference_search_tokens(text: str) -> list[str]:
    """Return tokens using the pre-fast-path search algorithm."""

    return transcript_search._WORD_RE.findall(text.lower())


class _FailingWordMatcher:
    """Reject any regular-expression call made on the all-alphanumeric path."""

    def findall(self, text: str) -> list[str]:
        """Fail with the unexpected candidate text for a useful regression trace."""

        raise AssertionError(f"regular expression must not inspect {text!r}")


class _RecordingWordMatcher:
    """Delegate matching while retaining every candidate sent to the regex."""

    def __init__(self, delegate) -> None:
        """Store the real matcher and initialize an ordered call receipt."""

        self._delegate = delegate
        self.calls: list[str] = []

    def findall(self, text: str) -> list[str]:
        """Record one candidate and return the real matcher result."""

        self.calls.append(text)
        return self._delegate.findall(text)


class TokenFastPathEquivalenceTests(unittest.TestCase):
    """Optimized tokenization must remain semantically identical."""

    def test_curated_multilingual_and_punctuation_cases(self) -> None:
        """Representative scripts and punctuation preserve exact output."""

        cases = [
            "Topic123",
            "Résumé42",
            "한글123",
            "東京2026",
            "alpha_beta",
            "can't stop",
            "(wrapped) punctuation!",
            "GreekΑ CyrillicА LatinA",
            "１２３ fullwidth",
            "emoji🙂boundary",
            "the and content",
            "",
        ]
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(
                    summarize._content_words(text),
                    reference_content_words(text),
                )
                self.assertEqual(
                    transcript_search.tokenize(text),
                    reference_search_tokens(text),
                )

    def test_seeded_randomized_equivalence(self) -> None:
        """A deterministic hostile character mix exercises both branches."""

        rng = random.Random(20260807)
        alphabet = (
            string.ascii_letters
            + string.digits
            + " _-'.,!?/\\()[]{}"
            + "éßΑА한글東京１２３🙂"
        )
        for index in range(512):
            text = "".join(
                rng.choice(alphabet) for _ in range(rng.randrange(0, 80))
            )
            with self.subTest(index=index, text=text):
                self.assertEqual(
                    summarize._content_words(text),
                    reference_content_words(text),
                )
                self.assertEqual(
                    transcript_search.tokenize(text),
                    reference_search_tokens(text),
                )

    def test_search_tokenizer_skips_regex_for_each_alphanumeric_token(self) -> None:
        """Whitespace-separated multilingual words independently use the fast path."""

        original = transcript_search._WORD_RE
        transcript_search._WORD_RE = _FailingWordMatcher()
        try:
            self.assertEqual(
                transcript_search.tokenize("Alpha123 한글456 東京2026"),
                ["alpha123", "한글456", "東京2026"],
            )
        finally:
            transcript_search._WORD_RE = original

    def test_search_tokenizer_sends_only_mixed_tokens_to_regex(self) -> None:
        """Punctuation fallback receives one token instead of the complete sentence."""

        original = transcript_search._WORD_RE
        matcher = _RecordingWordMatcher(original)
        transcript_search._WORD_RE = matcher
        try:
            self.assertEqual(
                transcript_search.tokenize("Alpha123 wrapped! 東京2026"),
                ["alpha123", "wrapped", "東京2026"],
            )
            self.assertEqual(matcher.calls, ["wrapped!"])
        finally:
            transcript_search._WORD_RE = original


if __name__ == "__main__":
    unittest.main()
