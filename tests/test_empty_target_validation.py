"""Focused contracts for target-size validation feedback state."""

from __future__ import annotations

import unittest
from pathlib import Path


SOURCE_TEXT = (Path(__file__).resolve().parents[1] / "saas_web.py").read_text(
    encoding="utf-8"
)


class EmptyTargetValidationTests(unittest.TestCase):
    """Both target-size inputs must expose and then clear required feedback."""

    @staticmethod
    def _handler_between(start_marker: str, end_marker: str) -> str:
        """Return one JavaScript handler body from the web source text."""

        start = SOURCE_TEXT.index(start_marker)
        end = SOURCE_TEXT.index(end_marker, start)
        return SOURCE_TEXT[start:end]

    def test_single_target_empty_branch_sets_required_feedback(self) -> None:
        """The single-file target handler sets required preview and accessibility state."""

        handler = self._handler_between(
            "document.getElementById('target_bytes').addEventListener('input'",
            "document.getElementById('batch_target_bytes').addEventListener('input'",
        )
        self._assert_empty_branch(handler)

    def test_batch_target_empty_branch_sets_required_feedback(self) -> None:
        """The batch target handler applies the identical empty-state contract."""

        handler = self._handler_between(
            "document.getElementById('batch_target_bytes').addEventListener('input'",
            "document.getElementById('shrink-form').addEventListener('submit'",
        )
        self._assert_empty_branch(handler)

    def test_exactly_two_empty_target_branches_exist(self) -> None:
        """No unrelated input handler inherits the target-size special case."""

        self.assertEqual(SOURCE_TEXT.count("if (this.value === '') {"), 2)

    def _assert_empty_branch(self, handler: str) -> None:
        """Assert one handler resets stale error state before validating its value."""

        empty_marker = "if (this.value === '') {"
        invalid_marker = "if (isNaN(val) || val <= 0) {"
        self.assertIn(empty_marker, handler)
        reset_prefix = handler[: handler.index(empty_marker)]
        self.assertIn("preview.classList.remove('error-text');", reset_prefix)
        self.assertNotIn("preview.classList.remove('required-star');", reset_prefix)
        self.assertIn("preview.innerText = 'This field is required.';", handler)
        self.assertIn("preview.classList.add('error-text');", handler)
        self.assertIn("this.setCustomValidity('This field is required.');", handler)
        self.assertIn("this.setAttribute('aria-invalid', 'true');", handler)
        self.assertIn(
            "return;",
            handler[handler.index(empty_marker) : handler.index(invalid_marker)],
        )
        self.assertLess(handler.index(empty_marker), handler.index(invalid_marker))


if __name__ == "__main__":
    unittest.main()
