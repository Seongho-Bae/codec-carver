"""Regression contract for server-authoritative target-size UI limits."""

import asyncio
import unittest

try:
    import saas_web

    _HAS_WEB = True
except (ImportError, RuntimeError):
    saas_web = None
    _HAS_WEB = False


@unittest.skipUnless(_HAS_WEB, "web integration dependencies are unavailable")
class TestTargetLimitContract(unittest.TestCase):
    """Keep client target-size validation bound to the server authority."""

    def test_rendered_ui_uses_server_target_limit(self):
        """Render both target controls from MAX_TARGET_BYTES without literals."""

        previous_limit = saas_web.MAX_TARGET_BYTES
        saas_web.MAX_TARGET_BYTES = 12_345
        try:
            html = asyncio.run(saas_web.get_ui())
        finally:
            saas_web.MAX_TARGET_BYTES = previous_limit

        self.assertEqual(html.count('max="12345"'), 2)
        self.assertIn("const MAX_TARGET_BYTES = 12345;", html)
        self.assertEqual(html.count("val > MAX_TARGET_BYTES"), 2)
        self.assertEqual(html.count("formatBinaryBytes(MAX_TARGET_BYTES)"), 2)
        self.assertNotIn("5368709120", html)

    def test_numeric_input_validation_uses_browser_number_semantics(self):
        """Keep exponent notation such as 6e9 from being truncated to 6."""

        html = asyncio.run(saas_web.get_ui())

        self.assertEqual(html.count("const val = this.valueAsNumber;"), 2)
        self.assertNotIn("parseInt(this.value, 10)", html)


if __name__ == "__main__":
    unittest.main()
