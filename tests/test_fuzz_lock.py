"""Keep the Atheris fuzz lock installable for central coverage-evidence."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
FUZZ_LOCK = ROOT / "fuzz" / "requirements-fuzz.txt"
FUZZ_IN = ROOT / "fuzz" / "requirements-fuzz.in"


class FuzzLockTests(unittest.TestCase):
    """Reject retired Atheris pins that fail trusted lock preflight."""

    def test_fuzz_lock_pins_published_atheris(self) -> None:
        """Require atheris 3.1.0; PyPI no longer publishes 3.0.0."""

        lock = FUZZ_LOCK.read_text(encoding="utf-8")
        spec = FUZZ_IN.read_text(encoding="utf-8")

        self.assertIn("atheris==3.1.0", spec)
        self.assertIn("atheris==3.1.0", lock)
        self.assertNotIn("atheris==3.0.0", spec)
        self.assertNotIn("atheris==3.0.0", lock)
        self.assertGreaterEqual(lock.count("--hash=sha256:"), 3)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
