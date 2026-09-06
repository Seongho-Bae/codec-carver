"""Regression contracts for original output-path collision probing."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import media_shrinker


class ResolveCollisionOriginalPathTests(unittest.TestCase):
    """The original output name is occupied unless lstat proves ENOENT."""

    def test_dangling_symlink_original_path_is_not_treated_as_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "output.flac"
            path.symlink_to("missing-target.flac")

            resolved = media_shrinker._resolve_collision(path, overwrite=False)

            self.assertEqual(resolved, Path(tmp) / "output-1.flac")


if __name__ == "__main__":
    unittest.main()
