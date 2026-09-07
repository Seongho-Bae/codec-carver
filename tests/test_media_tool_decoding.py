"""Real subprocess tests for hostile ffmpeg and ffprobe output bytes."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import media_shrinker
from media_shrinker import MediaShrinkerError, SilenceInterval


@unittest.skipIf(os.name == "nt", "POSIX executable fixtures use a shebang")
class MediaToolDecodingTests(unittest.TestCase):
    """The shared subprocess boundary must decode untrusted bytes tolerantly."""

    def _tool(
        self,
        *,
        stdout: bytes = b"",
        stderr: bytes = b"",
        returncode: int = 0,
    ) -> str:
        """Create an executable Python tool that emits exact raw byte streams."""

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "media_tool.py"
        path.write_text(
            "#!/usr/bin/env python3\n"
            "import os\n"
            f"os.write(1, {stdout!r})\n"
            f"os.write(2, {stderr!r})\n"
            f"raise SystemExit({returncode})\n",
            encoding="utf-8",
        )
        path.chmod(0o700)
        return str(path)

    def test_shared_runner_replaces_invalid_stdout_and_stderr(self) -> None:
        """Invalid bytes become replacement text without changing exit status."""

        completed = media_shrinker._run_media_tool(
            [self._tool(stdout=b"value\xff", stderr=b"warning\xfe")],
            tool="fixture",
            timeout=30,
        )

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stdout, "value\ufffd")
        self.assertEqual(completed.stderr, "warning\ufffd")

    def test_probe_media_parses_json_with_legacy_encoded_tag(self) -> None:
        """A bad metadata byte cannot prevent parsing valid ffprobe structure."""

        payload = (
            b'{"format":{"duration":"2.5","size":"4",'
            b'"format_name":"wav","tags":{"title":"caf\xff"}},'
            b'"streams":[{"codec_type":"audio","codec_name":"pcm_s16le",'
            b'"bit_rate":"128000"}]}'
        )
        probe = media_shrinker.probe_media(
            Path("source.wav"),
            ffprobe_path=self._tool(stdout=payload),
            source_size=4,
        )

        self.assertEqual(probe.duration_seconds, 2.5)
        self.assertEqual(probe.size_bytes, 4)
        self.assertEqual(probe.audio_codec, "pcm_s16le")
        self.assertFalse(probe.has_video)

    def test_silence_detection_preserves_ascii_markers(self) -> None:
        """Replacement decoding leaves silencedetect timestamps parseable."""

        stderr = (
            b"Metadata title: caf\xff\xfe\n"
            b"silence_start: 100.0\n"
            b"silence_end: 160.5 | silence_duration: 60.5\n"
        )
        intervals = media_shrinker.detect_silence_intervals(
            Path("source.wav"),
            ffmpeg_path=self._tool(stderr=stderr),
        )

        self.assertEqual(
            intervals,
            [SilenceInterval(start_seconds=100.0, end_seconds=160.5)],
        )

    def test_nonzero_error_remains_a_typed_media_error(self) -> None:
        """Nonzero execution is reported as MediaShrinkerError, not decode failure."""

        with self.assertRaises(MediaShrinkerError) as raised:
            media_shrinker.probe_media(
                Path("source.wav"),
                ffprobe_path=self._tool(stderr=b"bad\xffmetadata", returncode=7),
                source_size=4,
            )

        self.assertIn("ffprobe failed", str(raised.exception))
        self.assertIn("bad", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
