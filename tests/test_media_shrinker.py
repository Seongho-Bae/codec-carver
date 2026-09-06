import argparse
import json
import os
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

import media_shrinker
import presets
from media_shrinker import (
    ConversionPlan,
    MediaSegment,
    MediaShrinkerError,
    detect_silence_intervals,
    MediaProbe,
    SilenceInterval,
    build_audio_plan,
    build_icloud_download_command,
    build_opus_plan,
    build_segments,
    calculate_audio_bitrate,
    choose_worker_count,
    find_candidates,
    parse_silencedetect_intervals,
    write_report,
    preserve_file_attributes,
    probe_media,
    _execute_plan,
    _first_float,
    _first_int,
)


def _write_fake_ffmpeg(path: Path, output: bytes = b"converted") -> Path:
    """Create a tiny ffmpeg stand-in that writes the final argv path."""
    payload = repr(output)
    if os.name == "nt":
        path = path.with_suffix(".cmd")
        path.write_text(
            f'@echo off\n"{sys.executable}" -c "import pathlib, sys; pathlib.Path(sys.argv[-1]).write_bytes({payload})" %*\n',
            encoding="utf-8",
        )
        return path

    path.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        f"pathlib.Path(sys.argv[-1]).write_bytes({payload})\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _fake_lstat(mode: int, size: int = 0) -> MagicMock:
    result = MagicMock()
    result.st_mode = mode
    result.st_size = size
    return result


class FindCandidateTests(unittest.TestCase):
    def test_find_candidates_accepts_filesystem_root_without_extra_separator(
        self,
    ) -> None:
        with patch("media_shrinker.os.walk", return_value=[]):
            self.assertEqual(find_candidates(Path("/"), size_limit_bytes=1), [])

    def test_find_candidates_returns_supported_files_over_limit_case_insensitively(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            large_wav = root / "A.WAV"
            large_m4a = root / "nested" / "B.m4a"
            small_mp3 = root / "small.mp3"
            unsupported = root / "large.txt"

            large_m4a.parent.mkdir()
            large_wav.write_bytes(b"0" * 11)
            large_m4a.write_bytes(b"0" * 12)
            small_mp3.write_bytes(b"0" * 4)
            unsupported.write_bytes(b"0" * 99)

            candidates = [
                p[0].relative_to(root)
                for p in find_candidates(
                    root, size_limit_bytes=10, include_under_limit=False
                )
            ]

            self.assertEqual(candidates, [Path("A.WAV"), Path("nested/B.m4a")])

    def test_find_candidates_includes_under_limit_by_default_for_all_source_conversion(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            small_mp3 = root / "small.mp3"
            small_mp3.write_bytes(b"0" * 4)

            candidates = [
                p[0].relative_to(root)
                for p in find_candidates(root, size_limit_bytes=10)
            ]

            self.assertEqual(candidates, [Path("small.mp3")])

    def test_find_candidates_can_include_under_limit_and_skip_output_directory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.m4a"
            output = root / "under_2gb" / "source.flac"
            output.parent.mkdir()
            source.write_bytes(b"0" * 4)
            output.write_bytes(b"0" * 4)

            candidates = [
                p[0].relative_to(root)
                for p in find_candidates(
                    root,
                    size_limit_bytes=10,
                    include_under_limit=True,
                    exclude_paths=[output.parent],
                )
            ]

            self.assertEqual(candidates, [Path("source.m4a")])

    def test_find_candidates_skips_multiple_excluded_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            keep = root / "keep" / "source.wav"
            excluded = root / "excluded" / "hidden.wav"
            nested_excluded = root / "nested" / "output" / "hidden.wav"

            keep.parent.mkdir()
            excluded.parent.mkdir()
            nested_excluded.parent.mkdir(parents=True)
            keep.write_bytes(b"0" * 4)
            excluded.write_bytes(b"0" * 4)
            nested_excluded.write_bytes(b"0" * 4)

            candidates = [
                p[0].relative_to(root)
                for p in find_candidates(
                    root,
                    include_under_limit=True,
                    exclude_paths=[excluded.parent, nested_excluded.parent],
                )
            ]

            self.assertEqual(candidates, [Path("keep/source.wav")])

    def test_find_candidates_can_skip_generated_split_directories_by_prefix(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.wav"
            split = root / "split_over_1gb" / "source_part0001.wav"
            split.parent.mkdir()
            source.write_bytes(b"0" * 4)
            split.write_bytes(b"0" * 4)

            candidates = [
                p[0].relative_to(root)
                for p in find_candidates(
                    root,
                    include_under_limit=True,
                    exclude_dir_prefixes=["split_over"],
                )
            ]

            self.assertEqual(candidates, [Path("source.wav")])

    def test_find_candidates_skips_directory_when_current_dir_cannot_resolve(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = root / "good.mp3"
            broken = root / "broken"
            broken_media = broken / "hidden.mp3"

            broken.mkdir()
            good.write_bytes(b"0" * 4)
            broken_media.write_bytes(b"0" * 4)

            import os

            original_realpath = os.path.realpath

            def flaky_realpath(path: str, *args: object, **kwargs: object) -> str:
                if path == str(broken):
                    raise OSError("cannot resolve directory")
                return original_realpath(path, *args, **kwargs)

            with patch("os.path.realpath", flaky_realpath):
                candidates = [
                    p[0].relative_to(root)
                    for p in find_candidates(
                        root,
                        include_under_limit=True,
                        exclude_paths=[Path("/something")],
                    )
                ]

            self.assertEqual(candidates, [Path("good.mp3")])

    def test_find_candidates_skips_entries_when_symlink_check_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = root / "good.mp3"
            bad_file = root / "bad.mp3"
            bad_dir = root / "bad_dir"
            nested = bad_dir / "nested.mp3"

            bad_dir.mkdir()
            good.write_bytes(b"0" * 4)
            bad_file.write_bytes(b"0" * 4)
            nested.write_bytes(b"0" * 4)

            import os
            original_lstat = os.lstat

            def flaky_lstat(path):
                name = os.path.basename(str(path))
                if name in {"bad.mp3", "bad_dir"}:
                    raise OSError("cannot inspect symlink state")
                return original_lstat(path)

            with patch("os.lstat", flaky_lstat):
                candidates = [
                    p[0].relative_to(root)
                    for p in find_candidates(
                        root,
                        include_under_limit=True,
                        exclude_paths=[Path("/something")],
                    )
                ]

            self.assertEqual(candidates, [Path("good.mp3")])

    def test_find_candidates_skips_symlink_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root_str = str(root)
            link_file = os.path.join(root_str, "link.wav")

            original_lstat = os.lstat

            def fake_lstat(path: str) -> object:
                if path == link_file:
                    return _fake_lstat(stat.S_IFLNK)
                return original_lstat(path)

            with patch("os.walk", return_value=[(root_str, [], ["link.wav"])]):
                with patch("os.lstat", fake_lstat):
                    candidates = find_candidates(root, include_under_limit=True)

            self.assertEqual(candidates, [])

    def test_find_candidates_skips_mocked_symlink_dir_when_realpath_fails(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root_str = str(root)
            excluded = root / "excluded"
            excluded.mkdir()
            link_dir = os.path.join(root_str, "linked")

            original_lstat = os.lstat
            original_realpath = os.path.realpath
            failed_realpath_calls: list[str] = []

            def fake_lstat(path: str) -> object:
                if path == link_dir:
                    return _fake_lstat(stat.S_IFLNK)
                return original_lstat(path)

            def flaky_realpath(path: object, *args: object, **kwargs: object) -> str:
                if os.fspath(path) == link_dir:
                    failed_realpath_calls.append(link_dir)
                    raise OSError("cannot resolve symlink")
                return original_realpath(path, *args, **kwargs)

            with patch("os.walk", return_value=[(root_str, ["linked"], [])]):
                with patch("os.lstat", fake_lstat):
                    with patch("os.path.realpath", flaky_realpath):
                        candidates = find_candidates(
                            root,
                            include_under_limit=True,
                            exclude_paths=[excluded],
                        )

            self.assertEqual(candidates, [])
            self.assertEqual(failed_realpath_calls, [link_dir])

    def test_find_candidates_skips_symlink_dir_when_realpath_fails(self) -> None:
        # Regression test for a Python-3.10-only CI failure
        # ("OSError: [Errno 22] Invalid argument: '/tmp'"). On 3.10 pathlib
        # binds os.path.realpath at import time (_NormalAccessor.realpath), so
        # patch("os.path.realpath") never reaches Path.resolve(); the real
        # posixpath._joinrealpath ran instead and consumed a blanket os.lstat
        # mock that reported every path component (including /tmp) as a
        # symlink, which made it os.readlink() a regular directory. To stay
        # robust across pathlib implementations, this test uses a real
        # symlinked directory plus a single delegating realpath mock, so the
        # only faked behaviour is the one under test ("realpath fails for the
        # symlink dir") and no version-specific pathlib internals are hit.
        with (
            tempfile.TemporaryDirectory() as tmp,
            tempfile.TemporaryDirectory() as outside,
        ):
            root = Path(tmp)
            excluded = root / "excluded"
            excluded.mkdir()
            (Path(outside) / "hidden.wav").write_bytes(b"0" * 4)

            symlink_dir = os.path.join(str(root), "linked")
            try:
                os.symlink(outside, symlink_dir, target_is_directory=True)
            except (OSError, NotImplementedError):  # pragma: no cover
                self.skipTest("platform does not support symlinks")

            original_realpath = os.path.realpath
            failed_realpath_calls: list[str] = []

            def flaky_realpath(path: object, *args: object, **kwargs: object) -> str:
                if os.fspath(path) == symlink_dir:
                    failed_realpath_calls.append(symlink_dir)
                    raise OSError("cannot resolve symlink")
                return original_realpath(path, *args, **kwargs)

            with patch("os.path.realpath", flaky_realpath):
                candidates = find_candidates(
                    root,
                    include_under_limit=True,
                    exclude_paths=[excluded],
                )

            self.assertEqual(candidates, [])
            # The symlinked dir must have been skipped *because* realpath
            # failed, not merely ignored by os.walk.
            self.assertIn(symlink_dir, failed_realpath_calls)


class ProbeMediaTests(unittest.TestCase):
    @patch("media_shrinker.subprocess.run")
    def test_probe_media_raises_error_on_invalid_json(
        self, mock_run: MagicMock
    ) -> None:
        mock_completed = MagicMock()
        mock_completed.returncode = 0
        mock_completed.stdout = "invalid json"
        mock_run.return_value = mock_completed

        with self.assertRaises(MediaShrinkerError) as cm:
            probe_media(Path("test.wav"))

        self.assertIn("ffprobe returned invalid JSON for test.wav", str(cm.exception))

    @patch("media_shrinker.subprocess.run", side_effect=FileNotFoundError)
    def test_probe_media_raises_clear_error_when_ffprobe_missing(
        self, _mock_run: MagicMock
    ) -> None:
        with self.assertRaises(MediaShrinkerError) as cm:
            probe_media(Path("test.wav"), ffprobe_path="missing-ffprobe")

        message = str(cm.exception)
        self.assertIn("ffprobe not found", message)
        self.assertIn("missing-ffprobe", message)
        self.assertIn("Install ffmpeg", message)

    @patch("media_shrinker.subprocess.run", side_effect=FileNotFoundError)
    def test_detect_silence_intervals_raises_clear_error_when_ffmpeg_missing(
        self, _mock_run: MagicMock
    ) -> None:
        with self.assertRaises(MediaShrinkerError) as cm:
            media_shrinker.detect_silence_intervals(
                Path("test.wav"), ffmpeg_path="missing-ffmpeg"
            )

        message = str(cm.exception)
        self.assertIn("ffmpeg not found", message)
        self.assertIn("Install ffmpeg", message)

    def test_parse_probe_payload_uses_known_source_size_without_stat(self) -> None:
        payload = {
            "streams": [
                {
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "duration": "1.0",
                    "bit_rate": "128000",
                }
            ],
            "format": {"duration": "1.0", "format_name": "wav"},
        }

        probe = media_shrinker._parse_probe_payload(
            payload,
            Path("missing.wav"),
            source_size=123,
        )

        self.assertEqual(probe.size_bytes, 123)

    def test_parse_probe_payload_falls_back_to_format_duration_when_stream_is_zero(
        self,
    ) -> None:
        # Real ffprobe output: some containers report a stream-level
        # "duration": "0.000000" while the true duration lives in the format
        # section. The stream's unusable zero must not shadow the format value.
        payload = {
            "streams": [
                {
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "duration": "0.000000",
                    "bit_rate": "128000",
                }
            ],
            "format": {"duration": "123.5", "size": "456"},
        }

        probe = media_shrinker._parse_probe_payload(payload, Path("stream.mp4"))

        self.assertEqual(probe.duration_seconds, 123.5)


class PlanningTests(unittest.TestCase):
    def test_pcm_wav_uses_lossless_flac_first_and_preserves_container_metadata(
        self,
    ) -> None:
        probe = MediaProbe(
            duration_seconds=3600.0,
            size_bytes=4_294_808_936,
            audio_codec="pcm_s16le",
            audio_bit_rate=1_411_200,
            has_video=False,
            format_name="wav",
        )

        plan = build_audio_plan(
            Path("meeting.wav"),
            probe,
            target_bytes=1_900_000_000,
            output_dir=Path("out"),
        )

        self.assertEqual(plan.strategy, "flac-lossless")
        self.assertEqual(plan.output_path, Path("out/meeting.wav.flac"))
        self.assertIn("-map_metadata", plan.ffmpeg_args)
        self.assertIn("0", plan.ffmpeg_args)
        self.assertIn("flac", plan.ffmpeg_args)

    def test_conversion_command_resolves_input_and_output_overrides(self) -> None:
        plan = ConversionPlan(
            strategy="test",
            input_path=Path("input.wav"),
            output_path=Path("output.flac"),
            ffmpeg_args=["-n", "-i", "input.wav", "output.flac"],
        )

        command = plan.command(
            input_path=Path("-input.wav"),
            output_path=Path("-output.flac"),
        )

        self.assertEqual(
            command[command.index("-i") + 1],
            f"{Path('-input.wav').resolve()}",
        )
        self.assertEqual(command[-1], f"{Path('-output.flac').resolve()}")

    def test_lossy_audio_uses_highest_opus_bitrate_that_fits_target_with_safety_margin(
        self,
    ) -> None:
        probe = MediaProbe(
            duration_seconds=10_000.0,
            size_bytes=3_000_000_000,
            audio_codec="aac",
            audio_bit_rate=2_500_000,
            has_video=False,
            format_name="mov,mp4,m4a,3gp,3g2,mj2",
        )

        plan = build_audio_plan(
            Path("long.m4a"), probe, target_bytes=1_900_000_000, output_dir=Path("out")
        )

        self.assertEqual(plan.strategy, "opus-bitrate")
        self.assertEqual(plan.output_path, Path("out/long.m4a.opus"))
        self.assertEqual(
            plan.audio_bitrate_bps,
            calculate_audio_bitrate(10_000.0, 1_900_000_000, 2_500_000),
        )
        self.assertIn("libopus", plan.ffmpeg_args)

    def test_prefer_flac_converts_lossy_audio_to_flac_without_extra_loss(self) -> None:
        probe = MediaProbe(
            duration_seconds=3_600.0,
            size_bytes=50_000_000,
            audio_codec="aac",
            audio_bit_rate=128_000,
            has_video=False,
            format_name="mov,mp4,m4a,3gp,3g2,mj2",
        )

        plan = build_audio_plan(
            Path("voice memo.m4a"),
            probe,
            target_bytes=1_900_000_000,
            output_dir=Path("out"),
            prefer_flac=True,
            ffmpeg_threads=0,
        )

        self.assertEqual(plan.strategy, "flac-transcode")
        self.assertEqual(plan.output_path, Path("out/voice memo.m4a.flac"))
        self.assertIn("flac", plan.ffmpeg_args)
        self.assertIn("-threads", plan.ffmpeg_args)
        self.assertIn("0", plan.ffmpeg_args)

    def test_calculate_audio_bitrate_never_exceeds_source_bitrate(self) -> None:
        bitrate = calculate_audio_bitrate(
            duration_seconds=1_000.0,
            target_bytes=1_900_000_000,
            source_bitrate_bps=96_000,
        )

        self.assertEqual(bitrate, 96_000)

    def test_low_bitrate_source_that_fits_target_is_not_rejected(self) -> None:
        # A source already encoded below the reasonable floor (e.g. a 12 kbps
        # opus voice recording) fits a generous target with huge margin. The
        # floor guard exists to reject targets too small to fit, not sources
        # that are simply low bitrate, so this must return the source bitrate
        # rather than raise.
        bitrate = calculate_audio_bitrate(
            duration_seconds=1_000.0,
            target_bytes=1_900_000_000,
            source_bitrate_bps=12_000,
        )

        self.assertEqual(bitrate, 12_000)

    def test_same_stem_sources_keep_unique_output_paths(self) -> None:
        wav_probe = MediaProbe(
            duration_seconds=60.0,
            size_bytes=1_000,
            audio_codec="pcm_s16le",
            audio_bit_rate=1_411_200,
            has_video=False,
            format_name="wav",
        )
        m4a_probe = MediaProbe(
            duration_seconds=60.0,
            size_bytes=1_000,
            audio_codec="aac",
            audio_bit_rate=128_000,
            has_video=False,
            format_name="mov,mp4,m4a,3gp,3g2,mj2",
        )

        wav_plan = build_audio_plan(
            Path("clip.wav"),
            wav_probe,
            target_bytes=1_900_000_000,
            output_dir=Path("out"),
            prefer_flac=True,
        )
        m4a_plan = build_audio_plan(
            Path("clip.m4a"),
            m4a_probe,
            target_bytes=1_900_000_000,
            output_dir=Path("out"),
            prefer_flac=True,
        )

        self.assertEqual(wav_plan.output_path, Path("out/clip.wav.flac"))
        self.assertEqual(m4a_plan.output_path, Path("out/clip.m4a.flac"))

    def test_execute_plan_refuses_to_replace_source_path_even_with_overwrite(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.flac"
            fake_ffmpeg = _write_fake_ffmpeg(root / "fake_ffmpeg.py")
            source.write_bytes(b"original")
            plan = ConversionPlan(
                strategy="test",
                input_path=source,
                output_path=source,
                ffmpeg_args=["-nostdin", "-y", "-i", str(source), str(source)],
            )

            with self.assertRaises(MediaShrinkerError):
                _execute_plan(
                    plan, source, source, ffmpeg_path=str(fake_ffmpeg), overwrite=True
                )

            self.assertEqual(source.read_bytes(), b"original")

    def test_long_sources_plan_part_outputs_below_four_hours_at_latest_silence_point(
        self,
    ) -> None:
        segments = build_segments(
            duration_seconds=18_500.0,
            max_segment_duration_seconds=14_400.0,
            silence_intervals=[
                SilenceInterval(start_seconds=14_200.0, end_seconds=14_260.0)
            ],
        )

        self.assertEqual(
            segments,
            [
                MediaSegment(
                    index=1,
                    start_seconds=0.0,
                    duration_seconds=14_260.0,
                    total_segments=2,
                ),
                MediaSegment(
                    index=2,
                    start_seconds=14_260.0,
                    duration_seconds=4_240.0,
                    total_segments=2,
                ),
            ],
        )
        self.assertTrue(
            all(segment.duration_seconds < 14_400.0 for segment in segments)
        )

    def test_spanning_silence_split_advances_near_window_end_not_segment_start(
        self,
    ) -> None:
        split_point = media_shrinker._choose_silence_split_point(
            segment_start=10_000.0,
            window_end=24_400.0,
            silence_intervals=[
                SilenceInterval(start_seconds=0.0, end_seconds=30_000.0)
            ],
        )

        self.assertEqual(split_point, 24_399.999)

    def test_long_sources_fall_back_to_hard_splits_just_under_four_hours_without_silence(
        self,
    ) -> None:
        segments = build_segments(
            duration_seconds=30_000.0,
            max_segment_duration_seconds=14_400.0,
            silence_intervals=[],
        )

        self.assertEqual(len(segments), 3)
        self.assertTrue(
            all(segment.duration_seconds < 14_400.0 for segment in segments)
        )
        self.assertAlmostEqual(
            sum(segment.duration_seconds for segment in segments), 30_000.0, places=3
        )

    def test_segmented_conversion_plan_uses_part_name_seek_and_segment_duration(
        self,
    ) -> None:
        probe = MediaProbe(
            duration_seconds=18_500.0,
            size_bytes=4_000_000_000,
            audio_codec="pcm_s16le",
            audio_bit_rate=1_411_200,
            has_video=False,
            format_name="wav",
        )
        segment = MediaSegment(
            index=1, start_seconds=0.0, duration_seconds=14_230.0, total_segments=2
        )

        plan = build_audio_plan(
            Path("meeting.wav"),
            probe,
            target_bytes=1_900_000_000,
            output_dir=Path("out"),
            prefer_flac=True,
            segment=segment,
        )

        self.assertEqual(plan.output_path, Path("out/meeting.wav.part0001.flac"))
        self.assertIn("-ss", plan.ffmpeg_args)
        self.assertIn("0", plan.ffmpeg_args)
        self.assertIn("-t", plan.ffmpeg_args)
        self.assertIn("14230", plan.ffmpeg_args)

    def test_segment_duration_ffmpeg_arg_never_rounds_up_to_four_hours(self) -> None:
        probe = MediaProbe(
            duration_seconds=14_399.9996,
            size_bytes=1_000,
            audio_codec="pcm_s16le",
            audio_bit_rate=1_411_200,
            has_video=False,
            format_name="wav",
        )
        segment = MediaSegment(
            index=1, start_seconds=0.0, duration_seconds=14_399.9996, total_segments=2
        )

        plan = build_audio_plan(
            Path("meeting.wav"),
            probe,
            target_bytes=1_900_000_000,
            output_dir=Path("out"),
            prefer_flac=True,
            segment=segment,
        )
        t_value = plan.ffmpeg_args[plan.ffmpeg_args.index("-t") + 1]

        self.assertLess(float(t_value), 14_400.0)

    def test_convert_segment_marks_too_long_when_generated_output_probes_over_limit(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.wav"
            output_dir = root / "out"
            fake_ffmpeg = _write_fake_ffmpeg(root / "fake_ffmpeg.py")
            source.write_bytes(b"source")
            source_probe = MediaProbe(
                14_399.999, 1_000, "pcm_s16le", 1_411_200, False, "wav"
            )
            output_probe = MediaProbe(14_400.0, 1_000, "flac", 1_411_200, False, "flac")
            original_probe_media = media_shrinker.probe_media

            def fake_probe_media(
                path: Path,
                *,
                ffprobe_path: str = "ffprobe",
                source_size: int | None = None,
            ) -> MediaProbe:
                return output_probe if path.suffix == ".flac" else source_probe

            try:
                media_shrinker.probe_media = fake_probe_media
                result = media_shrinker._convert_segment(
                    source,
                    rel_source=Path("source.wav"),
                    probe=source_probe,
                    segment=MediaSegment(
                        index=1,
                        start_seconds=0.0,
                        duration_seconds=14_399.999,
                        total_segments=1,
                    ),
                    output_dir=output_dir,
                    target_bytes=1_900_000_000,
                    original_size=source.stat().st_size,
                    ffmpeg_path=str(fake_ffmpeg),
                    ffprobe_path="fake_ffprobe",
                    prefer_flac=True,
                    ffmpeg_threads=None,
                    overwrite=False,
                    max_segment_duration_seconds=14_400.0,
                )
            finally:
                media_shrinker.probe_media = original_probe_media

            self.assertEqual(result.status, "too_long")
            self.assertIsNone(result.output_path)
            self.assertFalse((output_dir / "source.wav.flac").exists())

    def test_convert_segment_deletes_generated_output_when_duration_mismatches_expected_segment(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.wav"
            output_dir = root / "out"
            fake_ffmpeg = _write_fake_ffmpeg(root / "fake_ffmpeg.py")
            source.write_bytes(b"source")
            source_probe = MediaProbe(60.0, 1_000, "pcm_s16le", 1_411_200, False, "wav")
            truncated_probe = MediaProbe(12.0, 1_000, "flac", 1_411_200, False, "flac")
            original_probe_media = media_shrinker.probe_media

            try:
                media_shrinker.probe_media = (
                    lambda path, *, ffprobe_path="ffprobe", source_size=None: (
                        truncated_probe
                    )
                )
                result = media_shrinker._convert_segment(
                    source,
                    rel_source=Path("source.wav"),
                    probe=source_probe,
                    segment=MediaSegment(
                        index=1,
                        start_seconds=0.0,
                        duration_seconds=60.0,
                        total_segments=1,
                    ),
                    output_dir=output_dir,
                    target_bytes=1_900_000_000,
                    original_size=source.stat().st_size,
                    ffmpeg_path=str(fake_ffmpeg),
                    ffprobe_path="fake_ffprobe",
                    prefer_flac=True,
                    ffmpeg_threads=None,
                    overwrite=False,
                    max_segment_duration_seconds=14_400.0,
                )
            finally:
                media_shrinker.probe_media = original_probe_media

            self.assertEqual(result.status, "duration_mismatch")
            self.assertIsNone(result.output_path)
            self.assertFalse((output_dir / "source.wav.flac").exists())

    def test_convert_segment_discards_output_that_has_no_probeable_duration(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.wav"
            output_dir = root / "out"
            source.write_bytes(b"source")
            fake_ffmpeg = _write_fake_ffmpeg(root / "fake_ffmpeg.py", output=b"junk")
            source_probe = MediaProbe(60.0, 1_000, "pcm_s16le", 1_411_200, False, "wav")
            original_probe_media = media_shrinker.probe_media

            def fake_probe_media(
                path: Path,
                *,
                ffprobe_path: str = "ffprobe",
                source_size: int | None = None,
            ) -> MediaProbe:
                # The source probes fine, but the generated output cannot be
                # probed for a usable duration (e.g. a truncated/empty file that
                # ffmpeg still exits 0 for). Real probe_media raises here.
                if path.suffix == ".wav":
                    return source_probe
                raise MediaShrinkerError(f"{path} has no usable duration")

            try:
                media_shrinker.probe_media = fake_probe_media
                result = media_shrinker._convert_segment(
                    source,
                    rel_source=Path("source.wav"),
                    probe=source_probe,
                    segment=MediaSegment(
                        index=1,
                        start_seconds=0.0,
                        duration_seconds=60.0,
                        total_segments=1,
                    ),
                    output_dir=output_dir,
                    target_bytes=1_900_000_000,
                    original_size=source.stat().st_size,
                    ffmpeg_path=str(fake_ffmpeg),
                    ffprobe_path="fake_ffprobe",
                    prefer_flac=True,
                    ffmpeg_threads=None,
                    overwrite=False,
                    max_segment_duration_seconds=14_400.0,
                )
            finally:
                media_shrinker.probe_media = original_probe_media

            self.assertEqual(result.status, "duration_mismatch")
            self.assertIsNone(result.output_path)
            self.assertIsNone(result.duration_seconds)
            self.assertFalse((output_dir / "source.wav.flac").exists())

    def test_existing_output_with_wrong_duration_is_replaced_not_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.wav"
            output_dir = root / "out"
            existing = output_dir / "source.wav.flac"
            fake_ffmpeg = _write_fake_ffmpeg(root / "fake_ffmpeg.py")
            output_dir.mkdir()
            source.write_bytes(b"source")
            existing.write_bytes(b"truncated")
            source_probe = MediaProbe(60.0, 1_000, "pcm_s16le", 1_411_200, False, "wav")
            probes = iter(
                [
                    MediaProbe(12.0, 1_000, "flac", 1_411_200, False, "flac"),
                    MediaProbe(60.0, 1_000, "flac", 1_411_200, False, "flac"),
                ]
            )
            original_probe_media = media_shrinker.probe_media

            try:
                media_shrinker.probe_media = (
                    lambda path, *, ffprobe_path="ffprobe", source_size=None: next(
                        probes
                    )
                )
                result = media_shrinker._convert_segment(
                    source,
                    rel_source=Path("source.wav"),
                    probe=source_probe,
                    segment=MediaSegment(
                        index=1,
                        start_seconds=0.0,
                        duration_seconds=60.0,
                        total_segments=1,
                    ),
                    output_dir=output_dir,
                    target_bytes=1_900_000_000,
                    original_size=source.stat().st_size,
                    ffmpeg_path=str(fake_ffmpeg),
                    ffprobe_path="fake_ffprobe",
                    prefer_flac=True,
                    ffmpeg_threads=None,
                    overwrite=False,
                    max_segment_duration_seconds=14_400.0,
                )
            finally:
                media_shrinker.probe_media = original_probe_media

            self.assertEqual(result.status, "converted")
            self.assertEqual(existing.read_bytes(), b"converted")

    def test_stale_oversized_existing_output_is_replaced_at_canonical_path(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.wav"
            output_dir = root / "out"
            existing = output_dir / "source.wav.flac"
            fake_ffmpeg = _write_fake_ffmpeg(root / "fake_ffmpeg.py", b"ok")
            output_dir.mkdir()
            source.write_bytes(b"source")
            existing.write_bytes(b"oversized")
            probe = MediaProbe(60.0, 1_000, "pcm_s16le", 1_411_200, False, "wav")
            original_probe_media = media_shrinker.probe_media

            try:
                media_shrinker.probe_media = (
                    lambda path, *, ffprobe_path="ffprobe", source_size=None: probe
                )
                result = media_shrinker._convert_segment(
                    source,
                    rel_source=Path("source.wav"),
                    probe=probe,
                    segment=MediaSegment(
                        index=1,
                        start_seconds=0.0,
                        duration_seconds=60.0,
                        total_segments=1,
                    ),
                    output_dir=output_dir,
                    target_bytes=5,
                    original_size=source.stat().st_size,
                    ffmpeg_path=str(fake_ffmpeg),
                    ffprobe_path="fake_ffprobe",
                    prefer_flac=True,
                    ffmpeg_threads=None,
                    overwrite=False,
                    max_segment_duration_seconds=14_400.0,
                    protected_sources={source},
                )
            finally:
                media_shrinker.probe_media = original_probe_media

            self.assertEqual(result.status, "converted")
            self.assertEqual(result.output_path, existing)
            self.assertFalse((output_dir / "source.wav-1.flac").exists())
            self.assertEqual(existing.read_bytes(), b"ok")

    def test_legacy_stem_output_over_duration_is_deleted_before_segmented_conversion(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.wav"
            output_dir = root / "out"
            legacy = output_dir / "source.flac"
            canonical = output_dir / "source.wav.part0001.flac"
            fake_ffmpeg = _write_fake_ffmpeg(root / "fake_ffmpeg.py", b"ok")
            output_dir.mkdir()
            source.write_bytes(b"source")
            legacy.write_bytes(b"legacy")
            source_probe = MediaProbe(
                18_000.0, 1_000, "pcm_s16le", 1_411_200, False, "wav"
            )
            probes = iter(
                [
                    MediaProbe(18_000.0, 1_000, "flac", 1_411_200, False, "flac"),
                    MediaProbe(14_399.0, 1_000, "flac", 1_411_200, False, "flac"),
                ]
            )
            original_probe_media = media_shrinker.probe_media

            try:
                media_shrinker.probe_media = (
                    lambda path, *, ffprobe_path="ffprobe", source_size=None: next(
                        probes
                    )
                )
                result = media_shrinker._convert_segment(
                    source,
                    rel_source=Path("source.wav"),
                    probe=source_probe,
                    segment=MediaSegment(
                        index=1,
                        start_seconds=0.0,
                        duration_seconds=14_399.0,
                        total_segments=2,
                    ),
                    output_dir=output_dir,
                    target_bytes=1_900_000_000,
                    original_size=source.stat().st_size,
                    ffmpeg_path=str(fake_ffmpeg),
                    ffprobe_path="fake_ffprobe",
                    prefer_flac=True,
                    ffmpeg_threads=None,
                    overwrite=False,
                    max_segment_duration_seconds=14_400.0,
                    protected_sources={source},
                )
            finally:
                media_shrinker.probe_media = original_probe_media

            self.assertEqual(result.status, "converted")
            self.assertFalse(legacy.exists())
            self.assertEqual(canonical.read_bytes(), b"ok")

    def test_cleanup_refuses_to_delete_another_protected_source_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.wav"
            other_source = root / "source.wav.flac"
            source.write_bytes(b"source")
            other_source.write_bytes(b"do-not-delete")
            probe = MediaProbe(60.0, 1_000, "pcm_s16le", 1_411_200, False, "wav")

            with self.assertRaises(MediaShrinkerError):
                media_shrinker._convert_segment(
                    source,
                    rel_source=Path("source.wav"),
                    probe=probe,
                    segment=MediaSegment(
                        index=1,
                        start_seconds=0.0,
                        duration_seconds=60.0,
                        total_segments=1,
                    ),
                    output_dir=root,
                    target_bytes=5,
                    original_size=source.stat().st_size,
                    ffmpeg_path="missing-ffmpeg",
                    ffprobe_path="fake_ffprobe",
                    prefer_flac=True,
                    ffmpeg_threads=None,
                    overwrite=False,
                    max_segment_duration_seconds=14_400.0,
                    protected_sources={source.resolve(), other_source.resolve()},
                )

            self.assertEqual(other_source.read_bytes(), b"do-not-delete")

    def test_prefer_flac_reuses_existing_opus_fallback_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.wav"
            output_dir = root / "out"
            existing_opus = output_dir / "source.wav.opus"
            existing_opus.parent.mkdir()
            source.write_bytes(b"source")
            existing_opus.write_bytes(b"opus")
            probe = MediaProbe(60.0, 1_000, "pcm_s16le", 1_411_200, False, "wav")
            original_probe_media = media_shrinker.probe_media

            try:
                media_shrinker.probe_media = (
                    lambda path, *, ffprobe_path="ffprobe", source_size=None: probe
                )
                result = media_shrinker._convert_segment(
                    source,
                    rel_source=Path("source.wav"),
                    probe=probe,
                    segment=MediaSegment(
                        index=1,
                        start_seconds=0.0,
                        duration_seconds=60.0,
                        total_segments=1,
                    ),
                    output_dir=output_dir,
                    target_bytes=1_900_000_000,
                    original_size=source.stat().st_size,
                    ffmpeg_path="missing-ffmpeg",
                    ffprobe_path="fake_ffprobe",
                    prefer_flac=True,
                    ffmpeg_threads=None,
                    overwrite=False,
                    max_segment_duration_seconds=14_400.0,
                )
            finally:
                media_shrinker.probe_media = original_probe_media

            self.assertEqual(result.status, "skipped_existing")
            self.assertEqual(result.output_path, existing_opus)

    def test_missing_source_size_fallback_keeps_failure_reporting_safe(self) -> None:
        missing_source = Path("/tmp/media-shrinker-test-missing-source.wav")

        self.assertEqual(media_shrinker.safe_source_size(missing_source), 0)

    def test_parse_silencedetect_intervals_pairs_long_silence_start_and_end(
        self,
    ) -> None:
        stderr = """
        [silencedetect @ 0x1] silence_start: 14200.125
        [silencedetect @ 0x1] silence_end: 14260.375 | silence_duration: 60.25
        [silencedetect @ 0x1] silence_start: 28000
        [silencedetect @ 0x1] silence_end: 28008 | silence_duration: 8
        """

        intervals = parse_silencedetect_intervals(stderr)

        self.assertEqual(
            intervals,
            [
                SilenceInterval(start_seconds=14_200.125, end_seconds=14_260.375),
                SilenceInterval(start_seconds=28_000.0, end_seconds=28_008.0),
            ],
        )

    def test_parse_silencedetect_ignores_orphan_end_marker(self) -> None:
        self.assertEqual(
            parse_silencedetect_intervals(
                "[silencedetect @ 0x1] silence_end: 4.0 | silence_duration: 4.0"
            ),
            [],
        )

    def test_parse_silencedetect_intervals_normalizes_negative_start_at_recording_head(
        self,
    ) -> None:
        # ffmpeg back-dates silence_start by the detection window, so a file that
        # begins in silence reports a slightly negative silence_start. The
        # interval must survive, but the segment boundary is normalized to the
        # start of the recording.
        stderr = """
        [silencedetect @ 0x1] silence_start: -0.00816327
        [silencedetect @ 0x1] silence_end: 150.5 | silence_duration: 150.5
        """

        intervals = parse_silencedetect_intervals(stderr)

        self.assertEqual(
            intervals,
            [SilenceInterval(start_seconds=0.0, end_seconds=150.5)],
        )

    def test_negative_start_silence_still_drives_split_point(self) -> None:
        # A silence that opens the recording but extends well past the midpoint is
        # the best split boundary; dropping it forces an unwanted mid-audio cut.
        intervals = parse_silencedetect_intervals(
            "[silencedetect @ 0x1] silence_start: -0.008\n"
            "[silencedetect @ 0x1] silence_end: 150.5 | silence_duration: 150.5\n"
        )
        segments = build_segments(
            duration_seconds=300.0,
            max_segment_duration_seconds=200.0,
            silence_intervals=intervals,
        )

        self.assertEqual(segments[0].duration_seconds, 150.5)


class LoudnessNormalizationTests(unittest.TestCase):
    _FLAC_PROBE = MediaProbe(
        duration_seconds=3_600.0,
        size_bytes=4_294_808_936,
        audio_codec="pcm_s16le",
        audio_bit_rate=1_411_200,
        has_video=False,
        format_name="wav",
    )
    _OPUS_PROBE = MediaProbe(
        duration_seconds=10_000.0,
        size_bytes=3_000_000_000,
        audio_codec="aac",
        audio_bit_rate=2_500_000,
        has_video=False,
        format_name="mov,mp4,m4a,3gp,3g2,mj2",
    )

    def _flac_plan(self, *, normalize: bool) -> ConversionPlan:
        return build_audio_plan(
            Path("meeting.wav"),
            self._FLAC_PROBE,
            target_bytes=1_900_000_000,
            output_dir=Path("out"),
            normalize=normalize,
        )

    def _opus_plan(self, *, normalize: bool) -> ConversionPlan:
        return build_audio_plan(
            Path("long.m4a"),
            self._OPUS_PROBE,
            target_bytes=1_900_000_000,
            output_dir=Path("out"),
            normalize=normalize,
        )

    def test_normalize_adds_loudnorm_filter_to_flac_plan(self) -> None:
        plan = self._flac_plan(normalize=True)
        self.assertEqual(plan.strategy, "flac-lossless")
        self.assertIn("-af", plan.ffmpeg_args)
        self.assertIn(media_shrinker.LOUDNORM_FILTER, plan.ffmpeg_args)
        self.assertIn("loudnorm", media_shrinker.LOUDNORM_FILTER)
        # The filter must precede the output path so ffmpeg applies it.
        self.assertLess(
            plan.ffmpeg_args.index(media_shrinker.LOUDNORM_FILTER),
            len(plan.ffmpeg_args) - 1,
        )

    def test_normalize_adds_loudnorm_filter_to_opus_plan(self) -> None:
        plan = self._opus_plan(normalize=True)
        self.assertEqual(plan.strategy, "opus-bitrate")
        self.assertIn("-af", plan.ffmpeg_args)
        self.assertIn(media_shrinker.LOUDNORM_FILTER, plan.ffmpeg_args)

    def test_default_off_flac_plan_is_byte_identical(self) -> None:
        plan = self._flac_plan(normalize=False)
        self.assertNotIn("-af", plan.ffmpeg_args)
        self.assertNotIn(media_shrinker.LOUDNORM_FILTER, plan.ffmpeg_args)
        # Default must equal the explicit normalize=False arg list unchanged.
        self.assertEqual(
            plan.ffmpeg_args,
            build_audio_plan(
                Path("meeting.wav"),
                self._FLAC_PROBE,
                target_bytes=1_900_000_000,
                output_dir=Path("out"),
            ).ffmpeg_args,
        )

    def test_default_off_opus_plan_is_byte_identical(self) -> None:
        plan = self._opus_plan(normalize=False)
        self.assertNotIn("-af", plan.ffmpeg_args)
        self.assertNotIn(media_shrinker.LOUDNORM_FILTER, plan.ffmpeg_args)
        self.assertEqual(
            plan.ffmpeg_args,
            build_audio_plan(
                Path("long.m4a"),
                self._OPUS_PROBE,
                target_bytes=1_900_000_000,
                output_dir=Path("out"),
            ).ffmpeg_args,
        )

    def test_build_opus_plan_normalize_flag_adds_filter(self) -> None:
        plan = media_shrinker.build_opus_plan(
            Path("long.m4a"),
            self._OPUS_PROBE,
            target_bytes=1_900_000_000,
            output_dir=Path("out"),
            normalize=True,
        )
        self.assertIn(media_shrinker.LOUDNORM_FILTER, plan.ffmpeg_args)


class MetadataTaggingTests(unittest.TestCase):
    """Cover --set-* CLI tags flowing into ffmpeg -metadata arguments."""

    @staticmethod
    def _lossless_probe() -> MediaProbe:
        return MediaProbe(
            duration_seconds=3600.0,
            size_bytes=4_294_808_936,
            audio_codec="pcm_s16le",
            audio_bit_rate=1_411_200,
            has_video=False,
            format_name="wav",
        )

    @staticmethod
    def _lossy_probe() -> MediaProbe:
        return MediaProbe(
            duration_seconds=10_000.0,
            size_bytes=3_000_000_000,
            audio_codec="aac",
            audio_bit_rate=2_500_000,
            has_video=False,
            format_name="mov,mp4,m4a,3gp,3g2,mj2",
        )

    def test_metadata_args_emits_pairs_in_sorted_key_order(self) -> None:
        args = media_shrinker._metadata_args({"title": "T", "artist": "A"})

        self.assertEqual(args, ["-metadata", "artist=A", "-metadata", "title=T"])

    def test_metadata_args_returns_empty_list_for_none_and_empty(self) -> None:
        self.assertEqual(media_shrinker._metadata_args(None), [])
        self.assertEqual(media_shrinker._metadata_args({}), [])

    def test_flac_plan_injects_metadata_overrides_after_map_metadata(self) -> None:
        plan = build_audio_plan(
            Path("meeting.wav"),
            self._lossless_probe(),
            target_bytes=1_900_000_000,
            output_dir=Path("out"),
            tags={"title": "Board Meeting", "album": "Archive 2026"},
        )

        args = plan.ffmpeg_args
        self.assertEqual(args.count("-metadata"), 2)
        album_index = args.index("album=Archive 2026")
        title_index = args.index("title=Board Meeting")
        self.assertLess(album_index, title_index)
        self.assertLess(args.index("-map_metadata"), album_index)
        self.assertLess(title_index, args.index("-vn"))
        self.assertEqual(args[album_index - 1], "-metadata")
        self.assertEqual(args[title_index - 1], "-metadata")

    def test_opus_plan_injects_metadata_overrides_after_map_metadata(self) -> None:
        plan = build_opus_plan(
            Path("long.m4a"),
            self._lossy_probe(),
            target_bytes=1_900_000_000,
            output_dir=Path("out"),
            tags={"artist": "Field Recorder", "comment": "auto-archived"},
        )

        args = plan.ffmpeg_args
        self.assertEqual(args.count("-metadata"), 2)
        artist_index = args.index("artist=Field Recorder")
        comment_index = args.index("comment=auto-archived")
        self.assertLess(artist_index, comment_index)
        self.assertLess(args.index("-map_metadata"), artist_index)
        self.assertLess(comment_index, args.index("-vn"))
        self.assertEqual(args[artist_index - 1], "-metadata")
        self.assertEqual(args[comment_index - 1], "-metadata")

    def test_forced_lossy_formats_inject_metadata_overrides(self) -> None:
        for output_format, suffix in (("aac", ".m4a"), ("mp3", ".mp3")):
            with self.subTest(output_format=output_format):
                plan = build_audio_plan(
                    Path("long.m4a"),
                    self._lossy_probe(),
                    target_bytes=1_900_000_000,
                    output_dir=Path("out"),
                    output_format=output_format,
                    tags={"title": "Board Meeting"},
                )

                args = plan.ffmpeg_args
                title_index = args.index("title=Board Meeting")
                self.assertEqual(plan.output_path.suffix, suffix)
                self.assertLess(args.index("-map_metadata"), title_index)
                self.assertLess(title_index, args.index("-vn"))
                self.assertEqual(args[title_index - 1], "-metadata")

    def test_no_tags_keeps_plan_args_byte_identical(self) -> None:
        for probe, source in (
            (self._lossless_probe(), Path("meeting.wav")),
            (self._lossy_probe(), Path("long.m4a")),
        ):
            with self.subTest(source=source):
                baseline = build_audio_plan(
                    source, probe, target_bytes=1_900_000_000, output_dir=Path("out")
                )
                with_none = build_audio_plan(
                    source,
                    probe,
                    target_bytes=1_900_000_000,
                    output_dir=Path("out"),
                    tags=None,
                )
                with_empty = build_audio_plan(
                    source,
                    probe,
                    target_bytes=1_900_000_000,
                    output_dir=Path("out"),
                    tags={},
                )
                self.assertEqual(baseline.ffmpeg_args, with_none.ffmpeg_args)
                self.assertEqual(baseline.ffmpeg_args, with_empty.ffmpeg_args)

    def test_no_tags_keeps_opus_plan_args_byte_identical(self) -> None:
        baseline = build_opus_plan(
            Path("long.m4a"),
            self._lossy_probe(),
            target_bytes=1_900_000_000,
            output_dir=Path("out"),
        )
        with_none = build_opus_plan(
            Path("long.m4a"),
            self._lossy_probe(),
            target_bytes=1_900_000_000,
            output_dir=Path("out"),
            tags=None,
        )

        self.assertEqual(baseline.ffmpeg_args, with_none.ffmpeg_args)

    def test_special_characters_pass_through_as_single_argv_items(self) -> None:
        value = "My \"Great\" Album; $(rm -rf /) && echo -n 'x' 한글 = tricky"
        plan = build_opus_plan(
            Path("long.m4a"),
            self._lossy_probe(),
            target_bytes=1_900_000_000,
            output_dir=Path("out"),
            tags={"album": value},
        )

        self.assertIn(f"album={value}", plan.ffmpeg_args)
        metadata_index = plan.ffmpeg_args.index("-metadata")
        self.assertEqual(plan.ffmpeg_args[metadata_index + 1], f"album={value}")

    def test_parse_args_collects_all_provided_tags(self) -> None:
        args = media_shrinker.parse_args(
            [
                "media",
                "--set-title",
                "Weekly Sync",
                "--set-artist",
                "Recorder",
                "--set-album",
                "Meetings",
                "--set-comment",
                "shrunk by codec-carver",
            ]
        )

        self.assertEqual(
            args.tags,
            {
                "title": "Weekly Sync",
                "artist": "Recorder",
                "album": "Meetings",
                "comment": "shrunk by codec-carver",
            },
        )

    def test_parse_args_collects_only_non_none_tags(self) -> None:
        args = media_shrinker.parse_args(
            ["media", "--set-title", "Weekly Sync", "--set-album", "Meetings"]
        )

        self.assertEqual(args.tags, {"title": "Weekly Sync", "album": "Meetings"})
        self.assertIsNone(args.set_artist)
        self.assertIsNone(args.set_comment)

    def test_parse_args_defaults_to_empty_tags(self) -> None:
        args = media_shrinker.parse_args(["media"])

        self.assertEqual(args.tags, {})
        self.assertIsNone(args.set_title)
        self.assertIsNone(args.set_artist)
        self.assertIsNone(args.set_album)
        self.assertIsNone(args.set_comment)

    def test_execute_conversions_forwards_tags_to_convert_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "ok.wav"
            source.write_bytes(b"ok")
            args = media_shrinker.parse_args(
                [str(root), "--workers", "1", "--set-title", "Weekly Sync"]
            )
            captured: dict[str, object] = {}

            def fake_convert_file(source_path: Path, **kwargs):
                captured.update(kwargs)
                return []

            with patch("media_shrinker.convert_file", side_effect=fake_convert_file):
                media_shrinker._execute_conversions(
                    [(source, 2)], args, root, root / "out"
                )

        self.assertEqual(captured["tags"], {"title": "Weekly Sync"})

    def test_execute_conversions_passes_none_when_no_tags(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "ok.wav"
            source.write_bytes(b"ok")
            args = media_shrinker.parse_args([str(root), "--workers", "1"])
            captured: dict[str, object] = {}

            def fake_convert_file(source_path: Path, **kwargs):
                captured.update(kwargs)
                return []

            with patch("media_shrinker.convert_file", side_effect=fake_convert_file):
                media_shrinker._execute_conversions(
                    [(source, 2)], args, root, root / "out"
                )

        self.assertIsNone(captured["tags"])


class SilenceDetectionTests(unittest.TestCase):
    @patch("media_shrinker.subprocess.run")
    def test_detect_silence_intervals_success(self, mock_run: MagicMock) -> None:
        mock_completed = MagicMock()
        mock_completed.returncode = 0
        mock_completed.stderr = """
        [silencedetect @ 0x1] silence_start: 14200.125
        [silencedetect @ 0x1] silence_end: 14260.375 | silence_duration: 60.25
        """
        mock_run.return_value = mock_completed

        intervals = detect_silence_intervals(
            Path("source.wav"),
            ffmpeg_path="custom-ffmpeg",
            silence_noise="-40dB",
            silence_min_duration_seconds=5.0,
        )

        self.assertEqual(
            intervals,
            [SilenceInterval(start_seconds=14_200.125, end_seconds=14_260.375)],
        )

        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        command = args[0]

        self.assertEqual(command[0], "custom-ffmpeg")
        self.assertIn("source.wav", str(command))
        self.assertIn("silencedetect=noise=-40dB:d=5", str(command))

        self.assertEqual(kwargs.get("check"), False)
        self.assertEqual(kwargs.get("capture_output"), True)
        self.assertEqual(kwargs.get("text"), True)

    @patch("media_shrinker.subprocess.run")
    def test_detect_silence_intervals_failure_raises_error(
        self, mock_run: MagicMock
    ) -> None:
        mock_completed = MagicMock()
        mock_completed.returncode = 1

        mock_completed.stderr = "ffmpeg error message"
        mock_run.return_value = mock_completed

        with self.assertRaisesRegex(
            MediaShrinkerError,
            "silencedetect failed for source.wav: ffmpeg error message",
        ):
            detect_silence_intervals(Path("source.wav"))


class MetadataPreservationTests(unittest.TestCase):
    @patch("os.setxattr", create=True)
    @patch("os.getxattr", create=True)
    @patch("os.listxattr", create=True)
    def test_preserve_file_attributes_ignores_getxattr_oserror(
        self, mock_list, mock_get, mock_set
    ) -> None:
        mock_list.return_value = ["user.attr1", "user.attr2"]

        def mock_getxattr_side_effect(src, name):
            if name == "user.attr1":
                raise OSError("Access denied")
            return b"value2"

        mock_get.side_effect = mock_getxattr_side_effect

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.wav"
            dest = root / "dest.flac"
            source.write_bytes(b"source")
            dest.write_bytes(b"dest")

            preserve_file_attributes(source, dest, setfile_path=None)

            self.assertEqual(mock_get.call_count, 2)
            mock_set.assert_called_once_with(dest, "user.attr2", b"value2")

    def test_preserve_file_attributes_copies_times_and_extended_attributes_when_supported(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.wav"
            dest = root / "dest.flac"
            source.write_bytes(b"source")
            dest.write_bytes(b"dest")

            atime_ns = 1_700_000_001_123_456_789
            mtime_ns = 1_700_000_002_987_654_321
            os.utime(source, ns=(atime_ns, mtime_ns))

            setxattr = getattr(os, "setxattr", None)
            getxattr = getattr(os, "getxattr", None)
            xattr_supported = setxattr is not None and getxattr is not None
            if xattr_supported:
                assert setxattr is not None
                try:
                    setxattr(source, b"user.media_shrinker_test", b"recording-date")
                except OSError:
                    xattr_supported = False

            preserve_file_attributes(source, dest, setfile_path=None)

            dest_stat = dest.stat()
            self.assertAlmostEqual(dest_stat.st_atime_ns, atime_ns, delta=1_000)
            self.assertAlmostEqual(dest_stat.st_mtime_ns, mtime_ns, delta=1_000)
            if xattr_supported:
                assert getxattr is not None
                self.assertEqual(
                    getxattr(dest, b"user.media_shrinker_test"), b"recording-date"
                )


class ICloudDownloadTests(unittest.TestCase):
    def test_build_icloud_download_command_uses_argument_list_for_paths_with_spaces(
        self,
    ) -> None:
        command = build_icloud_download_command(
            Path("folder/file with spaces.m4a"), brctl_path="brctl"
        )

        self.assertEqual(
            command,
            ["brctl", "download", f"{Path('folder/file with spaces.m4a').resolve()}"],
        )


class ParallelismTests(unittest.TestCase):
    def test_choose_worker_count_uses_requested_workers_when_positive(self) -> None:
        self.assertEqual(choose_worker_count(3, cpu_count=8), 3)

    def test_choose_worker_count_auto_uses_multiple_workers_without_exceeding_half_cores(
        self,
    ) -> None:
        self.assertEqual(choose_worker_count(0, cpu_count=10), 4)


class ReportingTests(unittest.TestCase):
    def test_write_report(self) -> None:
        result1 = media_shrinker.ConversionResult(
            source_path=Path("/scan/source1.wav"),
            output_path=Path("/scan/source1.wav.flac"),
            status="converted",
            original_size_bytes=100,
            output_size_bytes=50,
            strategy="flac-lossless",
        )
        result2 = media_shrinker.ConversionResult(
            source_path=Path("/scan/source2.wav"),
            output_path=None,
            status="skipped",
            original_size_bytes=200,
            strategy=None,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "report.json"
            write_report([result1, result2], report_path)

            self.assertTrue(report_path.exists())

            with open(report_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            self.assertEqual(len(payload), 2)
            self.assertEqual(payload[0]["source_path"], str(Path("/scan/source1.wav")))
            self.assertEqual(
                payload[0]["output_path"], str(Path("/scan/source1.wav.flac"))
            )
            self.assertEqual(payload[0]["status"], "converted")
            self.assertEqual(payload[0]["strategy"], "flac-lossless")
            self.assertEqual(payload[0]["original_size_bytes"], 100)
            self.assertEqual(payload[0]["output_size_bytes"], 50)

            self.assertEqual(payload[1]["source_path"], str(Path("/scan/source2.wav")))
            self.assertIsNone(payload[1]["output_path"])
            self.assertEqual(payload[1]["status"], "skipped")
            self.assertIsNone(payload[1]["strategy"])
            self.assertEqual(payload[1]["original_size_bytes"], 200)
            self.assertIsNone(payload[1]["output_size_bytes"])

    def test_format_result_handles_output_path_outside_scan_root(self) -> None:
        result = media_shrinker.ConversionResult(
            source_path=Path("/scan/source.wav"),
            output_path=Path("/external-output/source.wav.flac"),
            status="converted",
            original_size_bytes=100,
            output_size_bytes=50,
            strategy="flac-lossless",
        )

        line = media_shrinker._format_result(Path("/scan"), result)

        self.assertIn("source.wav", line)
        self.assertIn(str(Path("/external-output/source.wav.flac")), line)


class FirstFloatTests(unittest.TestCase):
    def test_first_float_returns_first_valid_number(self) -> None:
        self.assertEqual(_first_float(1.5, "2.0"), 1.5)
        self.assertEqual(_first_float(None, 2.0, 3.0), 2.0)
        self.assertEqual(_first_float("N/A", "1.1"), 1.1)

    def test_first_float_ignores_type_error_and_value_error(self) -> None:
        self.assertEqual(_first_float({}, "not-a-float", 3.14), 3.14)
        self.assertEqual(_first_float([], None, "bad", 42.0), 42.0)

    def test_first_float_returns_zero_on_all_failures(self) -> None:
        self.assertEqual(_first_float(), 0.0)
        self.assertEqual(_first_float(None, "N/A"), 0.0)
        self.assertEqual(_first_float({}, "bad"), 0.0)


class FirstIntTests(unittest.TestCase):
    def test_first_int_handles_uncastable_types(self) -> None:
        self.assertEqual(_first_int("invalid", "N/A", None, "12"), 12)
        self.assertIsNone(_first_int("invalid", object(), []))
        self.assertEqual(_first_int(10), 10)
        self.assertEqual(_first_int(10.5), 10)
        self.assertIsNone(_first_int())

    def test_first_int_more_cases(self) -> None:
        self.assertIsNone(_first_int("not a number"))
        self.assertIsNone(_first_int([1, 2]))
        self.assertIsNone(_first_int({"a": 1}))
        self.assertEqual(_first_int("not a number", "10", 20), 10)
        self.assertEqual(_first_int(None, "N/A", "1.5"), 1)


class FormatSecondsTests(unittest.TestCase):
    def test_format_seconds_truncates_to_three_decimals(self) -> None:
        from media_shrinker import _format_seconds

        self.assertEqual(_format_seconds(1.23456), "1.234")
        self.assertEqual(_format_seconds(10.0), "10")
        self.assertEqual(_format_seconds(0.0001), "0")
        self.assertEqual(_format_seconds(14400.0), "14400")


class CollisionResolutionTests(unittest.TestCase):
    def test_resolve_collision_returns_original_if_no_collision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "new.flac"
            resolved = media_shrinker._resolve_collision(path, overwrite=False)
            self.assertEqual(resolved, path)

    def test_resolve_collision_returns_numbered_variant_if_collision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "existing.flac"
            path.write_bytes(b"data")
            resolved = media_shrinker._resolve_collision(path, overwrite=False)
            self.assertEqual(resolved, Path(tmp) / "existing-1.flac")

    def test_resolve_collision_returns_original_if_overwrite_is_true(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "existing.flac"
            path.write_bytes(b"data")
            resolved = media_shrinker._resolve_collision(path, overwrite=True)
            self.assertEqual(resolved, path)

    def test_resolve_collision_handles_dangling_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "existing.flac"
            path.write_bytes(b"data")
            dangling = Path(tmp) / "existing-1.flac"
            dangling.symlink_to("does_not_exist.flac")
            resolved = media_shrinker._resolve_collision(path, overwrite=False)
            self.assertEqual(resolved, Path(tmp) / "existing-2.flac")

    def test_resolve_collision_handles_permission_errors(self) -> None:
        import errno
        original_lstat = os.lstat
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "existing.flac"
            path.write_bytes(b"data")

            def side_effect(candidate_str, *args, **kwargs):
                if isinstance(candidate_str, str) and "-1.flac" in candidate_str:
                    e = OSError()
                    e.errno = errno.EACCES
                    raise e
                elif isinstance(candidate_str, str) and "-2.flac" in candidate_str:
                    e = OSError()
                    e.errno = errno.ENOENT
                    raise e
                return original_lstat(candidate_str, *args, **kwargs)

            with patch("os.lstat", side_effect=side_effect):
                resolved = media_shrinker._resolve_collision(path, overwrite=False)
                self.assertEqual(resolved, Path(tmp) / "existing-2.flac")


class CliTests(unittest.TestCase):
    def test_normalize_argv_handles_silence_noise_values(self) -> None:
        self.assertIsNone(media_shrinker._normalize_argv(None))
        self.assertEqual(
            media_shrinker._normalize_argv(["--silence-noise", "quiet"]),
            ["--silence-noise", "quiet"],
        )
        self.assertEqual(
            media_shrinker._normalize_argv(["--silence-noise"]),
            ["--silence-noise"],
        )

    def test_parse_args_sets_conversion_options(self) -> None:
        args = media_shrinker.parse_args(
            [
                "media",
                "--size-limit-bytes",
                "10",
                "--target-bytes",
                "9",
                "--max-duration-seconds",
                "8.5",
                "--output-dir",
                "out",
                "--report",
                "report.json",
                "--ffmpeg",
                "ff",
                "--ffprobe",
                "probe",
                "--brctl",
                "br",
                "--download-icloud",
                "--over-limit-only",
                "--exclude-dir-prefix",
                "split_",
                "--flac-all",
                "--normalize",
                "--workers",
                "2",
                "--ffmpeg-threads",
                "-1",
                "--silence-noise",
                "-40dB",
                "--silence-min-duration-seconds",
                "3.5",
                "--execute",
                "--overwrite",
            ]
        )

        self.assertEqual(args.root, Path("media"))
        self.assertEqual(args.size_limit_bytes, 10)
        self.assertEqual(args.target_bytes, 9)
        self.assertEqual(args.max_duration_seconds, 8.5)
        self.assertEqual(args.output_dir, Path("out"))
        self.assertEqual(args.report, Path("report.json"))
        self.assertEqual(args.ffmpeg, "ff")
        self.assertEqual(args.ffprobe, "probe")
        self.assertEqual(args.brctl, "br")
        self.assertTrue(args.download_icloud)
        self.assertFalse(args.include_under_limit)
        self.assertEqual(args.exclude_dir_prefix, ["split_"])
        self.assertTrue(args.flac_all)
        self.assertTrue(args.normalize)
        self.assertEqual(args.workers, 2)
        self.assertEqual(args.ffmpeg_threads, -1)
        self.assertEqual(args.silence_noise, "-40dB")
        self.assertEqual(args.silence_min_duration_seconds, 3.5)
        self.assertTrue(args.execute)
        self.assertTrue(args.overwrite)

    def test_parse_args_normalize_defaults_false_and_opts_in(self) -> None:
        self.assertFalse(media_shrinker.parse_args(["root"]).normalize)
        self.assertTrue(media_shrinker.parse_args(["root", "--normalize"]).normalize)

    def test_main_dry_run_lists_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.wav"
            source.write_bytes(b"1234")

            with patch("builtins.print") as mock_print:
                rc = media_shrinker.main([str(root), "--size-limit-bytes", "1"])

        self.assertEqual(rc, 0)
        printed = ["\t".join(map(str, call.args)) for call in mock_print.call_args_list]
        self.assertIn(f"DRY-RUN\t4\t{source.name}", printed)
        self.assertIn("TOTAL_SELECTED=1", printed)

    def test_main_execute_writes_report_and_returns_success(self) -> None:
        result = media_shrinker.ConversionResult(
            source_path=Path("/scan/b.wav"),
            output_path=Path("/scan/out/b.flac"),
            status="converted",
            original_size_bytes=4,
            output_size_bytes=2,
            strategy="flac-lossless",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "b.wav").write_bytes(b"1234")
            report = root / "report.json"
            with patch("media_shrinker._execute_conversions", return_value=[result]):
                rc = media_shrinker.main(
                    [str(root), "--execute", "--report", str(report)]
                )

            payload = json.loads(report.read_text(encoding="utf-8"))

        self.assertEqual(rc, 0)
        self.assertEqual(payload[0]["status"], "converted")

    def test_main_execute_accepts_filesystem_root_without_extra_separator(self) -> None:
        with tempfile.TemporaryDirectory() as report_dir:
            report_path = Path(report_dir) / "ignored-report.json"
            with (
                patch("media_shrinker.find_candidates", return_value=[]),
                patch("media_shrinker._execute_conversions", return_value=[]),
                patch("media_shrinker.write_report") as write,
                patch("builtins.print"),
            ):
                rc = media_shrinker.main(
                    ["/", "--execute", "--report", str(report_path)]
                )

        self.assertEqual(rc, 0)
        write.assert_called_once()

    def test_main_execute_returns_failure_when_any_result_failed(self) -> None:
        result = media_shrinker.ConversionResult(
            source_path=Path("/scan/a.wav"),
            output_path=None,
            status="failed",
            original_size_bytes=4,
            message="boom",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "a.wav").write_bytes(b"1234")
            with patch("media_shrinker._execute_conversions", return_value=[result]):
                rc = media_shrinker.main([str(root), "--execute"])

        self.assertEqual(rc, 1)

    def test_main_summary_does_not_count_skipped_existing_as_converted(self) -> None:
        import contextlib
        import io

        converted = media_shrinker.ConversionResult(
            source_path=Path("/scan/a.wav"),
            output_path=Path("/scan/out/a.flac"),
            status="converted",
            original_size_bytes=4,
            output_size_bytes=2,
            strategy="flac-lossless",
        )
        skipped = media_shrinker.ConversionResult(
            source_path=Path("/scan/b.wav"),
            output_path=Path("/scan/out/b.flac"),
            status="skipped_existing",
            original_size_bytes=4,
            output_size_bytes=2,
            strategy="existing",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "a.wav").write_bytes(b"1234")
            (root / "b.wav").write_bytes(b"1234")
            buffer = io.StringIO()
            with (
                patch(
                    "media_shrinker._execute_conversions",
                    return_value=[converted, skipped],
                ),
                contextlib.redirect_stdout(buffer),
            ):
                rc = media_shrinker.main([str(root), "--execute"])

        summary_line = next(
            line
            for line in buffer.getvalue().splitlines()
            if line.startswith("SUMMARY")
        )
        self.assertEqual(rc, 0)
        # Only one file was actually converted; the other reused an existing
        # output, so it must not inflate the converted count.
        self.assertIn("converted=1", summary_line)
        self.assertIn("skipped_existing=1", summary_line)

    def test_execute_conversions_records_success_and_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ok = root / "ok.wav"
            bad = root / "bad.wav"
            ok.write_bytes(b"ok")
            bad.write_bytes(b"bad")
            args = media_shrinker.parse_args([str(root), "--workers", "1"])
            result = media_shrinker.ConversionResult(
                source_path=ok,
                output_path=root / "out" / "ok.flac",
                status="converted",
                original_size_bytes=2,
                output_size_bytes=1,
                strategy="flac-lossless",
            )

            def fake_convert_file(source: Path, **_kwargs):
                if source == bad:
                    raise RuntimeError("boom")
                return [result]

            with patch("media_shrinker.convert_file", side_effect=fake_convert_file):
                results = media_shrinker._execute_conversions(
                    [(ok, 2), (bad, 3)], args, root, root / "out"
                )

        statuses = sorted(item.status for item in results)
        self.assertEqual(statuses, ["converted", "failed"])
        self.assertEqual(
            [item.message for item in results if item.status == "failed"], ["boom"]
        )

    def test_format_progress_formats_counts_percent_and_eta(self) -> None:
        self.assertEqual(
            media_shrinker._format_progress(1, 4, 125.9),
            "PROGRESS\t1/4\t25%\tETA 02:05",
        )
        self.assertEqual(
            media_shrinker._format_progress(3, 3, 0.0),
            "PROGRESS\t3/3\t100%\tETA 00:00",
        )

    def test_format_progress_clamps_negative_eta_to_zero(self) -> None:
        self.assertEqual(
            media_shrinker._format_progress(2, 2, -5.0),
            "PROGRESS\t2/2\t100%\tETA 00:00",
        )

    def test_execute_conversions_prints_progress_lines_with_eta(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sources = []
            for name in ("a.wav", "b.wav", "c.wav"):
                source = root / name
                source.write_bytes(b"12")
                sources.append(source)
            args = media_shrinker.parse_args([str(root), "--workers", "1"])

            def fake_convert_file(source: Path, **_kwargs):
                return [
                    media_shrinker.ConversionResult(
                        source_path=source,
                        output_path=root / "out" / (source.stem + ".flac"),
                        status="converted",
                        original_size_bytes=2,
                        output_size_bytes=1,
                        strategy="flac-lossless",
                    )
                ]

            # One call at batch start, then one after each completion:
            # elapsed 10s per file -> average 10s -> ETA 20s, 10s, 0s.
            clock = iter([100.0, 110.0, 120.0, 130.0])
            with patch("media_shrinker.convert_file", side_effect=fake_convert_file):
                with patch(
                    "media_shrinker.time.monotonic",
                    side_effect=lambda: next(clock, 130.0),
                ):
                    with patch("builtins.print") as mock_print:
                        results = media_shrinker._execute_conversions(
                            [(source, 2) for source in sources],
                            args,
                            root,
                            root / "out",
                        )

        self.assertEqual(len(results), 3)
        printed = ["\t".join(map(str, call.args)) for call in mock_print.call_args_list]
        progress_lines = [line for line in printed if line.startswith("PROGRESS\t")]
        self.assertEqual(
            progress_lines,
            [
                "PROGRESS\t1/3\t33%\tETA 00:20",
                "PROGRESS\t2/3\t66%\tETA 00:10",
                "PROGRESS\t3/3\t100%\tETA 00:00",
            ],
        )

    def test_main_dry_run_prints_no_progress_lines(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "source.wav").write_bytes(b"1234")

            with patch("builtins.print") as mock_print:
                rc = media_shrinker.main([str(root), "--size-limit-bytes", "1"])

        self.assertEqual(rc, 0)
        printed = ["\t".join(map(str, call.args)) for call in mock_print.call_args_list]
        self.assertFalse(
            [line for line in printed if line.startswith("PROGRESS")], printed
        )

    def test_small_segment_returns_single_segment(self) -> None:
        segments = build_segments(
            duration_seconds=1.0, max_segment_duration_seconds=2.0
        )
        self.assertEqual(segments, [MediaSegment(1, 0.0, 1.0, 1)])

    def test_build_segments_rejects_invalid_durations(self) -> None:
        with self.assertRaisesRegex(ValueError, "duration_seconds must be positive"):
            build_segments(duration_seconds=0, max_segment_duration_seconds=2)
        with self.assertRaisesRegex(
            ValueError, "max_segment_duration_seconds must be greater than"
        ):
            build_segments(duration_seconds=10, max_segment_duration_seconds=0.001)

    def test_calculate_audio_bitrate_rejects_invalid_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "duration_seconds must be positive"):
            calculate_audio_bitrate(0, 1000, None)
        with self.assertRaisesRegex(ValueError, "target_bytes must be positive"):
            calculate_audio_bitrate(10, 0, None)
        with self.assertRaises(MediaShrinkerError):
            calculate_audio_bitrate(10_000, 1, None)

    def test_download_from_icloud_handles_missing_failure_and_success(self) -> None:
        with patch("media_shrinker.shutil.which", return_value=None):
            with self.assertRaisesRegex(MediaShrinkerError, "was not found"):
                media_shrinker.download_from_icloud(Path("source.wav"))

        completed = MagicMock(returncode=1, stderr="no cloud")
        with patch("media_shrinker.shutil.which", return_value="/usr/bin/brctl"):
            with patch("media_shrinker.subprocess.run", return_value=completed):
                with self.assertRaisesRegex(MediaShrinkerError, "no cloud"):
                    media_shrinker.download_from_icloud(Path("source.wav"))

        completed = MagicMock(returncode=0, stderr="")
        with (
            patch("media_shrinker.shutil.which", return_value="/usr/bin/brctl"),
            patch("media_shrinker.subprocess.run", return_value=completed) as run,
        ):
            media_shrinker.download_from_icloud(Path("source.wav"))
        run.assert_called_once()

    def test_conversion_plan_command_rejects_missing_input_placeholder(self) -> None:
        plan = ConversionPlan("bad", Path("in.wav"), Path("out.flac"), ["-n", "out"])
        with self.assertRaisesRegex(MediaShrinkerError, "missing '-i'"):
            plan.command(input_path=Path("other.wav"))

    def test_conversion_plan_command_can_disable_overwrite(self) -> None:
        plan = ConversionPlan(
            "test",
            Path("in.wav"),
            Path("out.flac"),
            ["-y", "-i", "in.wav", "out.flac"],
        )
        self.assertIn("-n", plan.command(overwrite=False))

    def test_convert_file_calls_segment_conversion_with_protected_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.wav"
            source.write_bytes(b"1234")
            result = media_shrinker.ConversionResult(
                source_path=source,
                output_path=root / "out/source.wav.flac",
                status="converted",
                original_size_bytes=4,
            )
            probe = MediaProbe(1.0, 4, "pcm_s16le", 128_000, False, "wav")
            resolved_sources = frozenset({source.resolve()})

            with patch("media_shrinker.probe_media", return_value=probe):
                with patch(
                    "media_shrinker._convert_segment", return_value=result
                ) as mocked:
                    results = media_shrinker.convert_file(
                        source,
                        root=root,
                        output_dir=root / "out",
                        original_size=4,
                        resolved_protected_sources=resolved_sources,
                    )

        self.assertEqual(results, [result])
        self.assertEqual(mocked.call_args.kwargs["original_size"], 4)
        self.assertEqual(mocked.call_args.kwargs["protected_sources"], resolved_sources)

    def test_convert_file_downloads_icloud_and_detects_silence_for_long_sources(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.wav"
            source.write_bytes(b"1234")
            result = media_shrinker.ConversionResult(
                source_path=source,
                output_path=root / "out/source.wav.flac",
                status="converted",
                original_size_bytes=4,
            )
            probe = MediaProbe(3.0, 4, "pcm_s16le", 128_000, False, "wav")

            with patch("media_shrinker.download_from_icloud") as mock_download:
                with patch("media_shrinker.probe_media", return_value=probe):
                    with patch(
                        "media_shrinker.detect_silence_intervals", return_value=[]
                    ):
                        with patch(
                            "media_shrinker._convert_segment", return_value=result
                        ):
                            results = media_shrinker.convert_file(
                                source,
                                root=root,
                                output_dir=root / "out",
                                download_icloud=True,
                                max_segment_duration_seconds=2,
                                original_size=4,
                            )

        self.assertEqual([item.status for item in results], ["converted", "converted"])
        mock_download.assert_called_once_with(source, brctl_path="brctl")

    def test_build_audio_plan_rejects_video_and_missing_audio(self) -> None:
        with self.assertRaisesRegex(MediaShrinkerError, "contains video"):
            build_audio_plan(
                Path("clip.mp4"),
                MediaProbe(10, 100, "aac", 96_000, True, "mp4"),
                target_bytes=100,
                output_dir=Path("out"),
            )
        with self.assertRaisesRegex(MediaShrinkerError, "has no audio stream"):
            build_audio_plan(
                Path("silent.wav"),
                MediaProbe(10, 100, None, None, False, "wav"),
                target_bytes=100,
                output_dir=Path("out"),
            )

    def test_find_candidates_skips_excluded_roots_files_and_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            excluded_root = root / "excluded"
            excluded_root.mkdir()
            (excluded_root / "source.wav").write_bytes(b"1234")
            source = root / "source.wav"
            source.write_bytes(b"1234")
            target = root / "target.wav"
            target.write_bytes(b"1234")
            try:
                (root / "link.wav").symlink_to(target)
                symlink_created = True
            except OSError:
                symlink_created = False

            self.assertEqual(
                find_candidates(excluded_root, exclude_paths=[excluded_root]), []
            )
            candidates = find_candidates(root, exclude_paths=[source])
            self.assertIn((target, 4), candidates)
            self.assertNotIn((source, 4), candidates)
            if symlink_created:
                self.assertNotIn((root / "link.wav", 4), candidates)

    def test_find_candidates_ignores_symlink_dir_realpath_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_dir = root / "target"
            target_dir.mkdir()
            link_dir = root / "linked"
            try:
                link_dir.symlink_to(target_dir, target_is_directory=True)
            except OSError:
                return
            original_realpath = os.path.realpath

            def flaky_realpath(path, *_args, **_kwargs):
                if Path(path) == link_dir:
                    raise OSError("bad link")
                return original_realpath(path)

            with patch("media_shrinker.os.path.realpath", side_effect=flaky_realpath):
                self.assertEqual(find_candidates(root, exclude_paths=[target_dir]), [])

    def test_probe_media_reports_ffprobe_failure(self) -> None:
        completed = MagicMock(returncode=1, stderr="bad probe")
        with patch("media_shrinker.subprocess.run", return_value=completed):
            with self.assertRaisesRegex(MediaShrinkerError, "bad probe"):
                probe_media(Path("source.wav"))

    def test_parse_probe_payload_rejects_missing_audio_and_duration(self) -> None:
        with self.assertRaisesRegex(MediaShrinkerError, "has no audio stream"):
            media_shrinker._parse_probe_payload(
                {"streams": [{"codec_type": "video"}], "format": {}},
                Path("silent.mp4"),
            )
        with self.assertRaisesRegex(MediaShrinkerError, "has no usable duration"):
            media_shrinker._parse_probe_payload(
                {
                    "streams": [
                        {"codec_type": "audio", "codec_name": "aac", "duration": "0"}
                    ],
                    "format": {},
                },
                Path("zero.wav"),
            )
        probe = media_shrinker._parse_probe_payload(
            {
                "streams": [
                    {"codec_type": "video"},
                    {"codec_type": "audio", "codec_name": "aac", "duration": "1"},
                ],
                "format": {"size": "10"},
            },
            Path("video.mp4"),
        )
        self.assertTrue(probe.has_video)

    def test_execute_plan_reports_ffmpeg_failures_and_existing_output(self) -> None:
        plan = ConversionPlan(
            "test",
            Path("in.wav"),
            Path("out.flac"),
            ["-n", "-i", "in.wav", "out.flac"],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.wav"
            source.write_bytes(b"source")
            output = root / "out.flac"

            with patch("media_shrinker.subprocess.run", side_effect=FileNotFoundError):
                with self.assertRaisesRegex(MediaShrinkerError, "ffmpeg not found"):
                    media_shrinker._execute_plan(
                        plan, source, output, ffmpeg_path="missing", overwrite=True
                    )

            completed = MagicMock(returncode=1, stderr="bad ffmpeg")
            with patch("media_shrinker.subprocess.run", return_value=completed):
                with self.assertRaisesRegex(MediaShrinkerError, "bad ffmpeg"):
                    media_shrinker._execute_plan(
                        plan, source, output, ffmpeg_path="ffmpeg", overwrite=True
                    )

            output.write_bytes(b"existing")
            completed = MagicMock(returncode=0, stderr="")
            with patch("media_shrinker.subprocess.run", return_value=completed):
                with self.assertRaisesRegex(FileExistsError, "already exists"):
                    media_shrinker._execute_plan(
                        plan, source, output, ffmpeg_path="ffmpeg", overwrite=False
                    )

    def test_execute_segment_conversion_falls_back_to_opus_then_discards_oversize(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.wav"
            source.write_bytes(b"source")
            probe = MediaProbe(1.0, 10, "pcm_s16le", 128_000, False, "wav")
            segment = MediaSegment(1, 0.0, 1.0, 1)

            def fake_execute_plan(_plan, _source, final_output, **_kwargs):
                final_output.parent.mkdir(parents=True, exist_ok=True)
                final_output.write_bytes(b"0" * 5000)
                return final_output

            with patch("media_shrinker._execute_plan", side_effect=fake_execute_plan):
                with patch("media_shrinker._probe_output_duration", return_value=1.0):
                    with patch("media_shrinker.preserve_file_attributes"):
                        result = media_shrinker._execute_segment_conversion(
                            source,
                            rel_source=Path("source.wav"),
                            probe=probe,
                            segment=segment,
                            output_dir=root / "out",
                            target_bytes=3000,
                            original_size=10,
                            ffmpeg_path="ffmpeg",
                            ffprobe_path="ffprobe",
                            prefer_flac=False,
                            ffmpeg_threads=None,
                            overwrite=False,
                            max_segment_duration_seconds=2,
                            resolved_protected_sources=frozenset({source.resolve()}),
                        )

        self.assertEqual(result.status, "too_large")
        self.assertIsNone(result.output_path)

    def test_remove_invalid_legacy_outputs_skips_canonical_and_removes_oversize(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "out"
            output_dir.mkdir()
            source = root / "source.wav"
            source.write_bytes(b"source")
            oversized_legacy = output_dir / "source.flac"
            oversized_legacy.write_bytes(b"0123456789")
            probe = MediaProbe(1.0, 10, "pcm_s16le", 128_000, False, "wav")

            media_shrinker._remove_invalid_legacy_outputs(
                source,
                rel_source=Path("source"),
                probe=probe,
                output_dir=output_dir,
                suffixes=[".flac"],
                target_bytes=5,
                ffprobe_path="ffprobe",
                max_segment_duration_seconds=2,
                protected_sources=frozenset({source.resolve()}),
            )
            media_shrinker._remove_invalid_legacy_outputs(
                source,
                rel_source=Path("source.wav"),
                probe=probe,
                output_dir=output_dir,
                suffixes=[".flac"],
                target_bytes=5,
                ffprobe_path="ffprobe",
                max_segment_duration_seconds=2,
                protected_sources=frozenset({source.resolve()}),
            )

            valid_legacy = output_dir / "other.flac"
            valid_legacy.write_bytes(b"valid")
            with patch("media_shrinker._probe_output_duration", return_value=1.0):
                media_shrinker._remove_invalid_legacy_outputs(
                    source,
                    rel_source=Path("other.wav"),
                    probe=probe,
                    output_dir=output_dir,
                    suffixes=[".flac"],
                    target_bytes=100,
                    ffprobe_path="ffprobe",
                    max_segment_duration_seconds=2,
                    protected_sources=frozenset({source.resolve()}),
                )

            self.assertTrue(valid_legacy.exists())

        self.assertFalse(oversized_legacy.exists())

    def test_build_segments_guards_nonadvancing_split_points(self) -> None:
        with patch("media_shrinker._choose_silence_split_point", return_value=0.0):
            segments = build_segments(
                duration_seconds=3.0, max_segment_duration_seconds=2.0
            )

        self.assertGreater(segments[0].duration_seconds, 0)

    def test_resolve_collision_reports_exhausted_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "existing.flac"
            path.write_bytes(b"data")
            with patch("os.lstat", return_value=True), patch("pathlib.Path.exists", return_value=True):
                with self.assertRaisesRegex(FileExistsError, "Could not find free"):
                    media_shrinker._resolve_collision(path, overwrite=False)

    def test_attribute_copy_helpers_ignore_platform_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.wav"
            dest = Path(tmp) / "dest.wav"
            source.write_bytes(b"source")
            dest.write_bytes(b"dest")

            with patch("media_shrinker.os.chmod", side_effect=OSError):
                preserve_file_attributes(source, dest, setfile_path=None)

            with patch("media_shrinker.os.listxattr", side_effect=OSError, create=True):
                with patch("media_shrinker.os.getxattr", return_value=b"", create=True):
                    with patch("media_shrinker.os.setxattr", create=True):
                        media_shrinker._copy_extended_attributes(source, dest)

            with patch(
                "media_shrinker.os.listxattr", return_value=["user.test"], create=True
            ):
                with patch(
                    "media_shrinker.os.getxattr", side_effect=OSError, create=True
                ):
                    with patch("media_shrinker.os.setxattr", create=True):
                        media_shrinker._copy_extended_attributes(source, dest)

            with patch(
                "media_shrinker.os.listxattr", return_value=["user.test"], create=True
            ):
                with patch(
                    "media_shrinker.os.getxattr", return_value=b"value", create=True
                ):
                    with patch(
                        "media_shrinker.os.setxattr", side_effect=OSError, create=True
                    ):
                        media_shrinker._copy_extended_attributes(source, dest)

            stat_result = os.stat(source)
            media_shrinker._copy_macos_creation_time(stat_result, dest, "SetFile")
            media_shrinker._copy_macos_creation_time(
                MagicMock(spec=[]), dest, "SetFile"
            )

            with patch("media_shrinker._get_setfile_path", return_value="SetFile"):
                with patch("media_shrinker._copy_macos_creation_time") as mock_copy:
                    media_shrinker.preserve_file_attributes(source, dest)
                    mock_copy.assert_called_once()

            with patch("media_shrinker.os.listxattr", side_effect=OSError, create=True):
                media_shrinker._copy_extended_attributes(source, dest)

            # Test unsupported OS for extended attributes
            with patch("builtins.hasattr", return_value=False):
                media_shrinker._copy_extended_attributes(source, dest)

            # Test macos creation time logic when birthtime is present
            with patch("media_shrinker.subprocess.run") as mock_run:
                mock_stat = MagicMock()
                mock_stat.st_birthtime = 1234567890.0
                media_shrinker._copy_macos_creation_time(mock_stat, dest, "SetFile")
                mock_run.assert_called_once()


class PresetTests(unittest.TestCase):
    """Scenario preset CLI wiring and precedence."""

    def test_music_preset_applies_bundled_overrides(self) -> None:
        """`--preset music` fills flac_all and gentler silence from the preset."""
        args = media_shrinker.parse_args(["root", "--preset", "music"])
        self.assertTrue(args.flac_all)
        self.assertEqual(args.silence_noise, "-45dB")
        self.assertEqual(args.silence_min_duration_seconds, 3.0)
        # target_bytes is not overridden by music -> stays the real default.
        self.assertEqual(args.target_bytes, media_shrinker.DEFAULT_TARGET_BYTES)

    def test_archive_preset_raises_target_and_prefers_flac(self) -> None:
        """`--preset archive` bumps target_bytes and enables flac_all."""
        args = media_shrinker.parse_args(["root", "--preset", "archive"])
        self.assertTrue(args.flac_all)
        self.assertEqual(args.target_bytes, 1_950_000_000)
        self.assertEqual(args.silence_noise, "-50dB")

    def test_voice_preset_keeps_opus_and_trims_aggressively(self) -> None:
        """`--preset voice` keeps Opus (flac_all False) with aggressive silence."""
        args = media_shrinker.parse_args(["root", "--preset", "voice"])
        self.assertFalse(args.flac_all)
        self.assertEqual(args.silence_noise, "-30dB")
        self.assertEqual(args.silence_min_duration_seconds, 0.5)

    def test_explicit_flag_overrides_preset(self) -> None:
        """A value the user sets on the CLI wins over the preset override."""
        args = media_shrinker.parse_args(
            ["root", "--preset", "music", "--silence-noise", "-99dB"]
        )
        self.assertEqual(args.silence_noise, "-99dB")
        # Unspecified options still come from the preset.
        self.assertTrue(args.flac_all)

    def test_explicit_store_true_flag_overrides_preset(self) -> None:
        """`--flac-all` set explicitly is honored even for a preset that omits it."""
        args = media_shrinker.parse_args(["root", "--preset", "voice", "--flac-all"])
        self.assertTrue(args.flac_all)

    def test_explicit_target_bytes_overrides_archive_preset(self) -> None:
        """A user-set --target-bytes beats the archive preset's larger value."""
        args = media_shrinker.parse_args(
            ["root", "--preset", "archive", "--target-bytes", "5"]
        )
        self.assertEqual(args.target_bytes, 5)

    def test_no_preset_leaves_defaults_unchanged(self) -> None:
        """Without --preset every tunable option equals the built-in default."""
        args = media_shrinker.parse_args(["root"])
        self.assertIsNone(args.preset)
        self.assertFalse(args.flac_all)
        self.assertEqual(args.silence_noise, media_shrinker.DEFAULT_SILENCE_NOISE)
        self.assertEqual(
            args.silence_min_duration_seconds,
            media_shrinker.DEFAULT_SILENCE_MIN_DURATION_SECONDS,
        )
        self.assertEqual(args.target_bytes, media_shrinker.DEFAULT_TARGET_BYTES)
        self.assertEqual(
            args.max_duration_seconds,
            media_shrinker.DEFAULT_MAX_SEGMENT_DURATION_SECONDS,
        )

    def test_unknown_preset_is_rejected(self) -> None:
        """An unknown preset name exits via argparse (SystemExit)."""
        with self.assertRaises(SystemExit):
            media_shrinker.parse_args(["root", "--preset", "bogus"])


class PresetModuleTests(unittest.TestCase):
    """Unit tests for the standalone presets module logic."""

    def test_preset_names_matches_registry(self) -> None:
        """preset_names reports every registered preset in definition order."""
        self.assertEqual(
            presets.preset_names(), ("voice", "podcast", "music", "archive")
        )

    def test_apply_preset_without_preset_restores_real_defaults(self) -> None:
        """No preset: every unset dest receives its real default."""
        args = argparse.Namespace(preset=None)
        for dest in presets.PRESET_TUNABLE_DESTS:
            setattr(args, dest, presets._UNSET)
        defaults = {dest: f"D_{dest}" for dest in presets.PRESET_TUNABLE_DESTS}
        presets.apply_preset(args, defaults)
        for dest, value in defaults.items():
            self.assertEqual(getattr(args, dest), value)

    def test_apply_preset_override_beats_default_but_not_explicit(self) -> None:
        """Preset override applies to unset dests; explicit values are untouched."""
        args = argparse.Namespace(preset="music")
        # flac_all explicitly set by the user -> must be preserved.
        args.flac_all = False
        args.silence_noise = presets._UNSET
        args.silence_min_duration_seconds = presets._UNSET
        args.target_bytes = presets._UNSET
        args.max_duration_seconds = presets._UNSET
        defaults = {dest: None for dest in presets.PRESET_TUNABLE_DESTS}
        presets.apply_preset(args, defaults)
        self.assertFalse(args.flac_all)  # explicit user value survives
        self.assertEqual(args.silence_noise, "-45dB")  # from music preset
        self.assertIsNone(args.target_bytes)  # not in preset -> default

    def test_apply_preset_returns_same_namespace(self) -> None:
        """apply_preset mutates and returns the same namespace object."""
        args = argparse.Namespace(preset=None)
        for dest in presets.PRESET_TUNABLE_DESTS:
            setattr(args, dest, presets._UNSET)
        result = presets.apply_preset(
            args, {d: 0 for d in presets.PRESET_TUNABLE_DESTS}
        )
        self.assertIs(result, args)


class FastPathTests(unittest.TestCase):
    def test_copy_extended_attributes_dummy(self) -> None:
        from media_shrinker import _copy_extended_attributes

        # We need to hit lines 703-704
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.txt"
            src.write_text("hello")
            dest = Path(tmp) / "dest.txt"
            dest.write_text("world")

            with patch(
                "os.listxattr", side_effect=OSError("Permission denied"), create=True
            ):
                _copy_extended_attributes(src, dest)

    def test_copy_macos_creation_time_dummy(self) -> None:
        from media_shrinker import _copy_macos_creation_time

        # Hit 1632
        class MockStat:
            st_birthtime = 12345.0

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "dest.txt"
            dest.write_text("hello")
            _copy_macos_creation_time(MockStat(), dest, "/bin/echo")

    def test_format_result_dummy(self) -> None:
        from media_shrinker import _format_result, ConversionResult

        # Hit 1654-1657
        result = ConversionResult(
            source_path=Path("/tmp/foo.txt"),
            output_path=Path("/tmp/foo.flac"),
            status="converted",
            original_size_bytes=200,
            output_size_bytes=100,
            strategy="flac-lossless",
        )
        s = _format_result(Path("/tmp"), result)
        self.assertIn("foo.txt", s)

    def test_copy_extended_attributes_dummy_success(self) -> None:
        from media_shrinker import _copy_extended_attributes

        # Hit 703-704
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.txt"
            src.write_text("hello")
            dest = Path(tmp) / "dest.txt"
            dest.write_text("world")

            with patch("os.listxattr", return_value=["user.test"], create=True):
                with patch("os.getxattr", side_effect=OSError("denied"), create=True):
                    _copy_extended_attributes(src, dest)

    def test_copy_macos_creation_time_dummy_none(self) -> None:
        from media_shrinker import _copy_macos_creation_time

        # Hit 1632
        class MockStat:
            st_birthtime = None

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "dest.txt"
            dest.write_text("hello")
            _copy_macos_creation_time(MockStat(), dest, "/bin/echo")

    def test_copy_extended_attributes_dummy_set_fail(self) -> None:
        from media_shrinker import _copy_extended_attributes

        # Hit 703-704
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.txt"
            src.write_text("hello")
            dest = Path(tmp) / "dest.txt"
            dest.write_text("world")

            with patch("os.listxattr", return_value=["user.test"], create=True):
                with patch("os.getxattr", return_value=b"value", create=True):
                    with patch(
                        "os.setxattr", side_effect=OSError("denied"), create=True
                    ):
                        _copy_extended_attributes(src, dest)

    def test_copy_macos_creation_time_dummy_not_found(self) -> None:
        from media_shrinker import _copy_macos_creation_time

        # Hit 1632
        class MockStat:
            st_birthtime = 12345.0

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "dest.txt"
            dest.write_text("hello")
            with patch("subprocess.run") as mock_run:
                _copy_macos_creation_time(MockStat(), dest, "/usr/bin/SetFile")
                mock_run.assert_called_once()

    def test_copy_macos_creation_time_dummy_success(self) -> None:
        from media_shrinker import _copy_macos_creation_time

        # Hit 1632
        class MockStat:
            st_birthtime = 12345.0

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "dest.txt"
            dest.write_text("hello")
            with patch("subprocess.run"):
                _copy_macos_creation_time(MockStat(), dest, "/bin/echo")

    def test_preserve_file_attributes_no_setfile(self) -> None:
        from media_shrinker import preserve_file_attributes

        # Hit 703-704
        class MockStat:
            st_atime_ns = 12345000
            st_mtime_ns = 67890000
            st_mode = 0o644

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.txt"
            src.write_text("hello")
            dest = Path(tmp) / "dest.txt"
            dest.write_text("world")
            with patch("os.stat", return_value=MockStat()):
                with patch("media_shrinker._get_setfile_path", return_value=None):
                    preserve_file_attributes(src, dest)

    def test_copy_extended_attributes_dummy_success_branch(self) -> None:
        from media_shrinker import _copy_extended_attributes

        # Hit 703-704
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.txt"
            src.write_text("hello")
            dest = Path(tmp) / "dest.txt"
            dest.write_text("world")

            with patch("os.listxattr", return_value=["user.test"], create=True):
                with patch("os.getxattr", return_value=b"value", create=True):
                    with patch("os.setxattr", create=True) as mock_set:
                        _copy_extended_attributes(src, dest)
                        mock_set.assert_called_once()

    def test_copy_extended_attributes_dummy_listxattr_missing(self) -> None:
        import builtins
        from media_shrinker import _copy_extended_attributes

        # Hit early return inside _copy_extended_attributes when OS doesn't support it
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.txt"
            src.write_text("hello")
            dest = Path(tmp) / "dest.txt"
            dest.write_text("world")

            original_hasattr = builtins.hasattr

            def fake_hasattr(obj, name):
                if name in ("listxattr", "getxattr", "setxattr"):
                    return False
                return original_hasattr(obj, name)

            with patch("builtins.hasattr", side_effect=fake_hasattr):
                _copy_extended_attributes(src, dest)

    def test_preserve_file_attributes_chmod_error(self) -> None:
        from media_shrinker import preserve_file_attributes

        # Hit 703-704
        class MockStat:
            st_atime_ns = 12345000
            st_mtime_ns = 67890000
            st_mode = 0o644

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.txt"
            src.write_text("hello")
            dest = Path(tmp) / "dest.txt"
            dest.write_text("world")
            with patch("os.stat", return_value=MockStat()):
                with patch("os.chmod", side_effect=OSError("denied")):
                    with patch("media_shrinker._get_setfile_path", return_value=None):
                        preserve_file_attributes(src, dest)

    def test_preserve_file_attributes_with_setfile(self) -> None:
        from media_shrinker import preserve_file_attributes

        # Hit 703-704
        class MockStat:
            st_atime_ns = 12345000
            st_mtime_ns = 67890000
            st_mode = 0o644

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.txt"
            src.write_text("hello")
            dest = Path(tmp) / "dest.txt"
            dest.write_text("world")
            with patch("os.stat", return_value=MockStat()):
                with patch("media_shrinker._copy_macos_creation_time"):
                    with patch(
                        "media_shrinker._get_setfile_path", return_value="/bin/echo"
                    ):
                        preserve_file_attributes(src, dest)

    def test_preserve_file_attributes_ignores_utime_error(self) -> None:
        """os.utime failure must not abort best-effort metadata copy.

        A read-only or timestamp-unsupporting destination filesystem can make
        os.utime raise OSError. The documented contract is best-effort, so the
        completed conversion must not be lost and the macOS creation-time step
        must still run.
        """
        from media_shrinker import preserve_file_attributes

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.txt"
            src.write_text("hello")
            dest = Path(tmp) / "dest.txt"
            dest.write_text("world")
            with patch("media_shrinker.os.utime", side_effect=OSError("read-only fs")):
                with patch("media_shrinker._copy_macos_creation_time") as mock_creation:
                    with patch(
                        "media_shrinker._get_setfile_path", return_value="/bin/echo"
                    ):
                        # Must not raise despite os.utime failing.
                        preserve_file_attributes(src, dest)
            # Best-effort continues to the creation-time step after utime fails.
            mock_creation.assert_called_once()

    @patch("subprocess.run")
    def test_ffprobe_timeout(self, mock_run: MagicMock) -> None:
        """Test ffprobe handles TimeoutExpired correctly."""
        from media_shrinker import probe_media, MediaShrinkerError
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["ffprobe"], timeout=60)
        with self.assertRaises(MediaShrinkerError) as ctx:
            probe_media(Path("/dummy.mp4"))

        self.assertIn("ffprobe timed out for", str(ctx.exception))

    @patch("subprocess.run")
    def test_silencedetect_timeout(self, mock_run: MagicMock) -> None:
        """Test silencedetect handles TimeoutExpired correctly."""
        from media_shrinker import detect_silence_intervals, MediaShrinkerError
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=3600)
        with self.assertRaises(MediaShrinkerError) as ctx:
            detect_silence_intervals(Path("/dummy.mp4"))

        self.assertIn("silencedetect timed out for", str(ctx.exception))

    @patch("shutil.which", return_value="/usr/bin/brctl")
    @patch("subprocess.run")
    def test_brctl_timeout(self, mock_run: MagicMock, mock_which: MagicMock) -> None:
        """Test brctl handles TimeoutExpired correctly."""
        from media_shrinker import download_from_icloud, MediaShrinkerError
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["brctl"], timeout=3600)
        with self.assertRaises(MediaShrinkerError) as ctx:
            download_from_icloud(Path("/dummy.mp4"))

        self.assertIn("iCloud download timed out for", str(ctx.exception))

    @patch("media_shrinker._ensure_not_source_path")
    @patch("media_shrinker._ensure_not_protected_source_path")
    @patch("tempfile.mkstemp", return_value=(0, "/tmp/.test.tmp"))
    @patch("os.close")
    @patch("pathlib.Path.unlink")
    @patch("subprocess.run")
    def test_convert_timeout(
        self,
        mock_run: MagicMock,
        mock_unlink: MagicMock,
        mock_close: MagicMock,
        mock_mkstemp: MagicMock,
        mock_ensure_prot: MagicMock,
        mock_ensure_src: MagicMock,
    ) -> None:
        """Test _execute_plan handles TimeoutExpired correctly."""
        from media_shrinker import _execute_plan, ConversionPlan, MediaShrinkerError
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=3600)
        plan = ConversionPlan(
            strategy="test",
            input_path=Path("/dummy.mp4"),
            output_path=Path("/tmp/dummy.mp4"),
            ffmpeg_args=["-i", "{input}", "-c:a", "copy", "{output}"],
            audio_bitrate_bps=128000,
        )

        with self.assertRaises(MediaShrinkerError) as ctx:
            _execute_plan(
                plan,
                source=Path("/dummy.mp4"),
                final_output=Path("/tmp/dummy.mp4"),
                overwrite=True,
                protected_sources=set(),
                ffmpeg_path="ffmpeg",
            )

        self.assertIn("ffmpeg timed out for", str(ctx.exception))

    @patch("subprocess.run")
    def test_copy_macos_creation_time_timeout(self, mock_run: MagicMock) -> None:
        """Test _copy_macos_creation_time ignores TimeoutExpired."""
        from media_shrinker import _copy_macos_creation_time
        import subprocess

        class MockStat:
            st_birthtime = 123456789.0

        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["SetFile"], timeout=60)
        # Should not raise
        _copy_macos_creation_time(MockStat(), Path("/dest"), "SetFile")


class VideoAllowTests(unittest.TestCase):
    """Opt-in audio-track extraction from video containers (--allow-video)."""

    def _video_probe(self, audio_codec):
        return MediaProbe(
            duration_seconds=1800.0,
            size_bytes=3_000_000_000,
            audio_codec=audio_codec,
            audio_bit_rate=192_000,
            has_video=True,
            format_name="mov,mp4,m4a,3gp,3g2,mj2",
        )

    def test_video_rejected_by_default(self) -> None:
        probe = self._video_probe("aac")
        with self.assertRaises(MediaShrinkerError) as cm:
            build_audio_plan(
                Path("zoom_meeting.mp4"),
                probe,
                target_bytes=1_900_000_000,
                output_dir=Path("out"),
            )
        self.assertIn("contains video", str(cm.exception))

    def test_video_with_audio_allowed_extracts_audio_track(self) -> None:
        probe = self._video_probe("aac")
        plan = build_audio_plan(
            Path("zoom_meeting.mp4"),
            probe,
            target_bytes=1_900_000_000,
            output_dir=Path("out"),
            allow_video=True,
        )
        # Audio-only extraction: video is dropped (-vn), output is audio.
        self.assertIn("-vn", plan.ffmpeg_args)
        self.assertTrue(str(plan.output_path).endswith(".opus"))

    def test_video_lossless_audio_allowed_uses_flac(self) -> None:
        probe = self._video_probe("flac")
        plan = build_audio_plan(
            Path("lecture.mov"),
            probe,
            target_bytes=1_900_000_000,
            output_dir=Path("out"),
            allow_video=True,
        )
        self.assertEqual(plan.strategy, "flac-lossless")
        self.assertIn("-vn", plan.ffmpeg_args)

    def test_video_without_audio_rejected_even_when_allowed(self) -> None:
        probe = self._video_probe(None)  # video container, no audio stream
        with self.assertRaises(MediaShrinkerError) as cm:
            build_audio_plan(
                Path("silent_clip.mp4"),
                probe,
                target_bytes=1_900_000_000,
                output_dir=Path("out"),
                allow_video=True,
            )
        self.assertIn("no audio stream", str(cm.exception))

    def test_parse_args_allow_video_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            off = media_shrinker.parse_args([tmp])
            on = media_shrinker.parse_args([tmp, "--allow-video"])
        self.assertFalse(off.allow_video)
        self.assertTrue(on.allow_video)


class OutputFormatTests(unittest.TestCase):
    """Selection of output codec/container via --format."""

    def _lossy_probe(self, audio_bit_rate: int | None = 192_000) -> MediaProbe:
        return MediaProbe(
            duration_seconds=1800.0,
            size_bytes=3_000_000_000,
            audio_codec="aac",
            audio_bit_rate=audio_bit_rate,
            has_video=False,
            format_name="mov",
        )

    def _plan(
        self,
        output_format: str,
        probe: MediaProbe | None = None,
        target_bytes: int = 1_900_000_000,
    ) -> ConversionPlan:
        return build_audio_plan(
            Path("meeting.wav"),
            probe or self._lossy_probe(),
            target_bytes=target_bytes,
            output_dir=Path("out"),
            output_format=output_format,
        )

    def test_aac_plan(self) -> None:
        plan = self._plan("aac")
        self.assertEqual(plan.strategy, "aac-bitrate")
        self.assertTrue(str(plan.output_path).endswith(".m4a"))
        self.assertIn("-vn", plan.ffmpeg_args)
        self.assertIn("aac", plan.ffmpeg_args)
        self.assertIn("0:a:0", plan.ffmpeg_args)
        self.assertIsNotNone(plan.audio_bitrate_bps)

    def test_mp3_plan_and_bitrate_cap(self) -> None:
        probe = MediaProbe(10.0, 3_000_000_000, "aac", None, False, "mov")
        plan = build_audio_plan(
            Path("clip.wav"),
            probe,
            target_bytes=1_900_000_000,
            output_dir=Path("out"),
            output_format="mp3",
        )
        self.assertEqual(plan.strategy, "mp3-bitrate")
        self.assertTrue(str(plan.output_path).endswith(".mp3"))
        self.assertIn("libmp3lame", plan.ffmpeg_args)
        self.assertEqual(plan.audio_bitrate_bps, media_shrinker.MP3_MAX_BITRATE_BPS)

    def test_opus_forced_even_for_lossless_source(self) -> None:
        lossless = MediaProbe(
            1800.0, 3_000_000_000, "pcm_s16le", 1_411_200, False, "wav"
        )
        self.assertEqual(self._plan("opus", probe=lossless).strategy, "opus-bitrate")

    def test_flac_forced_for_lossy_source(self) -> None:
        plan = self._plan("flac")
        self.assertEqual(plan.strategy, "flac-transcode")
        self.assertTrue(str(plan.output_path).endswith(".flac"))

    def test_auto_matches_legacy_behaviour(self) -> None:
        lossless = MediaProbe(1800.0, 3_000_000_000, "flac", 900_000, False, "flac")
        self.assertEqual(self._plan("auto", probe=lossless).strategy, "flac-lossless")
        self.assertEqual(self._plan("auto").strategy, "opus-bitrate")

    def test_forced_format_still_allows_opt_in_video_audio_extraction(self) -> None:
        video_probe = MediaProbe(
            1800.0,
            3_000_000_000,
            "aac",
            192_000,
            True,
            "mov,mp4,m4a,3gp,3g2,mj2",
        )
        plan = build_audio_plan(
            Path("meeting.mp4"),
            video_probe,
            target_bytes=1_900_000_000,
            output_dir=Path("out"),
            output_format="mp3",
            allow_video=True,
        )
        self.assertEqual(plan.strategy, "mp3-bitrate")
        self.assertTrue(str(plan.output_path).endswith(".mp3"))
        self.assertIn("-vn", plan.ffmpeg_args)

    def test_existing_output_suffixes_follow_selected_format(self) -> None:
        probe = self._lossy_probe()
        self.assertEqual(
            media_shrinker._existing_output_suffixes(
                "aac", prefer_flac=False, probe=probe
            ),
            (".m4a",),
        )
        self.assertEqual(
            media_shrinker._existing_output_suffixes(
                "mp3", prefer_flac=False, probe=probe
            ),
            (".mp3",),
        )
        self.assertEqual(
            media_shrinker._existing_output_suffixes(
                "opus", prefer_flac=False, probe=probe
            ),
            (".opus",),
        )
        self.assertEqual(
            media_shrinker._existing_output_suffixes(
                "flac", prefer_flac=False, probe=probe
            ),
            (".flac",),
        )
        self.assertEqual(
            media_shrinker._existing_output_suffixes(
                "auto", prefer_flac=False, probe=probe
            ),
            (".opus",),
        )
        self.assertEqual(
            media_shrinker._existing_output_suffixes(
                "auto", prefer_flac=True, probe=probe
            ),
            (".flac", ".opus"),
        )
        with self.assertRaisesRegex(MediaShrinkerError, "unsupported output format"):
            media_shrinker._existing_output_suffixes(
                "wav", prefer_flac=False, probe=probe
            )

    def test_unknown_output_format_rejected_for_direct_api_call(self) -> None:
        with self.assertRaisesRegex(MediaShrinkerError, "unsupported output format"):
            self._plan("wav")

    def test_parse_args_format_flag(self) -> None:
        self.assertEqual(media_shrinker.parse_args(["root"]).format, "auto")
        self.assertEqual(
            media_shrinker.parse_args(["root", "--format", "aac"]).format, "aac"
        )
        with self.assertRaises(SystemExit):
            media_shrinker.parse_args(["root", "--format", "wav"])


class MediaShrinkerParseCoverageTests(unittest.TestCase):
    def test_parse_silencedetect_intervals_no_silence(self) -> None:
        stderr = "some random ffmpeg progress output"
        intervals = parse_silencedetect_intervals(stderr)
        self.assertEqual(intervals, [])


if __name__ == "__main__":
    unittest.main()
