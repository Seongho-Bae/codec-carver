#!/usr/bin/env python3
"""Shrink large media files below a target size while preserving metadata.

The tool is intentionally conservative: it never overwrites or deletes source
files, can prefer lossless FLAC for all audio to avoid extra loss, falls back to
high-bitrate Opus when required by size constraints, and restores filesystem
metadata on generated files.
"""

import argparse
import functools
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import presets

import config_file


SUPPORTED_EXTENSIONS = {
    ".3gp",
    ".3gpp",
    ".ac3",
    ".aiff",
    ".amr",
    ".au",
    ".flac",
    ".m4a",
    ".mid",
    ".mp3",
    ".mxf",
    ".opus",
    ".ra",
    ".wav",
    ".weba",
    ".aac",
    ".asx",
    ".avi",
    ".ogm",
    ".ogv",
    ".m4v",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".wmv",
    ".webm",
}

SUPPORTED_EXTS_TUPLE = tuple(SUPPORTED_EXTENSIONS)

LOSSLESS_AUDIO_CODECS = {
    "alac",
    "flac",
    "pcm_alaw",
    "pcm_f32be",
    "pcm_f32le",
    "pcm_f64be",
    "pcm_f64le",
    "pcm_mulaw",
    "pcm_s16be",
    "pcm_s16le",
    "pcm_s24be",
    "pcm_s24le",
    "pcm_s32be",
    "pcm_s32le",
    "pcm_u8",
}

DEFAULT_SIZE_LIMIT_BYTES = 2_000_000_000
DEFAULT_TARGET_BYTES = 1_900_000_000
DEFAULT_MAX_SEGMENT_DURATION_SECONDS = 4 * 60 * 60
DEFAULT_SILENCE_NOISE = "-35dB"
LOUDNORM_FILTER = "loudnorm=I=-16:TP=-1.5:LRA=11"
DEFAULT_SILENCE_MIN_DURATION_SECONDS = 2.0
HARD_SPLIT_EPSILON_SECONDS = 0.001
DURATION_TOLERANCE_SECONDS = 2.0
BITRATE_SAFETY_MARGIN = 0.92
OPUS_MAX_BITRATE_BPS = 510_000
OPUS_MIN_REASONABLE_BITRATE_BPS = 16_000
MP3_MAX_BITRATE_BPS = 320_000  # libmp3lame ceiling
OUTPUT_FORMATS = ("auto", "flac", "opus", "aac", "mp3")
SILENCE_RE = re.compile(r"silence_(start|end):\s*(?P<value>-?[0-9]+(?:\.[0-9]+)?)")


class MediaShrinkerError(RuntimeError):
    """Raised when a media file cannot be processed safely."""


@dataclass(frozen=True)
class MediaProbe:
    """Relevant media properties discovered by ffprobe."""

    duration_seconds: float
    size_bytes: int
    audio_codec: str | None
    audio_bit_rate: int | None
    has_video: bool
    format_name: str


@dataclass(frozen=True)
class ConversionPlan:
    """A concrete ffmpeg conversion plan for one source file."""

    strategy: str
    input_path: Path
    output_path: Path
    ffmpeg_args: list[str]
    audio_bitrate_bps: int | None = None

    def command(
        self,
        *,
        ffmpeg_path: str = "ffmpeg",
        input_path: Path | None = None,
        output_path: Path | None = None,
        overwrite: bool = True,
    ) -> list[str]:
        """Return an executable ffmpeg command with optional path overrides."""

        args = list(self.ffmpeg_args)
        if input_path is not None:
            try:
                input_index = args.index("-i") + 1
            except ValueError as exc:
                raise MediaShrinkerError(
                    "ffmpeg argument template is missing '-i'"
                ) from exc
            args[input_index] = f"{input_path.resolve()}"
        if output_path is not None:
            args[-1] = f"{output_path.resolve()}"

        if overwrite:
            args = ["-y" if arg == "-n" else arg for arg in args]
        else:
            args = ["-n" if arg == "-y" else arg for arg in args]
        return [ffmpeg_path, *args]


@dataclass(frozen=True)
class ConversionResult:
    """Outcome for one processed source file."""

    source_path: Path
    output_path: Path | None
    status: str
    original_size_bytes: int
    output_size_bytes: int | None = None
    strategy: str | None = None
    message: str | None = None
    segment_index: int | None = None
    segment_count: int | None = None
    start_seconds: float | None = None
    duration_seconds: float | None = None


@dataclass(frozen=True)
class SilenceInterval:
    """A detected silence interval that can be used as a split boundary."""

    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class MediaSegment:
    """A source time window that will produce one output file."""

    index: int
    start_seconds: float
    duration_seconds: float
    total_segments: int


def find_candidates(
    root: Path,
    *,
    size_limit_bytes: int = DEFAULT_SIZE_LIMIT_BYTES,
    include_under_limit: bool = True,
    exclude_paths: Iterable[Path] = (),
    exclude_dir_prefixes: Iterable[str] = (),
) -> list[tuple[Path, int]]:
    """Return supported media files under root selected for conversion.

    The returned paths are absolute when root is absolute and are ordered by
    their relative path for repeatable batch runs.
    """

    root = Path(root)
    excluded = tuple(Path(item).resolve() for item in exclude_paths)
    excluded_prefixes = tuple(prefix.casefold() for prefix in exclude_dir_prefixes)
    candidates: list[tuple[Path, int]] = []

    excluded_exact_strs = tuple(str(p) for p in excluded)
    excluded_prefix_strs = tuple(s + os.sep for s in excluded_exact_strs)
    excluded_exact_set = frozenset(excluded_exact_strs)

    for dirpath_str, dirnames, filenames in os.walk(str(root)):
        if excluded_exact_strs:
            try:
                resolved_dir_str = os.path.realpath(dirpath_str)
            except OSError:
                continue

            if resolved_dir_str in excluded_exact_set or resolved_dir_str.startswith(
                excluded_prefix_strs
            ):
                dirnames[:] = []
                continue

        # Prune excluded directories
        valid_dirs = []
        for d in dirnames:
            if d.casefold().startswith(excluded_prefixes):
                continue

            if excluded_exact_strs:
                d_path_str = os.path.join(dirpath_str, d)
                try:
                    d_stat = os.lstat(d_path_str)
                    is_symlink = stat.S_ISLNK(d_stat.st_mode)
                except OSError:
                    continue

                if not is_symlink:
                    resolved_d_str = os.path.join(resolved_dir_str, d)
                else:
                    try:
                        resolved_d_str = os.path.realpath(d_path_str)
                    except OSError:
                        continue

                if resolved_d_str in excluded_exact_set or resolved_d_str.startswith(
                    excluded_prefix_strs
                ):
                    continue
            valid_dirs.append(d)
        dirnames[:] = valid_dirs

        for f in filenames:
            if not f.lower().endswith(SUPPORTED_EXTS_TUPLE):
                continue

            file_path_str = os.path.join(dirpath_str, f)

            try:
                st = os.lstat(file_path_str)
                if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
                    continue
                size = st.st_size
            except OSError:
                continue

            if excluded_exact_strs:
                resolved_file_str = os.path.join(resolved_dir_str, f)
                if (
                    resolved_file_str in excluded_exact_set
                    or resolved_file_str.startswith(excluded_prefix_strs)
                ):
                    continue

            if include_under_limit or size > size_limit_bytes:
                candidates.append((Path(file_path_str), size))

    # Fast path: Pre-compute the root string prefix to avoid slow Path.relative_to() instantiation in the sort loop
    # We add a trailing slash to handle cases where root represents a directory structure.
    root_prefix = root.as_posix()
    if not root_prefix.endswith("/"):
        root_prefix += "/"

    return sorted(
        candidates,
        key=lambda item: item[0].as_posix().removeprefix(root_prefix).casefold(),
    )


def calculate_audio_bitrate(
    duration_seconds: float,
    target_bytes: int,
    source_bitrate_bps: int | None,
    *,
    safety_margin: float = BITRATE_SAFETY_MARGIN,
) -> int:
    """Calculate the highest safe audio bitrate for target_bytes.

    Raises:
        ValueError: If duration or target size is not positive.
        MediaShrinkerError: If fitting the file would require an unusably low bitrate.
    """

    if duration_seconds <= 0:
        raise ValueError(f"duration_seconds must be positive, got {duration_seconds}")
    if target_bytes <= 0:
        raise ValueError(f"target_bytes must be positive, got {target_bytes}")

    fitting_bitrate = int((target_bytes * 8 * safety_margin) / duration_seconds)
    bitrate = min(fitting_bitrate, OPUS_MAX_BITRATE_BPS)
    # The floor guards against targets too small to fit at a usable quality.
    # It must be applied to the target-driven bitrate only: a source that is
    # already below the floor (e.g. a 12 kbps voice recording) still fits the
    # target and should be transcoded at its own bitrate, not rejected.
    if bitrate < OPUS_MIN_REASONABLE_BITRATE_BPS:
        raise MediaShrinkerError(
            f"Target size requires {bitrate} bps, below the safe floor of "
            f"{OPUS_MIN_REASONABLE_BITRATE_BPS} bps"
        )
    if source_bitrate_bps and source_bitrate_bps > 0:
        bitrate = min(bitrate, source_bitrate_bps)
    return bitrate


def _metadata_args(tags: dict[str, str] | None) -> list[str]:
    """Return ffmpeg ``-metadata key=value`` arguments for the given tags.

    Keys are emitted in sorted order so generated commands are deterministic.
    The pairs are meant to be inserted after ``-map_metadata 0`` in a plan:
    ffmpeg applies them on top of the metadata copied from the source, so each
    provided key overrides the corresponding source tag while every other
    source tag is preserved. Values are passed through verbatim as single
    argv items (plans run without a shell), so special characters are safe.

    Returns an empty list when tags is None or empty, keeping default plans
    byte-identical to those built before metadata tagging existed.
    """

    if not tags:
        return []
    args: list[str] = []
    for key in sorted(tags):
        args.extend(("-metadata", f"{key}={tags[key]}"))
    return args


def build_audio_plan(
    source_path: Path,
    probe: MediaProbe,
    *,
    target_bytes: int,
    output_dir: Path,
    prefer_flac: bool = False,
    ffmpeg_threads: int | None = None,
    segment: MediaSegment | None = None,
    tags: dict[str, str] | None = None,
    normalize: bool = False,
    output_format: str = "auto",
    allow_video: bool = False,
) -> ConversionPlan:
    """Build the preferred audio-only conversion plan for source_path.

    ``output_format`` selects the container/codec: ``auto`` (default) keeps the
    original behaviour (FLAC for lossless or ``--flac-all`` input, otherwise
    high-bitrate Opus); ``flac``/``opus`` force that codec; ``aac``/``mp3``
    produce a target-fitting lossy plan for broad device compatibility.

    When ``allow_video`` is False (the default) any file containing a video
    stream is rejected, preserving the audio-only contract. When True, the
    audio track is extracted from a video container (the ffmpeg plan already
    drops video with ``-vn``); a video file with no audio stream is still
    rejected below.

    When ``normalize`` is True the EBU R128 loudnorm audio filter is applied so
    generated audio has consistent loudness; when False the args are unchanged.

    When ``tags`` is provided, ``-metadata key=value`` pairs are injected after
    ``-map_metadata 0`` so they override those specific keys copied from the
    source while leaving all other source metadata intact.
    """

    if probe.has_video and not allow_video:
        raise MediaShrinkerError(
            f"{source_path} contains video; this tool is configured for audio "
            f"preservation (pass --allow-video to extract the audio track)"
        )
    if not probe.audio_codec:
        raise MediaShrinkerError(f"{source_path} has no audio stream")
    if output_format not in OUTPUT_FORMATS:
        raise MediaShrinkerError(f"unsupported output format: {output_format}")

    if output_format == "aac":
        return _build_lossy_plan(
            source_path,
            probe,
            target_bytes=target_bytes,
            output_dir=output_dir,
            suffix=".m4a",
            codec="aac",
            strategy="aac-bitrate",
            ffmpeg_threads=ffmpeg_threads,
            segment=segment,
            tags=tags,
            normalize=normalize,
        )
    if output_format == "mp3":
        return _build_lossy_plan(
            source_path,
            probe,
            target_bytes=target_bytes,
            output_dir=output_dir,
            suffix=".mp3",
            codec="libmp3lame",
            strategy="mp3-bitrate",
            max_bitrate=MP3_MAX_BITRATE_BPS,
            ffmpeg_threads=ffmpeg_threads,
            segment=segment,
            tags=tags,
            normalize=normalize,
        )
    if output_format == "opus":
        return build_opus_plan(
            source_path,
            probe,
            target_bytes=target_bytes,
            output_dir=output_dir,
            ffmpeg_threads=ffmpeg_threads,
            segment=segment,
            tags=tags,
            normalize=normalize,
        )

    force_flac = prefer_flac or output_format == "flac"
    suffix = ".flac" if force_flac or _is_lossless_probe(probe) else ".opus"
    output_path = _planned_output_path(
        _segment_source_path(source_path, segment), output_dir, suffix
    )

    if suffix == ".flac":
        strategy = "flac-lossless" if _is_lossless_probe(probe) else "flac-transcode"
        args = [
            "-nostdin",
            "-hide_banner",
            "-y",
        ]
        args.extend(_segment_input_args(segment))
        args.extend(
            [
                "-protocol_whitelist",
                "file,crypto,data",
                "-i",
                str(source_path),
                "-map",
                "0:a:0",
                "-map_metadata",
                "0",
                "-map_chapters",
                "0",
                *_metadata_args(tags),
                "-vn",
                "-c:a",
                "flac",
                "-compression_level",
                "12",
                str(output_path),
            ]
        )
        args = _with_loudnorm(args, normalize)
        args = _with_ffmpeg_threads(args, ffmpeg_threads)
        return ConversionPlan(
            strategy=strategy,
            input_path=source_path,
            output_path=output_path,
            ffmpeg_args=args,
        )

    return build_opus_plan(
        source_path,
        probe,
        target_bytes=target_bytes,
        output_dir=output_dir,
        ffmpeg_threads=ffmpeg_threads,
        segment=segment,
        tags=tags,
        normalize=normalize,
    )


def build_opus_plan(
    source_path: Path,
    probe: MediaProbe,
    *,
    target_bytes: int,
    output_dir: Path,
    ffmpeg_threads: int | None = None,
    segment: MediaSegment | None = None,
    tags: dict[str, str] | None = None,
    normalize: bool = False,
) -> ConversionPlan:
    """Build a high-quality Opus plan that fits the target size.

    When normalize is True the EBU R128 loudnorm audio filter is applied so
    generated audio has consistent loudness; when False the args are unchanged.
    When tags is provided, ``-metadata key=value`` pairs are injected after
    ``-map_metadata 0`` so they override those specific keys copied from the
    source while leaving all other source metadata intact.
    """

    duration_seconds = (
        segment.duration_seconds if segment is not None else probe.duration_seconds
    )
    bitrate = calculate_audio_bitrate(
        duration_seconds, target_bytes, probe.audio_bit_rate
    )
    output_path = _planned_output_path(
        _segment_source_path(source_path, segment), output_dir, ".opus"
    )
    args = [
        "-nostdin",
        "-hide_banner",
        "-y",
    ]
    args.extend(_segment_input_args(segment))
    args.extend(
        [
            "-protocol_whitelist",
            "file,crypto,data",
            "-i",
            str(source_path),
            "-map",
            "0:a:0",
            "-map_metadata",
            "0",
            "-map_chapters",
            "0",
            *_metadata_args(tags),
            "-vn",
            "-c:a",
            "libopus",
            "-application",
            "audio",
            "-b:a",
            str(bitrate),
            "-vbr",
            "on",
            "-compression_level",
            "10",
            str(output_path),
        ]
    )
    args = _with_loudnorm(args, normalize)
    args = _with_ffmpeg_threads(args, ffmpeg_threads)
    return ConversionPlan(
        strategy="opus-bitrate",
        input_path=source_path,
        output_path=output_path,
        ffmpeg_args=args,
        audio_bitrate_bps=bitrate,
    )


def _build_lossy_plan(
    source_path: Path,
    probe: MediaProbe,
    *,
    target_bytes: int,
    output_dir: Path,
    suffix: str,
    codec: str,
    strategy: str,
    max_bitrate: int | None = None,
    ffmpeg_threads: int | None = None,
    segment: MediaSegment | None = None,
    tags: dict[str, str] | None = None,
    normalize: bool = False,
) -> ConversionPlan:
    """Build a lossy audio plan (aac/mp3) whose bitrate fits the target size."""

    duration_seconds = (
        segment.duration_seconds if segment is not None else probe.duration_seconds
    )
    bitrate = calculate_audio_bitrate(
        duration_seconds, target_bytes, probe.audio_bit_rate
    )
    if max_bitrate is not None:
        bitrate = min(bitrate, max_bitrate)
    output_path = _planned_output_path(
        _segment_source_path(source_path, segment), output_dir, suffix
    )
    args = [
        "-nostdin",
        "-hide_banner",
        "-y",
    ]
    args.extend(_segment_input_args(segment))
    args.extend(
        [
            "-protocol_whitelist",
            "file,crypto,data",
            "-i",
            str(source_path),
            "-map",
            "0:a:0",
            "-map_metadata",
            "0",
            "-map_chapters",
            "0",
            *_metadata_args(tags),
            "-vn",
            "-c:a",
            codec,
            "-b:a",
            str(bitrate),
            str(output_path),
        ]
    )
    args = _with_loudnorm(args, normalize)
    args = _with_ffmpeg_threads(args, ffmpeg_threads)
    return ConversionPlan(
        strategy=strategy,
        input_path=source_path,
        output_path=output_path,
        ffmpeg_args=args,
        audio_bitrate_bps=bitrate,
    )


def _existing_output_suffixes(
    output_format: str, *, prefer_flac: bool, probe: MediaProbe
) -> tuple[str, ...]:
    """Return generated output suffixes that may satisfy the selected format."""

    if output_format == "aac":
        return (".m4a",)
    if output_format == "mp3":
        return (".mp3",)
    if output_format == "opus":
        return (".opus",)
    if output_format == "flac":
        return (".flac",)
    if output_format != "auto":
        raise MediaShrinkerError(f"unsupported output format: {output_format}")
    if prefer_flac or _is_lossless_probe(probe):
        return (".flac", ".opus")
    return (".opus",)


def _run_media_tool(
    command: list[str],
    *,
    tool: str,
    timeout: float | None = None,
    timeout_message: str | None = None,
) -> "subprocess.CompletedProcess[str]":
    """Run an ffmpeg/ffprobe command, mapping missing binaries clearly.

    ffmpeg/ffprobe echo source-file metadata into their output. Those
    attacker-influenceable legacy-encoded bytes may not be valid UTF-8.
    Replacement decoding preserves the ASCII parser structure while preventing
    an untrusted metadata byte from raising ``UnicodeDecodeError``.
    """

    try:
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise MediaShrinkerError(
            f"{tool} not found: could not run '{command[0]}'. "
            "Install ffmpeg (it provides both ffmpeg and ffprobe) and make sure "
            "it is on your PATH; for example, 'brew install ffmpeg' (macOS) or "
            "'sudo apt install ffmpeg' (Debian/Ubuntu)."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaShrinkerError(timeout_message or f"{tool} timed out") from exc


def probe_media(
    source_path: Path,
    *,
    ffprobe_path: str = "ffprobe",
    source_size: int | None = None,
) -> MediaProbe:
    """Probe source_path with ffprobe and return normalized media properties."""

    command = [
        ffprobe_path,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        "-protocol_whitelist",
        "file,crypto,data",
        "-i",
        f"{Path(source_path).resolve()}",
    ]
    completed = _run_media_tool(
        command,
        tool="ffprobe",
        timeout=60,
        timeout_message=f"ffprobe timed out for {source_path}",
    )
    if completed.returncode != 0:
        raise MediaShrinkerError(
            f"ffprobe failed for {source_path}: {completed.stderr.strip()}"
        )

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise MediaShrinkerError(
            f"ffprobe returned invalid JSON for {source_path}"
        ) from exc

    return _parse_probe_payload(payload, source_path, source_size=source_size)


def build_silencedetect_command(
    source_path: Path,
    *,
    ffmpeg_path: str = "ffmpeg",
    silence_noise: str = DEFAULT_SILENCE_NOISE,
    silence_min_duration_seconds: float = DEFAULT_SILENCE_MIN_DURATION_SECONDS,
) -> list[str]:
    """Build an ffmpeg command that detects long silence intervals."""

    if not re.match(r"^[+-]?[0-9]+(\.[0-9]+)?(?:dB)?$", silence_noise):
        raise MediaShrinkerError(f"Invalid silence_noise value: {silence_noise}")

    return [
        ffmpeg_path,
        "-nostdin",
        "-hide_banner",
        "-protocol_whitelist",
        "file,crypto,data",
        "-i",
        f"{Path(source_path).resolve()}",
        "-af",
        f"silencedetect=noise={silence_noise}:d={_format_seconds(silence_min_duration_seconds)}",
        "-f",
        "null",
        "-",
    ]


def detect_silence_intervals(
    source_path: Path,
    *,
    ffmpeg_path: str = "ffmpeg",
    silence_noise: str = DEFAULT_SILENCE_NOISE,
    silence_min_duration_seconds: float = DEFAULT_SILENCE_MIN_DURATION_SECONDS,
) -> list[SilenceInterval]:
    """Run ffmpeg silencedetect and return paired silence intervals."""

    completed = _run_media_tool(
        build_silencedetect_command(
            source_path,
            ffmpeg_path=ffmpeg_path,
            silence_noise=silence_noise,
            silence_min_duration_seconds=silence_min_duration_seconds,
        ),
        tool="ffmpeg",
        timeout=3600,
        timeout_message=f"silencedetect timed out for {source_path}",
    )
    if completed.returncode != 0:
        raise MediaShrinkerError(
            f"silencedetect failed for {source_path}: {completed.stderr.strip()}"
        )
    return parse_silencedetect_intervals(completed.stderr)


def parse_silencedetect_intervals(stderr: str) -> list[SilenceInterval]:
    """Parse ffmpeg silencedetect stderr into complete silence intervals."""

    intervals: list[SilenceInterval] = []
    current_start: float | None = None

    if "silence_" not in stderr:
        return intervals

    # Fast path: Using re.finditer directly on the raw string avoids
    # OOM issues and overhead from str.splitlines() on massive ffmpeg logs.
    for match in SILENCE_RE.finditer(stderr):
        kind = match.group(1)
        value = float(match.group("value"))
        if kind == "start":
            current_start = max(value, 0.0)
        elif kind == "end" and current_start is not None:
            if value > current_start:
                intervals.append(
                    SilenceInterval(
                        start_seconds=current_start, end_seconds=value
                    )
                )
            current_start = None
    return intervals


def build_segments(
    *,
    duration_seconds: float,
    max_segment_duration_seconds: float = DEFAULT_MAX_SEGMENT_DURATION_SECONDS,
    silence_intervals: Iterable[SilenceInterval] = (),
) -> list[MediaSegment]:
    """Build source windows that are each shorter than max_segment_duration_seconds.

    Split points prefer the latest safe point inside a detected silence interval
    that fits inside the current four-hour window. If no silence exists in that
    window, a hard split is placed just below the maximum duration.
    """

    if duration_seconds <= 0:
        raise ValueError(f"duration_seconds must be positive, got {duration_seconds}")
    if max_segment_duration_seconds <= HARD_SPLIT_EPSILON_SECONDS:
        raise ValueError(
            "max_segment_duration_seconds must be greater than "
            f"{HARD_SPLIT_EPSILON_SECONDS}"
        )

    if duration_seconds < max_segment_duration_seconds:
        return [
            MediaSegment(
                index=1,
                start_seconds=0.0,
                duration_seconds=duration_seconds,
                total_segments=1,
            )
        ]

    sorted_intervals = sorted(silence_intervals, key=lambda item: item.start_seconds)
    split_points: list[float] = []
    segment_start = 0.0
    while duration_seconds - segment_start >= max_segment_duration_seconds:
        window_end = segment_start + max_segment_duration_seconds
        split_point = _choose_silence_split_point(
            segment_start, window_end, sorted_intervals
        )
        if split_point is None:
            split_point = window_end - HARD_SPLIT_EPSILON_SECONDS
        if split_point <= segment_start:
            split_point = min(
                duration_seconds,
                segment_start
                + max_segment_duration_seconds
                - HARD_SPLIT_EPSILON_SECONDS,
            )
        split_points.append(split_point)
        segment_start = split_point

    boundaries = [0.0, *split_points, duration_seconds]
    total_segments = len(boundaries) - 1
    return [
        MediaSegment(
            index=index,
            start_seconds=start,
            duration_seconds=end - start,
            total_segments=total_segments,
        )
        for index, (start, end) in enumerate(zip(boundaries, boundaries[1:]), start=1)
    ]


def build_icloud_download_command(
    source_path: Path, *, brctl_path: str = "brctl"
) -> list[str]:
    """Build a safe iCloud download command for source_path."""

    return [brctl_path, "download", f"{source_path.resolve()}"]


def download_from_icloud(source_path: Path, *, brctl_path: str = "brctl") -> None:
    """Ask macOS iCloud Drive to materialize source_path before media reads."""

    if not shutil.which(brctl_path):
        raise MediaShrinkerError(
            f"iCloud download requested but '{brctl_path}' was not found"
        )
    try:
        completed = subprocess.run(
            build_icloud_download_command(source_path, brctl_path=brctl_path),
            check=False,
            capture_output=True,
            text=True,
            shell=False,
            timeout=3600,
        )
    except subprocess.TimeoutExpired as exc:
        raise MediaShrinkerError(f"iCloud download timed out for {source_path}") from exc
    if completed.returncode != 0:
        raise MediaShrinkerError(
            f"iCloud download failed for {source_path}: {completed.stderr.strip()}"
        )


def choose_worker_count(requested_workers: int, *, cpu_count: int | None = None) -> int:
    """Choose a safe parallel conversion worker count.

    requested_workers <= 0 means automatic: use multiple workers, but leave CPU
    headroom because each ffmpeg process can also use multiple threads.
    """

    if requested_workers > 0:
        return requested_workers
    cores = cpu_count or os.cpu_count() or 1
    return max(1, min(4, cores // 2))


@functools.cache
def _get_setfile_path() -> str | None:
    """Return the path to the SetFile executable, cached for efficiency."""
    return shutil.which("SetFile")


def preserve_file_attributes(
    source: Path, dest: Path, *, setfile_path: str | None = None
) -> None:
    """Best-effort copy of filesystem metadata from source to dest.

    This preserves permissions, atime/mtime with nanosecond precision, extended
    attributes where the OS exposes them, and macOS creation date when SetFile is
    available. Non-critical metadata copy failures are ignored so a completed
    conversion is not lost.
    """

    source = Path(source)
    dest = Path(dest)
    source_stat = source.stat()

    try:
        os.chmod(dest, stat.S_IMODE(source_stat.st_mode) & 0o777)
    except OSError:
        pass

    _copy_extended_attributes(source, dest)

    _restore_timestamps(source_stat, dest)

    resolved_setfile = setfile_path if setfile_path is not None else _get_setfile_path()
    if resolved_setfile:
        _copy_macos_creation_time(source_stat, dest, resolved_setfile)
        _restore_timestamps(source_stat, dest)


def convert_file(
    source: Path,
    *,
    root: Path,
    output_dir: Path,
    target_bytes: int = DEFAULT_TARGET_BYTES,
    ffmpeg_path: str = "ffmpeg",
    ffprobe_path: str = "ffprobe",
    download_icloud: bool = False,
    brctl_path: str = "brctl",
    prefer_flac: bool = False,
    output_format: str = "auto",
    ffmpeg_threads: int | None = None,
    overwrite: bool = False,
    max_segment_duration_seconds: float = DEFAULT_MAX_SEGMENT_DURATION_SECONDS,
    silence_noise: str = DEFAULT_SILENCE_NOISE,
    silence_min_duration_seconds: float = DEFAULT_SILENCE_MIN_DURATION_SECONDS,
    protected_sources: Iterable[Path] = (),
    resolved_protected_sources: frozenset[Path] | None = None,
    original_size: int | None = None,
    tags: dict[str, str] | None = None,
    normalize: bool = False,
    post_process: Callable[[ConversionResult], None] | None = None,
    allow_video: bool = False,
) -> list[ConversionResult]:
    """Convert one file and return generated segment results without deleting the source.

    ``post_process`` is an optional hook invoked once per successfully generated
    output (a ``converted`` result with an output path). It is the extension
    seam used for follow-on steps such as transcription; a failing hook must not
    abort the conversion, so callers are expected to handle their own errors.
    ``allow_video`` preserves the default audio-only contract unless callers opt
    into extracting the audio stream from video containers.
    When ``normalize`` is True generated audio is loudness-normalized with the
    EBU R128 loudnorm filter; the default of False keeps prior output.
    When ``tags`` is provided, each generated output is stamped with those
    ``-metadata`` key/value pairs on top of the metadata copied from the source.
    """

    source = Path(source)
    root = Path(root)
    output_dir = Path(output_dir)
    resolved_source = source.resolve()
    resolved_root = root.resolve()

    if not resolved_source.is_relative_to(resolved_root):
        raise MediaShrinkerError("Source path is outside the permitted root directory")

    original_size = (
        original_size if original_size is not None else safe_source_size(source)
    )

    rel_source = resolved_source.relative_to(resolved_root)
    if download_icloud:
        download_from_icloud(source, brctl_path=brctl_path)
    probe = probe_media(source, ffprobe_path=ffprobe_path, source_size=original_size)

    silence_intervals: list[SilenceInterval] = []
    if probe.duration_seconds >= max_segment_duration_seconds:
        silence_intervals = detect_silence_intervals(
            source,
            ffmpeg_path=ffmpeg_path,
            silence_noise=silence_noise,
            silence_min_duration_seconds=silence_min_duration_seconds,
        )

    segments = build_segments(
        duration_seconds=probe.duration_seconds,
        max_segment_duration_seconds=max_segment_duration_seconds,
        silence_intervals=silence_intervals,
    )

    resolved_sources = resolved_protected_sources
    if resolved_sources is None:
        resolved_sources = frozenset(Path(item).resolve() for item in protected_sources)

    results = [
        _convert_segment(
            source,
            rel_source=rel_source,
            probe=probe,
            segment=segment,
            output_dir=output_dir,
            target_bytes=target_bytes,
            original_size=original_size,
            ffmpeg_path=ffmpeg_path,
            ffprobe_path=ffprobe_path,
            prefer_flac=prefer_flac,
            ffmpeg_threads=ffmpeg_threads,
            overwrite=overwrite,
            max_segment_duration_seconds=max_segment_duration_seconds,
            protected_sources=resolved_sources,
            tags=tags,
            normalize=normalize,
            output_format=output_format,
            allow_video=allow_video,
        )
        for segment in segments
    ]
    _run_post_process(results, post_process)
    return results


def _run_post_process(
    results: list[ConversionResult],
    post_process: Callable[[ConversionResult], None] | None,
) -> None:
    """Invoke ``post_process`` for each successfully generated output."""
    if post_process is None:
        return
    for result in results:
        if result.status == "converted" and result.output_path is not None:
            post_process(result)


def safe_source_size(source: Path) -> int:
    """Return source size for reports without letting stat failures abort a batch."""

    try:
        return Path(source).stat().st_size
    except OSError:
        return 0


def _find_valid_existing_output(
    source: Path,
    *,
    segment_rel_source: Path,
    segment: MediaSegment,
    output_dir: Path,
    target_bytes: int,
    original_size: int,
    ffprobe_path: str,
    max_segment_duration_seconds: float,
    resolved_protected_sources: frozenset[Path],
    existing_suffixes: tuple[str, ...],
) -> ConversionResult | None:
    """Return a ConversionResult if a valid output already exists on disk."""
    existing_output: Path | None = None
    existing_duration: float | None = None
    for suffix in existing_suffixes:
        candidate = _planned_output_path(segment_rel_source, output_dir, suffix)
        # Fast path: Rely on stat() throwing OSError to check existence and get size simultaneously,
        # avoiding a redundant exists() syscall. Also defers collision checks for non-existent files.
        try:
            candidate_size = candidate.stat().st_size
        except OSError:
            continue
        _ensure_not_source_path(source, candidate)
        _ensure_not_protected_source_path(resolved_protected_sources, candidate)
        if candidate_size > target_bytes:
            _remove_generated_output(
                source, candidate, protected_sources=resolved_protected_sources
            )
            continue
        candidate_duration = _probe_output_duration(
            candidate, ffprobe_path=ffprobe_path, output_size=candidate_size
        )
        if (
            candidate_duration >= max_segment_duration_seconds
            or not _duration_matches_expected(
                candidate_duration,
                segment.duration_seconds,
            )
        ):
            _remove_generated_output(
                source, candidate, protected_sources=resolved_protected_sources
            )
            continue
        existing_output = candidate
        existing_duration = candidate_duration
        existing_size = candidate_size
        break

    if existing_output is not None:
        return ConversionResult(
            source_path=source,
            output_path=existing_output,
            status="skipped_existing",
            original_size_bytes=original_size,
            output_size_bytes=existing_size,
            strategy="existing",
            message="Existing output is already under target",
            segment_index=segment.index,
            segment_count=segment.total_segments,
            start_seconds=segment.start_seconds,
            duration_seconds=existing_duration,
        )
    return None


def _execute_segment_conversion(
    source: Path,
    *,
    rel_source: Path,
    probe: MediaProbe,
    segment: MediaSegment,
    output_dir: Path,
    target_bytes: int,
    original_size: int,
    ffmpeg_path: str,
    ffprobe_path: str,
    prefer_flac: bool,
    ffmpeg_threads: int | None,
    overwrite: bool,
    max_segment_duration_seconds: float,
    resolved_protected_sources: frozenset[Path],
    tags: dict[str, str] | None = None,
    normalize: bool = False,
    output_format: str = "auto",
    allow_video: bool = False,
) -> ConversionResult:
    """Execute the conversion plan(s) for a single segment.

    ``output_format``/``allow_video`` select the conversion contract, while
    ``normalize`` applies EBU R128 loudnorm to generated audio when requested.
    """
    plan = build_audio_plan(
        rel_source,
        probe,
        target_bytes=target_bytes,
        output_dir=output_dir,
        prefer_flac=prefer_flac,
        ffmpeg_threads=ffmpeg_threads,
        segment=segment,
        tags=tags,
        normalize=normalize,
        output_format=output_format,
        allow_video=allow_video,
    )

    final_output = _resolve_collision(plan.output_path, overwrite=overwrite)
    first_result = _execute_plan(
        plan,
        source,
        final_output,
        ffmpeg_path=ffmpeg_path,
        overwrite=overwrite,
        protected_sources=resolved_protected_sources,
    )
    output_size = first_result.stat().st_size
    if output_size > target_bytes and plan.strategy in {
        "flac-lossless",
        "flac-transcode",
    }:
        _remove_generated_output(
            source, first_result, protected_sources=resolved_protected_sources
        )
        opus_plan = build_opus_plan(
            rel_source,
            probe,
            target_bytes=target_bytes,
            output_dir=output_dir,
            ffmpeg_threads=ffmpeg_threads,
            segment=segment,
            tags=tags,
            normalize=normalize,
        )
        final_output = _resolve_collision(opus_plan.output_path, overwrite=overwrite)
        first_result = _execute_plan(
            opus_plan,
            source,
            final_output,
            ffmpeg_path=ffmpeg_path,
            overwrite=overwrite,
            protected_sources=resolved_protected_sources,
        )
        plan = opus_plan
        output_size = first_result.stat().st_size

    try:
        output_duration = _probe_output_duration(
            first_result, ffprobe_path=ffprobe_path, output_size=output_size
        )
    except MediaShrinkerError as exc:
        return _discard_invalid_generated_output(
            source,
            first_result,
            {
                "source_path": source,
                "output_path": first_result,
                "original_size_bytes": original_size,
                "output_size_bytes": output_size,
                "strategy": plan.strategy,
                "segment_index": segment.index,
                "segment_count": segment.total_segments,
                "start_seconds": segment.start_seconds,
                "duration_seconds": None,
            },
            status="duration_mismatch",
            message=f"Generated output has no usable duration: {exc}",
            protected_sources=resolved_protected_sources,
        )
    preserve_file_attributes(source, first_result)
    common_fields = {
        "source_path": source,
        "output_path": first_result,
        "original_size_bytes": original_size,
        "output_size_bytes": output_size,
        "strategy": plan.strategy,
        "segment_index": segment.index,
        "segment_count": segment.total_segments,
        "start_seconds": segment.start_seconds,
        "duration_seconds": output_duration,
    }

    if output_size > target_bytes:
        return _discard_invalid_generated_output(
            source,
            first_result,
            common_fields,
            status="too_large",
            message=f"Output remains above target: {output_size} > {target_bytes}",
            protected_sources=resolved_protected_sources,
        )

    if output_duration >= max_segment_duration_seconds:
        return _discard_invalid_generated_output(
            source,
            first_result,
            common_fields,
            status="too_long",
            message=(
                "Output segment duration remains at or above the configured maximum: "
                f"{output_duration} >= {max_segment_duration_seconds}"
            ),
            protected_sources=resolved_protected_sources,
        )

    if not _duration_matches_expected(output_duration, segment.duration_seconds):
        return _discard_invalid_generated_output(
            source,
            first_result,
            common_fields,
            status="duration_mismatch",
            message=(
                "Output segment duration does not match the planned segment: "
                f"{output_duration} vs {segment.duration_seconds}"
            ),
            protected_sources=resolved_protected_sources,
        )

    return ConversionResult(**common_fields, status="converted")


def _convert_segment(
    source: Path,
    *,
    rel_source: Path,
    probe: MediaProbe,
    segment: MediaSegment,
    output_dir: Path,
    target_bytes: int,
    original_size: int,
    ffmpeg_path: str,
    ffprobe_path: str,
    prefer_flac: bool,
    ffmpeg_threads: int | None,
    overwrite: bool,
    max_segment_duration_seconds: float,
    protected_sources: frozenset[Path] = frozenset(),
    tags: dict[str, str] | None = None,
    normalize: bool = False,
    output_format: str = "auto",
    allow_video: bool = False,
) -> ConversionResult:
    """Convert one media segment fitting the target size limit.

    Format selection, video opt-in, and loudness normalization are threaded
    through to the concrete conversion plan.
    """
    # protected_sources is passed from convert_file where it is already fully resolved.
    # _ensure_not_source_path explicitly checks against the current source independently.
    resolved_protected_sources = protected_sources
    existing_suffixes = _existing_output_suffixes(
        output_format, prefer_flac=prefer_flac, probe=probe
    )
    segment_rel_source = _segment_source_path(rel_source, segment)
    _remove_invalid_legacy_outputs(
        source,
        rel_source=rel_source,
        probe=probe,
        output_dir=output_dir,
        suffixes=existing_suffixes,
        target_bytes=target_bytes,
        ffprobe_path=ffprobe_path,
        max_segment_duration_seconds=max_segment_duration_seconds,
        protected_sources=resolved_protected_sources,
    )

    existing_result = _find_valid_existing_output(
        source,
        segment_rel_source=segment_rel_source,
        segment=segment,
        output_dir=output_dir,
        target_bytes=target_bytes,
        original_size=original_size,
        ffprobe_path=ffprobe_path,
        max_segment_duration_seconds=max_segment_duration_seconds,
        resolved_protected_sources=resolved_protected_sources,
        existing_suffixes=existing_suffixes,
    )
    if existing_result is not None:
        return existing_result

    return _execute_segment_conversion(
        source,
        rel_source=rel_source,
        probe=probe,
        segment=segment,
        output_dir=output_dir,
        target_bytes=target_bytes,
        original_size=original_size,
        ffmpeg_path=ffmpeg_path,
        ffprobe_path=ffprobe_path,
        prefer_flac=prefer_flac,
        ffmpeg_threads=ffmpeg_threads,
        overwrite=overwrite,
        max_segment_duration_seconds=max_segment_duration_seconds,
        resolved_protected_sources=resolved_protected_sources,
        tags=tags,
        normalize=normalize,
        output_format=output_format,
        allow_video=allow_video,
    )


def _remove_invalid_legacy_outputs(
    source: Path,
    *,
    rel_source: Path,
    probe: MediaProbe,
    output_dir: Path,
    suffixes: Iterable[str],
    target_bytes: int,
    ffprobe_path: str,
    max_segment_duration_seconds: float,
    protected_sources: frozenset[Path],
) -> None:
    """Remove invalid stem-only outputs created by earlier tool versions."""

    for suffix in suffixes:
        legacy_output = output_dir / rel_source.with_suffix(suffix)
        canonical_output = _planned_output_path(rel_source, output_dir, suffix)
        if legacy_output == canonical_output:
            continue
        # Fast path: Rely on stat() throwing OSError to check existence and get size simultaneously,
        # avoiding a redundant exists() syscall. Also defers collision checks for non-existent files.
        try:
            legacy_size = legacy_output.stat().st_size
        except OSError:
            continue
        _ensure_not_source_path(source, legacy_output)
        _ensure_not_protected_source_path(protected_sources, legacy_output)
        if legacy_size > target_bytes:
            _remove_generated_output(
                source, legacy_output, protected_sources=protected_sources
            )
            continue
        legacy_duration = _probe_output_duration(
            legacy_output, ffprobe_path=ffprobe_path, output_size=legacy_size
        )
        if (
            legacy_duration >= max_segment_duration_seconds
            or not _duration_matches_expected(
                legacy_duration,
                probe.duration_seconds,
            )
        ):
            _remove_generated_output(
                source, legacy_output, protected_sources=protected_sources
            )


def write_report(results: Iterable[ConversionResult], report_path: Path) -> None:
    """Write a machine-readable JSON conversion report."""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "source_path": str(result.source_path),
            "output_path": str(result.output_path) if result.output_path else None,
            "status": result.status,
            "strategy": result.strategy,
            "original_size_bytes": result.original_size_bytes,
            "output_size_bytes": result.output_size_bytes,
            "segment_index": result.segment_index,
            "segment_count": result.segment_count,
            "start_seconds": result.start_seconds,
            "duration_seconds": result.duration_seconds,
            "message": result.message,
        }
        for result in results
    ]
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _normalize_argv(argv: list[str] | None) -> list[str] | None:
    """Normalize option values that older argparse versions treat as options."""

    if argv is None:
        return None

    normalized: list[str] = []
    iterator = iter(argv)
    for arg in iterator:
        if arg == "--silence-noise":
            try:
                value = next(iterator)
            except StopIteration:
                normalized.append(arg)
                break
            if value.startswith("-"):
                normalized.append(f"{arg}={value}")
            else:
                normalized.extend((arg, value))
            continue
        normalized.append(arg)
    return normalized


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "root", nargs="?", default=".", type=Path, help="Folder to scan"
    )
    parser.add_argument(
        "--size-limit-bytes",
        type=int,
        default=DEFAULT_SIZE_LIMIT_BYTES,
        help="Convert files larger than this size in bytes",
    )
    parser.add_argument(
        "--target-bytes",
        type=int,
        default=DEFAULT_TARGET_BYTES,
        help="Target size in bytes for converted files",
    )
    parser.add_argument(
        "--max-duration-seconds",
        type=float,
        default=DEFAULT_MAX_SEGMENT_DURATION_SECONDS,
        help="Maximum duration in seconds per output segment",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("under_2gb"),
        help="Directory to save converted files",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("under_2gb/conversion_report.json"),
        help="Path to save the JSON conversion report",
    )
    parser.add_argument(
        "--ffmpeg", default="ffmpeg", help="Path to the ffmpeg executable"
    )
    parser.add_argument(
        "--ffprobe", default="ffprobe", help="Path to the ffprobe executable"
    )
    parser.add_argument("--brctl", default="brctl", help="Path to the brctl executable")
    parser.add_argument(
        "--download-icloud",
        action="store_true",
        help="Run 'brctl download' before reading each file",
    )
    parser.set_defaults(include_under_limit=True)
    parser.add_argument(
        "--include-under-limit",
        dest="include_under_limit",
        action="store_true",
        help="Also convert supported media files at or below the size limit. This is the default.",
    )
    parser.add_argument(
        "--over-limit-only",
        dest="include_under_limit",
        action="store_false",
        help="Only convert files above --size-limit-bytes.",
    )
    parser.add_argument(
        "--exclude-dir-prefix",
        action="append",
        default=[],
        help="Skip directories whose name starts with this prefix. Can be repeated.",
    )
    parser.add_argument(
        "--flac-all",
        action="store_true",
        help="Prefer FLAC for every audio-only source, including lossy input",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=False,
        help=(
            "Apply EBU R128 loudness normalization (loudnorm=I=-16:TP=-1.5:LRA=11) "
            "to generated audio. Off by default; output is byte-identical when omitted."
        ),
    )
    parser.add_argument(
        "--format",
        choices=OUTPUT_FORMATS,
        default="auto",
        help=(
            "Output audio format. 'auto' (default) keeps FLAC-for-lossless / "
            "Opus behaviour; 'flac'/'opus' force that codec; 'aac' (.m4a) and "
            "'mp3' produce broadly-compatible lossy output fitted to the target."
        ),
    )
    parser.add_argument(
        "--workers", type=int, default=0, help="Parallel ffmpeg jobs. 0 = auto"
    )
    parser.add_argument(
        "--ffmpeg-threads",
        type=int,
        default=0,
        help="Pass -threads to ffmpeg. 0 = ffmpeg auto",
    )
    parser.add_argument(
        "--silence-noise",
        default=DEFAULT_SILENCE_NOISE,
        help="ffmpeg silencedetect noise threshold",
    )
    parser.add_argument(
        "--silence-min-duration-seconds",
        type=float,
        default=DEFAULT_SILENCE_MIN_DURATION_SECONDS,
        help="Minimum silence duration used as a preferred split boundary",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually convert files. Omit for a dry run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting generated output paths",
    )
    parser.add_argument(
        "--set-title",
        default=None,
        help="Set the 'title' metadata tag on every generated output",
    )
    parser.add_argument(
        "--set-artist",
        default=None,
        help="Set the 'artist' metadata tag on every generated output",
    )
    parser.add_argument(
        "--set-album",
        default=None,
        help="Set the 'album' metadata tag on every generated output",
    )
    parser.add_argument(
        "--set-comment",
        default=None,
        help="Set the 'comment' metadata tag on every generated output",
    )
    parser.add_argument(
        "--preset",
        choices=presets.preset_names(),
        default=None,
        help=(
            "Apply bundled defaults for a scenario. Explicit flags always win "
            "over the preset."
        ),
    )
    parser.add_argument(
        "--transcribe",
        action="store_true",
        help=(
            "After each successful conversion, write a text + JSON transcript "
            "sidecar next to the generated audio (requires the optional "
            "'faster-whisper' dependency; skipped with a notice if unavailable)."
        ),
    )
    parser.add_argument(
        "--transcribe-model",
        default="base",
        help="faster-whisper model name to use when --transcribe is set (default: base)",
    )
    parser.add_argument(
        "--allow-video",
        action="store_true",
        help=(
            "Extract and shrink the audio track from video containers "
            "(e.g. .mp4/.mov/.mkv Zoom/Teams/lecture recordings). By default "
            "video files are rejected. Video files with no audio stream are "
            "always rejected."
        ),
    )
    return parser


def _explicitly_set_dests(
    parser: argparse.ArgumentParser,
    argv: list[str] | None,
    dests: Iterable[str],
) -> set[str]:
    """Return the subset of dests the user explicitly set on the command line.

    Uses the documented argparse behavior that pre-set namespace attributes
    are kept unless an option is actually provided: each dest is seeded with
    a unique sentinel, argv is re-parsed onto that namespace, and any dest
    that no longer holds the sentinel was explicitly given by the user.
    """

    sentinel = object()
    dests = tuple(dests)
    namespace = argparse.Namespace(**{dest: sentinel for dest in dests})
    parser.parse_args(argv, namespace=namespace)
    return {dest for dest in dests if getattr(namespace, dest) is not sentinel}


def _apply_config_file(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    argv: list[str] | None,
) -> None:
    """Overlay `.codec-carver.json` values onto args for unset CLI options.

    Explicit CLI flags always win over config-file values; with no config
    file present, args is returned untouched. Config problems (unknown keys,
    type mismatches, malformed JSON) abort with a normal argparse error
    message instead of a traceback.
    """

    try:
        config = config_file.load_config(Path(args.root))
    except config_file.ConfigFileError as exc:
        parser.error(str(exc))
    if not config:
        return
    explicit = _explicitly_set_dests(parser, argv, config)
    for dest, value in config.items():
        if dest not in explicit:
            setattr(args, dest, value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments, resolving presets then config-file defaults.

    Precedence is command line, then `.codec-carver.json`, then `--preset`
    defaults, then built-in defaults. Config problems abort with a normal
    argparse error message instead of a traceback. The returned namespace also
    carries a ``tags`` dict collecting the metadata values provided by
    ``--set-title``/``--set-artist``/``--set-album``/``--set-comment``.
    """

    parser = _build_parser()
    normalized = _normalize_argv(argv)

    # Capture the tool's built-in defaults, then seed the namespace with a
    # sentinel for every preset-tunable option so argparse leaves it untouched
    # unless the user passes the flag. That lets apply_preset tell "unset" apart
    # from "set to the default value" and give explicit flags precedence.
    real_defaults = {
        dest: parser.get_default(dest) for dest in presets.PRESET_TUNABLE_DESTS
    }
    namespace = argparse.Namespace()
    for dest in presets.PRESET_TUNABLE_DESTS:
        setattr(namespace, dest, presets._UNSET)
    args = parser.parse_args(normalized, namespace)
    args = presets.apply_preset(args, real_defaults)
    _apply_config_file(parser, args, normalized)
    provided_tags = {
        "title": args.set_title,
        "artist": args.set_artist,
        "album": args.set_album,
        "comment": args.set_comment,
    }
    args.tags = {key: value for key, value in provided_tags.items() if value is not None}
    return args


def _build_transcription_hook(
    args: argparse.Namespace,
) -> Callable[[ConversionResult], None] | None:
    """Return a per-output transcription hook when --transcribe is requested.

    Transcription is optional: the module is imported lazily and any failure is
    reported without aborting the conversion batch.
    """
    if not getattr(args, "transcribe", False):
        return None

    import transcribe as _transcribe

    def hook(result: ConversionResult) -> None:
        """Transcribe one generated output; never raise into the conversion."""
        try:
            txt_path, _ = _transcribe.transcribe_output(
                result.output_path, model=args.transcribe_model
            )
            print(f"TRANSCRIBED\t{txt_path}", flush=True)
        except _transcribe.TranscriptionUnavailableError as exc:
            print(f"TRANSCRIBE_SKIP\t{exc}", flush=True)
        except Exception as exc:  # noqa: BLE001 - a bad transcript must not fail conversion.
            print(f"TRANSCRIBE_FAIL\t{result.output_path}\t{exc}", flush=True)

    return hook


def _format_progress(completed: int, total: int, eta_seconds: float) -> str:
    """Format a tab-separated batch progress line with an mm:ss ETA.

    ``completed`` and ``total`` are the number of finished and queued
    candidates respectively (``total`` must be positive). ``eta_seconds`` is
    the estimated remaining wall time; it is clamped at zero and truncated to
    whole seconds before formatting, and minutes may exceed 59 for long
    batches.
    """
    percent = completed * 100 // total
    eta = max(0, int(eta_seconds))
    return f"PROGRESS\t{completed}/{total}\t{percent}%\tETA {eta // 60:02d}:{eta % 60:02d}"


def _execute_conversions(
    candidates: list[tuple[Path, int]],
    args: argparse.Namespace,
    root: Path,
    output_dir: Path,
) -> list[ConversionResult]:
    """Execute conversions in parallel, printing per-file results and progress.

    After each candidate finishes, a ``PROGRESS`` line reports the completed
    count, percentage, and an ETA extrapolated from the average per-file wall
    time so far (measured with a monotonic clock).
    """
    results: list[ConversionResult] = []
    workers = choose_worker_count(args.workers)
    ffmpeg_threads = args.ffmpeg_threads if args.ffmpeg_threads >= 0 else None
    post_process = _build_transcription_hook(args)

    resolved_candidates = frozenset(Path(item[0]).resolve() for item in candidates)
    protected_sources = [c[0] for c in candidates]

    def process_candidate(candidate_tuple: tuple[Path, int]) -> list[ConversionResult]:
        """Convert one queued candidate and return a failure result instead of aborting the batch."""
        candidate, size = candidate_tuple
        try:
            return convert_file(
                candidate,
                root=root,
                output_dir=output_dir,
                target_bytes=args.target_bytes,
                ffmpeg_path=args.ffmpeg,
                ffprobe_path=args.ffprobe,
                download_icloud=args.download_icloud,
                brctl_path=args.brctl,
                prefer_flac=args.flac_all,
                output_format=args.format,
                ffmpeg_threads=ffmpeg_threads,
                overwrite=args.overwrite,
                max_segment_duration_seconds=args.max_duration_seconds,
                silence_noise=args.silence_noise,
                silence_min_duration_seconds=args.silence_min_duration_seconds,
                protected_sources=protected_sources,
                resolved_protected_sources=resolved_candidates,
                original_size=size,
                tags=args.tags or None,
                normalize=args.normalize,
                post_process=post_process,
                allow_video=args.allow_video,
            )
        except Exception as exc:  # noqa: BLE001 - batch processing records per-file failures.
            return [
                ConversionResult(
                    source_path=candidate,
                    output_path=None,
                    status="failed",
                    original_size_bytes=safe_source_size(candidate),
                    message=str(exc),
                )
            ]

    print(f"WORKERS\t{workers}\tFFMPEG_THREADS\t{ffmpeg_threads}")
    total = len(candidates)
    completed = 0
    start_time = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_candidate = {
            executor.submit(process_candidate, candidate): candidate
            for candidate in candidates
        }
        for future in as_completed(future_to_candidate):
            candidate_results = future.result()
            results.extend(candidate_results)
            for result in candidate_results:
                print(_format_result(root, result), flush=True)
            completed += 1
            elapsed = time.monotonic() - start_time
            eta_seconds = (elapsed / completed) * (total - completed)
            print(_format_progress(completed, total, eta_seconds), flush=True)

    return results


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    args = parse_args(argv)
    root = args.root.resolve()
    output_dir = (
        args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    )
    report_path = args.report if args.report.is_absolute() else root / args.report

    candidates = find_candidates(
        root,
        size_limit_bytes=args.size_limit_bytes,
        include_under_limit=args.include_under_limit,
        exclude_paths=[output_dir],
        exclude_dir_prefixes=args.exclude_dir_prefix,
    )
    if not args.execute:
        for candidate, size in candidates:
            print(f"DRY-RUN\t{size}\t{candidate.relative_to(root)}")
        print(f"TOTAL_SELECTED={len(candidates)}")
        return 0

    results = _execute_conversions(candidates, args, root, output_dir)

    # Fast path: Pre-compute the root string prefix to avoid slow Path.relative_to() instantiation in the sort loop
    root_prefix = root.as_posix()
    if not root_prefix.endswith("/"):
        root_prefix += "/"

    results.sort(
        key=lambda result: (
            result.source_path.as_posix().removeprefix(root_prefix).casefold()
        )
    )
    write_report(results, report_path)
    converted = [result for result in results if result.status == "converted"]
    skipped = [result for result in results if result.status == "skipped_existing"]
    failed = [
        result
        for result in results
        if result.status not in {"converted", "skipped_existing"}
    ]
    print(f"REPORT\t{report_path}")
    print(
        f"SUMMARY\tconverted={len(converted)}\tskipped_existing={len(skipped)}"
        f"\tfailed_or_too_large={len(failed)}"
    )
    return 1 if failed else 0


def _is_lossless_probe(probe: MediaProbe) -> bool:
    """Return True if the probed codec is considered lossless."""
    return (probe.audio_codec or "").lower() in LOSSLESS_AUDIO_CODECS


def _probe_output_duration(
    output_path: Path, *, ffprobe_path: str, output_size: int | None = None
) -> float:
    """Return the duration of a generated output file."""
    return probe_media(
        output_path, ffprobe_path=ffprobe_path, source_size=output_size
    ).duration_seconds


def _duration_matches_expected(
    actual_seconds: float,
    expected_seconds: float,
    *,
    tolerance_seconds: float = DURATION_TOLERANCE_SECONDS,
) -> bool:
    """Return True if actual duration is close enough to expected duration."""
    return abs(actual_seconds - expected_seconds) <= tolerance_seconds


def _discard_invalid_generated_output(
    source: Path,
    output_path: Path,
    common_fields: dict[str, Any],
    *,
    status: str,
    message: str,
    protected_sources: frozenset[Path] = frozenset(),
) -> ConversionResult:
    """Delete an invalid output and return a failure ConversionResult."""
    _remove_generated_output(source, output_path, protected_sources=protected_sources)
    invalid_fields = dict(common_fields)
    invalid_fields["output_path"] = None
    return ConversionResult(**invalid_fields, status=status, message=message)


def _remove_generated_output(
    source: Path,
    output_path: Path,
    *,
    protected_sources: frozenset[Path] = frozenset(),
) -> None:
    """Safely remove a generated file without touching protected sources."""
    _ensure_not_source_path(source, output_path)
    _ensure_not_protected_source_path(protected_sources, output_path)
    output_path.unlink(missing_ok=True)


def _ensure_not_protected_source_path(
    protected_sources: frozenset[Path], output: Path
) -> None:
    """Raise MediaShrinkerError if output would overwrite a protected source."""
    resolved_output = output.resolve()
    if resolved_output in protected_sources:
        raise MediaShrinkerError(
            f"Refusing to use protected source path as generated output: {output}"
        )


def _choose_silence_split_point(
    segment_start: float,
    window_end: float,
    silence_intervals: Iterable[SilenceInterval],
) -> float | None:
    """Return the latest safe split point inside a silence interval."""
    latest_safe_end = window_end - HARD_SPLIT_EPSILON_SECONDS
    candidates = [
        min(interval.end_seconds, latest_safe_end)
        for interval in silence_intervals
        if interval.end_seconds > segment_start + HARD_SPLIT_EPSILON_SECONDS
        and interval.start_seconds < latest_safe_end
    ]
    candidates = [
        candidate for candidate in candidates if segment_start < candidate < window_end
    ]
    return max(candidates) if candidates else None


def _segment_source_path(source_path: Path, segment: MediaSegment | None) -> Path:
    """Return the output filename for a specific segment part."""
    if segment is None or segment.total_segments <= 1:
        return source_path
    return source_path.with_name(f"{source_path.name}.part{segment.index:04d}")


def _segment_input_args(segment: MediaSegment | None) -> list[str]:
    """Return ffmpeg -ss and -t arguments for a segment if applicable."""
    if segment is None or segment.total_segments <= 1:
        return []
    return [
        "-ss",
        _format_seconds(segment.start_seconds),
        "-t",
        _format_seconds(segment.duration_seconds),
    ]


def _format_seconds(value: float) -> str:
    """Format a second value for ffmpeg arguments with three decimals."""
    truncated = int(value * 1000) / 1000
    return f"{truncated:.3f}".rstrip("0").rstrip(".") or "0"


def _planned_output_path(source_path: Path, output_dir: Path, suffix: str) -> Path:
    """Return the canonical output path for a source and suffix."""
    relative_source = (
        Path(source_path.name) if source_path.is_absolute() else source_path
    )
    return output_dir / relative_source.with_name(f"{relative_source.name}{suffix}")


def _with_ffmpeg_threads(args: list[str], ffmpeg_threads: int | None) -> list[str]:
    """Insert ffmpeg -threads before the output path when requested."""

    if ffmpeg_threads is None:
        return args
    return [*args[:-1], "-threads", str(ffmpeg_threads), args[-1]]


def _with_loudnorm(args: list[str], normalize: bool) -> list[str]:
    """Insert the EBU R128 loudnorm audio filter before the output path when requested.

    When normalize is False the argument list is returned unchanged so default
    conversions remain byte-identical to prior behavior.
    """

    if not normalize:
        return args
    return [*args[:-1], "-af", LOUDNORM_FILTER, args[-1]]


def _parse_probe_payload(
    payload: dict[str, Any],
    source_path: Path,
    source_size: int | None = None,
) -> MediaProbe:
    """Parse raw ffprobe JSON payload into a MediaProbe object."""
    if not isinstance(payload, dict):
        raise MediaShrinkerError(f"{source_path} ffprobe payload was not an object")
    streams = payload.get("streams", [])
    if not isinstance(streams, list):
        raise MediaShrinkerError(f"{source_path} ffprobe payload has invalid streams")

    # Fast path: O(N) loop to find audio stream and check for video in one pass
    # Avoids multiple generator expressions and any/next calls for measurable CPU savings on large files
    audio_stream = None
    has_video = False
    for stream in streams:
        # ffprobe output is untrusted; a non-object stream entry cannot be the
        # audio stream, so skip it rather than let stream.get() raise AttributeError.
        if not isinstance(stream, dict):
            continue
        codec_type = stream.get("codec_type")
        if codec_type == "audio" and audio_stream is None:
            audio_stream = stream
        elif codec_type == "video":
            has_video = True

    if audio_stream is None:
        raise MediaShrinkerError(f"{source_path} has no audio stream")

    # codec_name is untrusted; MediaProbe.audio_codec is typed str | None and
    # _is_lossless_probe() calls .lower() on it, so a non-string value (e.g. a
    # JSON list) would raise AttributeError downstream. Reject it here instead.
    audio_codec = audio_stream.get("codec_name")
    if audio_codec is not None and not isinstance(audio_codec, str):
        raise MediaShrinkerError(
            f"{source_path} ffprobe audio stream has invalid codec_name"
        )

    format_section = payload.get("format", {})
    if not isinstance(format_section, dict):
        raise MediaShrinkerError(f"{source_path} ffprobe payload has invalid format")
    # Prefer the stream duration, but a stream-level "0"/"0.000000" (reported by
    # some containers) is unusable and must fall back to the format duration.
    duration = _first_float(audio_stream.get("duration"))
    if duration <= 0:
        duration = _first_float(format_section.get("duration"))
    if duration <= 0:
        raise MediaShrinkerError(f"{source_path} has no usable duration")

    parsed_size = _first_int(format_section.get("size"))
    if parsed_size is None:
        parsed_size = (
            source_size if source_size is not None else source_path.stat().st_size
        )

    audio_bit_rate = _first_int(
        audio_stream.get("bit_rate"), format_section.get("bit_rate")
    )
    return MediaProbe(
        duration_seconds=duration,
        size_bytes=parsed_size,
        audio_codec=audio_codec,
        audio_bit_rate=audio_bit_rate,
        has_video=has_video,
        format_name=str(format_section.get("format_name", "")),
    )


def _first_float(*values: Any) -> float:
    """Return the first non-null float from a list of values."""
    for value in values:
        if value is None or value == "N/A":
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(parsed):
            continue
        return parsed
    return 0.0


def _first_int(*values: Any) -> int | None:
    """Return the first non-null int from a list of values."""
    for value in values:
        if value is None or value == "N/A":
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(parsed):
            continue
        return int(parsed)
    return None


def _execute_plan(
    plan: ConversionPlan,
    source: Path,
    final_output: Path,
    *,
    ffmpeg_path: str,
    overwrite: bool,
    protected_sources: frozenset[Path] = frozenset(),
) -> Path:
    """Execute a conversion plan using ffmpeg."""
    _ensure_not_protected_source_path(protected_sources, final_output)
    _ensure_not_source_path(source, final_output)
    final_output.parent.mkdir(parents=True, exist_ok=True)

    fd, temp_output_str = tempfile.mkstemp(
        suffix=final_output.suffix,
        prefix=f".{final_output.stem}.",
        dir=final_output.parent,
    )
    os.close(fd)
    temp_output = Path(temp_output_str)

    _ensure_not_protected_source_path(protected_sources, temp_output)
    _ensure_not_source_path(source, temp_output)

    try:
        command = plan.command(
            ffmpeg_path=ffmpeg_path,
            input_path=source,
            output_path=temp_output,
            overwrite=True,
        )
        completed = _run_media_tool(
            command,
            tool="ffmpeg",
            timeout=3600,
            timeout_message=f"ffmpeg timed out for {source}",
        )

        if completed.returncode != 0:
            raise MediaShrinkerError(
                f"ffmpeg failed for {source}: {completed.stderr.strip()}"
            )

        if final_output.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {final_output}")

        temp_output.replace(final_output)
    finally:
        temp_output.unlink(missing_ok=True)

    return final_output


def _ensure_not_source_path(source: Path, output: Path) -> None:
    """Reject generated paths that would overwrite or delete the source file."""

    if source.resolve() == output.resolve():
        raise MediaShrinkerError(
            f"Refusing to use source path as generated output: {output}"
        )


def _resolve_collision(path: Path, *, overwrite: bool) -> Path:
    """Return path or a numbered variant if path already exists."""
    if overwrite or not path.exists():
        return path
    for index in range(1, 10_000):
        candidate = path.with_name(f"{path.stem}-{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Could not find free output path for {path}")


def _restore_timestamps(source_stat: os.stat_result, dest: Path) -> None:
    """Best-effort copy of nanosecond atime/mtime from source_stat onto dest.

    A read-only destination or a filesystem lacking timestamp support can make
    os.utime raise OSError; that is non-critical metadata, so the failure is
    swallowed to keep attribute preservation best-effort.
    """
    try:
        os.utime(dest, ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns))
    except OSError:
        pass


def _copy_extended_attributes(source: Path, dest: Path) -> None:
    """Copy extended attributes from source to dest if supported by OS."""
    if not all(hasattr(os, attr) for attr in ("listxattr", "getxattr", "setxattr")):
        return

    try:
        names = os.listxattr(source)  # type: ignore[attr-defined]
    except OSError:
        return

    for name in names:
        try:
            value = os.getxattr(source, name)  # type: ignore[attr-defined]
            os.setxattr(dest, name, value)  # type: ignore[attr-defined]
        except OSError:
            continue


def _copy_macos_creation_time(
    source_stat: os.stat_result, dest: Path, setfile_path: str
) -> None:
    """Copy macOS creation time using SetFile if available."""
    birthtime = getattr(source_stat, "st_birthtime", None)
    if birthtime is None:
        return
    creation_date = datetime.fromtimestamp(float(birthtime)).strftime(
        "%m/%d/%Y %H:%M:%S"
    )
    try:
        subprocess.run(
            [setfile_path, "-d", creation_date, f"{dest.resolve()}"],
            check=False,
            capture_output=True,
            text=True,
            shell=False,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass  # Metadata restoration failure is non-fatal


def _format_result(root: Path, result: ConversionResult) -> str:
    """Format a single conversion result for CLI output."""
    source = _display_path(root, result.source_path)
    output = (
        ""
        if result.output_path is None
        else str(_display_path(root, result.output_path))
    )
    return (
        f"{result.status.upper()}\t{result.strategy or ''}\t"
        f"{result.original_size_bytes}\t{result.output_size_bytes or ''}\t{source}\t{output}\t{result.message or ''}"
    )


def _display_path(root: Path, path: Path) -> Path:
    """Return path relative to root if possible, otherwise absolute."""
    return path.relative_to(root) if path.is_relative_to(root) else path


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
