"""Regression contracts for actionable CLI default-value help."""

import argparse

from audio_library import build_parser


def _subcommand_help(name: str) -> str:
    """Return rendered help for one real audio-library subcommand."""

    parser = build_parser()
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    return subparsers.choices[name].format_help()


def test_inventory_help_exposes_timeout_default() -> None:
    """A user should see the long inspection timeout before running inventory."""

    help_text = _subcommand_help("inventory")

    assert "--inspect-timeout-seconds" in help_text
    assert "(default: 14400)" in help_text


def test_transcribe_help_exposes_language_and_diarization_defaults() -> None:
    """Subcommand help should expose defaults even when options had no help prose."""

    help_text = _subcommand_help("transcribe")

    assert "--language" in help_text
    assert "(default: ko)" in help_text
    assert "--speaker-diarization" in help_text
    assert "(default: True)" in help_text


def test_top_level_help_does_not_advertise_missing_optional_values_as_defaults() -> None:
    """Unset optional values should not be described as a user-meaningful default."""

    help_text = build_parser().format_help()

    assert "--state-dir" in help_text
    assert "(default: None)" not in help_text


def test_top_level_help_does_not_advertise_suppressed_defaults() -> None:
    """Suppressed defaults must remain absent from rendered help output."""

    parser = build_parser()
    parser.add_argument(
        "--test-suppressed-default",
        default=argparse.SUPPRESS,
        help="test-only option",
    )

    help_text = parser.format_help()

    assert "--test-suppressed-default" in help_text
    assert "==SUPPRESS==" not in help_text
