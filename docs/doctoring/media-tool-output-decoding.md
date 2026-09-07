# Media-tool output decoding boundary

## Decision

Codec Carver applies replacement error handling when Python converts stdout and stderr from the shared ffmpeg/ffprobe subprocess boundary into text. Media metadata is attacker-influenceable and can contain legacy encodings. A decoding exception must not escape before the JSON or silencedetect parser examines the valid ASCII structure.

The policy is deliberately narrower than accepting arbitrary malformed media. Command arguments, protocol allowlists, `shell=False`, timeouts, missing-executable handling, return codes, JSON shape validation, numeric validation, and executable/path trust boundaries remain unchanged. Undecodable metadata bytes become replacement text rather than raw bytes in user-facing errors.

## Verification and rollback

Real executable fixtures emit invalid bytes to stdout and stderr. Tests cover the shared runner, ffprobe JSON containing a legacy-encoded tag, silencedetect output with valid timestamps beside invalid metadata, and a nonzero typed error. The existing suite continues to cover timeout and missing-executable behavior.

Rollback may select another explicit bounded decoding policy, but must not restore strict decoding exceptions at this untrusted boundary.

## References

Python Software Foundation. (2026). *subprocess — Subprocess management* (Python 3.14 documentation). https://docs.python.org/3.14/library/subprocess.html

The Unicode Consortium. (2025). *The Unicode Standard, Version 17.0.0*. https://www.unicode.org/versions/Unicode17.0.0/

FFmpeg Project. (2026). *FFmpeg documentation*. https://ffmpeg.org/ffmpeg.html
