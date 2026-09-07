"""Per-API-key usage metering and monthly quota enforcement.

This module is the missing piece between API-key authentication (*who* is
calling) and billing (*how much* they owe): it durably records per-key usage
counters (conversion count, input bytes, output bytes) in a SQLite database
and enforces monthly quotas against those counters.

Design notes:

* Storage is stdlib :mod:`sqlite3` in WAL mode so readers never block the
  writer and counters survive process restarts.
* Usage is bucketed by *period*, a ``"YYYY-MM"`` string derived from a
  caller-supplied :class:`datetime.datetime`. Callers pass ``now`` explicitly
  (the module never calls ``datetime.now()`` itself) so tests are fully
  deterministic.
* Every operation opens a short-lived connection, which combined with a
  process-level lock makes :class:`UsageStore` safe to share across threads.

Example::

    store = UsageStore("/var/lib/carver/usage.db")
    now = datetime.now(timezone.utc)
    store.check_quota(key, now, max_conversions=100)  # raises if over
    store.record(key, input_bytes=len(payload), output_bytes=len(result), now=now)
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import closing
from datetime import datetime
from pathlib import Path

__all__ = ["QuotaExceededError", "UsageStore"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage (
    api_key TEXT NOT NULL,
    period TEXT NOT NULL,
    conversions INTEGER NOT NULL DEFAULT 0,
    input_bytes INTEGER NOT NULL DEFAULT 0,
    output_bytes INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (api_key, period)
)
"""


class QuotaExceededError(Exception):
    """Raised when an API key has exhausted its quota for the current period.

    Attributes:
        api_key: The offending API key.
        limit_name: Which limit was exceeded (``"max_conversions"`` or
            ``"max_bytes"``).
        limit: The configured limit value.
        used: The usage value that met or exceeded the limit.
    """

    def __init__(self, api_key: str, limit_name: str, limit: int, used: int) -> None:
        """Store the quota-violation details and build a readable message.

        Args:
            api_key: The API key whose quota was exceeded.
            limit_name: Name of the exceeded limit.
            limit: Configured maximum for that limit.
            used: Current usage that hit the limit.
        """
        self.api_key = api_key
        self.limit_name = limit_name
        self.limit = limit
        self.used = used
        super().__init__(
            f"quota exceeded for API key {api_key!r}: "
            f"{limit_name}={limit} reached (used={used})"
        )


def _period(now: datetime) -> str:
    """Return the monthly billing period for ``now`` as a ``"YYYY-MM"`` string.

    Args:
        now: The instant to bucket. Naive or aware datetimes both work; the
            period is derived from the datetime's own year and month.

    Returns:
        The zero-padded period string, e.g. ``"2026-07"``.
    """
    return f"{now.year:04d}-{now.month:02d}"


class UsageStore:
    """Durable per-API-key usage counters with monthly quota checks.

    Each instance owns one SQLite database file. Connections are short-lived
    (opened per operation) and writes are serialized behind an internal lock,
    so a single instance may be shared freely across threads.
    """

    def __init__(self, db_path: str | Path) -> None:
        """Open (creating if necessary) the usage database at ``db_path``.

        Args:
            db_path: Filesystem path for the SQLite database. The parent
                directory must already exist.

        Raises:
            ValueError: If the parent directory of ``db_path`` does not exist
                or ``db_path`` points at a directory, so misconfiguration
                surfaces as a clear error instead of an opaque sqlite failure.
        """
        path = Path(db_path)
        if path.is_dir():
            raise ValueError(f"db_path points at a directory, not a file: {path}")
        if not path.parent.is_dir():
            raise ValueError(
                f"cannot create usage database: parent directory does not exist: "
                f"{path.parent}"
            )
        self._db_path = path
        self._lock = threading.Lock()
        with closing(self._connect()) as conn:
            # WAL mode is persistent per database file; set it once during initialization
            conn.executescript("PRAGMA journal_mode=WAL;\n" + _SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        """Open a new short-lived connection.

        Returns:
            A fresh :class:`sqlite3.Connection` to the store's database.
        """
        return sqlite3.connect(self._db_path, timeout=30.0)

    def record(
        self, api_key: str, *, input_bytes: int, output_bytes: int, now: datetime
    ) -> None:
        """Record one completed conversion for ``api_key`` in ``now``'s period.

        Increments the key's conversion count by one and adds the given byte
        counts to its running totals for the current monthly period. The write
        is durable once this method returns.

        Args:
            api_key: The API key that performed the conversion.
            input_bytes: Size of the input media in bytes (must be >= 0).
            output_bytes: Size of the converted output in bytes (must be >= 0).
            now: The current time, supplied by the caller for determinism.

        Raises:
            ValueError: If ``input_bytes`` or ``output_bytes`` is negative.
        """
        if input_bytes < 0 or output_bytes < 0:
            raise ValueError("input_bytes and output_bytes must be non-negative")
        period = _period(now)
        with self._lock, closing(self._connect()) as conn:
            with conn:
                conn.execute(
                    """
                    INSERT INTO usage (api_key, period, conversions, input_bytes, output_bytes)
                    VALUES (?, ?, 1, ?, ?)
                    ON CONFLICT (api_key, period) DO UPDATE SET
                        conversions = conversions + 1,
                        input_bytes = input_bytes + excluded.input_bytes,
                        output_bytes = output_bytes + excluded.output_bytes
                    """,
                    (api_key, period, input_bytes, output_bytes),
                )

    def usage(self, api_key: str, now: datetime) -> dict:
        """Return ``api_key``'s usage totals for the period containing ``now``.

        Args:
            api_key: The API key to look up.
            now: The current time; selects the monthly period to report.

        Returns:
            A dict with keys ``period``, ``conversions``, ``input_bytes``, and
            ``output_bytes``. Unknown keys (or keys with no usage this period)
            report zero for all counters.
        """
        period = _period(now)
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT conversions, input_bytes, output_bytes FROM usage "
                "WHERE api_key = ? AND period = ?",
                (api_key, period),
            ).fetchone()
        conversions, input_bytes, output_bytes = row if row else (0, 0, 0)
        return {
            "period": period,
            "conversions": conversions,
            "input_bytes": input_bytes,
            "output_bytes": output_bytes,
        }

    def check_quota(
        self,
        api_key: str,
        now: datetime,
        *,
        max_conversions: int | None = None,
        max_bytes: int | None = None,
    ) -> bool:
        """Check whether ``api_key`` may perform another conversion right now.

        Call this *before* doing work; it compares current-period usage
        against the given limits. A key exactly at a limit is denied (the
        limit is the total number allowed, so the next request would exceed
        it).

        Args:
            api_key: The API key to check.
            now: The current time; selects the monthly period to check.
            max_conversions: Maximum conversions allowed per period, or
                ``None`` for unlimited.
            max_bytes: Maximum combined input+output bytes allowed per period,
                or ``None`` for unlimited.

        Returns:
            ``True`` if the key is within all supplied limits.

        Raises:
            QuotaExceededError: If any supplied limit has been reached, with
                :attr:`~QuotaExceededError.limit_name` naming which one.
        """
        current = self.usage(api_key, now)
        if max_conversions is not None and current["conversions"] >= max_conversions:
            raise QuotaExceededError(
                api_key, "max_conversions", max_conversions, current["conversions"]
            )
        total_bytes = current["input_bytes"] + current["output_bytes"]
        if max_bytes is not None and total_bytes >= max_bytes:
            raise QuotaExceededError(api_key, "max_bytes", max_bytes, total_bytes)
        return True

    def reset(self, api_key: str) -> None:
        """Delete all recorded usage for ``api_key`` across every period.

        Admin operation, e.g. after a plan change or a billing dispute.

        Args:
            api_key: The API key whose usage rows should be removed.
        """
        with self._lock, closing(self._connect()) as conn:
            with conn:
                conn.execute("DELETE FROM usage WHERE api_key = ?", (api_key,))

    def all_usage(self, period: str) -> dict:
        """Return usage for every API key active in ``period``.

        Admin/billing-export operation.

        Args:
            period: The monthly period to report, as a ``"YYYY-MM"`` string.

        Returns:
            A dict mapping each API key with recorded usage in ``period`` to
            a dict of its ``conversions``, ``input_bytes``, and
            ``output_bytes``. Keys with no usage that period are absent.
        """
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT api_key, conversions, input_bytes, output_bytes "
                "FROM usage WHERE period = ? ORDER BY api_key",
                (period,),
            ).fetchall()
        return {
            api_key: {
                "conversions": conversions,
                "input_bytes": input_bytes,
                "output_bytes": output_bytes,
            }
            for api_key, conversions, input_bytes, output_bytes in rows
        }
