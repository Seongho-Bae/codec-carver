"""SQLite-backed durable job store for async/worker job tracking.

The async web layer currently tracks jobs in an in-memory dict, which
loses all state on restart and cannot be shared across processes. This
module provides :class:`JobStore`, a small stdlib-only persistence layer
(``sqlite3`` with WAL journaling) that any async or worker process can
use so jobs survive restarts and are visible across processes.

Design notes:

- Every public method opens a short-lived connection guarded by a lock,
  so a single ``JobStore`` instance is safe to share across threads.
- WAL mode allows concurrent readers alongside a writer, which suits a
  web process polling job status while a worker updates it.
- Callers pass ``now`` (a :class:`datetime.datetime`) explicitly; the
  store never calls ``datetime.now()`` itself, keeping tests
  deterministic.

Example:
    >>> from datetime import datetime, timezone
    >>> store = JobStore("/tmp/jobs.db")  # doctest: +SKIP
    >>> store.create("job-1", temp_dir="/tmp/job-1",
    ...              now=datetime.now(timezone.utc))  # doctest: +SKIP
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime

#: Allowed job lifecycle states.
VALID_STATUSES = frozenset({"queued", "processing", "done", "failed"})

_SCHEMA = """
-- 데이터베이스 초기화 시 WAL 모드를 한 번만 설정하여 짧은 연결의 오버헤드를 줄입니다.
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    status      TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    output_path TEXT,
    output_name TEXT,
    error       TEXT,
    temp_dir    TEXT
)
"""

_COLUMNS = (
    "id",
    "status",
    "created_at",
    "updated_at",
    "output_path",
    "output_name",
    "error",
    "temp_dir",
)


class DuplicateJobError(ValueError):
    """Raised by :meth:`JobStore.create` when the job id already exists."""


class JobStore:
    """Durable, thread-safe job store backed by a SQLite database file.

    Multiple processes may open independent ``JobStore`` instances on the
    same ``db_path``; SQLite's file locking plus WAL mode keeps their
    reads and writes consistent. Within one process, a single instance
    may be shared freely across threads.

    Args:
        db_path: Filesystem path of the SQLite database. Created (along
            with the schema) if it does not exist. ``":memory:"`` is not
            supported because each operation opens a fresh connection,
            which would discard an in-memory database every time.
    """

    def __init__(self, db_path: str) -> None:
        """Initialize the store and create the schema if needed.

        Args:
            db_path: Path to the SQLite database file.

        Raises:
            ValueError: If ``db_path`` is ``":memory:"``.
        """
        if db_path == ":memory:":
            raise ValueError(
                "JobStore requires a file path; ':memory:' databases do "
                "not survive the short-lived connections this store uses"
            )
        self._db_path = str(db_path)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open a new WAL-mode connection to the underlying database.

        Yields:
            A short-lived ``sqlite3.Connection`` with WAL journaling and
            a row factory that yields ``sqlite3.Row`` objects.
        """
        conn = sqlite3.connect(self._db_path, timeout=30.0)
        try:
            conn.row_factory = sqlite3.Row
            yield conn
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _validate_status(status: str) -> None:
        """Reject statuses outside the allowed lifecycle set.

        Args:
            status: Candidate status string.

        Raises:
            ValueError: If ``status`` is not one of ``VALID_STATUSES``.
        """
        if status not in VALID_STATUSES:
            allowed = ", ".join(sorted(VALID_STATUSES))
            raise ValueError(
                f"invalid status {status!r}; must be one of: {allowed}"
            )

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        """Convert a database row into a plain job dict.

        Args:
            row: A ``sqlite3.Row`` from the ``jobs`` table.

        Returns:
            A dict with the keys ``id``, ``status``, ``created_at``,
            ``updated_at``, ``output_path``, ``output_name``, ``error``,
            and ``temp_dir``.
        """
        return {key: row[key] for key in _COLUMNS}

    def create(self, job_id: str, *, temp_dir: str, now: datetime) -> None:
        """Insert a new job in the ``queued`` state.

        Args:
            job_id: Unique identifier for the job.
            temp_dir: Working directory associated with the job (stored
                so a cleanup pass can remove it later).
            now: Timestamp recorded as both ``created_at`` and
                ``updated_at`` (ISO 8601 via ``datetime.isoformat()``).

        Raises:
            DuplicateJobError: If a job with ``job_id`` already exists.
                (Subclass of ``ValueError``, so callers may catch either.)
        """
        timestamp = now.isoformat()
        with self._lock, self._connect() as conn:
            try:
                conn.execute(
                    "INSERT INTO jobs (id, status, created_at, updated_at,"
                    " temp_dir) VALUES (?, 'queued', ?, ?, ?)",
                    (job_id, timestamp, timestamp, temp_dir),
                )
            except sqlite3.IntegrityError as exc:
                raise DuplicateJobError(
                    f"job {job_id!r} already exists"
                ) from exc

    def set_status(
        self,
        job_id: str,
        status: str,
        *,
        now: datetime,
        output_path: str | None = None,
        output_name: str | None = None,
        error: str | None = None,
    ) -> None:
        """Update a job's status, timestamp, and optional result fields.

        Only the fields passed as non-``None`` keyword arguments are
        overwritten; previously stored values for ``output_path``,
        ``output_name``, and ``error`` are preserved otherwise.

        Args:
            job_id: Identifier of the job to update.
            status: New status; one of ``queued``, ``processing``,
                ``done``, or ``failed``.
            now: Timestamp recorded as ``updated_at``.
            output_path: Path of the finished output file, if any.
            output_name: Client-facing download name, if any.
            error: Human-readable failure message, if any.

        Raises:
            ValueError: If ``status`` is not allowed.
            KeyError: If no job with ``job_id`` exists.
        """
        self._validate_status(status)
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                "UPDATE jobs SET status = ?, updated_at = ?,"
                " output_path = COALESCE(?, output_path),"
                " output_name = COALESCE(?, output_name),"
                " error = COALESCE(?, error)"
                " WHERE id = ?",
                (status, now.isoformat(), output_path, output_name,
                 error, job_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"job {job_id!r} does not exist")

    def get(self, job_id: str) -> dict | None:
        """Fetch a single job by id.

        Args:
            job_id: Identifier of the job to look up.

        Returns:
            The job as a dict (see :meth:`_row_to_dict` for keys), or
            ``None`` if no such job exists.
        """
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return self._row_to_dict(row) if row is not None else None

    def list_jobs(self, status: str | None = None) -> list[dict]:
        """List jobs, optionally filtered by status.

        Args:
            status: If given, only jobs in this state are returned; must
                be one of the allowed statuses.

        Returns:
            Jobs as dicts, ordered by ``created_at`` then id for a
            stable listing.

        Raises:
            ValueError: If ``status`` is given but not allowed.
        """
        with self._lock, self._connect() as conn:
            if status is None:
                rows = conn.execute(
                    "SELECT * FROM jobs ORDER BY created_at, id"
                ).fetchall()
            else:
                self._validate_status(status)
                rows = conn.execute(
                    "SELECT * FROM jobs WHERE status = ?"
                    " ORDER BY created_at, id",
                    (status,),
                ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def delete(self, job_id: str) -> None:
        """Remove a job record if it exists.

        Deleting an unknown id is a no-op, so cleanup passes can call
        this without checking existence first.

        Args:
            job_id: Identifier of the job to remove.
        """
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
