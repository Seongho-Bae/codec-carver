## 2026-08-15 - [Sentinel: Uncontrolled Resource Consumption in Job Cleanup]
**Vulnerability:** Resource Exhaustion (CWE-400 / CWE-770) via unretrieved job results.
**Learning:** When successful jobs only clean up their temporary directories upon result download, an attacker can intentionally create jobs and abandon them to exhaust disk space or inodes over time.
**Prevention:** Implement an automatic cleanup mechanism (like a background sweep or TTL) for jobs that complete but are never retrieved.
## 2026-07-25 - [Cross-platform upload basename normalization]
**Behavior:** Upload metadata now interprets both forward slashes and backslashes as path separators before extracting a basename.
**Learning:** On POSIX systems, `pathlib.Path(filename).name` retains backslashes because they are ordinary characters there. That caused inconsistent manifest and converter filenames for Windows-style client paths. The upload itself is still written inside a trusted temporary workspace, and batch archive entry names are generated outputs; this change does not establish a filesystem traversal or archive-entry escape.
**Prevention:** Normalize client path separators before extracting a basename, retain the existing empty/`.`/`..` fallback, and test the persisted source name and manifest metadata. Treat the normalization as cross-platform consistency and defense in depth, not as evidence of a demonstrated Zip Slip exploit.

## 2026-05-28 - [Sentinel Fixes: Temp Files & Injection]
**Vulnerability:** Predictable Temp Files (CWE-377) and Insecure Default Permissions (CWE-276), plus Command Injection via FFmpeg Filtergraph (CWE-20).
**Learning:** Python's `Path.with_name` plus a suffix string to make a temp file opens a race condition because it's predictable and the permissions default to system `umask` which might expose secret `0600` data. Additionally, interpolating variables directly into FFmpeg filtergraph strings allows arbitrary filter injection.
**Prevention:** Use `tempfile.mkstemp` which generates unguessable names and creates the file with secure `0600` permissions automatically. Use strict regex allow-lists for string parameters passed into complex shell-like arguments such as FFmpeg's `-af`.

## 2026-05-29 - [Sentinel: Unsafe Metadata Copying]
**Vulnerability:** Use of `shutil.copymode(source, dest)` preserves potentially dangerous permission bits (setuid, setgid, sticky).
**Learning:** Utilities that copy file metadata (like `shutil.copymode`) can inadvertently transfer elevated execution privileges from an untrusted source to a generated output. This can lead to privilege escalation if the destination file is later executed.
**Prevention:** Explicitly mask file permissions when restoring metadata. Use `os.chmod(dest, stat.S_IMODE(source_stat.st_mode) & 0o777)` to ensure only standard read/write/execute permissions are copied, dropping the setuid, setgid, and sticky bits.
## 2026-05-31 - [Sentinel: Unhandled FastAPI Upload Vulnerability Leading to Temporary Directory Leak]
**Vulnerability:** Path edge cases in uploaded filenames (`.`, `..`, or empty strings) triggering unhandled exceptions (`IsADirectoryError`) before reaching cleanup blocks, causing unbounded temporary directory accumulation on disk (CWE-400 / CWE-770 Resource Exhaustion / DoS).
**Learning:** In FastAPI/Starlette, `file.filename` can be unsafe or empty. Using `Path(file.filename).name` may resolve to `.` or `..`, leading to OS-level exceptions when attempting to write data. If resource allocation (like `tempfile.mkdtemp()`) occurs outside the scope of the `try...finally` (or `BackgroundTasks` cleanup) that handles these errors, an attacker can intentionally leak resources by sending manipulated paths.
**Prevention:** Always place resource allocation inside or immediately before the associated `try...finally` block. Sanitize and validate filenames retrieved from `UploadFile.filename` by ensuring they are non-empty and are not relative references (`.` or `..`), providing a safe default fallback.

## 2026-06-07 - FFmpeg SSRF/LFI Vulnerability Fix
**Vulnerability:** Local File Inclusion and Server-Side Request Forgery via unrestricted FFmpeg/FFprobe protocols.
**Learning:** The application executed FFmpeg and FFprobe on user-supplied media files without protocol restrictions. Malicious files (like HLS playlists) could leverage protocols like `http` to exfiltrate data or access internal services.
**Prevention:** Always enforce `"-protocol_whitelist", "file,crypto,data"` before the input flag when invoking FFmpeg/FFprobe to restrict processing to safe local protocols.

## 2026-06-09 - [Sentinel: FFmpeg Argument Injection Vulnerability Fix]
**Vulnerability:** Argument injection via maliciously crafted filenames.
**Learning:** Command-line utilities (like `ffprobe`) interpret arguments starting with a hyphen (e.g., `-version`, `-help`) as options. If user input (like a file path) is directly passed to the command list without an explicit input flag (like `-i`), a maliciously named file could inject arguments and alter the command execution flow, even with `shell=False`.
**Prevention:** When passing file paths to command-line tools like `ffmpeg` or `ffprobe` via `subprocess.run`, explicitly use the input flag (e.g., `-i`) immediately before the file path. This prevents argument injection vulnerabilities where a filename starting with a hyphen (e.g., `-version`) is misinterpreted as a command-line option.

## 2026-06-15 - [Sentinel: Uncontrolled Resource Consumption in Uploads]
**Vulnerability:** Uncontrolled Resource Consumption (CWE-400) / Missing input length limits via unbound file uploads.
**Learning:** Using `shutil.copyfileobj` blindly copies an uploaded stream directly to disk without size constraints. An attacker could upload an infinitely large file or a file large enough to exhaust server storage space, causing a Denial of Service.
**Prevention:** Do not use unbounded `shutil.copyfileobj` for web uploads. Implement chunked reads and track bytes written, raising an exception safely if a predefined strict maximum file size is exceeded.

## 2026-06-20 - [Sentinel: FastAPI request size limits]
**Vulnerability:** Uncontrolled Resource Consumption (CWE-400) via oversized HTTP request bodies.
**Learning:** A `Content-Length` check rejects known-oversized requests early, but requests without a usable length header still need byte counting while the ASGI body stream is consumed.
**Prevention:** Validate malformed or negative `Content-Length` values, reject declared oversized requests with `413`, and wrap the request receive function so chunked or lengthless uploads cannot exceed the same global limit.

## 2026-06-25 - [Sentinel: Unsafe Subprocess Paths leading to Argument Injection]
**Vulnerability:** Argument Injection via relative paths starting with a hyphen in command-line utilities.
**Learning:** Even when `ffmpeg` inputs are protected by `-i`, the output paths, as well as arguments to other utilities like `brctl` and `SetFile`, can be maliciously crafted to start with `-` and be interpreted as options if relative paths are used.
**Prevention:** Resolve file paths before passing them to `subprocess.run` when a tool does not support an explicit input flag or `--` delimiter. Absolute paths use a root, drive, or UNC prefix rather than a leading hyphen, so they cannot be parsed as command-line options.

## 2026-06-25 - [Sentinel: Strix CI Command Injection False Positives]
**Vulnerability:** CI security scanners (like Strix) falsely reporting command injection vulnerabilities when `shell=False` is omitted.
**Learning:** Some static analysis security tools flag `subprocess.run` calls as vulnerable to command injection if the `shell` argument is missing, even when the command is passed safely as a list of strings.
**Prevention:** Explicitly include `shell=False` in all `subprocess.run` calls, even when passing arguments as a list, to prevent false positive command injection alerts from CI security scanners like Strix.

## 2026-07-05 - [Sentinel: Fix Argument Injection Vulnerability]
**Vulnerability:** Argument Injection via relative paths starting with a hyphen in command-line utilities (CWE-88).
**Learning:** Even when `ffmpeg` inputs are protected by `-i`, command-line utilities (like `ffprobe` and `ffmpeg` filters) can interpret user input (like a file path) starting with a hyphen (e.g., `-version.wav`) as options if passed as a relative path.
**Prevention:** File paths must be converted to absolute paths using `.resolve()` before they are passed to `subprocess.run`. This prefixes the path with a root, drive, or UNC prefix rather than a leading hyphen, thereby averting the possibility of argument injection.
## 2026-07-06 - [Sentinel: Uncontrolled Resource Consumption (DoS) via Subprocess Timeouts]
**Vulnerability:** Uncontrolled Resource Consumption (CWE-400) via missing subprocess timeouts.
**Learning:** `subprocess.run` calls without explicit `timeout` arguments can cause the application to hang indefinitely if the spawned process (e.g., `ffmpeg`, `ffprobe`, `brctl`) deadlocks or takes an unreasonable amount of time due to maliciously crafted input files or underlying system issues.
**Prevention:** Always specify an explicit, appropriate `timeout` parameter for `subprocess.run` calls (e.g., 60s for probes/metadata, 3600s+ for intensive processing) and handle the resulting `subprocess.TimeoutExpired` exception to ensure the application fails securely and releases resources.

## 2026-07-09 - [Sentinel: FastAPI Missing Defense-in-Depth Headers]
**Vulnerability:** Missing defense-in-depth security headers like `Referrer-Policy` and `Permissions-Policy`.
**Learning:** To enhance security in FastAPI applications, missing HTTP response headers could leak referrers or give access to APIs (e.g. geolocation) without explicit intent.
**Prevention:** Implement an `@app.middleware('http')` function to globally inject defense-in-depth security headers such as `Content-Security-Policy`, `X-Frame-Options`, `Strict-Transport-Security`, `X-Content-Type-Options`, `X-XSS-Protection`, `Referrer-Policy` (e.g., `strict-origin-when-cross-origin`), and `Permissions-Policy` (e.g., `geolocation=(), microphone=(), camera=()`).

## 2026-07-10 - [Sentinel: Media Source Path Traversal]
**Vulnerability:** Path traversal in `media_shrinker.py` via unresolved `..` segments or symlink escapes before deriving conversion output paths.
**Learning:** `Path.relative_to()` is only a lexical containment check unless both the source and root have first been resolved into canonical absolute paths. Relative paths and symlinks can otherwise bypass root-boundary assumptions.
**Prevention:** Resolve both source and root once, reject sources outside the resolved root with a sanitized `MediaShrinkerError`, and derive `rel_source` from the resolved paths before planning outputs.
