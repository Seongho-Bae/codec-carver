"""FastAPI upload UI for shrinking one media file through Codec Carver."""

import json
import hmac
import logging
import os
import shutil
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Form, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from job_store import JobStore
import media_shrinker

app = FastAPI(title="Codec Carver SaaS")
MAX_UPLOAD_BYTES = 5 * 1024 * 1024 * 1024
MAX_REQUEST_BYTES = MAX_UPLOAD_BYTES + 10 * 1024 * 1024
MAX_BATCH_FILES = 20
# A shrink target larger than the biggest accepted upload is meaningless; cap it
# to keep numeric input bounded.
MAX_TARGET_BYTES = MAX_UPLOAD_BYTES
# This service only processes audio/video. Uploaded files are never executed or
# served as web content — they are handed to ffmpeg, which rejects non-media —
# but validating the declared content type rejects obviously-wrong uploads early.
_ALLOWED_CONTENT_PREFIXES = ("audio/", "video/")


def _validate_request(file: "UploadFile", target_bytes: int) -> str | None:
    """Return an error message for an invalid upload request, or None if valid."""
    if target_bytes <= 0:
        return "Invalid target_bytes value. Must be greater than 0."
    if target_bytes > MAX_TARGET_BYTES:
        return "Invalid target_bytes value. Exceeds the maximum allowed size."
    if not file.filename:
        return "No file uploaded or filename missing"
    content_type = getattr(file, "content_type", None)
    if content_type and not content_type.startswith(_ALLOWED_CONTENT_PREFIXES):
        return "Unsupported content type; upload an audio or video file."
    return None


class RequestTooLarge(Exception):
    """Raised when streamed request bytes exceed the accepted upload envelope."""

    pass


@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    """Reject declared or streamed request bodies above the service limit."""

    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared_size = int(content_length)
        except ValueError:
            return JSONResponse(status_code=400, content={"error": "Invalid Content-Length"})
        if declared_size < 0:
            return JSONResponse(status_code=400, content={"error": "Invalid Content-Length"})
        if declared_size > MAX_REQUEST_BYTES:
            return JSONResponse(status_code=413, content={"error": "Payload Too Large"})

    received = 0
    receive = request._receive

    async def limited_receive():
        """Count streamed request bytes before handing them to FastAPI."""

        nonlocal received
        message = await receive()
        if message.get("type") == "http.request":
            received += len(message.get("body", b""))
            if received > MAX_REQUEST_BYTES:
                raise RequestTooLarge
        return message

    request._receive = limited_receive
    try:
        return await call_next(request)
    except RequestTooLarge:
        return JSONResponse(status_code=413, content={"error": "Payload Too Large"})

def get_configured_api_keys():
    """Return the API keys configured via the CODEC_CARVER_API_KEYS env var.

    The variable holds a comma-separated list of keys. Whitespace around each
    key is stripped and empty entries are ignored. Keys are read from the
    environment at request time (not import time) so tests can patch the
    environment easily and key rotation needs no server restart. Returns an
    empty list when the variable is unset or contains no usable keys, which
    leaves the service open (today's default behaviour).
    """

    raw = os.environ.get("CODEC_CARVER_API_KEYS", "")
    return [key.strip() for key in raw.split(",") if key.strip()]


@app.middleware("http")
async def require_api_key(request: Request, call_next):
    """Enforce opt-in API-key authentication on all endpoints except GET /.

    When one or more keys are configured via CODEC_CARVER_API_KEYS, every
    request other than GET / (the upload UI page) must carry an X-API-Key
    header matching a configured key; comparison uses hmac.compare_digest to
    stay constant-time. Requests failing the check receive a 401 JSON error
    without echoing any key material. When no keys are configured, all
    requests pass through unchanged.
    """

    configured_keys = get_configured_api_keys()
    if configured_keys and not (request.method == "GET" and request.url.path == "/"):
        provided_key = request.headers.get("x-api-key", "")
        if not any(
            hmac.compare_digest(provided_key, key) for key in configured_keys
        ):
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid or missing API key"},
            )
    return await call_next(request)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Attach conservative browser security headers to every response."""

    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Content-Security-Policy"] = "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    if request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response
logger = logging.getLogger(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Codec Carver SaaS</title>
    <style>
        body { font-family: sans-serif; max-width: 600px; margin: 40px auto; padding: 20px; }
        .box { border: 1px solid #ccc; padding: 20px; border-radius: 8px; }
        button { padding: 10px 20px; background-color: #0056b3; color: white; border: none; border-radius: 4px; cursor: pointer; }
        button:hover:not(:disabled) { background-color: #004085; }
        button:disabled { background-color: #6c757d; cursor: not-allowed; }
        button:focus-visible, input:focus-visible { outline: 2px solid #004085; outline-offset: 2px; }
        .required-star { color: #dc3545; }
        .help-text { color: #6c757d; font-size: 0.85em; display: inline-block; margin-top: 4px; }
        .spinner { display: inline-block; width: 1em; height: 1em; vertical-align: -0.125em; border: 2px solid currentColor; border-right-color: transparent; border-radius: 50%; animation: spinner-border .75s linear infinite; margin-right: 8px; }
        @keyframes spinner-border { to { transform: rotate(360deg); } }
        .box { transition: background-color 0.2s, border-color 0.2s; }
        .box.dragover { background-color: #f8f9fa; border-color: #0056b3; border-style: dashed; }
        .preset-container { margin-top: 8px; display: flex; gap: 8px; flex-wrap: wrap; }
        .preset-btn { padding: 4px 8px; font-size: 0.85em; background-color: #e9ecef; color: #495057; border: 1px solid #ced4da; border-radius: 4px; cursor: pointer; }
        .preset-btn:hover { background-color: #dde2e6; color: #212529; }
        .preset-btn[aria-pressed="true"] { background-color: #0056b3; color: white; border-color: #004085; font-weight: bold; }
        input[aria-invalid="true"] { border-color: #dc3545; outline: 2px solid #dc3545; }
    </style>
</head>
<body>
    <div class="box" id="drop-zone">
        <h2>Shrink Media File</h2>
        <form action="/shrink" method="post" enctype="multipart/form-data" id="shrink-form">
            <p>
                <label for="file">Media File: <span class="required-star" aria-hidden="true">*</span></label><br>
                <input type="file" id="file" name="file" accept="audio/*,video/*" aria-describedby="file_help file_size_preview" required onchange="updateFileSizePreview(this)">
                <br><span id="file_help" class="help-text">Select an audio or video file to shrink, or drag and drop it here.</span>
                <br><span id="file_size_preview" class="help-text" aria-live="polite" style="font-weight: bold; color: #0f6674;"></span>
            </p>
            <p>
                <label for="target_bytes">Target Bytes: <span class="required-star" aria-hidden="true">*</span></label><br>
                <input type="number" id="target_bytes" name="target_bytes" value="2000000000" min="1" aria-describedby="target_bytes_help target_bytes_preview" required>
                <br><span id="target_bytes_help" class="help-text">Maximum allowed file size in bytes (e.g., 2000000000 for ~1.86 GiB)</span>
                <br><span id="target_bytes_preview" class="help-text" aria-live="polite" style="font-weight: bold; color: #1e7e34;">1.86 GiB</span>
                <div id="preset_buttons_container" class="preset-container" role="group" aria-label="Preset target sizes">
                    <button type="button" class="preset-btn" data-bytes="26214400" aria-pressed="false">25 MiB</button>
                    <button type="button" class="preset-btn" data-bytes="104857600" aria-pressed="false">100 MiB</button>
                    <button type="button" class="preset-btn" data-bytes="524288000" aria-pressed="false">500 MiB</button>
                    <button type="button" class="preset-btn" data-bytes="1073741824" aria-pressed="false">1 GiB</button>
                </div>
            </p>
            <button type="submit" id="submit-btn">Upload and Shrink</button>
        </form>
        <script>
            const MAX_UPLOAD_BYTES = 5 * 1024 * 1024 * 1024;
            function formatBinaryBytes(value) {
                const units = ['B', 'KiB', 'MiB', 'GiB'];
                let size = value;
                let unit = 0;
                while (size >= 1024 && unit < units.length - 1) {
                    size = size / 1024;
                    unit += 1;
                }
                return unit === 0 ? size + ' ' + units[unit] : size.toFixed(2) + ' ' + units[unit];
            }
            document.getElementById('preset_buttons_container').addEventListener('click', function(e) {
                if (e.target.classList.contains('preset-btn')) {
                    const input = document.getElementById('target_bytes');
                    input.value = e.target.dataset.bytes;
                    input.dispatchEvent(new Event('input', { bubbles: true }));
                }
            });

            document.getElementById('batch_preset_buttons_container').addEventListener('click', function(e) {
                if (e.target.classList.contains('preset-btn')) {
                    const input = document.getElementById('batch_target_bytes');
                    input.value = e.target.dataset.bytes;
                    input.dispatchEvent(new Event('input', { bubbles: true }));
                }
            });

            function updateFileSizePreview(input) {
                const file = input.files[0];
                const preview = document.getElementById('file_size_preview');
                input.setCustomValidity('');
                input.removeAttribute('aria-invalid');
                preview.style.color = '#0f6674';
                if (!file) {
                    preview.innerText = '';
                    return;
                }
                const text = formatBinaryBytes(file.size);
                if (file.size > MAX_UPLOAD_BYTES) {
                    const limitText = formatBinaryBytes(MAX_UPLOAD_BYTES);
                    input.setCustomValidity('File exceeds ' + limitText + ' limit.');
                    input.setAttribute('aria-invalid', 'true');
                    preview.innerText = 'Selected file size: ' + text + ' (exceeds ' + limitText + ' limit)';
                    preview.style.color = '#dc3545';
                    return;
                }
                preview.innerText = 'Selected file size: ' + text;
            }

            document.getElementById('target_bytes').addEventListener('input', function(e) {
                const val = parseInt(this.value, 10);
                const preview = document.getElementById('target_bytes_preview');
                this.setCustomValidity('');
                this.removeAttribute('aria-invalid');
                preview.style.color = '#1e7e34';

                const buttons = document.querySelectorAll('#preset_buttons_container .preset-btn');
                buttons.forEach(btn => {
                    const presetValue = Number.parseInt(btn.dataset.bytes, 10);
                    btn.setAttribute(
                        'aria-pressed',
                        !e.isTrusted && presetValue === val ? 'true' : 'false'
                    );
                });

                if (this.value === '') {
                    preview.innerText = '';
                    this.setCustomValidity('');
                    this.removeAttribute('aria-invalid');
                    return;
                }

                if (isNaN(val) || val <= 0) {
                    preview.innerText = 'Must be greater than 0.';
                    preview.style.color = '#dc3545';
                    this.setCustomValidity('Must be greater than 0.');
                    this.setAttribute('aria-invalid', 'true');
                } else {
                    preview.innerText = formatBinaryBytes(val);
                }

            });

            document.getElementById('batch_target_bytes').addEventListener('input', function(e) {
                const val = parseInt(this.value, 10);
                const preview = document.getElementById('batch_target_bytes_preview');
                this.setCustomValidity('');
                this.removeAttribute('aria-invalid');
                preview.style.color = '#1e7e34';

                const buttons = document.querySelectorAll('#batch_preset_buttons_container .preset-btn');
                buttons.forEach(btn => {
                    const presetValue = Number.parseInt(btn.dataset.bytes, 10);
                    btn.setAttribute(
                        'aria-pressed',
                        !e.isTrusted && presetValue === val ? 'true' : 'false'
                    );
                });

                if (this.value === '') {
                    preview.innerText = '';
                    this.setCustomValidity('');
                    this.removeAttribute('aria-invalid');
                    return;
                }

                if (isNaN(val) || val <= 0) {
                    preview.innerText = 'Must be greater than 0.';
                    preview.style.color = '#dc3545';
                    this.setCustomValidity('Must be greater than 0.');
                    this.setAttribute('aria-invalid', 'true');
                } else {
                    preview.innerText = formatBinaryBytes(val);
                }
            });

            document.getElementById('shrink-form').addEventListener('submit', function() {
                const btn = document.getElementById('submit-btn');
                setTimeout(() => {
                    btn.disabled = true;
                    btn.innerHTML = '<span class="spinner" aria-hidden="true"></span>Processing...';
                    btn.setAttribute('aria-busy', 'true');
                }, 10);
            });

            function updateBatchFilePreview(input) {
                const preview = document.getElementById('batch_files_preview');
                input.setCustomValidity('');
                input.removeAttribute('aria-invalid');
                preview.style.color = '#0f6674';

                const files = input.files;
                if (!files || files.length === 0) {
                    preview.innerText = '';
                    return;
                }

                let totalSize = 0;
                for (let i = 0; i < files.length; i++) {
                    totalSize += files[i].size;
                }

                if (files.length > 20) {
                    input.setCustomValidity('Maximum is 20 files per batch.');
                    input.setAttribute('aria-invalid', 'true');
                    preview.innerText = 'Selected ' + files.length + ' files (' + formatBinaryBytes(totalSize) + ', exceeds 20 files limit)';
                    preview.style.color = '#dc3545';
                    return;
                }

                if (totalSize > MAX_UPLOAD_BYTES) {
                    const limitText = formatBinaryBytes(MAX_UPLOAD_BYTES);
                    input.setCustomValidity('Total file size exceeds ' + limitText + ' limit.');
                    input.setAttribute('aria-invalid', 'true');
                    preview.innerText = 'Selected ' + files.length + ' file(s) (' + formatBinaryBytes(totalSize) + ', exceeds ' + limitText + ' limit)';
                    preview.style.color = '#dc3545';
                    return;
                }
                preview.innerText = 'Selected ' + files.length + ' file(s) (' + formatBinaryBytes(totalSize) + ')';
            }

            document.getElementById('shrink-batch-form').addEventListener('submit', function() {
                const btn = document.getElementById('batch-submit-btn');
                setTimeout(() => {
                    btn.disabled = true;
                    btn.innerHTML = '<span class="spinner" aria-hidden="true"></span>Processing...';
                    btn.setAttribute('aria-busy', 'true');
                }, 10);
            });

        const dropZone = document.getElementById('drop-zone');
        const batchDropZone = document.getElementById('batch-drop-zone');
        const fileInput = document.getElementById('file');
        const batchFileInput = document.getElementById('batch_files');

        [dropZone, batchDropZone].forEach(zone => {
            if (!zone) return;
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                zone.addEventListener(eventName, preventDefaults, false);
            });
            ['dragenter', 'dragover'].forEach(eventName => {
                zone.addEventListener(eventName, () => zone.classList.add('dragover'), false);
            });
            ['dragleave', 'drop'].forEach(eventName => {
                zone.addEventListener(eventName, () => zone.classList.remove('dragover'), false);
            });
        });
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            document.body.addEventListener(eventName, preventDefaults, false);
        });
        function preventDefaults (e) {
            e.preventDefault();
            e.stopPropagation();
        }
        dropZone.addEventListener('drop', (e) => {
            let dt = e.dataTransfer;
            let files = dt.files;
            if (files.length) {
                fileInput.files = files;
                updateFileSizePreview(fileInput);
            }
        }, false);
        if (batchDropZone) {
            batchDropZone.addEventListener('drop', (e) => {
                let dt = e.dataTransfer;
                let files = dt.files;
                if (files.length) {
                    batchFileInput.files = files;
                    updateBatchFilePreview(batchFileInput);
                }
            }, false);
        }
        </script>
    </div>
    <div class="box" id="batch-drop-zone" style="margin-top: 20px;">
        <h2>Shrink Multiple Files</h2>
        <form action="/shrink-batch" method="post" enctype="multipart/form-data" id="shrink-batch-form">
            <p>
                <label for="batch_files">Media Files (up to 20): <span class="required-star" aria-hidden="true">*</span></label><br>
                <input type="file" id="batch_files" name="files" accept="audio/*,video/*" multiple aria-describedby="batch_files_help batch_files_preview" required onchange="updateBatchFilePreview(this)">
                <br><span id="batch_files_help" class="help-text">Select several audio or video files, or drag and drop them here. You get back one zip with every output plus a results.json manifest.</span>
                <br><span id="batch_files_preview" class="help-text" aria-live="polite" style="font-weight: bold; color: #0f6674;"></span>
            </p>
            <p>
                <label for="batch_target_bytes">Target Bytes (per file): <span class="required-star" aria-hidden="true">*</span></label><br>
                <input type="number" id="batch_target_bytes" name="target_bytes" value="2000000000" min="1" aria-describedby="batch_target_bytes_help batch_target_bytes_preview" required>
                <br><span id="batch_target_bytes_help" class="help-text">Maximum allowed size in bytes for each output file</span>
                <br><span id="batch_target_bytes_preview" class="help-text" aria-live="polite" style="font-weight: bold; color: #1e7e34;">1.86 GiB</span>
                <div id="batch_preset_buttons_container" class="preset-container" role="group" aria-label="Preset target sizes for batch">
                    <button type="button" class="preset-btn" data-bytes="26214400" aria-pressed="false">25 MiB</button>
                    <button type="button" class="preset-btn" data-bytes="104857600" aria-pressed="false">100 MiB</button>
                    <button type="button" class="preset-btn" data-bytes="524288000" aria-pressed="false">500 MiB</button>
                    <button type="button" class="preset-btn" data-bytes="1073741824" aria-pressed="false">1 GiB</button>
                </div>
            </p>
            <button type="submit" id="batch-submit-btn">Upload and Shrink Batch</button>
        </form>
    </div>
</body>
</html>
"""

def cleanup_temp_dir(temp_dir_path: Path):
    """Clean up the temporary directory after the response is sent."""
    if temp_dir_path.exists():
        shutil.rmtree(temp_dir_path, ignore_errors=True)


def _zip_outputs(outputs: list[Path], dest_dir: Path, archive_name: str) -> Path:
    """Bundle multiple generated outputs into a single (uncompressed) zip archive."""
    archive_path = dest_dir / archive_name
    # ZIP_STORED: the audio is already compressed, so re-compressing wastes CPU.
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_STORED) as archive:
        for output in outputs:
            archive.write(output, arcname=output.name)
    return archive_path


def _existing_outputs(results) -> list[Path]:
    """Return generated output paths that still exist on disk."""

    if not results:
        return []
    return [
        result.output_path
        for result in results
        if result.output_path and result.output_path.exists()
    ]


def _download_path_for_outputs(
    outputs: list[Path], dest_dir: Path, archive_name: str
) -> Path:
    """Return the single download path for one output or a zip for many outputs."""

    if len(outputs) == 1:
        return outputs[0]
    return _zip_outputs(outputs, dest_dir, archive_name)


def _persist_upload(file: UploadFile) -> tuple[Path, Path, Path, Path]:
    """Save an uploaded file into a fresh temp workspace.

    Returns ``(temp_dir_path, input_dir, output_dir, source_path)``. Any
    filesystem or size-limit failure raises after cleaning up its own partial
    workspace, so callers can map it to an error response.
    """
    temp_dir_path: Path | None = None
    try:
        temp_dir_path = Path(tempfile.mkdtemp(prefix="codec_carver_"))
        input_dir = temp_dir_path / "input"
        output_dir = temp_dir_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        safe_filename = Path((file.filename or "").replace("\\", "/")).name
        if not safe_filename or safe_filename in (".", ".."):
            safe_filename = "upload.tmp"

        source_path = input_dir / safe_filename
        bytes_written = 0
        with open(source_path, "wb") as f:
            while chunk := file.file.read(1024 * 1024):  # 1 MB chunks
                bytes_written += len(chunk)
                if bytes_written > MAX_UPLOAD_BYTES:
                    raise ValueError("File exceeds maximum allowed upload size")
                f.write(chunk)
        return temp_dir_path, input_dir, output_dir, source_path
    except Exception:
        if temp_dir_path is not None:
            cleanup_temp_dir(temp_dir_path)
        raise


@app.get("/", response_class=HTMLResponse)
async def get_ui():
    """Return the single-page upload form."""

    return HTML_TEMPLATE


@app.post("/shrink")
def shrink_media(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_bytes: int = Form(2_000_000_000)
):
    """Persist an uploaded media file, shrink it, and return the generated file.

    Security model for the uploaded bytes (self-contained; no trust in
    downstream internals): the upload is (1) validated (audio/video content
    type + bounded target size) by ``_validate_request``, (2) written under a
    private per-request temp directory with a sanitized filename (never a
    web-served or executable location), and (3) passed to ``media_shrinker``
    only as a **file-path argument** to ``ffmpeg``/``ffprobe`` invoked via
    ``subprocess.run`` with an explicit argument list and ``shell=False`` — the
    bytes are never executed, ``eval``/``exec``'d, or interpolated into a shell.
    The generated output is returned as an ``application/octet-stream`` download,
    or as ``application/zip`` when conversion produces multiple segments. The
    uploaded file itself is never served back. The temp workspace is removed
    after the response.
    """

    error = _validate_request(file, target_bytes)
    if error is not None:
        return {"error": error}

    try:
        temp_dir_path, input_dir, output_dir, source_path = _persist_upload(file)
    except Exception:
        logger.exception("Failed to prepare uploaded media")
        return {"error": "Upload processing failed"}

    # Process the file using media_shrinker
    try:
        results = media_shrinker.convert_file(
            source=source_path,
            root=input_dir,
            output_dir=output_dir,
            target_bytes=target_bytes,
        )

        # Collect every generated output. Long recordings are split into several
        # segments; returning only the first would silently drop the rest.
        outputs = _existing_outputs(results)
        background_tasks.add_task(cleanup_temp_dir, temp_dir_path)

        if not outputs:
            logger.error("Processing produced no output: %r", results)
            return {"error": "Processing failed or no output generated"}

        output_path = _download_path_for_outputs(
            outputs, temp_dir_path, source_path.stem + "_shrunk.zip"
        )
        media_type = (
            "application/zip"
            if output_path.suffix == ".zip"
            else "application/octet-stream"
        )
        return FileResponse(
            path=output_path,
            filename=output_path.name,
            media_type=media_type,
        )

    except Exception:
        cleanup_temp_dir(temp_dir_path)
        logger.exception("Media processing failed")
        return {"error": "Upload processing failed"}


@app.post("/shrink-batch")
def shrink_media_batch(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(default=[]),
    target_bytes: int = Form(2_000_000_000),
):
    """Shrink several uploaded media files and return one zip archive."""
    if target_bytes <= 0:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid target_bytes value. Must be greater than 0."},
        )
    if target_bytes > MAX_TARGET_BYTES:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid target_bytes value. Exceeds the maximum allowed size."},
        )
    if not files:
        return JSONResponse(status_code=400, content={"error": "No files uploaded"})
    if len(files) > MAX_BATCH_FILES:
        return JSONResponse(
            status_code=400,
            content={"error": f"Too many files. Maximum is {MAX_BATCH_FILES} files per batch."},
        )

    try:
        temp_dir_path = Path(tempfile.mkdtemp(prefix="codec_carver_batch_"))
    except Exception:
        logger.exception("Failed to create batch upload workspace")
        return JSONResponse(status_code=500, content={"error": "Upload processing failed"})

    workspace_root = temp_dir_path.resolve()
    manifest = []
    zip_path = temp_dir_path / "codec_carver_batch.zip"
    try:
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as archive:
            for index, upload in enumerate(files):
                safe_filename = Path((upload.filename or "").replace("\\", "/")).name
                if not safe_filename or safe_filename in (".", ".."):
                    safe_filename = "upload.tmp"
                entry = {
                    "index": index,
                    "filename": safe_filename,
                    "status": "error",
                    "output_name": None,
                    "output_bytes": None,
                    "error": None,
                }
                manifest.append(entry)

                error = _validate_request(upload, target_bytes)
                if error is not None:
                    entry["error"] = error
                    continue

                input_dir = temp_dir_path / f"input_{index}"
                output_dir = temp_dir_path / f"output_{index}"
                try:
                    input_dir.mkdir()
                    output_dir.mkdir()
                    source_path = input_dir / safe_filename
                    bytes_written = 0
                    with open(source_path, "wb") as f:
                        while chunk := upload.file.read(1024 * 1024):
                            bytes_written += len(chunk)
                            if bytes_written > MAX_UPLOAD_BYTES:
                                raise ValueError("File exceeds maximum allowed upload size")
                            f.write(chunk)
                except Exception:
                    logger.exception("Failed to prepare batch upload #%d", index)
                    entry["error"] = "Upload processing failed"
                    continue

                try:
                    results = media_shrinker.convert_file(
                        source=source_path,
                        root=input_dir,
                        output_dir=output_dir,
                        target_bytes=target_bytes,
                    )
                except Exception:
                    logger.exception("Batch media processing failed for upload #%d", index)
                    entry["error"] = "Upload processing failed"
                    continue

                outputs = _existing_outputs(results)
                if not outputs:
                    logger.error("Batch processing produced no output for upload #%d: %r", index, results)
                    entry["error"] = "Processing failed or no output generated"
                    continue

                for output_index, output_path in enumerate(outputs, start=1):
                    output_path = output_path.resolve()
                    if not (output_path.is_file() and output_path.is_relative_to(workspace_root)):
                        logger.error("Batch output for upload #%d is missing or outside the workspace", index)
                        entry["error"] = "Processing failed or no output generated"
                        break
                    suffix = "" if len(outputs) == 1 else f".part{output_index:04d}"
                    arcname = f"{index + 1:02d}_{output_path.stem}{suffix}{output_path.suffix}"
                    archive.write(output_path, arcname=arcname)
                    entry["status"] = "ok"
                    entry["output_name"] = arcname
                    entry["output_bytes"] = (entry["output_bytes"] or 0) + output_path.stat().st_size

            archive.writestr(
                "results.json",
                json.dumps({"target_bytes": target_bytes, "results": manifest}, indent=2),
            )
    except Exception:
        cleanup_temp_dir(temp_dir_path)
        logger.exception("Failed to build batch archive")
        return JSONResponse(status_code=500, content={"error": "Upload processing failed"})

    background_tasks.add_task(cleanup_temp_dir, temp_dir_path)
    return FileResponse(
        path=zip_path,
        filename="codec_carver_batch.zip",
        media_type="application/zip",
    )

# --- Async job model --------------------------------------------------------
# The synchronous /shrink endpoint blocks for the whole conversion, which is
# impractical for long recordings. These endpoints let a client submit a job,
# poll its status, and download the result when ready (Upload -> Processing ->
# Result). SQLite keeps status durable across restarts and visible across
# worker/web processes.


def _default_job_store_path() -> Path:
    """Return the configured SQLite path for async job state."""

    configured = os.environ.get("CODEC_CARVER_JOB_DB")
    if configured:
        return Path(configured)
    return Path(tempfile.gettempdir()) / "codec_carver_jobs.sqlite3"


JOB_STORE = JobStore(str(_default_job_store_path()))


def _now() -> datetime:
    """Return an aware UTC timestamp for job-store writes."""

    return datetime.now(timezone.utc)


def _get_job_store() -> JobStore:
    """Return the active job store; tests replace ``JOB_STORE`` directly."""

    return JOB_STORE


def _run_job(
    job_id: str,
    source_path: Path,
    input_dir: Path,
    output_dir: Path,
    target_bytes: int,
    temp_dir_path: Path,
) -> None:
    """Background worker: shrink one uploaded file and record the outcome."""
    store = _get_job_store()
    try:
        store.set_status(job_id, "processing", now=_now())
    except KeyError:
        logger.error("Job %s disappeared before processing", job_id)
        cleanup_temp_dir(temp_dir_path)
        return

    try:
        results = media_shrinker.convert_file(
            source=source_path,
            root=input_dir,
            output_dir=output_dir,
            target_bytes=target_bytes,
        )
    except Exception:
        logger.exception("Job processing failed")
        try:
            store.set_status(job_id, "failed", now=_now(), error="Processing failed")
        except KeyError:
            logger.error("Job %s disappeared while recording failure", job_id)
        cleanup_temp_dir(temp_dir_path)
        return

    outputs = _existing_outputs(results)
    if outputs:
        output_path = _download_path_for_outputs(
            outputs, temp_dir_path, source_path.stem + "_shrunk.zip"
        )
        try:
            store.set_status(
                job_id,
                "done",
                now=_now(),
                output_path=str(output_path),
                output_name=output_path.name,
            )
        except KeyError:
            logger.error("Job %s disappeared while recording result", job_id)
            cleanup_temp_dir(temp_dir_path)
    else:
        logger.error("Job produced no output: %r", results)
        try:
            store.set_status(
                job_id,
                "failed",
                now=_now(),
                error="Processing failed or no output generated",
            )
        except KeyError:
            logger.error("Job %s disappeared while recording empty output", job_id)
        cleanup_temp_dir(temp_dir_path)


@app.post("/jobs")
def submit_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_bytes: int = Form(2_000_000_000),
):
    """Enqueue a shrink job and return its id for asynchronous status polling."""
    error = _validate_request(file, target_bytes)
    if error is not None:
        return JSONResponse(status_code=400, content={"error": error})

    try:
        temp_dir_path, input_dir, output_dir, source_path = _persist_upload(file)
    except Exception:
        logger.exception("Failed to prepare uploaded media")
        return JSONResponse(
            status_code=500, content={"error": "Upload processing failed"}
        )

    job_id = uuid.uuid4().hex
    try:
        _get_job_store().create(job_id, temp_dir=str(temp_dir_path), now=_now())
    except ValueError:
        cleanup_temp_dir(temp_dir_path)
        logger.exception("Failed to create async job record")
        return JSONResponse(
            status_code=500, content={"error": "Upload processing failed"}
        )

    background_tasks.add_task(
        _run_job, job_id, source_path, input_dir, output_dir, target_bytes, temp_dir_path
    )
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    """Return the current status of a previously submitted job."""
    job = _get_job_store().get(job_id)
    if job is None:
        return JSONResponse(status_code=404, content={"error": "Unknown job"})
    return {"job_id": job_id, "status": job["status"], "error": job.get("error")}


def _cleanup_job(job_id: str) -> None:
    """Forget a job and remove its temporary workspace."""
    store = _get_job_store()
    job = store.get(job_id)
    store.delete(job_id)
    if job is not None and job.get("temp_dir"):
        cleanup_temp_dir(Path(job["temp_dir"]))


@app.get("/jobs/{job_id}/result")
def job_result(job_id: str, background_tasks: BackgroundTasks):
    """Download a finished job's output, then clean up its workspace."""
    job = _get_job_store().get(job_id)
    if job is None:
        return JSONResponse(status_code=404, content={"error": "Unknown job"})
    if job["status"] != "done":
        return JSONResponse(
            status_code=409, content={"error": f"Job is {job['status']}"}
        )
    # Defense in depth: only ever serve a regular file that lives inside this
    # job's own temp workspace. `job_id` is an opaque store key and is never
    # used to build a path, but confining the served path makes traversal
    # impossible even if the store were ever populated from untrusted data.
    output_path_text = job.get("output_path")
    temp_dir_text = job.get("temp_dir")
    if not output_path_text or not temp_dir_text:
        return JSONResponse(
            status_code=410, content={"error": "Result no longer available"}
        )
    output_path = Path(output_path_text).resolve()
    workspace = Path(temp_dir_text).resolve()
    if not output_path.is_relative_to(workspace) or not output_path.is_file():
        return JSONResponse(
            status_code=410, content={"error": "Result no longer available"}
        )
    background_tasks.add_task(_cleanup_job, job_id)
    media_type = (
        "application/zip" if output_path.suffix == ".zip" else "application/octet-stream"
    )
    return FileResponse(
        path=output_path,
        filename=job["output_name"],
        media_type=media_type,
    )


if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
