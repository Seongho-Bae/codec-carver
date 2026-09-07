import asyncio
import io
import json
import os
import tempfile
import unittest
import zipfile
from unittest.mock import patch, MagicMock
from pathlib import Path
from types import SimpleNamespace

try:
    from fastapi import BackgroundTasks
    from fastapi.testclient import TestClient
    from fastapi.responses import Response

    import saas_web
    from saas_web import app

    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False

from media_shrinker import ConversionResult
from job_store import JobStore

if _HAS_FASTAPI:
    client = TestClient(app)


@unittest.skipUnless(
    _HAS_FASTAPI, "fastapi not installed (optional integration dependency)"
)
class TestSaasWeb(unittest.TestCase):
    def test_get_ui(self):
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Codec Carver SaaS", response.content)

    def test_get_ui_includes_accessible_file_input_helpers(self):
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.text

        self.assertIn('accept="audio/*,video/*"', html)
        self.assertIn('aria-describedby="file_help file_size_preview"', html)
        self.assertIn('id="file_help"', html)
        self.assertIn('class="required-star" aria-hidden="true"', html)

    def test_get_ui_includes_binary_file_size_validation(self):
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.text

        self.assertIn("const MAX_UPLOAD_BYTES = 5 * 1024 * 1024 * 1024;", html)
        self.assertIn("['B', 'KiB', 'MiB', 'GiB']", html)
        self.assertIn("const limitText = formatBinaryBytes(MAX_UPLOAD_BYTES);", html)
        self.assertIn("File exceeds ' + limitText + ' limit.", html)
        self.assertIn("Total file size exceeds ' + limitText + ' limit.", html)
        self.assertIn("preview.style.color = '#0f6674';", html)
        self.assertIn('onchange="updateFileSizePreview(this)"', html)

    def test_security_headers_present_without_plain_http_hsts(self):
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(response.headers["X-Frame-Options"], "DENY")
        self.assertEqual(response.headers["X-XSS-Protection"], "1; mode=block")
        self.assertEqual(
            response.headers["Content-Security-Policy"],
            "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'",
        )
        self.assertEqual(
            response.headers["Referrer-Policy"],
            "strict-origin-when-cross-origin",
        )
        self.assertEqual(
            response.headers["Permissions-Policy"],
            "geolocation=(), microphone=(), camera=()",
        )
        self.assertNotIn("Strict-Transport-Security", response.headers)

    def test_hsts_header_present_for_forwarded_https(self):
        response = client.get("/", headers={"X-Forwarded-Proto": "https"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["Strict-Transport-Security"],
            "max-age=31536000; includeSubDomains",
        )

    def test_request_size_limit_rejects_oversized_declared_body(self):
        response = client.post(
            "/shrink",
            headers={"Content-Length": str(saas_web.MAX_REQUEST_BYTES + 1)},
        )
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json(), {"error": "Payload Too Large"})

    def test_request_size_limit_rejects_invalid_content_length(self):
        response = client.post(
            "/shrink",
            headers={"Content-Length": "not-a-number"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"error": "Invalid Content-Length"})

    def test_request_size_limit_rejects_negative_content_length(self):
        response = client.post(
            "/shrink",
            headers={"Content-Length": "-1"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"error": "Invalid Content-Length"})

    @patch("saas_web.media_shrinker.convert_file")
    def test_shrink_media_endpoint(self, mock_convert_file):
        # Create a dummy output file for the FileResponse
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_output = Path(temp_dir) / "output.flac"
            temp_output.write_bytes(b"dummy audio data")

            # Setup mock return value
            mock_result = MagicMock(spec=ConversionResult)
            mock_result.output_path = temp_output
            mock_convert_file.return_value = [mock_result]

            # Create a dummy upload file
            dummy_file_path = Path(temp_dir) / "input.wav"
            dummy_file_path.write_bytes(b"dummy wav data")

            with open(dummy_file_path, "rb") as f:
                response = client.post(
                    "/shrink",
                    files={"file": ("input.wav", f, "audio/wav")},
                    data={"target_bytes": 10000},
                )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content, b"dummy audio data")

            # Verify the mock was called
            mock_convert_file.assert_called_once()

    @patch("saas_web.media_shrinker.convert_file")
    def test_shrink_media_failure(self, mock_convert_file):
        # Setup mock to return empty or error
        mock_convert_file.return_value = []

        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            dummy_file_path = Path(temp_dir) / "input.wav"
            dummy_file_path.write_bytes(b"dummy wav data")

            with open(dummy_file_path, "rb") as f:
                response = client.post(
                    "/shrink",
                    files={"file": ("input.wav", f, "audio/wav")},
                    data={"target_bytes": 10000},
                )

            self.assertEqual(
                response.status_code, 200
            )  # Returns 200 with JSON error dict currently
            self.assertIn(b"error", response.content)
            self.assertNotIn("details", response.json())

    def test_shrink_media_rejects_nonpositive_target_bytes(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            dummy_file_path = Path(temp_dir) / "input.wav"
            dummy_file_path.write_bytes(b"dummy wav data")

            with open(dummy_file_path, "rb") as f:
                response = client.post(
                    "/shrink",
                    files={"file": ("input.wav", f, "audio/wav")},
                    data={"target_bytes": 0},
                )

        self.assertEqual(
            response.json(),
            {"error": "Invalid target_bytes value. Must be greater than 0."},
        )

    def test_shrink_media_rejects_missing_filename(self):
        response = saas_web.shrink_media(
            BackgroundTasks(),
            file=SimpleNamespace(filename="", file=io.BytesIO(b"dummy wav data")),
            target_bytes=10000,
        )

        self.assertEqual(response, {"error": "No file uploaded or filename missing"})

    @patch("saas_web.tempfile.mkdtemp", side_effect=OSError("disk full"))
    def test_shrink_media_handles_temp_dir_failure(self, _mock_mkdtemp):
        response = saas_web.shrink_media(
            BackgroundTasks(),
            file=SimpleNamespace(
                filename="input.wav", file=io.BytesIO(b"dummy wav data")
            ),
            target_bytes=10000,
        )

        self.assertEqual(response, {"error": "Upload processing failed"})

    @patch("saas_web.media_shrinker.convert_file")
    def test_shrink_media_uses_safe_fallback_filename(self, mock_convert_file):
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "output.flac"
            output.write_bytes(b"audio")
            mock_result = MagicMock(spec=ConversionResult)
            mock_result.output_path = output
            mock_convert_file.return_value = [mock_result]

            response = saas_web.shrink_media(
                BackgroundTasks(),
                file=SimpleNamespace(filename=".", file=io.BytesIO(b"dummy wav data")),
                target_bytes=10000,
            )

        self.assertEqual(Path(response.path), output)
        self.assertEqual(
            mock_convert_file.call_args.kwargs["source"].name, "upload.tmp"
        )

    def test_shrink_media_rejects_uploaded_body_over_limit(self):
        previous_limit = saas_web.MAX_UPLOAD_BYTES
        saas_web.MAX_UPLOAD_BYTES = 3
        try:
            response = saas_web.shrink_media(
                BackgroundTasks(),
                file=SimpleNamespace(filename="input.wav", file=io.BytesIO(b"1234")),
                target_bytes=10000,
            )
        finally:
            saas_web.MAX_UPLOAD_BYTES = previous_limit

        self.assertEqual(response, {"error": "Upload processing failed"})

    @patch("saas_web.Path.mkdir", side_effect=OSError("mkdir failed"))
    def test_shrink_media_handles_workspace_prepare_failure(self, _mock_mkdir):
        response = saas_web.shrink_media(
            BackgroundTasks(),
            file=SimpleNamespace(
                filename="input.wav", file=io.BytesIO(b"dummy wav data")
            ),
            target_bytes=10000,
        )

        self.assertEqual(response, {"error": "Upload processing failed"})

    @patch("saas_web.media_shrinker.convert_file")
    def test_shrink_media_exception_does_not_expose_internal_path(
        self, mock_convert_file
    ):
        mock_convert_file.side_effect = RuntimeError(
            "/tmp/codec_carver_secret/input.wav"
        )

        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            dummy_file_path = Path(temp_dir) / "input.wav"
            dummy_file_path.write_bytes(b"dummy wav data")

            with open(dummy_file_path, "rb") as f:
                response = client.post(
                    "/shrink",
                    files={"file": ("input.wav", f, "audio/wav")},
                    data={"target_bytes": 10000},
                )

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload, {"error": "Upload processing failed"})
            self.assertNotIn("/tmp/codec_carver_secret", response.text)

    @patch("saas_web.media_shrinker.convert_file")
    def test_shrink_media_failed_result_does_not_expose_internal_path(
        self, mock_convert_file
    ):
        mock_result = MagicMock(spec=ConversionResult)
        mock_result.output_path = Path("/tmp/codec_carver_secret/output.flac")
        mock_convert_file.return_value = [mock_result]

        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            dummy_file_path = Path(temp_dir) / "input.wav"
            dummy_file_path.write_bytes(b"dummy wav data")

            with open(dummy_file_path, "rb") as f:
                response = client.post(
                    "/shrink",
                    files={"file": ("input.wav", f, "audio/wav")},
                    data={"target_bytes": 10000},
                )

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(
                payload, {"error": "Processing failed or no output generated"}
            )
            self.assertNotIn("/tmp/codec_carver_secret", response.text)

    def test_get_ui_includes_target_bytes_validation_feedback(self):
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.text
        self.assertIn("preview.innerText = 'Must be greater than 0.';", html)
        self.assertIn("preview.innerText = 'This field is required.';", html)
        self.assertIn("preview.style.color = '#dc3545';", html)

    def test_request_size_limit_rejects_streamed_body_over_limit(self):
        async def receive():
            return {"type": "http.request", "body": b"1234"}

        async def call_next(request):
            await request._receive()
            return Response()

        request = SimpleNamespace(headers={}, _receive=receive)
        previous_limit = saas_web.MAX_REQUEST_BYTES
        saas_web.MAX_REQUEST_BYTES = 3
        try:
            response = asyncio.run(saas_web.limit_request_size(request, call_next))
        finally:
            saas_web.MAX_REQUEST_BYTES = previous_limit

        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.body, b'{"error":"Payload Too Large"}')

    def test_request_size_limit_passes_non_request_asgi_messages(self):
        async def receive():
            return {"type": "http.disconnect"}

        async def call_next(request):
            self.assertEqual(await request._receive(), {"type": "http.disconnect"})
            return Response(status_code=204)

        request = SimpleNamespace(headers={}, _receive=receive)
        response = asyncio.run(saas_web.limit_request_size(request, call_next))

        self.assertEqual(response.status_code, 204)

    def test_get_ui_includes_preset_buttons(self):
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.text

        self.assertIn('class="preset-container"', html)
        self.assertIn('data-bytes="26214400"', html)
        self.assertIn('data-bytes="104857600"', html)
        self.assertIn('data-bytes="524288000"', html)
        self.assertIn('data-bytes="1073741824"', html)
        self.assertIn(
            "document.getElementById('preset_buttons_container').addEventListener('click'",
            html,
        )
        self.assertIn('aria-pressed="false"', html)
        self.assertIn('role="group" aria-label="Preset target sizes"', html)
        self.assertNotIn('onclick="setTargetBytes(', html)
        self.assertIn(
            "const presetValue = Number.parseInt(btn.dataset.bytes, 10);", html
        )
        self.assertIn("!e.isTrusted && presetValue === val", html)
        self.assertNotIn("btn.dataset.bytes === this.value", html)


@unittest.skipUnless(
    _HAS_FASTAPI, "fastapi not installed (optional integration dependency)"
)
class TestShrinkBatch(unittest.TestCase):
    """Tests for the POST /shrink-batch multi-file endpoint."""

    @staticmethod
    def _fake_convert(source, root, output_dir, target_bytes):
        """Fake convert_file that writes a shrunk copy into output_dir."""
        output_path = Path(output_dir) / (Path(source).stem + ".flac")
        output_path.write_bytes(b"shrunk:" + Path(source).read_bytes())
        result = MagicMock(spec=ConversionResult)
        result.output_path = output_path
        return [result]

    @staticmethod
    def _read_zip(response):
        """Return (namelist, manifest dict, zipfile) for a zip response."""
        archive = zipfile.ZipFile(io.BytesIO(response.content))
        manifest = json.loads(archive.read("results.json"))
        return archive.namelist(), manifest, archive

    @patch("saas_web.media_shrinker.convert_file")
    def test_shrink_batch_two_files_returns_zip_with_outputs_and_manifest(
        self, mock_convert_file
    ):
        mock_convert_file.side_effect = self._fake_convert

        response = client.post(
            "/shrink-batch",
            files=[
                ("files", ("a.wav", b"audio-a", "audio/wav")),
                ("files", ("b.mp4", b"video-b", "video/mp4")),
            ],
            data={"target_bytes": 10000},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "application/zip")
        names, manifest, archive = self._read_zip(response)
        self.assertIn("01_a.flac", names)
        self.assertIn("02_b.flac", names)
        self.assertIn("results.json", names)
        self.assertEqual(archive.read("01_a.flac"), b"shrunk:audio-a")
        self.assertEqual(archive.read("02_b.flac"), b"shrunk:video-b")
        self.assertEqual(manifest["target_bytes"], 10000)
        self.assertEqual(len(manifest["results"]), 2)
        self.assertEqual(manifest["results"][0]["status"], "ok")
        self.assertEqual(manifest["results"][0]["filename"], "a.wav")
        self.assertEqual(manifest["results"][0]["output_name"], "01_a.flac")
        self.assertEqual(manifest["results"][0]["output_bytes"], len(b"shrunk:audio-a"))
        self.assertEqual(manifest["results"][1]["status"], "ok")
        self.assertEqual(mock_convert_file.call_count, 2)

    @patch("saas_web.media_shrinker.convert_file")
    def test_shrink_batch_one_failure_does_not_abort_batch(self, mock_convert_file):
        def convert(source, root, output_dir, target_bytes):
            if Path(source).name == "bad.wav":
                raise RuntimeError("/tmp/codec_carver_secret/bad.wav")
            return self._fake_convert(source, root, output_dir, target_bytes)

        mock_convert_file.side_effect = convert

        response = client.post(
            "/shrink-batch",
            files=[
                ("files", ("bad.wav", b"broken", "audio/wav")),
                ("files", ("good.wav", b"fine", "audio/wav")),
            ],
            data={"target_bytes": 10000},
        )

        self.assertEqual(response.status_code, 200)
        names, manifest, archive = self._read_zip(response)
        self.assertNotIn("01_bad.flac", names)
        self.assertIn("02_good.flac", names)
        self.assertEqual(manifest["results"][0]["status"], "error")
        self.assertEqual(manifest["results"][0]["error"], "Upload processing failed")
        self.assertNotIn("codec_carver_secret", archive.read("results.json").decode())
        self.assertEqual(manifest["results"][1]["status"], "ok")

    def test_shrink_batch_rejects_zero_files(self):
        response = client.post("/shrink-batch", data={"target_bytes": 10000})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"error": "No files uploaded"})

    def test_shrink_batch_rejects_too_many_files(self):
        uploads = [
            ("files", (f"f{i}.wav", b"x", "audio/wav"))
            for i in range(saas_web.MAX_BATCH_FILES + 1)
        ]
        response = client.post(
            "/shrink-batch", files=uploads, data={"target_bytes": 10000}
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json(),
            {
                "error": f"Too many files. Maximum is {saas_web.MAX_BATCH_FILES} files per batch."
            },
        )

    def test_shrink_batch_rejects_nonpositive_target_bytes(self):
        response = client.post(
            "/shrink-batch",
            files=[("files", ("a.wav", b"audio", "audio/wav"))],
            data={"target_bytes": 0},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json(),
            {"error": "Invalid target_bytes value. Must be greater than 0."},
        )

    def test_shrink_batch_rejects_oversized_target_bytes(self):
        response = client.post(
            "/shrink-batch",
            files=[("files", ("a.wav", b"audio", "audio/wav"))],
            data={"target_bytes": saas_web.MAX_TARGET_BYTES + 1},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json(),
            {"error": "Invalid target_bytes value. Exceeds the maximum allowed size."},
        )

    @patch("saas_web.media_shrinker.convert_file")
    def test_shrink_batch_rejects_disallowed_content_type_per_file(
        self, mock_convert_file
    ):
        mock_convert_file.side_effect = self._fake_convert

        response = client.post(
            "/shrink-batch",
            files=[
                ("files", ("evil.sh", b"#!/bin/sh", "application/x-sh")),
                ("files", ("good.wav", b"fine", "audio/wav")),
            ],
            data={"target_bytes": 10000},
        )

        self.assertEqual(response.status_code, 200)
        names, manifest, _archive = self._read_zip(response)
        self.assertEqual(names, ["02_good.flac", "results.json"])
        self.assertEqual(manifest["results"][0]["status"], "error")
        self.assertIn("Unsupported content type", manifest["results"][0]["error"])
        self.assertEqual(manifest["results"][1]["status"], "ok")
        mock_convert_file.assert_called_once()

    @patch("saas_web.media_shrinker.convert_file")
    def test_shrink_batch_records_no_output_as_error(self, mock_convert_file):
        mock_convert_file.return_value = []

        response = client.post(
            "/shrink-batch",
            files=[("files", ("a.wav", b"audio", "audio/wav"))],
            data={"target_bytes": 10000},
        )

        self.assertEqual(response.status_code, 200)
        names, manifest, _archive = self._read_zip(response)
        self.assertEqual(names, ["results.json"])
        self.assertEqual(manifest["results"][0]["status"], "error")
        self.assertEqual(
            manifest["results"][0]["error"],
            "Processing failed or no output generated",
        )

    @patch("saas_web.media_shrinker.convert_file")
    def test_shrink_batch_never_serves_output_outside_workspace(
        self, mock_convert_file
    ):
        with tempfile.TemporaryDirectory() as outside_dir:
            outside_file = Path(outside_dir) / "secret.flac"
            outside_file.write_bytes(b"secret contents")
            mock_result = MagicMock(spec=ConversionResult)
            mock_result.output_path = outside_file
            mock_convert_file.return_value = [mock_result]

            response = client.post(
                "/shrink-batch",
                files=[("files", ("a.wav", b"audio", "audio/wav"))],
                data={"target_bytes": 10000},
            )

        self.assertEqual(response.status_code, 200)
        names, manifest, _archive = self._read_zip(response)
        self.assertEqual(names, ["results.json"])
        self.assertEqual(manifest["results"][0]["status"], "error")
        self.assertEqual(
            manifest["results"][0]["error"],
            "Processing failed or no output generated",
        )
        self.assertNotIn(b"secret contents", response.content)

    def test_shrink_batch_records_oversized_file_as_error(self):
        previous_limit = saas_web.MAX_UPLOAD_BYTES
        saas_web.MAX_UPLOAD_BYTES = 3
        try:
            response = client.post(
                "/shrink-batch",
                files=[("files", ("big.wav", b"12345", "audio/wav"))],
                data={"target_bytes": 10000},
            )
        finally:
            saas_web.MAX_UPLOAD_BYTES = previous_limit

        self.assertEqual(response.status_code, 200)
        names, manifest, _archive = self._read_zip(response)
        self.assertEqual(names, ["results.json"])
        self.assertEqual(manifest["results"][0]["status"], "error")
        self.assertEqual(manifest["results"][0]["error"], "Upload processing failed")

    @patch("saas_web.tempfile.mkdtemp", side_effect=OSError("disk full"))
    def test_shrink_batch_handles_workspace_creation_failure(self, _mock_mkdtemp):
        response = client.post(
            "/shrink-batch",
            files=[("files", ("a.wav", b"audio", "audio/wav"))],
            data={"target_bytes": 10000},
        )
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"error": "Upload processing failed"})

    @patch("saas_web.zipfile.ZipFile", side_effect=OSError("cannot write zip"))
    def test_shrink_batch_handles_archive_failure(self, _mock_zipfile):
        response = client.post(
            "/shrink-batch",
            files=[("files", ("a.wav", b"audio", "audio/wav"))],
            data={"target_bytes": 10000},
        )
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"error": "Upload processing failed"})

    @patch("saas_web.media_shrinker.convert_file")
    def test_shrink_batch_uses_safe_fallback_filename_with_backslashes(self, mock_convert_file):
        mock_convert_file.return_value = []

        response = saas_web.shrink_media_batch(
            BackgroundTasks(),
            files=[
                SimpleNamespace(
                    filename="..\\..\\windows.ini",
                    content_type="audio/wav",
                    file=io.BytesIO(b"dummy"),
                )
            ],
            target_bytes=10000,
        )

        try:
            with zipfile.ZipFile(response.path) as archive:
                manifest = json.loads(archive.read("results.json"))
            self.assertEqual(manifest["results"][0]["filename"], "windows.ini")
            self.assertEqual(
                mock_convert_file.call_args.kwargs["source"].name, "windows.ini"
            )
        finally:
            saas_web.cleanup_temp_dir(Path(response.path).parent)

    @patch("saas_web.media_shrinker.convert_file")
    def test_shrink_batch_uses_safe_fallback_filename(self, mock_convert_file):
        mock_convert_file.return_value = []

        response = saas_web.shrink_media_batch(
            BackgroundTasks(),
            files=[
                SimpleNamespace(
                    filename="..",
                    content_type="audio/wav",
                    file=io.BytesIO(b"dummy"),
                )
            ],
            target_bytes=10000,
        )

        try:
            with zipfile.ZipFile(response.path) as archive:
                manifest = json.loads(archive.read("results.json"))
            self.assertEqual(manifest["results"][0]["filename"], "upload.tmp")
            self.assertEqual(
                mock_convert_file.call_args.kwargs["source"].name, "upload.tmp"
            )
        finally:
            saas_web.cleanup_temp_dir(Path(response.path).parent)

    def test_get_ui_includes_batch_upload_form(self):
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.text
        self.assertIn('action="/shrink-batch"', html)
        self.assertIn('id="batch_files"', html)
        self.assertIn("multiple", html)
        self.assertIn('accept="audio/*,video/*"', html)
        self.assertIn('aria-describedby="batch_files_help batch_files_preview"', html)
        self.assertIn('onchange="updateBatchFilePreview(this)"', html)
        self.assertIn('id="batch_files_preview"', html)
        self.assertIn("function updateBatchFilePreview(input)", html)


@unittest.skipUnless(
    _HAS_FASTAPI, "fastapi not installed (optional integration dependency)"
)
class TestApiKeyAuth(unittest.TestCase):
    """Tests for the opt-in CODEC_CARVER_API_KEYS authentication middleware."""

    def _post_shrink(self, headers=None):
        """POST a minimal /shrink request and return the response."""

        return client.post(
            "/shrink",
            files={"file": ("input.wav", io.BytesIO(b"dummy wav data"), "audio/wav")},
            data={"target_bytes": 0},
            headers=headers or {},
        )

    def test_no_env_var_leaves_endpoints_open(self):
        with patch.dict(os.environ):
            os.environ.pop("CODEC_CARVER_API_KEYS", None)
            response = self._post_shrink()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"error": "Invalid target_bytes value. Must be greater than 0."},
        )

    def test_missing_header_rejected_when_keys_configured(self):
        with patch.dict(os.environ, {"CODEC_CARVER_API_KEYS": "secret-key"}):
            response = self._post_shrink()

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json(), {"error": "Invalid or missing API key"})
        self.assertNotIn("secret-key", response.text)

    def test_wrong_key_rejected(self):
        with patch.dict(os.environ, {"CODEC_CARVER_API_KEYS": "secret-key"}):
            response = self._post_shrink(headers={"X-API-Key": "wrong-key"})

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json(), {"error": "Invalid or missing API key"})
        self.assertNotIn("secret-key", response.text)

    def test_correct_key_reaches_handler(self):
        with patch.dict(os.environ, {"CODEC_CARVER_API_KEYS": "secret-key"}):
            response = self._post_shrink(headers={"X-API-Key": "secret-key"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"error": "Invalid target_bytes value. Must be greater than 0."},
        )

    def test_get_ui_always_open_without_key(self):
        with patch.dict(os.environ, {"CODEC_CARVER_API_KEYS": "secret-key"}):
            response = client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Codec Carver SaaS", response.content)

    def test_job_api_requires_key_when_configured(self):
        with patch.dict(os.environ, {"CODEC_CARVER_API_KEYS": "secret-key"}):
            response = client.get("/jobs/missing")
            allowed = client.get("/jobs/missing", headers={"X-API-Key": "secret-key"})

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json(), {"error": "Invalid or missing API key"})
        self.assertEqual(allowed.status_code, 404)

    def test_multiple_comma_separated_keys_all_valid(self):
        with patch.dict(
            os.environ, {"CODEC_CARVER_API_KEYS": "key-one,key-two,key-three"}
        ):
            for key in ("key-one", "key-two", "key-three"):
                response = self._post_shrink(headers={"X-API-Key": key})
                self.assertEqual(response.status_code, 200, key)
            rejected = self._post_shrink(headers={"X-API-Key": "key-four"})

        self.assertEqual(rejected.status_code, 401)

    def test_whitespace_around_keys_is_stripped(self):
        with patch.dict(os.environ, {"CODEC_CARVER_API_KEYS": "  key-one , key-two  "}):
            response = self._post_shrink(headers={"X-API-Key": "key-one"})
            self.assertEqual(response.status_code, 200)
            response = self._post_shrink(headers={"X-API-Key": "key-two"})
            self.assertEqual(response.status_code, 200)
            rejected = self._post_shrink(headers={"X-API-Key": " key-one "})

        self.assertEqual(rejected.status_code, 401)

    def test_empty_entries_are_ignored(self):
        with patch.dict(os.environ, {"CODEC_CARVER_API_KEYS": "key-one,,  ,"}):
            response = self._post_shrink(headers={"X-API-Key": "key-one"})
            self.assertEqual(response.status_code, 200)
            rejected = self._post_shrink(headers={"X-API-Key": ""})

        self.assertEqual(rejected.status_code, 401)

    def test_only_empty_entries_leave_endpoints_open(self):
        with patch.dict(os.environ, {"CODEC_CARVER_API_KEYS": " , ,"}):
            response = self._post_shrink()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"error": "Invalid target_bytes value. Must be greater than 0."},
        )

    def test_get_configured_api_keys_parsing(self):
        with patch.dict(os.environ, {"CODEC_CARVER_API_KEYS": " a ,, b ,"}):
            self.assertEqual(saas_web.get_configured_api_keys(), ["a", "b"])
        with patch.dict(os.environ):
            os.environ.pop("CODEC_CARVER_API_KEYS", None)
            self.assertEqual(saas_web.get_configured_api_keys(), [])


@unittest.skipUnless(
    _HAS_FASTAPI, "fastapi not installed (optional integration dependency)"
)
class MultiSegmentZipTests(unittest.TestCase):
    """Long recordings split into multiple segments must all be returned (as a zip)."""

    @patch("saas_web.media_shrinker.convert_file")
    def test_multiple_segments_returned_as_zip(self, mock_convert_file):
        import io as _io
        import tempfile
        import zipfile

        with tempfile.TemporaryDirectory() as temp_dir:
            part1 = Path(temp_dir) / "rec.wav.part0001.flac"
            part2 = Path(temp_dir) / "rec.wav.part0002.flac"
            part1.write_bytes(b"segment-one")
            part2.write_bytes(b"segment-two")
            r1 = MagicMock(spec=ConversionResult)
            r1.output_path = part1
            r2 = MagicMock(spec=ConversionResult)
            r2.output_path = part2
            mock_convert_file.return_value = [r1, r2]

            response = client.post(
                "/shrink",
                files={"file": ("rec.wav", _io.BytesIO(b"wav data"), "audio/wav")},
                data={"target_bytes": 10000},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "application/zip")
        names = zipfile.ZipFile(_io.BytesIO(response.content)).namelist()
        self.assertEqual(
            sorted(names), ["rec.wav.part0001.flac", "rec.wav.part0002.flac"]
        )


@unittest.skipUnless(
    _HAS_FASTAPI, "fastapi not installed (optional integration dependency)"
)
class JobModelTests(unittest.TestCase):
    """Async job API: submit -> status -> result, plus all error paths."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._old_store = saas_web.JOB_STORE
        self.addCleanup(setattr, saas_web, "JOB_STORE", self._old_store)
        saas_web.JOB_STORE = JobStore(str(Path(self._tmp.name) / "jobs.db"))

    def tearDown(self) -> None:
        for job in saas_web.JOB_STORE.list_jobs():
            temp = job.get("temp_dir")
            if temp:
                saas_web.cleanup_temp_dir(Path(temp))
            saas_web.JOB_STORE.delete(job["id"])

    def _create_job(
        self,
        job_id: str,
        status: str,
        temp_dir: str,
        output_path: str | None = None,
        output_name: str | None = None,
        error: str | None = None,
    ) -> None:
        """Insert a job-store row for direct endpoint edge-case tests."""

        saas_web.JOB_STORE.create(job_id, temp_dir=temp_dir, now=saas_web._now())
        if status != "queued":
            saas_web.JOB_STORE.set_status(
                job_id,
                status,
                now=saas_web._now(),
                output_path=output_path,
                output_name=output_name,
                error=error,
            )

    def _make_workspace(self) -> tuple[Path, Path, Path, Path]:
        temp_dir = Path(tempfile.mkdtemp(prefix="codec_carver_"))
        input_dir = temp_dir / "input"
        output_dir = temp_dir / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        source_path = input_dir / "in.wav"
        source_path.write_bytes(b"wav data")
        return temp_dir, input_dir, output_dir, source_path

    def test_default_job_store_path_uses_env(self):
        with patch.dict(
            saas_web.os.environ,
            {"CODEC_CARVER_JOB_DB": "custom-jobs.sqlite3"},
        ):
            self.assertEqual(
                saas_web._default_job_store_path(), Path("custom-jobs.sqlite3")
            )

    @patch("saas_web.media_shrinker.convert_file")
    def test_job_lifecycle_submit_status_result(self, mock_convert_file):
        def fake_convert(**kwargs):
            # Write the output inside the job's real workspace (output_dir), as
            # the engine does, so the served path passes the confinement check.
            output = kwargs["output_dir"] / "out.flac"
            output.write_bytes(b"audio-bytes")
            mock_result = MagicMock(spec=ConversionResult)
            mock_result.output_path = output
            return [mock_result]

        mock_convert_file.side_effect = fake_convert

        submit = client.post(
            "/jobs",
            files={"file": ("in.wav", io.BytesIO(b"wav data"), "audio/wav")},
            data={"target_bytes": 10000},
        )
        self.assertEqual(submit.status_code, 200)
        job_id = submit.json()["job_id"]
        self.assertEqual(submit.json()["status"], "queued")

        status = client.get(f"/jobs/{job_id}")
        self.assertEqual(status.status_code, 200)
        self.assertEqual(status.json()["status"], "done")

        result = client.get(f"/jobs/{job_id}/result")
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.content, b"audio-bytes")

    @patch("saas_web.media_shrinker.convert_file")
    def test_job_result_returns_zip_for_multiple_segments(self, mock_convert_file):
        import zipfile

        def fake_convert(**kwargs):
            part1 = kwargs["output_dir"] / "in.wav.part0001.flac"
            part2 = kwargs["output_dir"] / "in.wav.part0002.flac"
            part1.write_bytes(b"segment-one")
            part2.write_bytes(b"segment-two")
            result1 = MagicMock(spec=ConversionResult)
            result1.output_path = part1
            result2 = MagicMock(spec=ConversionResult)
            result2.output_path = part2
            return [result1, result2]

        mock_convert_file.side_effect = fake_convert

        submit = client.post(
            "/jobs",
            files={"file": ("in.wav", io.BytesIO(b"wav data"), "audio/wav")},
            data={"target_bytes": 10000},
        )
        job_id = submit.json()["job_id"]

        result = client.get(f"/jobs/{job_id}/result")

        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.headers["content-type"], "application/zip")
        names = zipfile.ZipFile(io.BytesIO(result.content)).namelist()
        self.assertEqual(
            sorted(names), ["in.wav.part0001.flac", "in.wav.part0002.flac"]
        )

    def test_result_outside_workspace_rejected(self):
        # A "done" job whose output escaped its workspace must not be served.
        import tempfile

        workspace = Path(tempfile.mkdtemp(prefix="codec_carver_"))
        outside_dir = Path(tempfile.mkdtemp())
        escaped = outside_dir / "escaped.flac"
        escaped.write_bytes(b"secret")
        try:
            self._create_job(
                "escape",
                "done",
                str(workspace),
                output_path=str(escaped),
                output_name="escaped.flac",
            )
            response = client.get("/jobs/escape/result")
            self.assertEqual(response.status_code, 410)
        finally:
            saas_web.cleanup_temp_dir(workspace)
            saas_web.cleanup_temp_dir(outside_dir)

    def test_submit_rejects_nonpositive_target(self):
        response = client.post(
            "/jobs",
            files={"file": ("in.wav", io.BytesIO(b"wav data"), "audio/wav")},
            data={"target_bytes": 0},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("greater than 0", response.json()["error"])

    def test_submit_rejects_missing_filename(self):
        response = saas_web.submit_job(
            BackgroundTasks(),
            file=SimpleNamespace(filename="", file=io.BytesIO(b"wav data")),
            target_bytes=10000,
        )
        self.assertEqual(response.status_code, 400)

    @patch("saas_web._persist_upload", side_effect=OSError("disk full"))
    def test_submit_handles_persist_failure(self, _mock_persist):
        response = saas_web.submit_job(
            BackgroundTasks(),
            file=SimpleNamespace(filename="in.wav", file=io.BytesIO(b"wav data")),
            target_bytes=10000,
        )
        self.assertEqual(response.status_code, 500)

    @patch("saas_web.media_shrinker.convert_file")
    def test_run_job_cleans_unknown_job_before_processing(self, mock_convert_file):
        temp_dir, input_dir, output_dir, source_path = self._make_workspace()

        saas_web._run_job(
            "missing", source_path, input_dir, output_dir, 10000, temp_dir
        )

        self.assertFalse(temp_dir.exists())
        mock_convert_file.assert_not_called()

    @patch("saas_web.media_shrinker.convert_file", side_effect=RuntimeError("boom"))
    def test_run_job_records_failure_on_exception(self, _mock_convert):
        submit = client.post(
            "/jobs",
            files={"file": ("in.wav", io.BytesIO(b"wav data"), "audio/wav")},
            data={"target_bytes": 10000},
        )
        job_id = submit.json()["job_id"]
        status = client.get(f"/jobs/{job_id}")
        self.assertEqual(status.json()["status"], "failed")
        self.assertEqual(status.json()["error"], "Processing failed")

    @patch("saas_web._get_job_store")
    @patch("saas_web.media_shrinker.convert_file", side_effect=RuntimeError("boom"))
    def test_run_job_handles_missing_job_while_recording_failure(
        self, _mock_convert, mock_get_store
    ):
        class VanishingFailureStore:
            def set_status(self, _job_id, status, **_kwargs):
                if status == "processing":
                    return None
                raise KeyError("gone")

        mock_get_store.return_value = VanishingFailureStore()
        temp_dir, input_dir, output_dir, source_path = self._make_workspace()

        saas_web._run_job(
            "missing-after-error",
            source_path,
            input_dir,
            output_dir,
            10000,
            temp_dir,
        )

        self.assertFalse(temp_dir.exists())

    @patch("saas_web.media_shrinker.convert_file", return_value=[])
    def test_run_job_records_failure_on_empty_output(self, _mock_convert):
        submit = client.post(
            "/jobs",
            files={"file": ("in.wav", io.BytesIO(b"wav data"), "audio/wav")},
            data={"target_bytes": 10000},
        )
        job_id = submit.json()["job_id"]
        status = client.get(f"/jobs/{job_id}")
        self.assertEqual(status.json()["status"], "failed")
        self.assertIn("no output", status.json()["error"])

    @patch("saas_web._get_job_store")
    @patch("saas_web.media_shrinker.convert_file")
    def test_run_job_handles_missing_job_while_recording_result(
        self, mock_convert_file, mock_get_store
    ):
        class VanishingResultStore:
            def set_status(self, _job_id, status, **_kwargs):
                if status == "processing":
                    return None
                raise KeyError("gone")

        temp_dir, input_dir, output_dir, source_path = self._make_workspace()
        output = output_dir / "out.flac"
        output.write_bytes(b"audio")
        mock_result = MagicMock(spec=ConversionResult)
        mock_result.output_path = output
        mock_convert_file.return_value = [mock_result]
        mock_get_store.return_value = VanishingResultStore()

        saas_web._run_job(
            "missing-after-output",
            source_path,
            input_dir,
            output_dir,
            10000,
            temp_dir,
        )

        self.assertFalse(temp_dir.exists())

    @patch("saas_web._get_job_store")
    @patch("saas_web.media_shrinker.convert_file", return_value=[])
    def test_run_job_handles_missing_job_while_recording_empty_output(
        self, _mock_convert, mock_get_store
    ):
        class VanishingEmptyOutputStore:
            def set_status(self, _job_id, status, **_kwargs):
                if status == "processing":
                    return None
                raise KeyError("gone")

        mock_get_store.return_value = VanishingEmptyOutputStore()
        temp_dir, input_dir, output_dir, source_path = self._make_workspace()

        saas_web._run_job(
            "missing-after-empty",
            source_path,
            input_dir,
            output_dir,
            10000,
            temp_dir,
        )

        self.assertFalse(temp_dir.exists())

    @patch("saas_web._get_job_store")
    @patch("saas_web._persist_upload")
    def test_submit_handles_job_store_create_failure(
        self, mock_persist_upload, mock_get_store
    ):
        class RejectingStore:
            def create(self, *_args, **_kwargs):
                raise ValueError("duplicate")

        temp_dir, input_dir, output_dir, source_path = self._make_workspace()
        mock_persist_upload.return_value = (
            temp_dir,
            input_dir,
            output_dir,
            source_path,
        )
        mock_get_store.return_value = RejectingStore()

        response = saas_web.submit_job(
            BackgroundTasks(),
            file=SimpleNamespace(filename="in.wav", file=io.BytesIO(b"wav data")),
            target_bytes=10000,
        )

        self.assertEqual(response.status_code, 500)
        self.assertFalse(temp_dir.exists())

    def test_status_unknown_job_returns_404(self):
        response = client.get("/jobs/does-not-exist")
        self.assertEqual(response.status_code, 404)

    def test_result_unknown_job_returns_404(self):
        response = client.get("/jobs/does-not-exist/result")
        self.assertEqual(response.status_code, 404)

    def test_result_not_ready_returns_409(self):
        self._create_job("pending", "processing", "")
        response = client.get("/jobs/pending/result")
        self.assertEqual(response.status_code, 409)
        self.assertIn("processing", response.json()["error"])

    def test_result_missing_file_returns_410(self):
        self._create_job(
            "gone",
            "done",
            "",
            output_path="/nonexistent/output.flac",
            output_name="output.flac",
        )
        response = client.get("/jobs/gone/result")
        self.assertEqual(response.status_code, 410)

    def test_cleanup_job_removes_workspace(self):
        import tempfile

        temp_dir = Path(tempfile.mkdtemp(prefix="codec_carver_"))
        self._create_job("c", "done", str(temp_dir))
        saas_web._cleanup_job("c")
        self.assertFalse(temp_dir.exists())
        self.assertIsNone(saas_web.JOB_STORE.get("c"))

    def test_cleanup_job_tolerates_unknown_job(self):
        saas_web._cleanup_job("unknown-cleanup-job")
        self.assertIsNone(saas_web.JOB_STORE.get("unknown-cleanup-job"))


@unittest.skipUnless(
    _HAS_FASTAPI, "fastapi not installed (optional integration dependency)"
)
class UploadValidationTests(unittest.TestCase):
    """Input hardening surfaced by the SAST review: target bound + content type."""

    def test_shrink_rejects_oversized_target_bytes(self):
        response = client.post(
            "/shrink",
            files={"file": ("in.wav", io.BytesIO(b"wav data"), "audio/wav")},
            data={"target_bytes": saas_web.MAX_TARGET_BYTES + 1},
        )
        self.assertEqual(
            response.json(),
            {"error": "Invalid target_bytes value. Exceeds the maximum allowed size."},
        )

    def test_shrink_rejects_non_media_content_type(self):
        response = client.post(
            "/shrink",
            files={"file": ("shell.php", io.BytesIO(b"<?php ?>"), "application/x-php")},
            data={"target_bytes": 10000},
        )
        self.assertEqual(
            response.json(),
            {"error": "Unsupported content type; upload an audio or video file."},
        )

    def test_submit_rejects_non_media_content_type(self):
        response = client.post(
            "/jobs",
            files={"file": ("shell.php", io.BytesIO(b"<?php ?>"), "application/x-php")},
            data={"target_bytes": 10000},
        )
        self.assertEqual(response.status_code, 400)

    def test_video_content_type_accepted_by_validator(self):
        self.assertIsNone(
            saas_web._validate_request(
                SimpleNamespace(filename="clip.mp4", content_type="video/mp4"),
                10000,
            )
        )


if __name__ == "__main__":
    unittest.main()
