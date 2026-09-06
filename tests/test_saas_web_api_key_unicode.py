import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    from fastapi.responses import Response

    import saas_web

    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False


@unittest.skipUnless(
    _HAS_FASTAPI, "fastapi not installed (optional integration dependency)"
)
class TestApiKeyUnicodeContract(unittest.TestCase):
    @staticmethod
    def _request(api_key: str):
        return SimpleNamespace(
            headers={"x-api-key": api_key},
            method="POST",
            url=SimpleNamespace(path="/shrink"),
        )

    @staticmethod
    async def _allow(_request):
        return Response(status_code=204)

    def test_non_ascii_unconfigured_key_is_rejected_without_compare_digest_type_error(self):
        with patch.dict(os.environ, {"CODEC_CARVER_API_KEYS": "secret-key"}):
            response = asyncio.run(
                saas_web.require_api_key(self._request("공격자-키"), self._allow)
            )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.body, b'{"error":"Invalid or missing API key"}')

    def test_matching_non_ascii_configured_key_reaches_handler(self):
        with patch.dict(os.environ, {"CODEC_CARVER_API_KEYS": "운영-보안키"}):
            response = asyncio.run(
                saas_web.require_api_key(self._request("운영-보안키"), self._allow)
            )

        self.assertEqual(response.status_code, 204)


if __name__ == "__main__":
    unittest.main()
