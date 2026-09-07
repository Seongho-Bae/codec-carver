"""Lock the local OpenCode config to the NVIDIA NIM-only org contract."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OPENCODE_CONFIG = ROOT / "opencode.jsonc"
AGENTS_GUIDE = ROOT / "AGENTS.md"
DOCTORING = ROOT / "docs" / "doctoring" / "opencode-nvidia-nim-contract.md"

FORBIDDEN_FRAGMENTS = (
    "github-models",
    "STRIX_GITHUB_MODELS_TOKEN",
    "COPILOT_GITHUB_TOKEN",
    "openai/gpt-5",
    "openai/o3",
    "openai/o4-mini",
    "models.github.ai",
    "{env:NVIDIA_NIM_API_KEY}",
    "reasoningEffort",
)


class OpenCodeNvidiaNimContractTests(unittest.TestCase):
    """Keep local OpenCode on NVIDIA NIM and reject retired GitHub Models IDs."""

    def test_opencode_uses_nvidia_nim_only(self) -> None:
        """Require the exact NIM provider, models, endpoint, and process-env bind."""

        text = OPENCODE_CONFIG.read_text(encoding="utf-8")
        config = json.loads(text)

        self.assertEqual(
            config["model"],
            "nvidia-nim/nvidia/llama-3.3-nemotron-super-49b-v1.5",
        )
        self.assertEqual(
            config["small_model"],
            "nvidia-nim/meta/llama-3.3-70b-instruct",
        )
        self.assertEqual(config["enabled_providers"], ["nvidia-nim"])
        self.assertEqual(config["share"], "disabled")
        self.assertEqual(list(config["provider"]), ["nvidia-nim"])

        provider = config["provider"]["nvidia-nim"]
        self.assertEqual(provider["npm"], "@ai-sdk/openai-compatible")
        self.assertEqual(
            provider["options"]["baseURL"],
            "https://integrate.api.nvidia.com/v1",
        )
        self.assertEqual(provider["options"]["apiKey"], "{env:NVIDIA_API_KEY}")
        self.assertIn(
            "nvidia/llama-3.3-nemotron-super-49b-v1.5",
            provider["models"],
        )
        self.assertIn("meta/llama-3.3-70b-instruct", provider["models"])

        for fragment in FORBIDDEN_FRAGMENTS:
            with self.subTest(fragment=fragment):
                self.assertNotIn(fragment, text)

    def test_opencode_forbids_reasoning_effort_anywhere(self) -> None:
        """Ban OpenAI-style reasoningEffort at every config level, not just model options."""

        text = OPENCODE_CONFIG.read_text(encoding="utf-8")
        self.assertNotIn("reasoningEffort", text)

    def test_current_policy_docs_record_the_nim_contract(self) -> None:
        """Keep AGENTS.md and doctoring aligned with the locked OpenCode file."""

        agents = AGENTS_GUIDE.read_text(encoding="utf-8")
        doctoring = DOCTORING.read_text(encoding="utf-8")

        for document in (agents, doctoring):
            with self.subTest(document=document[:24]):
                self.assertIn("nvidia-nim", document)
                self.assertIn("{env:NVIDIA_API_KEY}", document)
                self.assertIn("NVIDIA_NIM_API_KEY", document)
                self.assertIn("https://integrate.api.nvidia.com/v1", document)
                self.assertIn("llama-3.3-nemotron-super-49b-v1.5", document)
                self.assertNotIn("github-models/openai/gpt-5", document)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
