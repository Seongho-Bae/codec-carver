# Local OpenCode NVIDIA NIM contract

## Decision

Local OpenCode in this repository uses NVIDIA NIM only. GitHub Models is
retired for ContextualWisdomLab. The checked-in `opencode.jsonc` therefore
enables a single provider, `nvidia-nim`, against the hosted OpenAI-compatible
endpoint `https://integrate.api.nvidia.com/v1`.

The default model is `nvidia-nim/nvidia/llama-3.3-nemotron-super-49b-v1.5`.
The small model is `nvidia-nim/meta/llama-3.3-70b-instruct`. Session sharing
is disabled. Existing MCP servers stay as-is; this change does not grant
generic exec, read, or write authority.

The local API-key bind remains `{env:NVIDIA_API_KEY}`. The organization
Actions secret name remains `NVIDIA_NIM_API_KEY`. CI and operator bootstrap
may map that secret onto the process alias `NVIDIA_API_KEY`. Do not rename
the OpenCode bind to `{env:NVIDIA_NIM_API_KEY}`.

OpenAI-style `reasoningEffort` is forbidden in this file. Llama 3.3 Nemotron
Super 49B v1.5 is a reasoning model; NVIDIA documents reasoning on/off through
the chat template and the `/no_think` system prompt, not through that OpenAI
request knob.

## Technical basis

NVIDIA NIM exposes chat completions at the shared hosted base URL
`https://integrate.api.nvidia.com/v1` (NVIDIA, n.d.-a). OpenCode consumes
OpenAI-compatible providers through `@ai-sdk/openai-compatible` and reads
`{env:…}` interpolations from the process environment (OpenCode, n.d.).
Llama-3.3-Nemotron-Super-49B-v1.5 is post-trained for reasoning, RAG, and
tool calling, with a 128K-token context (NVIDIA, n.d.-b). The model card
documents reasoning control via the system prompt rather than an
`reasoningEffort` field (NVIDIA, n.d.-b).

This file is developer-client configuration. Application runtime in
`saas_web.py` / `media_shrinker.py` does not read `NVIDIA_API_KEY`. Org
runtime-secret policy (KV, not `os.getenv`) is unchanged.

## Verification and rollback

- `tests/test_opencode_nvidia_nim_contract.py` locks the provider id, model
  ids, endpoint, `{env:NVIDIA_API_KEY}` bind, `share: "disabled"`, and the
  leftover-id deny list.
- The same module independently asserts `"reasoningEffort"` is absent from
  the whole config text.
- Roll back by restoring the previous `opencode.jsonc`; do not re-enable
  GitHub Models, `STRIX_GITHUB_MODELS_TOKEN`, or `COPILOT_GITHUB_TOKEN`.

## References

NVIDIA. (n.d.-a). *LLM APIs*. NVIDIA API Catalog.
https://docs.api.nvidia.com/nim/reference/llm-apis

NVIDIA. (n.d.-b). *nvidia / llama-3.3-nemotron-super-49b-v1.5*. NVIDIA API
Catalog.
https://docs.api.nvidia.com/nim/reference/nvidia-llama-3_3-nemotron-super-49b-v1_5

OpenCode. (n.d.). *Providers*.
https://opencode.ai/docs/providers/
