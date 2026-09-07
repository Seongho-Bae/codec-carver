# AGENTS.md

Cross-agent conventions for **codec-carver** — a Python CLI/FastAPI tool for
carving long recordings into metadata-preserved FLAC/Opus files. These notes
apply to any coding agent (Claude, Codex, Cursor, opencode, …) working in this
repo.

<!-- BEGIN cwl-agent-guidance -->
## Agent guidance (CWL governance)

### Security & review gate
- Every PR runs a central **Security Scan** required gate: `osv-scan` +
  `dependency-review` (diff-scoped) and `trivy-fs` (repo-wide; CRITICAL/HIGH,
  fixable only). It runs on every PR base, **including stacked PRs**.
- A **failing `trivy-fs` is a REAL finding, not a flake.** Read the job log — it
  prints each finding's rule id, severity, and file — or inspect the run's SARIF
  results. Then **remediate**:
  - This repo's dependencies are Python packages declared in `requirements.txt`
    (currently `fastapi`, `uvicorn`, `python-multipart`, `mcp`, `aiofiles`,
    `httpx`). A dependency CVE means bumping/pinning the affected package there.
  - There is **no Dockerfile or k8s manifest** today; if you add one, a trivy
    misconfig finding means fixing that IaC file directly.
  - Only for a genuine false positive, add a narrow, documented
    `.trivyignore` (or `.trivyignore.yaml`) entry — one CVE id with a reason.
- **Do NOT weaken or disable the gate.** A local scan with a stale DB misses
  findings: run `trivy --download-db-only` first, then scan the **merge ref**
  (not just the PR head), e.g. `trivy fs --severity CRITICAL,HIGH --ignore-unfixed .`.
- The org `code_scanning` ruleset is intentionally **CodeQL-only** (multiple
  code-scanning tools can't converge on one PR ref). Gating is by the Security
  Scan **job result**, not the code_scanning rule — **don't add tools to that rule.**

### Config & secrets (KV, not env)
- **Org rule: do NOT read config/secrets via `os.getenv()` / raw environment
  variables at runtime.** Read them from a KV / credential registry. Org Actions
  secrets (e.g. `OPENAI_API_KEY`) flow **into** the KV via a bootstrap/CI step;
  runtime reads from the KV — env is only transport into the KV, never the
  runtime source.
- **Reference implementation:** xtrmLLMBatchPython's pgcrypto-encrypted Postgres
  credential registry (`get_credential(name)`). Reuse that pattern (a DB-backed
  KV is fine — this repo already ships a stdlib `sqlite3` store in
  `job_store.py`) unless a dedicated KV is adopted.
- **Applies here:** `saas_web.py` is a FastAPI service being productized (durable
  job store, and open PRs adding API-key auth and usage metering), so it *will*
  read runtime secrets/config (API keys, DB creds, endpoints). When you add them,
  source them from the KV, not `os.getenv`.
- **Known deviation to migrate:** the in-flight API-key auth work reads keys from
  a `CODEC_CARVER_API_KEYS` environment variable — that is exactly the anti-pattern
  above. Move it to read from the credential registry (env may still be the
  bootstrap transport that *populates* the KV, never the runtime source).

### Code exploration
- There is no `.codegraph/` index in this repo today, so use normal search
  (grep/find) to locate and understand code. If a `.codegraph/` directory is
  ever added at the repo root, prefer CodeGraph first —
  `codegraph explore "<query>"` or the code-review-graph MCP tools — before
  grep/find; it surfaces callers, callees, and impact that text search misses.

### This repo's role in the ecosystem
- **codec-carver** is the **speech/video conversion module for STT and
  omni-modal LLM input** — carving and transcoding recordings into
  metadata-preserved streams that feed speech-to-text and multimodal LLM
  pipelines.
- The org is an ecosystem around **naruon** (the hub: email/PIM that
  DOM-decomposes emails and files into a persisted knowledge graph). Each
  component is a **standalone program that must ALSO work as a git submodule**,
  grown separately and together: **wardnet** (WAF / IDS / AI SOC / LB /
  APIM), **clearfolio** (document viewer), **pg-erd-cloud** (ERD tool),
  **contextual-orchestrator** (LLM cost/perf/upstream-LB gateway beyond
  LiteLLM), **codec-carver** (this repo — STT/omni-modal speech-video codec),
  **fast-mlsirm** (LLM-as-a-Judge calibration + evaluation-item quality, using
  aFIPC FIPC + kaefa item-fit), **keyverse** (passwordless SSO —
  OIDC/SCIM/ADFS/LDAP/FIDO2/OAuth2.1, eliminate passwords), **newsdom-api**
  (PDF→DOM sidecar), and **semantic-data-portal** (upper ontology / catalog /
  governance plane with its own graph engine).

### Research grounding (attach paper PDFs)
- **Org rule:** substantive feature/process PRs should find the relevant
  academic papers and **commit their PDFs into the PR** (e.g. a `docs/papers/`
  or `references/` directory) with full citations. Respect copyright — attach
  the PDF only when redistribution is permissible; otherwise **cite + link +
  summarize** instead of committing the file.
- **For this repo:** ground speech/codec work in the primary literature — e.g.
  ASR/STT acoustic modeling, neural audio/speech codecs (Opus/EnCodec-class),
  and audio-visual / omni-modal representation papers behind any new transcode
  or feature-extraction path.
<!-- END cwl-agent-guidance -->

## Local OpenCode (NVIDIA NIM only)

ContextualWisdomLab no longer uses GitHub Models. Local OpenCode in this repo
is NVIDIA NIM only:

- Provider `nvidia-nim` at `https://integrate.api.nvidia.com/v1`
- Local bind `{env:NVIDIA_API_KEY}` — do **not** rename this to
  `{env:NVIDIA_NIM_API_KEY}`. The org Actions secret remains
  `NVIDIA_NIM_API_KEY` and is mapped onto the process alias `NVIDIA_API_KEY`.
- `enabled_providers`: `["nvidia-nim"]`
- Default: `nvidia-nim/nvidia/llama-3.3-nemotron-super-49b-v1.5`
- Small: `nvidia-nim/meta/llama-3.3-70b-instruct`
- No `COPILOT_GITHUB_TOKEN`, `STRIX_GITHUB_MODELS_TOKEN`, or GitHub Models
- Do not set OpenAI-style `reasoningEffort`; Nemotron Super reasoning is
  controlled via the NIM system prompt (`/no_think`), not that knob

See [`docs/doctoring/opencode-nvidia-nim-contract.md`](docs/doctoring/opencode-nvidia-nim-contract.md)
and [`tests/test_opencode_nvidia_nim_contract.py`](tests/test_opencode_nvidia_nim_contract.py).

## Code-owner review gates — disabled (on hold)

As of 2026-08-04, code-owner review requirements (`require_code_owner_reviews` in branch
protection, `require_code_owner_review` in rulesets) are disabled across the ContextualWisdomLab
org: there is a single maintainer (solo developer), so a code-owner approval gate can never be
satisfied. This is ON HOLD until the org has multiple maintainers — do NOT re-enable these
settings or add CODEOWNERS-based merge gates before then.
