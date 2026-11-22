# `cloture` — Autonomous Exploration & Build State Log (`COMPANY_STATE.md`)

**Date:** 2026-11-22  
**Status:** Release Candidate `v0.1.0` — All 266 tests passing (`100%`), all 7 competitive benchmark suites executed and verified in `benchmarks/results.json`, 100% anonymized, zero external dependencies.

---

## 1. Strategic Niche & Thesis

**Problem Identified:** Modern low-latency, high-volume multi-agent AI stacks (parallel Best-of-N coding/reasoning swarms, LLM-as-a-judge evaluation panels, online agent A/B racing, live multi-agent ensembles) face an unsolved statistical-systems bottleneck:
- Existing sequential-testing libraries (`confseq`, `expectation`, `ppi-python`, `online-cp`, `bayesbench`) are either **slow** (`100 µs–30 s`), **fixed-horizon** (breaking under continuous peeking), **narrow** (lacking late-2026 multi-agent frontiers like grey-box `ASC-D`, unbounded `CITE`, correlated-panel $n_{\text{eff}}$ diagnostics, realized-vs-population rollout certification, poisoning quarantine, and distributed monoids), or **heavy** (requiring Boost, SciPy, scikit-learn, statsmodels, or 2s cold imports).
- Meanwhile, multi-agent systems waste **40%–92%** of LLM tokens/calls on fixed-`N` sampling, suffer **28%–54%** false-positive explosions under naive peeking, pay for **9 judges delivering ~2.2 effective votes**, and misclassify correlated within-prompt rollouts as population proof at **33%–40%** error.

**Product Positioning (`cloture`):**
> *End the multi-agent debate the millisecond the math proves who won — and catch when your judge panel is 2 correlated votes wearing 9 trench coats.*

**Defensible Moat (Double Edge):**
1. **Frontier Statistical Depth:** Unifies 5 cutting-edge 2024–Nov 2026 non-asymptotic statistical frameworks (`ASC-D`, `CITE` + Kraft-IUT + deterministic lock, `PopulationCert`, correlated-panel `PanelAudit` + LOO pruning, `EmpiricalBernsteinCS`/`BettingCS`/`AgentRace`/`e-BH`/online affine `ActivePPI`, `SafeFuse` unclamped Fan-inequality quarantine, `AbortCascade`/`AgACI`, and `712`-byte `EvidenceSketch`) in one mathematically coherent engine.
2. **Systems & Distribution Edge:** Zero external Rust crates (`15`-line `Cargo.lock`), zero Python runtime dependencies (`26.5 ms` cold import), single `2.47 MiB` static binary (`3.05 ms` CLI cold start), native JSON-RPC 2.0 MCP server (`76.7 µs` roundtrip), C ABI (`include/cloture.h`), and built-in `cloture bench` — beating existing Python/C++/Rust competitors by `4x–11,112x` on streaming/multi-agent workloads.

---

## 2. Autonomous Engineering & Adversarial Verification Log

| Milestone | Scope | Outcome |
|---|---|---|
| **M1 — Market & Namespace Verification** | Verified `cloture` HTTP 404 availability on PyPI & crates.io; surveyed real installed competitors (`confseq`, `ppi-python`, `expectation`, `online-cp`, `bayesbench`). | Clean namespace, orthogonal frontier scope. |
| **M2 — Zero-Dep Rust Core (`cloture-core`)** | Implemented `math.rs`, `mode.rs`, `panel.rs`, `betting.rs`, `race.rs`, `ppi.rs`, `fuse.rs`, `cascade.rs`, `sketch.rs`, `ffi.rs` (~11,000 LOC, `0` external crates). | Compiled with `-D warnings` (0 warnings); `55 / 55` Rust unit & Monte Carlo tests PASS. |
| **M3 — CLI, MCP Server, C ABI & Python SDK** | Built `cloture` static CLI (`stop`, `audit`, `race`, `ppi`, `bench`, `mcp`), `include/cloture.h`, and zero-dep `python/cloture/` (`_native.py` + `async_tools.py`). | `8 / 8` Python/CLI/MCP integration tests PASS; all 3 runnable demos verified. |
| **M4 — ActivePPI Affine Debiasing Upgrade** | Discovered uncentered `lam * f` PPI incurred `(1-pi)/pi * mu^2` burn-in variance and judge-bias variance penalty. Upgraded to strictly predictable centered affine proxy `y_hat = clamp(m_y + lam*(f - m_f), 0, 1)` with exact `[z_min, z_max]` support and Fan-damped step `lam = q/(1+q)`. | `6.29x` variance reduction on biased judges (`+0.15` bias), `99%` anytime coverage, `0.0000` `lambda_hat` on inverted judges. |
| **M5 — Independent Adversarial Breaker Audit** | Launched independent breaker subagent writing 203 torture tests (`tests/breaker/`). Exposed **10 real bugs** (4 Critical, 3 High, 2 Medium, 1 Low): `C1` (`SafeFuse` skewed-null clamping 96% FPR), `C2` (`AscdStopper` `K=2` tie-null 7.85% FPR), `C3` (`panic = "abort"` disabling `catch_unwind` + `usize::MAX` OOM `SIGABRT`), `C4` (4 Python batch buffer overreads), `H1` (`u64 >= 2^63` `CiteStopper` sentinel wrap), `H2` (`Json::as_u64` float truncation), `H3` (`max_val - min_val = Inf` EBCS span), `M1` (`EvidenceSketch` `1e154` `CompSum` overflow), `M2` (silent CLI flag fallbacks), `L1` (left-identity wire bytes). | **All 10 fixed at root cause** (including upgrading `SafeFuse` to an unclamped Fan-inequality supermartingale over `max(LOO mean, LOO median)` — simultaneously fixing `C1` from `96.3% -> 0.0–0.75%` FPR AND accelerating `SafeFuse` `1.6x` to `1.24 µs`!). Re-run result: **`203 / 203` breaker tests PASS** (`266 / 266` total). |
| **M6 — Finalized 7-Suite Competitive Benchmarks** | Re-ran full `benchmarks/run_benchmarks.py` against hardened release binary (`194.2s`, strict RFC 8259 JSON receipt). | Saved verbatim in `benchmarks/results.json`; all numbers audited. |
| **M7 — GEO/AEO Launch Package** | Authored pure-Markdown `README.md`, `docs/benchmarks.md`, `docs/comparisons.md`, `docs/ARCHITECTURE.md`, `SKILL.md`, `llms.txt`, `llms-full.txt`, and `deploy/cloudflare-agent-markdown-worker.js`. | Zero hype, radical honesty on scope boundaries, 100% anonymized. |

---

## 3. Verified Key Metrics (Snapshot from `benchmarks/results.json`)

- **Consensus Savings:** `67.2%` fewer calls vs. Fixed-64 (`91.8%` vs. Fixed-256) on Easy `ASC-D`, `72.3%` fewer vs. Fixed-256 on open-ended `K=200` `CITE`, `0.0%–0.5%` tie-null FPR (vs. `28.5%–30.5%` naive peeking); `28.9x` faster than NumPy (`2.21 µs` vs. `63.8 µs`).
- **Panel Audit (`9 x 1,000`):** `317.7 ms` (`4.8x` faster than NumPy/SciPy, `2.66e-15` max diff); pruning 2 redundant judges yields `+2.72 pp` higher accuracy at `-22.2%` judge API cost.
- **Correlated Rollouts:** `0.7%` false fire vs. `33.0%–40.0%` pooled tests and `99.3%` any-prompt single-unit misfire; `optimal_rollouts` allocation triples population power (`27.3% -> 88.0%`) at identical compute budget.
- **CS & Racing:** `88.9 ns` Rust EB-CS (`4.0x` faster batch than `confseq`, `219x` faster streaming than `expectation`); `BettingCS` `28.4x` faster & tighter than `confseq.hedged_cs`; `AgentRace` saves `75.0%` evals (`100%` acc, `0.0%` null FWER) and is `11,112x` faster than `bayesbench.BayesianRanker`.
- **Active-PPI++:** `99.0%` anytime coverage (`6.29x` variance reduction) vs. `ppi-python` collapsing to `71.0%` under peeking and naive judges getting `0.0%`.
- **OnlineConformal & Monoid:** `64.5 ns` Rust AgACI (`2,930x` faster than `online-cp` `SimpleJumper`, `94.4%` post-shift recovery vs. `68.0%` CRR); `712 B` `EvidenceSketch` merges in `76.3 ns` with `0.0` exact 1,000-worker & LOO error.

---

## 4. Horizon-Labs Organic Distribution & Launch Playbook (`pgg898200-ship-it/cloture`)

This repository follows a strictly **organic, zero-astroturf** distribution strategy modeled on empirical developer adoption:

1. **Automated Cross-Platform Releases (`.github/workflows/release.yml`):**
   - Tagging `v0.1.0` builds static musl/macOS/Windows CLI binaries + `SHA256SUMS.txt`, 5 universal `py3-none-<platform>` Python wheels (`scripts/build_wheel.py`) covering Python 3.9–3.14+, publishes to **PyPI via OIDC Trusted Publishing** (`pypa/gh-action-pypi-publish`), and optionally publishes `cloture-core` + `cloture-cli` to **crates.io**.
   - Verified one-line installer: `curl -fsSL https://raw.githubusercontent.com/pgg898200-ship-it/cloture/main/scripts/install.sh | sh`.
2. **AI-Agent Discovery & Pure-Markdown Docs (`.github/workflows/pages.yml` + `deploy/cloudflare-agent-markdown-worker.js`):**
   - GitHub Pages deploys `https://pgg898200-ship-it.github.io/cloture` serving both a fast static landing page (`site/index.html`) and raw Markdown/JSON endpoints (`/llms.txt`, `/llms-full.txt`, `/SKILL.md`, `/README.md`, `/docs/*.md`, `/benchmarks/results.json`, `/server.json`).
   - Official MCP Registry manifest (`server.json` for `io.github.pgg898200-ship-it/cloture`) ready for registration and Claude Code / Cursor (`claude mcp add cloture -- cloture mcp`).
3. **Drop-In Ecosystem Recipes & Authentic Community Launches:**
   - Copy-paste OpenAI/vLLM logprobs, judge-panel audit, MCP, CI gate, and C FFI recipes live in `docs/integrations.md`.
   - Zero-hype empirical launch posts live in `docs/LAUNCH_DRAFTS.md` (Show HN, r/LocalLLaMA, r/MachineLearning, r/rust, X/Twitter, MCP directories).
4. **Daily Evidence-Driven Telemetry Loop (`scripts/telemetry.py`):**
   - Run `python3 scripts/telemetry.py --append telemetry/snapshots.jsonl` to log public GitHub stars/forks/release downloads, PyPI `pypistats`, and crates.io download velocity and prioritize next releases from real user demand.
5. **Optional Next Frontier (Requires GPU / Live LLM API Budget):**
   - Run live end-to-end token-cost benchmarks against real vLLM / OpenRouter logprobs on AIME-2025, SWE-bench-Lite, and Chatbot Arena judge panels and append live API receipts to `benchmarks/results.json`.
