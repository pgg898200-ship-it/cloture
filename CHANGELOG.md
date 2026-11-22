# Changelog

All notable changes to `cloture` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-11-22

Major performance, WebAssembly, real-world frontier benchmark, and drop-in adapter release — still **0 external Rust crates** and **0 Python runtime dependencies**.

### Performance Breakthroughs (All Verified in `benchmarks/results.json`)
- **`EmpiricalBernsteinCS`:** `16.0 ns/step` batch-100k (`29.7 ns/step` Rust single-step, `3.68 µs` Python stream) — **`25.7×` faster than `confseq` C++ (`410 ns`)**, **`297×` faster than `expectation` in Python streaming**, and **`1.10×` faster than `expectation` (`17.6 ns`) even on single-call offline batches** (eliminating our previous batch speed caveat). Implemented via incremental exact `ln(t+1)`, zero-libm `< 6e-17` `psi_e(lam)`, and a 4-wide SIMD-decoupled steady-state batch kernel with exact bit-for-bit sequential parity.
- **`BettingCS(G=201)`:** `281.4 ns/step` Rust / `330.2 ns/step` Py batch (**`415×` faster than `confseq.hedged_cs`** `137 µs`, while tighter `0.095` vs `0.119`) via exact running-intersection active-cell pruning and branchless IEEE-754 `fast_ln_1p`.
- **`PanelAudit (9 × 1,000, 2k perms / 2k MC / 1k boot)`:** `26.0 ms` vs NumPy/SciPy `1,446.7 ms` (**`55.7×` faster**, up from `4.8×`, `2.66e-15` max diff) via exact `O(K²)` Poisson-Binomial DP on binary buckets (zero MC variance), L1-resident multiclass frequency tables (`C <= 16`), in-place Fisher-Yates permutations, bitmask-deduplicated bootstrap CI, and zero-dep `std::thread::scope` multi-core chunking (single-threaded on `wasm32`).
- **`AscdStopper(K=4)`:** `667.8 ns/step` Py batch / `683.9 ns/step` Rust (**`85.5×` faster than NumPy** `57.1 µs`, up from `28.9×`, `25/25` exact stop parity) via `(a,b)/(b,a)` antisymmetry + tiered SSE2 Chebyshev log pairs (`add_bet_grid_pair`) and dominated-challenger pruning.
- **Python SDK Streaming:** Added `slots=True` on result dataclasses and cached bound `ctypes` function handles (`import cloture` `24.8 ms`, `OnlineConformal` `59.7 ns` Rust / `3,294×` faster than `online-cp`, `cloture mcp` `67.6 µs` median).

### Added
- **Zero-Import WebAssembly & JS/TS SDK (`js/cloture.mjs`, `js/cloture.d.ts`, `js/package.json`):** Compiles `cloture-core` to **`243 KB` raw / `81.4 KB` gzipped** standalone WebAssembly with **`0` host imports** for browsers, Node.js, Deno, Bun, and Cloudflare Workers, including `AbortController`-aware async `consensus()`.
- **Live Interactive Browser Playground (`site/playground.html`):** Run real Rust `cloture-core` locally in browser WASM (zero backend upload) — paste your own judge CSV, click **"Real RewardBench (11 LLMs × 2,985)"**, race `ASC-D`/`CITE` vs. naive peeking, or stress-test `ActivePPI` under adversarial judge inversion.
- **Real-World AllenAI RewardBench Study (`benchmarks/real_world_rewardbench.py`, `docs/REAL_WORLD_EVAL.md`):** Verified on `2,985` real human-verified items across `11` frontier models (`23` strata): 7 generative LLMs collapse to `2.21` effective votes (`within-family φ = 0.504` vs `cross-family φ = 0.317`), pruning `11 → 3` raises accuracy **`+3.85 pp`** (`92.03% → 95.88%`) at **`-72.7%` cost**, paired `AgentRace` saves **`80.4%`** of evals (`50/50` exact winner), and `ActivePPI` achieves `100%` anytime coverage (`10.46×` variance reduction).
- **Native Zero-Dep CLI CSV Auditing (`cloture audit --csv <path|-> --gold-col gold --strata-col domain`):** Built-in RFC-4180 CSV parser with judge-name reporting and prunable-judge highlights.
- **Drop-In Python Adapters (`python/cloture/adapters.py`):** Zero-dependency `audit_csv`, `audit_jsonl`, `audit_promptfoo`, and `extract_openai_logprobs`.
- **Distribution & Discovery:** Homebrew formula (`Formula/cloture.rb`), rate-limit-free `scripts/install.sh` (using `releases/latest` redirect), publication-grade SVG benchmark charts (`docs/assets/*.svg`), `robots.txt`/`sitemap.xml`/Schema.org JSON-LD, `CONTRIBUTING.md`, `SECURITY.md`, and `docs/OUTREACH_PLAYBOOK.md`.
- **Expanded Verification:** **`282 / 282` passing tests** (`65` Rust core + `3` Rust CLI + `213` Python/CLI/MCP/adversarial breaker + Node.js WASM suite).

## [0.1.0] - 2026-11-22

Initial public release of `cloture` across GitHub Releases, PyPI (`pip install cloture`), and GitHub Pages.
