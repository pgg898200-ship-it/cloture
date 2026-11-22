# Changelog

All notable changes to `cloture` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-11-22

Initial public release of `cloture`.

### Added
- **Zero-dependency Rust core (`cloture-core`)** (`std` only, 0 external crates in `Cargo.lock`):
  - `mode::{AscdStopper, CiteStopper, PopulationCert}`: grey-box `ASC-D` stopping over probability vectors, category-agnostic unbounded Kraft-IUT `CITE` stopping with deterministic budget lock, and unit-level `PopulationCert` with `optimal_rollouts` allocation under correlated rollouts.
  - `panel::{PanelAudit, OnlinePanelTracker}`: Kish & Jacobi-eigenvalue effective sample size ($n_{\text{eff}}$), stratified Condorcet independence gap, exact permutation omnibus p-value, Krippendorff's $\alpha$, and leave-one-out (LOO) redundant-judge pruning.
  - `betting::{EmpiricalBernsteinCS, BettingCS, PairedEProcess}`: finite-sample anytime-valid confidence sequences and paired sequential e-processes.
  - `race::{AgentRace, e_bh_discoveries}`: multi-arm LUCB and paired e-process sequential racing with FWER control and Wang-Ramdas e-BH FDR control.
  - `ppi::ActivePPI`: online centered-affine prediction-powered confidence sequence with Fan-damped step (`lambda_hat -> 0` under adversarial or inverted judges).
  - `fuse::SafeFuse`: Scale-Free Bernstein Online Aggregation (BOA) with unclamped Fan-inequality Ville e-process quarantine over `max(LOO weighted mean, LOO median)`.
  - `cascade::{AbortCascade, OnlineConformal}`: Conformal Risk Control (CRC) multi-stage trajectory abort calibration and 4-rate `AgACI` adaptive conformal inference.
  - `sketch::EvidenceSketch`: fixed `712`-byte associative, commutative wire monoid (`b"CLTR"` header + FNV-1a 64-bit integrity checksum) with exact Leave-One-Out subtraction.
  - Panic-safe C ABI (`include/cloture.h`, `panic = "unwind"` with `catch_unwind` at every FFI boundary).
- **Single static CLI & JSON-RPC 2.0 MCP server (`cloture-cli`)**:
  - Commands: `cloture stop`, `cloture audit`, `cloture race`, `cloture ppi`, `cloture bench`, `cloture mcp`.
  - 6 MCP tools over stdio: `cloture_mode_stop`, `cloture_audit_panel`, `cloture_population_cert`, `cloture_agent_race`, `cloture_active_ppi`, `cloture_safe_fuse`.
- **Zero-dependency Python SDK (`python/cloture/`)**:
  - PEP 562 lazy-loading ctypes wrapper (`26.5 ms` cold import) with zero-copy NumPy pointer fast-path when NumPy is installed.
  - Async orchestration helpers: `consensus`, `consensus_dist`, `extract_logprob_distribution`, `race_agents`.
  - Universal `py3-none-<platform>` wheel builder (`scripts/build_wheel.py`) covering Python 3.9+.
- **Verification & Benchmarks**:
  - `266 / 266` passing tests (`55` Rust release tests, `8` Python/CLI/MCP integration tests, `203` independent adversarial statistical, FFI, and fuzzing breaker tests).
  - Reproducible 7-suite head-to-head benchmark harness (`benchmarks/run_benchmarks.py` + `benchmarks/results.json`) against `confseq`, `ppi-python`, `expectation`, `online-cp`, `bayesbench`, and NumPy/SciPy.
- **Distribution & Agent Discovery**:
  - GitHub Actions CI, cross-platform release builds + `SHA256SUMS.txt` + PyPI OIDC Trusted Publishing, GitHub Pages docs site, Official MCP Registry manifest (`server.json`), `llms.txt`, `llms-full.txt`, `SKILL.md`, and Cloudflare pure-Markdown worker.
