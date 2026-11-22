# Security Policy

## Scope

`cloture` is designed for high-integrity production multi-agent pipelines:
- **Zero third-party crate/package dependencies** across Rust (`Cargo.lock`), Python (`pyproject.toml`), and WebAssembly (`js/cloture.mjs`), eliminating transitive supply-chain risk.
- **Panic-safe C ABI** (`panic = "unwind"` with `catch_unwind` at every FFI boundary) and bounded memory allocation caps across CLI, MCP JSON-RPC, and FFI constructors.
- **SHA-256 verified release artifacts** (`SHA256SUMS.txt`) verified fail-closed by `scripts/install.sh`.

## Reporting a Vulnerability or Statistical Soundness Bug

Please report security issues (memory safety, FFI UB, DoS via crafted JSON/wire bytes, or statistical validity violations under valid null hypotheses) via **GitHub Private Security Advisories**:

`https://github.com/pgg898200-ship-it/cloture/security/advisories/new`

Please include a minimal reproduction input (JSON payload, C/Python/JS snippet, or Monte Carlo seed) and expected vs. observed behavior.
