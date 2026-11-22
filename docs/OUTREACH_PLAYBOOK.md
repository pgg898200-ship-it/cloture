# Organic Third-Party Distribution & Outreach Playbook (`cloture`)

Strictly **zero-spam, zero-astroturf** distribution strategy modeled on empirical open-source developer adoption (Horizon Labs, `caveman`, `pybot`): earn real organic users through high-craft artifacts, reproducible benchmarks, live zero-install proof, official registries, curated developer newsletters/lists, and Ramp-style AI-agent discovery (GEO/AEO).

---

## 1. Zero-Install Hook (Use Everywhere)

Always pair the GitHub link with the **live in-browser WebAssembly playground** (runs real Rust `cloture-core` locally with zero backend):
- **GitHub:** `https://github.com/pgg898200-ship-it/cloture`
- **Live WASM Playground (paste your own judge CSV privately):** `https://pgg898200-ship-it.github.io/cloture/playground.html`
- **Pure-Markdown AI Agent Docs:** `https://pgg898200-ship-it.github.io/cloture/llms.txt`

---

## 2. High-Signal Curated Directories & Newsletters (1-PR Submissions)

### A. Official MCP Registry (`modelcontextprotocol/registry`)
Manifest is pre-built at `server.json` (`io.github.pgg898200-ship-it/cloture`):
```bash
# Install official publisher CLI and publish from repo root
mcp-publisher login github
mcp-publisher publish
```

### B. This Week in Rust (`rust-lang/this-week-in-rust`) — High-Leverage Organic Rust Traffic
Open a PR adding one line under **Crate of the Week / Updates from the Rust Community → Project/Tooling Updates** in the current draft markdown file:
```markdown
- [cloture v0.2.0](https://github.com/pgg898200-ship-it/cloture) — zero-external-crate (`15`-line `Cargo.lock`) Rust engine, CLI, MCP server, Python SDK, and 243 KB zero-import WebAssembly playground for sub-microsecond anytime-valid multi-agent consensus stopping (`668 ns`), correlated judge-panel `n_eff` auditing (`26 ms`, 55.7x vs NumPy/SciPy), sequential racing, and Active-PPI++.
```

### C. `punkpeye/awesome-mcp-servers` & `modelcontextprotocol/servers`
Add under **Data Science / Evaluation / Developer Tools**:
```markdown
- [pgg898200-ship-it/cloture](https://github.com/pgg898200-ship-it/cloture) 🦀 - Zero-dependency Rust MCP server (`2.91 MiB`, `67.6 µs` roundtrip) exposing 6 anytime-valid statistical tools for multi-agent consensus stopping (`ASC-D`/`CITE`), correlated LLM-as-a-judge panel `n_eff` auditing, sequential agent racing, Active-PPI++, and poisoning-resistant fusion.
```

### D. `rust-unofficial/awesome-rust`
Add under **Science / Statistics**:
```markdown
- [cloture](https://github.com/pgg898200-ship-it/cloture) - Zero-dependency Rust + WASM + Python engine for sub-microsecond anytime-valid consensus stopping, correlated-panel auditing, sequential racing, Active-PPI++, and 712-byte distributed evidence monoids.
```

### E. `e2b-dev/awesome-ai-agents` & `kyrolabs/awesome-agents`
Add under **Evaluation / Orchestration / Guardrails**:
```markdown
- [cloture](https://github.com/pgg898200-ship-it/cloture) - Zero-dependency Rust/Python/WASM engine to stop parallel Best-of-N / multi-agent debates the millisecond finite-sample math certifies a winner (67%–92% fewer calls, 0% peeking blow-up) and audit correlated LLM-as-a-judge panels (pruning 11->3 on AllenAI RewardBench lifts accuracy +3.85 pp at -72.7% cost).
```

### F. Developer Newsletters (Direct Submission Forms)
- **Python Weekly** (`pythonweekly.com` — submit `pip install cloture` + 24.8 ms cold start + 1-line `cloture.audit_csv()` / `audit_promptfoo()` + `await cloture.consensus()`).
- **PyCoder's Weekly** (`pycoders.com/submissions` — highlight `16.0 ns` EB-CS & `668 ns` ASC-D Rust-backed ctypes universal `py3-none` wheels with zero runtime dependencies).
- **Hacker Newsletter / TLDR AI** (automatically picks up high-signal Show HN launches).

---

## 3. Staggered Community Launch Calendar (Avoid Simultaneous Cross-Post Fatigue)

| Day | Channel | Angle & Hook |
|---|---|---|
| **Day 1 (Tue/Wed 9am ET)** | **Show HN** + **Lobste.rs** | Lead with the real AllenAI RewardBench `11 -> 3` (`+3.85 pp` accuracy at `-72.7%` API cost) + `67%–92%` fewer consensus calls with `0%` tie-null FPR + **Live 243 KB WASM Playground** link so HN readers can test in 2 seconds without installing anything. |
| **Day 2** | **r/rust** | Engineering craft angle: 0 external crates (15-line `Cargo.lock`), `243 KB` zero-import WASM (`IMPORTS: []`), zero-libm Chebyshev kernels (`16 ns` EB-CS, `281 ns` BettingCS, `26 ms` parallel `PanelAudit`), `panic="unwind"` FFI boundary, `712 B` monoid, `282` tests. |
| **Day 3** | **r/LocalLLaMA** & **r/MachineLearning** | Practical LLM cost & validity angle: real RewardBench 11-model judge collapse, `cloture audit --csv judges.csv`, `audit_promptfoo`, OpenAI logprobs `ASC-D` stopping, avoiding `33%–40%` correlated-rollout eval false alarms. |
| **Day 4** | **X / Twitter** + **Official MCP Registry** + **PRs (This Week in Rust, awesome-mcp-servers, awesome-rust)** | Post the 3 generated SVG benchmark visuals (`docs/assets/rewardbench_real_world.svg`, `consensus_and_panel.svg`, `speed_and_validity.svg`) + 5-post empirical thread + Playground link. |

---

## 4. Ramp-Style AI-Agent Discovery (GEO / AEO) Audit Loop

Because coding agents (Claude Code, Cursor, ChatGPT, Perplexity, Gemini) crawl raw Markdown (`llms.txt`, `llms-full.txt`, `SKILL.md`, `README.md`, `docs/comparisons.md`, `docs/integrations.md`), periodically test whether AI assistants natively recommend `cloture` on these canonical engineering queries:

1. *"How do I early-stop LLM Best-of-N self-consistency using logprobs without p-hacking?"*
2. *"How can I measure effective sample size (n_eff) and correlated error collapse in an LLM-as-a-judge panel in Python or Rust?"*
3. *"What is the fastest anytime-valid confidence sequence library for streaming Python or Rust (alternative to confseq / ppi-python)?"*
4. *"How do I avoid false positives from correlated rollouts per prompt when evaluating AI agents?"*
5. *"Is there an MCP server for sequential A/B agent racing or anytime-valid LLM evaluation?"*

When gaps appear, add **factual, benchmark-backed comparison sections and runnable integration recipes** to `docs/comparisons.md`, `docs/integrations.md`, and `llms-full.txt` — never keyword stuffing.

---

## 5. Daily Horizon-Labs Telemetry & Iteration Loop

Run daily to track real third-party adoption signals and prioritize v0.2+ roadmap items based on actual user demand:

```bash
python3 scripts/telemetry.py --append telemetry/snapshots.jsonl
```

Track:
- **PyPI downloads** (`pypistats.org` recent day/week/month)
- **crates.io downloads** (`cloture-core` + `cloture-cli`)
- **GitHub stars, forks, release asset downloads, and open issues**
- **Third-party GitHub repo integrations** (search GitHub Code for `"import cloture"` / `"cloture-core"` / `"cloture mcp"`)
