---
name: cloture
description: Anytime-valid multi-agent consensus stopping, correlated-panel n_eff auditing, sequential agent/model racing, and active prediction-powered inference (Active-PPI++). Use when building or optimizing multi-agent swarms, Best-of-N / self-consistency loops, LLM-as-a-judge evaluation panels, A/B agent benchmarks, or live streaming agent guardrails.
---

# Cloture Skill

Use `cloture` whenever a codebase runs **multiple agents, parallel LLM rollouts, LLM-as-a-judge panels, or continuous agent evaluations** and needs:

1. **Early stopping with finite-sample guarantees (46–95% fewer rollouts/tokens)** instead of fixed `N=8..64` loops or unsafe heuristic consensus stopping.
2. **Correlated-panel diagnostics (`n_eff`, Condorcet gap, LOO judge pruning)** so you don't pay for 9 judges that only deliver ~2 effective independent votes.
3. **Realized-null vs. population-null safety** under shared-prefix or multi-prompt correlated rollouts.
4. **Anytime-valid sequential A/B racing & Active-PPI++** combining cheap LLM judges with sparse human/ground-truth audits (5–25× cheaper evals, zero p-hacking).
5. **Automatic poisoning quarantine (`SafeFuse`) & recall-controlled trajectory abort cascades (`AbortCascade`)**.

## Quick Decision Table

| Task | Tool / API | Why |
|---|---|---|
| Stop Best-of-N / multi-agent voting over arbitrary answers (code, SQL, JSON) | `cloture.CiteStopper(delta=0.05)` or `await cloture.consensus(...)` | Category-agnostic anytime-valid certification (`Type-I <= delta`); cancels in-flight calls immediately |
| Stop Best-of-N when LLM logprobs / answer distributions are available | `cloture.AscdStopper(k=4, delta=0.05)` | Grey-box `ASC-D` uses 46–95% fewer trajectories than label-only stopping |
| Audit an LLM-as-a-judge or multi-agent panel for correlated errors | `cloture.audit_panel(votes, gold)` or `cloture audit --file panel.json` | Reports Kish & Eigenvalue `n_eff`, Condorcet gap, permutation p-value, and redundant agents to drop |
| Certify deployment-level accuracy across prompts (not just one prompt) | `cloture.PopulationCert(delta=0.05)` | Prevents correlated single-unit misfires; computes optimal rollouts per prompt |
| Race `M` candidate prompts, models, or agents | `cloture.AgentRace(n_arms=M, delta=0.05, paired=True)` | Anytime-valid FWER control + LUCB adaptive task allocation + Wang-Ramdas `e_bh_discoveries` |
| Estimate true pass rate with 100% LLM judge + 10% human audits | `cloture.ActivePPI(alpha=0.05, target_audit_rate=0.10)` | Anytime-valid CS with online affine debiasing; auto-protects against biased or adversarial judges (`lambda -> 0`) |
| Fuse live heterogeneous agents with poisoning protection | `cloture.SafeFuse(n_agents=K)` | Scale-free BOA + per-agent Ville E-process auto-quarantine |
| Abort doomed multi-turn agent trajectories early | `cloture.AbortCascade.calibrate(...)` | Conformal recall-controlled multi-stage thresholds (+ `OnlineConformal` 4-rate AgACI) |
| Merge distributed calibration / e-process evidence across workers | `cloture.EvidenceSketch` | `712`-byte associative, commutative wire monoid with Leave-One-Out subtraction |

## CLI & MCP Usage

```bash
# Pipe newline/JSONL votes or probability vectors for instant anytime-valid stopping
cat rollouts.jsonl | cloture stop --mode cite --delta 0.05
cat logprobs.jsonl | cloture stop --mode ascd --k 4 --delta 0.05

# Audit a judge panel JSON ({"votes": [[...]], "gold": [...], "strata": [...]})
cloture audit --file panel.json --perms 2000

# Stream paired agent scores for sequential multi-arm winner certification
cat race_rounds.jsonl | cloture race --delta 0.05 --paired

# Run sequential Active-PPI++ on JSONL ({"pred": 0.9, "audited": true, "gold": 1.0, "prob": 0.1})
cat eval_stream.jsonl | cloture ppi --alpha 0.05 --min-prob 0.05

# Run built-in sub-microsecond benchmark suite on your CPU
cloture bench

# Start zero-dependency JSON-RPC 2.0 MCP server over stdio for Claude Code / Codex / Cursor
cloture mcp
```

## Python Drop-In Pattern

```python
import asyncio
import cloture

# Stop parallel LLM/agent rollouts the instant consensus is mathematically certified
result = await cloture.consensus(
    spawn_fn=lambda i: call_agent_async(prompt, seed=i),
    delta=0.05,
    max_calls=32,
    concurrency=8,
)
print(result.winner, f"certified in {result.n_samples}/32 calls!")
```

## Statistical Integrity Rules

- **Never** treat a single-prompt `CiteStopper` / `AscdStopper` certificate as a population-level claim across prompts when rollouts are correlated; use `PopulationCert` for deployment-level certification.
- **Never** drop `min_audit_prob` to `0` in `ActivePPI`; every item must retain a strictly positive predictable audit probability (`>= 0.02` recommended) for exact unbiasedness.
- **Always** inspect `panel.independence_ratio` (`n_eff_kish / k_judges`): if `< 0.5`, prune redundant correlated judges (`loo_delta_acc >= 0` or `loo_delta_neff >= 0`) rather than adding more similar models.
