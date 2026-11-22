"""Comprehensive integration tests for cloture Python SDK, C ABI, CLI, and MCP server."""
from __future__ import annotations

import asyncio
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import cloture
from cloture import (
    AbortCascade,
    ActivePPI,
    AgentRace,
    AscdStopper,
    BettingCS,
    CiteStopper,
    EmpiricalBernsteinCS,
    EvidenceSketch,
    OnlineConformal,
    OnlinePanelTracker,
    PairedEProcess,
    PopulationCert,
    SafeFuse,
    audit_panel,
    consensus,
    consensus_dist,
    e_bh_discoveries,
    extract_logprob_distribution,
    race_agents,
)

CLI_BIN = ROOT / "target" / "release" / ("cloture.exe" if sys.platform == "win32" else "cloture")


def test_import_cold_latency_under_50ms():
    code = (
        "import sys, time; sys.path.insert(0, %r); "
        "t0 = time.perf_counter(); import cloture; "
        "print((time.perf_counter() - t0) * 1000.0)"
    ) % str(ROOT / "python")
    out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    ms = float(out)
    limit = 250.0 if sys.platform == "win32" else 50.0
    assert ms < limit, f"Cold import took {ms:.2f} ms (limit {limit:.0f} ms)"


def test_ascd_and_cite_stopping_and_deterministic_lock():
    stopper = AscdStopper(k=4, delta=0.05, max_budget=16)
    dec = None
    for _ in range(16):
        dec = stopper.update([0.85, 0.08, 0.04, 0.03])
        if dec.certified:
            break
    assert dec is not None
    assert dec.certified
    assert dec.winner == 0
    assert dec.n_samples <= 12

    # Batch zero-copy NumPy fast path
    stopper.reset()
    arr = np.tile(np.array([0.05, 0.88, 0.04, 0.03], dtype=np.float64), (20, 1))
    dec_b = stopper.update_batch(arr)
    assert dec_b.certified
    assert dec_b.winner == 1
    assert dec_b.n_samples < 20

    # Category-agnostic CITE preserves exact label_id (not just slot 0!)
    cite = CiteStopper(delta=0.05, max_budget=12)
    cdec = None
    for _ in range(12):
        cdec = cite.update(999_123)
        if cdec.certified:
            break
    assert cdec is not None
    assert cdec.certified
    assert cdec.winner == 999_123
    assert cdec.leader == 999_123

    # Deterministic budget lock: 6 votes out of max_budget=10 -> challenger can get at most 4
    cite_lock = CiteStopper(delta=1e-9, max_budget=10)
    for _ in range(6):
        ldec = cite_lock.update(77)
    assert ldec.certified
    assert ldec.deterministic_lock
    assert ldec.scope == "deterministic_budget"
    assert ldec.winner == 77


def test_population_cert_prevents_correlated_rollout_fallacy():
    rng = np.random.default_rng(7)
    # Null across units: theta_q ~ N(0, 0.5) clipped to [-0.8, 0.8], 32 rollouts per unit
    pop = PopulationCert(delta=0.05)
    for _ in range(40):
        unit_mean = float(np.clip(rng.normal(0.0, 0.4), -0.8, 0.8))
        rollouts = np.clip(unit_mean + rng.normal(0.0, 0.15, size=32), -1.0, 1.0)
        dec = pop.add_unit(rollouts)
    assert dec.n_units == 40
    assert dec.total_rollouts == 40 * 32
    assert dec.rho_hat > 0.5
    assert dec.n_eff_within < 5.0
    assert pop.optimal_rollouts(1.0, 1.0) >= 1

    # Strong positive signal across units certifies cleanly
    pop_pos = PopulationCert(delta=0.05)
    for _ in range(30):
        dec = pop_pos.add_unit(np.full(8, 0.65, dtype=np.float64))
    assert dec.certified
    assert dec.theta_hat > 0.6


def test_panel_audit_and_online_tracker():
    rng = np.random.default_rng(42)
    n_items, k_judges = 300, 9
    gold = rng.integers(0, 2, size=n_items, dtype=np.uint32)
    # Shared difficulty factor creates high error correlation (~2 effective votes)
    latent = rng.normal(size=n_items)
    votes = np.zeros((n_items, k_judges), dtype=np.uint32)
    for j in range(k_judges):
        err = (0.75 * latent + 0.66 * rng.normal(size=n_items)) > 0.7
        votes[:, j] = np.where(err, 1 - gold, gold)

    rep = audit_panel(votes, gold, n_condorcet_sims=500, n_perms=500, n_boot=200, seed=42)
    assert rep.k_judges == 9
    assert rep.n_items == 300
    assert rep.n_eff_kish < 5.0
    assert rep.n_eff_eigen < 5.0
    assert rep.mean_phi > 0.15
    assert rep.condorcet_gap > 0.0
    assert rep.permutation_p_value < 0.05
    assert len(rep.phi_matrix) == 9 and len(rep.phi_matrix[0]) == 9
    assert len(rep.loo_delta_acc) == 9 and len(rep.loo_delta_neff) == 9

    tracker = OnlinePanelTracker(k_judges)
    for i in range(n_items):
        errs = [1 if votes[i, j] != gold[i] else 0 for j in range(k_judges)]
        tracker.observe(errs)
    assert abs(tracker.n_eff_kish() - rep.n_eff_kish) < 1e-6
    assert abs(tracker.n_eff_eigen() - rep.n_eff_eigen) < 1e-6


def test_confidence_sequences_racing_ebh_and_ppi():
    rng = np.random.default_rng(123)
    xs = rng.binomial(1, 0.72, size=400).astype(np.float64)

    eb = EmpiricalBernsteinCS(alpha=0.05)
    b_eb = eb.update_batch(xs)
    assert b_eb.lower <= 0.72 <= b_eb.upper

    bet = BettingCS(alpha=0.05, grid_points=201)
    b_bet = bet.update_batch(xs)
    assert b_bet.lower <= 0.72 <= b_bet.upper
    assert b_bet.width <= b_eb.width + 0.05

    pe = PairedEProcess(alpha=0.05)
    a = np.clip(0.8 + rng.normal(0.0, 0.1, size=100), 0.0, 1.0)
    b = np.clip(0.5 + rng.normal(0.0, 0.1, size=100), 0.0, 1.0)
    pr = pe.update_batch(a, b)
    assert pr.winner == 0
    assert pr.anytime_p_value < 0.05

    race = AgentRace(n_arms=4, delta=0.05, paired=True)
    winner = None
    for _ in range(150):
        shared = rng.normal(0.0, 0.15)
        scores = np.clip(np.array([0.85, 0.55, 0.50, 0.45]) + shared + rng.normal(0.0, 0.05, 4), 0.0, 1.0)
        winner = race.observe_paired(scores)
        if winner is not None:
            break
    assert winner == 0
    assert race.active_count() == 1

    # e-BH discoveries
    disc = e_bh_discoveries([0.5, 50.0, 100.0, 0.2, 40.0], alpha=0.05)
    assert set(disc) == {1, 2, 4}

    # Active-PPI++
    ppi = ActivePPI(alpha=0.05, target_audit_rate=0.15, min_audit_prob=0.05)
    for _ in range(600):
        y = float(rng.binomial(1, 0.65))
        f = float(np.clip(0.8 * y + 0.1 + rng.normal(0.0, 0.1), 0.0, 1.0))
        pi = ppi.suggest_prob(f)
        aud = bool(rng.random() < pi)
        res = ppi.observe(f, aud, y if aud else 0.0, pi)
    assert res.lower <= 0.65 <= res.upper
    assert res.lambda_hat > 0.3
    assert res.variance_reduction > 1.2


def test_safe_fuse_cascade_conformal_and_sketch():
    # SafeFuse quarantines poisoned agent 3
    fuse = SafeFuse(n_agents=4, quarantine_alpha=0.01, fallback_alpha=0.05)
    for _ in range(60):
        val, fb = fuse.predict([0.8, 0.82, 0.79, 0.05])
        assert math.isfinite(val)
        fuse.update([0.10, 0.11, 0.10, 0.95], baseline_loss=0.15)
    assert fuse.is_quarantined(3)
    assert not fuse.is_quarantined(0)
    w = fuse.weights()
    assert len(w) == 4 and abs(sum(w) - 1.0) < 1e-6 and w[3] == 0.0

    # AbortCascade
    rng = np.random.default_rng(99)
    n_cal = 400
    succ = rng.binomial(1, 0.5, size=n_cal).astype(np.uint8)
    scores = np.zeros((n_cal, 3), dtype=np.float64)
    for i in range(n_cal):
        base = 0.78 if succ[i] else 0.22
        scores[i] = np.clip(base + rng.normal(0.0, 0.12, size=3), 0.0, 1.0)
    cascade = AbortCascade.calibrate(scores, succ, target_recall=0.92)
    thresholds = cascade.thresholds()
    assert len(thresholds) == 3
    assert cascade.should_abort(0, 0.01)
    assert not cascade.should_abort(0, 0.99)

    # OnlineConformal
    oc = OnlineConformal(target_alpha=0.10)
    for i in range(200):
        oc.update_pvalue((i % 20) / 20.0)
    assert 0.0 <= oc.alpha_t <= 1.0
    assert 0.7 <= oc.empirical_coverage <= 1.0

    # EvidenceSketch monoid + LOO subtraction + <1KB wire format
    s1, s2 = EvidenceSketch(), EvidenceSketch()
    for i in range(100):
        s1.observe(i / 100.0, 0.1)
        s2.observe((i + 50) / 200.0, -0.05)
    wire = s1.to_bytes()
    assert len(wire) == 712
    s1_rt = EvidenceSketch.from_bytes(wire)
    assert abs(s1_rt.mean() - s1.mean()) < 1e-12
    assert abs(s1_rt.e_value() - s1.e_value()) < 1e-10

    merged = EvidenceSketch.from_bytes(wire)
    merged.merge(s2)
    merged.subtract(s1)
    assert abs(merged.mean() - s2.mean()) < 1e-9
    assert abs(merged.e_value() - s2.e_value()) < 1e-9


def test_async_wrappers_and_logprobs():
    dist = extract_logprob_distribution({"A": -0.1, "B": -2.5, "C": -4.0}, ["A", "B", "C"])
    assert abs(sum(dist) - 1.0) < 1e-9 and dist[0] > 0.85

    async def _run():
        async def spawn_ans(idx: int) -> str:
            await asyncio.sleep(0.001)
            return "42" if idx % 10 != 9 else "41"

        res = await consensus(spawn_ans, delta=0.05, max_calls=32, concurrency=4)
        assert res.certified and res.winner == "42" and res.n_samples < 32

        async def spawn_dist(idx: int):
            await asyncio.sleep(0.001)
            return [0.90, 0.05, 0.03, 0.02]

        dres = await consensus_dist(
            spawn_dist, extract_probs=lambda x: x, labels=["A", "B", "C", "D"], delta=0.05, max_calls=20
        )
        assert dres.certified and dres.winner == "A" and dres.n_samples <= 10

        async def eval_arm(arm: int, task: int) -> float:
            await asyncio.sleep(0.0005)
            return 0.9 if arm == 1 else 0.4

        winner, evals, active = await race_agents(eval_arm, n_arms=3, tasks=list(range(60)), delta=0.05)
        assert winner == 1 and active == [1] and evals < 3 * 60

    asyncio.run(_run())


def test_cli_and_mcp_server():
    assert CLI_BIN.exists(), f"Missing binary at {CLI_BIN}"

    # Cold CLI startup check
    t0 = time.perf_counter()
    ver = subprocess.check_output([str(CLI_BIN), "--version"], text=True).strip()
    dt_ms = (time.perf_counter() - t0) * 1000.0
    assert ver.startswith("cloture ")
    assert dt_ms < (250.0 if sys.platform == "win32" else 50.0)

    # Streaming stop subcommand
    votes = "\n".join(["42"] * 25) + "\n"
    out = subprocess.check_output(
        [str(CLI_BIN), "stop", "--mode", "cite", "--delta", "0.05"], input=votes, text=True
    )
    payload = json.loads(out)
    assert payload["certified"] is True
    assert payload["winner"] == "42"
    assert payload["n_samples"] < 25

    # MCP stdio JSON-RPC handshake + tools/list + tools/call
    reqs = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "cloture_mode_stop",
                "arguments": {
                    "mode": "ascd",
                    "samples": [[0.9, 0.05, 0.05]] * 10,
                    "delta": 0.05,
                },
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "cloture_audit_panel",
                "arguments": {
                    "votes": [[0, 0, 0], [1, 1, 1], [0, 0, 0], [1, 1, 0], [0, 1, 0]] * 10,
                    "gold": [0, 1, 0, 1, 1] * 10,
                    "n_perms": 200,
                },
            },
        },
    ]
    mcp_in = "\n".join(json.dumps(r) for r in reqs) + "\n"
    mcp_out = subprocess.check_output([str(CLI_BIN), "mcp"], input=mcp_in, text=True)
    lines = [json.loads(ln) for ln in mcp_out.strip().splitlines()]
    assert len(lines) == 4  # notification yields no response
    assert lines[0]["result"]["serverInfo"]["name"] == "cloture"
    assert len(lines[1]["result"]["tools"]) == 6
    mode_res = json.loads(lines[2]["result"]["content"][0]["text"])
    assert mode_res["certified"] is True and mode_res["winner"] == 0
    panel_res = json.loads(lines[3]["result"]["content"][0]["text"])
    assert panel_res["k_judges"] == 3 and panel_res["n_items"] == 50
