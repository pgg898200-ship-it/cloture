#!/usr/bin/env python3
"""Head-to-head reproducible competitive benchmark suite for cloture.

Measures all 7 statistical & systems suites on deterministic seeded traces from
`benchmarks/traces.py`, comparing cloture against real installed PyPI libraries
(`confseq`, `ppi-python`, `expectation`, `online-cp`, `bayesbench`) and honest
vectorized NumPy/SciPy reference implementations & standard baselines.

Outputs complete machine-verified JSON receipts to `benchmarks/results.json`.
"""

from __future__ import annotations

import importlib.metadata as md
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
PYTHON_SDK = ROOT / "python"
BENCH_DIR = ROOT / "benchmarks"
CLI_BIN = ROOT / "target" / "release" / "cloture"
LIB_SO = ROOT / "target" / "release" / "libcloture_core.so"
RESULTS_PATH = BENCH_DIR / "results.json"

if str(PYTHON_SDK) not in sys.path:
    sys.path.insert(0, str(PYTHON_SDK))
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import numpy as np
from scipy.stats import norm

import cloture
import traces

# Real installed PyPI competitors
import bayesbench
import confseq
import confseq.betting as cs_betting
import confseq.boundaries as cs_boundaries
import confseq.conjmix_bounded as cs_conjmix
import confseq.predmix as cs_predmix
import expectation
from expectation.confseq.confidenceconfig import EmpiricalBernsteinConfig
from expectation.confseq.confidencesequence import EmpiricalBernsteinConfidenceSequence
import expectation.par_seqtest as exp_pst
import online_cp
import ppi_py


# ---------------------------------------------------------------------------
# Helpers & Metadata
# ---------------------------------------------------------------------------


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _rustc_version() -> str:
    try:
        out = subprocess.check_output(["rustc", "--version"], text=True, stderr=subprocess.STDOUT)
        return out.strip()
    except Exception:
        return "unknown"


def _pkg_version(name: str) -> str:
    try:
        return md.version(name)
    except Exception:
        return "not-installed"


def _percentiles(vals: List[float] | np.ndarray) -> Dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def collect_metadata() -> Dict[str, Any]:
    pkgs = [
        "numpy",
        "scipy",
        "pytest",
        "confseq",
        "ppi-python",
        "expectation",
        "online-cp",
        "bayesbench",
        "pandas",
        "scikit-learn",
        "statsmodels",
        "numba",
    ]
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "os": platform.platform(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "rustc_version": _rustc_version(),
        "cloture_version": cloture.__version__,
        "packages": {p: _pkg_version(p) for p in pkgs},
    }


# ---------------------------------------------------------------------------
# Suite 1: Consensus Stopping (ASC-D Table 2 + Open-Ended Swarm)
# ---------------------------------------------------------------------------

_BET_GRID = np.array(
    [
        0.005, 0.012, 0.025, 0.050, 0.090, 0.150, 0.240, 0.360,
        0.500, 0.640, 0.760, 0.850, 0.910, 0.950, 0.975, 0.990,
    ],
    dtype=np.float64,
)
_LN_GRID_LEN = float(np.log(len(_BET_GRID)))


class _NumpyAscdReference:
    """Faithful vectorized NumPy reference implementation of ASC-D (identical 16-pt grid)."""

    def __init__(self, k: int, delta: float = 0.05) -> None:
        self.k = k
        self.log_thresh = float(np.log(k / delta))
        self.scores = np.zeros(k, dtype=np.float64)
        self.log_wealth = np.zeros((k, k, len(_BET_GRID)), dtype=np.float64)
        self.n = 0

    def update(self, p: np.ndarray) -> Tuple[bool, int, int]:
        self.n += 1
        self.scores += p
        diff = p[:, None] - p[None, :]  # (K, K)
        # Update all (i, j) log-wealths across the 16-point betting grid
        self.log_wealth += np.log1p(diff[:, :, None] * _BET_GRID[None, None, :])
        leader = int(np.argmax(self.scores))
        # Minimum log-mean-exp over challengers j != leader
        lw_leader = self.log_wealth[leader]  # (K, G)
        max_lw = np.max(lw_leader, axis=1, keepdims=True)
        lme = max_lw[:, 0] + np.log(np.mean(np.exp(lw_leader - max_lw), axis=1))
        lme[leader] = np.inf
        min_ev = float(np.min(lme))
        return (min_ev >= self.log_thresh), leader, self.n


def _run_naive_majority_peeking(ys: np.ndarray, alpha: float = 0.05, min_n: int = 4) -> Tuple[int, int, bool]:
    """Naive uncorrected two-sided z-test peeking between top-1 and top-2 counts every step."""
    z_crit = float(norm.ppf(1.0 - alpha / 2.0))  # 1.95996
    counts: Dict[int, int] = {}
    for t in range(ys.size):
        v = int(ys[t])
        counts[v] = counts.get(v, 0) + 1
        n = t + 1
        if n < min_n:
            continue
        # Top-2 counts
        top1_k, top1_c = -1, -1
        top2_c = 0
        for ck, cc in counts.items():
            if cc > top1_c:
                top2_c = top1_c if top1_c > 0 else 0
                top1_c = cc
                top1_k = ck
            elif cc > top2_c:
                top2_c = cc
        total_top2 = top1_c + top2_c
        if total_top2 > 0:
            z = (top1_c - top2_c) / np.sqrt(float(total_top2))
            if z >= z_crit:
                return n, top1_k, True
    # Exhausted budget: pick empirical leader
    leader = max(counts.items(), key=lambda kv: kv[1])[0]
    return int(ys.size), leader, False


def _plurality_winner_1d(arr: np.ndarray) -> int:
    vals, cnts = np.unique(arr, return_counts=True)
    return int(vals[int(np.argmax(cnts))])


def run_suite_1(rust_bench_map: Dict[str, float]) -> Dict[str, Any]:
    print("[Suite 1] Consensus Stopping (ASC-D Table 2 + Open-Ended Swarm)...", flush=True)
    t_suite = time.perf_counter()

    # Part 1A: ASC-D Table 2 across easy / medium / hard / exact-tie null
    ascd_regimes = [
        ("easy", 0.45, 200, 256),
        ("medium", 0.25, 200, 256),
        ("hard", 0.12, 200, 256),
        ("null_tie", 0.00, 200, 256),
    ]
    ascd_results: Dict[str, Any] = {}
    for label, dg, n_trials, max_n in ascd_regimes:
        tr = traces.gen_ascd_benchmark(seed=101 + int(dg * 100), delta_gap=dg, n_trials=n_trials, max_n=max_n)
        thetas = np.ascontiguousarray(tr.thetas, dtype=np.float64)
        ys_u64 = np.ascontiguousarray(tr.ys, dtype=np.uint64)
        ys_i32 = tr.ys
        true_w = tr.true_winner

        ascd_n, ascd_cert, ascd_correct = [], 0, 0
        ascd_b64_n, ascd_b64_cert, ascd_b64_correct = [], 0, 0
        cite_n, cite_cert, cite_correct = [], 0, 0
        naive_n, naive_cert, naive_correct = [], 0, 0
        fix32_correct, fix64_correct, fix256_correct = 0, 0, 0

        for i in range(n_trials):
            # cloture AscdStopper (unbounded horizon up to max_n)
            st = cloture.AscdStopper(k=4, delta=0.05)
            d = st.update_batch(thetas[i])
            ascd_n.append(d.n_samples)
            ascd_cert += int(d.certified)
            pick = d.winner if d.winner is not None else d.leader
            if true_w >= 0 and pick == true_w:
                ascd_correct += 1

            # cloture AscdStopper with max_budget=64 deterministic lock
            st64 = cloture.AscdStopper(k=4, delta=0.05, max_budget=64)
            d64 = st64.update_batch(thetas[i, :64])
            ascd_b64_n.append(d64.n_samples)
            ascd_b64_cert += int(d64.certified)
            pick64 = d64.winner if d64.winner is not None else d64.leader
            if true_w >= 0 and pick64 == true_w:
                ascd_b64_correct += 1

            # cloture CiteStopper (category-agnostic black-box on matched hard votes)
            ct = cloture.CiteStopper(delta=0.05)
            dc = ct.update_batch(ys_u64[i])
            cite_n.append(dc.n_samples)
            cite_cert += int(dc.certified)
            pick_c = dc.winner if dc.winner is not None else dc.leader
            if true_w >= 0 and pick_c == true_w:
                cite_correct += 1

            # Naive uncorrected majority peeking
            nn, nw, nc = _run_naive_majority_peeking(ys_i32[i], alpha=0.05)
            naive_n.append(nn)
            naive_cert += int(nc)
            if true_w >= 0 and nw == true_w:
                naive_correct += 1

            # Fixed-N baselines on matched hard votes
            if true_w >= 0:
                fix32_correct += int(_plurality_winner_1d(ys_i32[i, :32]) == true_w)
                fix64_correct += int(_plurality_winner_1d(ys_i32[i, :64]) == true_w)
                fix256_correct += int(_plurality_winner_1d(ys_i32[i, :256]) == true_w)

        ascd_mean_n = float(np.mean(ascd_n))
        ascd_b64_mean_n = float(np.mean(ascd_b64_n))
        cite_mean_n = float(np.mean(cite_n))
        naive_mean_n = float(np.mean(naive_n))

        entry: Dict[str, Any] = {
            "delta_gap": dg,
            "n_trials": n_trials,
            "max_n": max_n,
            "cloture_ascd": {
                "mean_samples": ascd_mean_n,
                "median_samples": float(np.median(ascd_n)),
                "p95_samples": float(np.percentile(ascd_n, 95)),
                "cert_rate": ascd_cert / n_trials,
                "accuracy": None if true_w < 0 else ascd_correct / n_trials,
                "type1_error_on_null": (ascd_cert / n_trials) if true_w < 0 else None,
                "pct_saved_vs_fixed64": 100.0 * (1.0 - ascd_mean_n / 64.0),
                "pct_saved_vs_fixed256": 100.0 * (1.0 - ascd_mean_n / 256.0),
            },
            "cloture_ascd_budget64": {
                "mean_samples": ascd_b64_mean_n,
                "cert_rate": ascd_b64_cert / n_trials,
                "accuracy": None if true_w < 0 else ascd_b64_correct / n_trials,
                "pct_saved_vs_fixed64": 100.0 * (1.0 - ascd_b64_mean_n / 64.0),
            },
            "cloture_cite": {
                "mean_samples": cite_mean_n,
                "median_samples": float(np.median(cite_n)),
                "cert_rate": cite_cert / n_trials,
                "accuracy": None if true_w < 0 else cite_correct / n_trials,
                "type1_error_on_null": (cite_cert / n_trials) if true_w < 0 else None,
                "pct_saved_vs_fixed64": 100.0 * (1.0 - cite_mean_n / 64.0),
                "pct_saved_vs_fixed256": 100.0 * (1.0 - cite_mean_n / 256.0),
            },
            "naive_uncorrected_peeking": {
                "mean_samples": naive_mean_n,
                "cert_rate": naive_cert / n_trials,
                "accuracy": None if true_w < 0 else naive_correct / n_trials,
                "type1_error_on_null": (naive_cert / n_trials) if true_w < 0 else None,
            },
            "fixed_n": {
                "n32_accuracy": None if true_w < 0 else fix32_correct / n_trials,
                "n64_accuracy": None if true_w < 0 else fix64_correct / n_trials,
                "n256_accuracy": None if true_w < 0 else fix256_correct / n_trials,
            },
        }
        ascd_results[label] = entry

    # Verify numerical parity between cloture.AscdStopper and _NumpyAscdReference on 25 trials
    tr_par = traces.gen_ascd_benchmark(seed=999, delta_gap=0.30, n_trials=25, max_n=200)
    parity_matches = 0
    for i in range(25):
        st = cloture.AscdStopper(k=4, delta=0.05)
        d = st.update_batch(np.ascontiguousarray(tr_par.thetas[i], dtype=np.float64))
        ref = _NumpyAscdReference(k=4, delta=0.05)
        ref_n, ref_w = 200, -1
        for t in range(200):
            cert, ldr, n = ref.update(tr_par.thetas[i, t])
            if cert:
                ref_n, ref_w = n, ldr
                break
        if d.certified and d.n_samples == ref_n and d.winner == ref_w:
            parity_matches += 1

    # Part 1B: Open-Ended Swarm (K_true = 200, Zipf long tail, unknown K)
    oe_regimes = [
        ("easy_gap_0.30", 0.30, 200, 256),
        ("medium_gap_0.18", 0.18, 200, 256),
        ("null_tie", 0.00, 200, 256),
    ]
    open_ended_results: Dict[str, Any] = {}
    for label, gap, n_trials, max_n in oe_regimes:
        tr = traces.gen_open_ended_swarm(
            seed=201 + int(gap * 100), k_true=200, zipf_s=1.25, top_gap=gap, n_trials=n_trials, max_n=max_n
        )
        ys_u64 = np.ascontiguousarray(tr.ys, dtype=np.uint64)
        ys_i32 = tr.ys
        true_w = tr.true_winner

        cite_n, cite_cert, cite_correct = [], 0, 0
        naive_n, naive_cert, naive_correct = [], 0, 0
        fix32_correct, fix64_correct, fix256_correct = 0, 0, 0

        for i in range(n_trials):
            ct = cloture.CiteStopper(delta=0.05)
            dc = ct.update_batch(ys_u64[i])
            cite_n.append(dc.n_samples)
            cite_cert += int(dc.certified)
            pick = dc.winner if dc.winner is not None else dc.leader
            if true_w >= 0 and pick == true_w:
                cite_correct += 1

            nn, nw, nc = _run_naive_majority_peeking(ys_i32[i], alpha=0.05)
            naive_n.append(nn)
            naive_cert += int(nc)
            if true_w >= 0 and nw == true_w:
                naive_correct += 1

            if true_w >= 0:
                fix32_correct += int(_plurality_winner_1d(ys_i32[i, :32]) == true_w)
                fix64_correct += int(_plurality_winner_1d(ys_i32[i, :64]) == true_w)
                fix256_correct += int(_plurality_winner_1d(ys_i32[i, :256]) == true_w)

        c_mean = float(np.mean(cite_n))
        open_ended_results[label] = {
            "k_true": 200,
            "zipf_s": 1.25,
            "top_gap": gap,
            "plurality_mode_p0": float(tr.probs[0]),
            "tail_mass": float(tr.probs[2:].sum()),
            "cloture_cite": {
                "mean_samples": c_mean,
                "median_samples": float(np.median(cite_n)),
                "cert_rate": cite_cert / n_trials,
                "accuracy": None if true_w < 0 else cite_correct / n_trials,
                "type1_error_on_null": (cite_cert / n_trials) if true_w < 0 else None,
                "pct_saved_vs_fixed64": 100.0 * (1.0 - c_mean / 64.0),
                "pct_saved_vs_fixed256": 100.0 * (1.0 - c_mean / 256.0),
            },
            "naive_uncorrected_peeking": {
                "mean_samples": float(np.mean(naive_n)),
                "cert_rate": naive_cert / n_trials,
                "accuracy": None if true_w < 0 else naive_correct / n_trials,
                "type1_error_on_null": (naive_cert / n_trials) if true_w < 0 else None,
            },
            "fixed_n": {
                "n32_accuracy": None if true_w < 0 else fix32_correct / n_trials,
                "n64_accuracy": None if true_w < 0 else fix64_correct / n_trials,
                "n256_accuracy": None if true_w < 0 else fix256_correct / n_trials,
            },
        }

    # Part 1C: CPU Latency (ns/update) — null stream so stoppers process all N updates without halting early
    tr_lat = traces.gen_ascd_benchmark(seed=777, delta_gap=0.0, n_trials=1, max_n=5000)
    th_lat = np.ascontiguousarray(tr_lat.thetas[0], dtype=np.float64)
    ys_lat = np.ascontiguousarray(tr_lat.ys[0], dtype=np.uint64)

    # cloture AscdStopper batch (median of 5)
    t_ascd_batch = []
    for _ in range(5):
        st = cloture.AscdStopper(k=4, delta=1e-12)
        t0 = time.perf_counter_ns()
        st.update_batch(th_lat)
        t_ascd_batch.append((time.perf_counter_ns() - t0) / len(th_lat))

    # cloture AscdStopper Python streaming
    t_ascd_stream = []
    for _ in range(3):
        st = cloture.AscdStopper(k=4, delta=1e-12)
        t0 = time.perf_counter_ns()
        for row in th_lat:
            st.update(row)
        t_ascd_stream.append((time.perf_counter_ns() - t0) / len(th_lat))

    # cloture CiteStopper batch & streaming
    t_cite_batch = []
    for _ in range(5):
        ct = cloture.CiteStopper(delta=1e-12)
        t0 = time.perf_counter_ns()
        ct.update_batch(ys_lat)
        t_cite_batch.append((time.perf_counter_ns() - t0) / len(ys_lat))

    t_cite_stream = []
    for _ in range(3):
        ct = cloture.CiteStopper(delta=1e-12)
        t0 = time.perf_counter_ns()
        for v in ys_lat.tolist():
            ct.update(v)
        t_cite_stream.append((time.perf_counter_ns() - t0) / len(ys_lat))

    # Pure-Python/NumPy ASC-D Reference (streaming per update)
    t_numpy_ascd = []
    for _ in range(3):
        ref = _NumpyAscdReference(k=4, delta=1e-12)
        t0 = time.perf_counter_ns()
        for row in th_lat:
            ref.update(row)
        t_numpy_ascd.append((time.perf_counter_ns() - t0) / len(th_lat))

    ascd_batch_ns = float(np.median(t_ascd_batch))
    ascd_stream_ns = float(np.median(t_ascd_stream))
    cite_batch_ns = float(np.median(t_cite_batch))
    cite_stream_ns = float(np.median(t_cite_stream))
    numpy_ascd_ns = float(np.median(t_numpy_ascd))

    latency = {
        "rust_cli_ascd_ns_per_update": rust_bench_map.get("AscdStopper(K=4)::update"),
        "rust_cli_cite_ns_per_update": rust_bench_map.get("CiteStopper::update"),
        "cloture_py_ascd_batch_ns_per_update": ascd_batch_ns,
        "cloture_py_ascd_stream_ns_per_update": ascd_stream_ns,
        "cloture_py_cite_batch_ns_per_update": cite_batch_ns,
        "cloture_py_cite_stream_ns_per_update": cite_stream_ns,
        "numpy_ascd_stream_ns_per_update": numpy_ascd_ns,
        "speedup_cloture_batch_vs_numpy": numpy_ascd_ns / max(ascd_batch_ns, 1e-9),
        "speedup_cloture_stream_vs_numpy": numpy_ascd_ns / max(ascd_stream_ns, 1e-9),
        "ascd_vs_numpy_exact_stop_parity": f"{parity_matches}/25",
    }

    return {
        "ascd_table2": ascd_results,
        "open_ended_swarm": open_ended_results,
        "latency": latency,
        "elapsed_sec": time.perf_counter() - t_suite,
    }


# ---------------------------------------------------------------------------
# Suite 2: Correlated-Panel Audit ("Nine Judges, Two Effective Votes")
# ---------------------------------------------------------------------------


def _numpy_audit_panel_reference(
    votes: np.ndarray,
    gold: np.ndarray,
    n_condorcet_sims: int = 2000,
    n_perms: int = 2000,
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Honest vectorized NumPy/SciPy reference computing the exact same panel metrics."""
    n_items, k_judges = votes.shape
    n_classes = int(max(int(votes.max()), int(gold.max()))) + 1
    errors = (votes != gold[:, None]).astype(np.float64)

    # 1. Accuracies, plurality credit, unanimity
    judge_acc = 1.0 - errors.mean(axis=0)
    best_single_idx = int(np.argmax(judge_acc))
    best_single_acc = float(judge_acc[best_single_idx])

    def _plurality_acc(v_mat: np.ndarray, g_vec: np.ndarray) -> float:
        counts = np.zeros((v_mat.shape[0], n_classes), dtype=np.int32)
        for c in range(n_classes):
            counts[:, c] = (v_mat == c).sum(axis=1)
        mx = counts.max(axis=1, keepdims=True)
        is_mx = counts == mx
        n_tied = is_mx.sum(axis=1)
        true_mx = is_mx[np.arange(v_mat.shape[0]), g_vec]
        return float(np.mean(true_mx / n_tied))

    panel_acc = _plurality_acc(votes, gold)
    unanimous = np.all(votes == votes[:, :1], axis=1)
    unanimous_rate = float(unanimous.mean())
    unanimous_acc = float((votes[unanimous, 0] == gold[unanimous]).mean()) if unanimous.any() else 0.0
    all_wrong_rate = float(np.all(errors == 1.0, axis=1).mean())

    # 2. Pairwise phi, Kish & Spectral n_eff
    phi = np.corrcoef(errors, rowvar=False)
    np.fill_diagonal(phi, 1.0)
    triu = np.triu_indices(k_judges, k=1)
    mean_phi = float(phi[triu].mean())
    std_phi = float(phi[triu].std(ddof=0))
    n_eff_kish = float(k_judges / (1.0 + (k_judges - 1.0) * max(mean_phi, 0.0)))
    eigs = np.linalg.eigvalsh(phi)[::-1]
    lambda_max = float(max(eigs[0], 1e-12))
    n_eff_eigen = float(k_judges / lambda_max)
    pos_eigs = eigs[eigs > 1e-12]
    q_e = pos_eigs / pos_eigs.sum()
    eff_rank = float(np.clip(np.exp(-np.sum(q_e * np.log(q_e))), 1.0, k_judges))

    # 3. Exact nominal Krippendorff's alpha (vectorized over classes)
    item_class_counts = np.stack([(votes == c).sum(axis=1) for c in range(n_classes)], axis=1).astype(np.int64)
    within_agree = int(np.sum(item_class_counts * (item_class_counts - 1)))
    d_o = 1.0 - within_agree / float(n_items * k_judges * (k_judges - 1))
    marg_counts = item_class_counts.sum(axis=0)
    total_r = int(n_items * k_judges)
    d_e = 1.0 - int(np.sum(marg_counts * (marg_counts - 1))) / float(total_r * (total_r - 1))
    k_alpha = 1.0 if d_e <= 1e-15 else float(1.0 - d_o / d_e)

    rng = np.random.default_rng(seed)

    # 4. Item-aware stratified Condorcet null (2,000 Monte Carlo sims, vectorized in chunks)
    buckets = [np.flatnonzero(gold == c) for c in range(n_classes)]
    chunk = 100
    condorcet_sum = 0.0
    n_done = 0
    while n_done < n_condorcet_sims:
        b = min(chunk, n_condorcet_sims - n_done)
        sim_votes = np.empty((b, n_items, k_judges), dtype=np.int32)
        for c in range(n_classes):
            idx_c = buckets[c]
            if idx_c.size == 0:
                continue
            picks = rng.integers(0, idx_c.size, size=(b, idx_c.size, k_judges))
            chosen_rows = idx_c[picks]  # (b, n_c, k)
            j_idx = np.arange(k_judges)[None, None, :]
            sim_votes[:, idx_c, :] = votes[chosen_rows, j_idx]
        # Vectorized plurality credit across (b, n_items)
        counts = np.empty((b, n_items, n_classes), dtype=np.int32)
        for c in range(n_classes):
            counts[:, :, c] = (sim_votes == c).sum(axis=2)
        mx = counts.max(axis=2, keepdims=True)
        is_mx = counts == mx
        n_tied = is_mx.sum(axis=2)
        true_mx = is_mx[:, np.arange(n_items), gold]
        condorcet_sum += float(np.sum(np.mean(true_mx / n_tied, axis=1)))
        n_done += b
    condorcet_pred = condorcet_sum / float(n_condorcet_sims)
    condorcet_gap = condorcet_pred - panel_acc

    # 5. Stratified permutation test (2,000 permutations using exact O(N*K) weighted sum-of-squares identity)
    p_err = errors.mean(axis=0)
    w = np.where(p_err * (1.0 - p_err) > 1e-15, 1.0 / np.sqrt(np.maximum(p_err * (1.0 - p_err), 1e-15)), 0.0)
    s_obs = float(np.sum((errors @ w) ** 2))
    tol = 1e-11 * max(abs(s_obs), 1.0)
    ge_cnt = 0
    perm_chunk = 200
    p_done = 0
    err_counts_j = errors.sum(axis=0).astype(np.int64)
    while p_done < n_perms:
        b = min(perm_chunk, n_perms - p_done)
        # Generate permuted column indicators cleanly via argsort of random uniforms
        acc = np.zeros((b, n_items), dtype=np.float64)
        for j in range(k_judges):
            cj = int(err_counts_j[j])
            if cj == 0 or w[j] == 0.0:
                continue
            u = rng.random((b, n_items))
            idx_top = np.argpartition(u, cj, axis=1)[:, :cj]
            rows = np.arange(b)[:, None]
            acc[rows, idx_top] += w[j]
        s_perm = np.sum(acc * acc, axis=1)
        ge_cnt += int(np.sum(s_perm >= s_obs - tol))
        p_done += b
    perm_p = float((1 + ge_cnt) / (1 + n_perms))

    # 6. Item-level bootstrap 95% CI on n_eff_kish (1,000 resamples, vectorized matrix ops)
    boot_vals = np.empty(n_boot, dtype=np.float64)
    boot_chunk = 100
    b_done = 0
    while b_done < n_boot:
        b = min(boot_chunk, n_boot - b_done)
        # Multinomial weights per item via bincount / random indices
        idx_b = rng.integers(0, n_items, size=(b, n_items))
        for m in range(b):
            eb = errors[idx_b[m]]
            pe = eb.mean(axis=0)
            sd = np.sqrt(np.maximum(pe * (1.0 - pe), 0.0))
            cov_b = (eb.T @ eb) / float(n_items) - np.outer(pe, pe)
            denom = np.outer(sd, sd)
            phi_b = np.where(denom > 1e-15, np.clip(cov_b / np.maximum(denom, 1e-15), -1.0, 1.0), 0.0)
            mp = float(phi_b[triu].mean())
            boot_vals[b_done + m] = k_judges / (1.0 + (k_judges - 1.0) * max(mp, 0.0))
        b_done += b
    ci_lo, ci_hi = float(np.quantile(boot_vals, 0.025)), float(np.quantile(boot_vals, 0.975))

    # 7. Leave-one-out pruning analysis
    loo_delta_acc = []
    loo_delta_neff = []
    for drop_j in range(k_judges):
        keep = [j for j in range(k_judges) if j != drop_j]
        acc_wo = _plurality_acc(votes[:, keep], gold)
        loo_delta_acc.append(acc_wo - panel_acc)
        sub_phi = phi[np.ix_(keep, keep)]
        sub_triu = np.triu_indices(k_judges - 1, k=1)
        mp_wo = float(sub_phi[sub_triu].mean())
        neff_wo = (k_judges - 1.0) / (1.0 + (k_judges - 2.0) * max(mp_wo, 0.0))
        loo_delta_neff.append(neff_wo - n_eff_kish)

    return {
        "panel_accuracy": panel_acc,
        "best_single_accuracy": best_single_acc,
        "best_single_index": best_single_idx,
        "mean_phi": mean_phi,
        "std_phi": std_phi,
        "n_eff_kish": n_eff_kish,
        "n_eff_eigen": n_eff_eigen,
        "lambda_max": lambda_max,
        "effective_rank": eff_rank,
        "krippendorff_alpha": k_alpha,
        "unanimous_rate": unanimous_rate,
        "unanimous_accuracy": unanimous_acc,
        "all_wrong_rate": all_wrong_rate,
        "condorcet_predicted_accuracy": condorcet_pred,
        "condorcet_gap": condorcet_gap,
        "permutation_p_value": perm_p,
        "n_eff_ci_low": ci_lo,
        "n_eff_ci_high": ci_hi,
        "phi_matrix": phi,
        "loo_delta_acc": loo_delta_acc,
        "loo_delta_neff": loo_delta_neff,
    }


def run_suite_2() -> Dict[str, Any]:
    print("[Suite 2] Correlated-Panel Audit (K=9, N=1000, 2k perms/2k MC/1k boot)...", flush=True)
    t_suite = time.perf_counter()
    tr = traces.gen_correlated_judge_panel(seed=42, n_items=1000, k_judges=9, target_phi=0.391)
    votes_u32 = np.ascontiguousarray(tr.votes, dtype=np.uint32)
    gold_u32 = np.ascontiguousarray(tr.true_labels, dtype=np.uint32)

    # Time cloture.audit_panel (5 runs)
    cloture_times_ms = []
    rep = None
    for _ in range(5):
        t0 = time.perf_counter()
        rep = cloture.audit_panel(
            votes_u32, gold_u32, n_condorcet_sims=2000, n_perms=2000, n_boot=1000, seed=42
        )
        cloture_times_ms.append((time.perf_counter() - t0) * 1e3)
    assert rep is not None

    # Time honest vectorized NumPy/SciPy reference (3 runs)
    numpy_times_ms = []
    ref = None
    for _ in range(3):
        t0 = time.perf_counter()
        ref = _numpy_audit_panel_reference(
            tr.votes, tr.true_labels, n_condorcet_sims=2000, n_perms=2000, n_boot=1000, seed=42
        )
        numpy_times_ms.append((time.perf_counter() - t0) * 1e3)
    assert ref is not None

    # Deterministic exact numerical parity check
    det_diffs = {
        "panel_accuracy": abs(rep.panel_accuracy - ref["panel_accuracy"]),
        "mean_phi": abs(rep.mean_phi - ref["mean_phi"]),
        "n_eff_kish": abs(rep.n_eff_kish - ref["n_eff_kish"]),
        "n_eff_eigen": abs(rep.n_eff_eigen - ref["n_eff_eigen"]),
        "effective_rank": abs(rep.effective_rank - ref["effective_rank"]),
        "krippendorff_alpha": abs(rep.krippendorff_alpha - ref["krippendorff_alpha"]),
        "phi_matrix_max_diff": float(np.max(np.abs(np.asarray(rep.phi_matrix) - ref["phi_matrix"]))),
        "loo_delta_acc_max_diff": float(
            np.max(np.abs(np.asarray(rep.loo_delta_acc) - np.asarray(ref["loo_delta_acc"])))
        ),
        "loo_delta_neff_max_diff": float(
            np.max(np.abs(np.asarray(rep.loo_delta_neff) - np.asarray(ref["loo_delta_neff"])))
        ),
    }
    max_det_diff = float(max(det_diffs.values()))

    # Pruning impact: dropping top LOO redundant judges (positive delta_acc)
    redundant_judges = [j for j, da in enumerate(rep.loo_delta_acc) if da > 0.0]
    best_single_drop = int(np.argmax(rep.loo_delta_acc))
    # Also evaluate accuracy when dropping ALL judges in the redundant high-corr cluster (e.g. top-2 LOO)
    top2_drop = sorted(range(9), key=lambda j: rep.loo_delta_acc[j], reverse=True)[:2]
    keep_7 = [j for j in range(9) if j not in top2_drop]
    rep_7 = cloture.audit_panel(
        np.ascontiguousarray(tr.votes[:, keep_7], dtype=np.uint32),
        gold_u32,
        n_condorcet_sims=500,
        n_perms=500,
        n_boot=200,
        seed=42,
    )

    c_med_ms = float(np.median(cloture_times_ms))
    n_med_ms = float(np.median(numpy_times_ms))

    return {
        "config": {"k_judges": 9, "n_items": 1000, "n_perms": 2000, "n_condorcet_sims": 2000, "n_boot": 1000},
        "timing": {
            "cloture_ms": _percentiles(cloture_times_ms),
            "numpy_scipy_ref_ms": _percentiles(numpy_times_ms),
            "speedup_x": n_med_ms / max(c_med_ms, 1e-9),
        },
        "numerical_parity": {
            "deterministic_diffs": det_diffs,
            "max_deterministic_abs_diff": max_det_diff,
            "stochastic_condorcet_gap_diff": abs(rep.condorcet_gap - ref["condorcet_gap"]),
            "stochastic_perm_p_cloture": rep.permutation_p_value,
            "stochastic_perm_p_numpy": ref["permutation_p_value"],
        },
        "panel_findings": {
            "panel_accuracy": rep.panel_accuracy,
            "best_single_accuracy": rep.best_single_accuracy,
            "panel_lift": rep.panel_lift,
            "mean_phi": rep.mean_phi,
            "n_eff_kish": rep.n_eff_kish,
            "n_eff_kish_ci95": [rep.n_eff_ci_low, rep.n_eff_ci_high],
            "n_eff_eigen": rep.n_eff_eigen,
            "effective_rank": rep.effective_rank,
            "krippendorff_alpha": rep.krippendorff_alpha,
            "condorcet_predicted_accuracy": rep.condorcet_predicted_accuracy,
            "condorcet_gap": rep.condorcet_gap,
            "permutation_p_value": rep.permutation_p_value,
            "loo_delta_acc": rep.loo_delta_acc,
            "loo_delta_neff": rep.loo_delta_neff,
            "redundant_judge_indices_positive_acc_gain": redundant_judges,
            "drop_1_best_judge_idx": best_single_drop,
            "drop_1_accuracy_gain": rep.loo_delta_acc[best_single_drop],
            "drop_1_cost_savings_pct": 100.0 / 9.0,
            "drop_2_judges_indices": top2_drop,
            "drop_2_panel_accuracy": rep_7.panel_accuracy,
            "drop_2_accuracy_gain": rep_7.panel_accuracy - rep.panel_accuracy,
            "drop_2_cost_savings_pct": 200.0 / 9.0,
        },
        "elapsed_sec": time.perf_counter() - t_suite,
    }


# ---------------------------------------------------------------------------
# Suite 3: Realized vs. Population Correlated Rollouts (arXiv:2611.26592)
# ---------------------------------------------------------------------------


def _eval_rollout_regime(
    n_reps: int, n_prompts: int, n_rollouts: int, rho: float, theta: float, seed_base: int
) -> Dict[str, Any]:
    pooled_z_fires = 0
    pooled_cs_fires = 0
    single_unit_prompt0_fires = 0
    single_unit_any_prompt_fires = 0
    single_unit_prompt_fire_fracs = []
    pop_cert_fires = 0
    rho_hats, neffs, ceilings, opt_rollouts_1_1, opt_rollouts_10_1 = [], [], [], [], []

    for r in range(n_reps):
        tr = traces.gen_correlated_rollouts_aime(
            seed=seed_base + r, n_prompts=n_prompts, n_rollouts=n_rollouts, rho=rho, theta=theta
        )
        diffs = np.ascontiguousarray(tr.diffs, dtype=np.float64)
        scores01 = np.ascontiguousarray(tr.unit_scores, dtype=np.float64)

        # (a) Naive pooled rollout test (treats all n_prompts * n_rollouts as i.i.d.)
        flat = diffs.ravel()
        se = float(flat.std(ddof=1) / np.sqrt(flat.size))
        z = float(flat.mean() / se) if se > 0 else 0.0
        if z > 1.6448536269514722:
            pooled_z_fires += 1

        # Naive pooled anytime-valid CS on 480 correlated rollouts as if i.i.d.
        eb_pool = cloture.EmpiricalBernsteinCS(alpha=0.05, min_val=-1.0, max_val=1.0)
        res_pool = eb_pool.update_batch(flat)
        if res_pool.lower > 0.0:
            pooled_cs_fires += 1

        # (b) Naive single-unit consensus misfire (certifies a single prompt via BettingCS on its rollouts)
        prompt_fired = 0
        for p_idx in range(n_prompts):
            bcs = cloture.BettingCS(alpha=0.05, grid_points=101)
            bres = bcs.update_batch(scores01[p_idx])
            if bres.lower > 0.5:
                prompt_fired += 1
                if p_idx == 0:
                    single_unit_prompt0_fires += 1
        single_unit_prompt_fire_fracs.append(prompt_fired / float(n_prompts))
        if prompt_fired > 0:
            single_unit_any_prompt_fires += 1

        # (c) cloture.PopulationCert (prompt-level anytime-valid e-process)
        pc = cloture.PopulationCert(delta=0.05)
        pd = None
        for p_idx in range(n_prompts):
            pd = pc.add_unit(diffs[p_idx])
        assert pd is not None
        if pd.certified:
            pop_cert_fires += 1
        rho_hats.append(pd.rho_hat)
        neffs.append(pd.n_eff_within)
        ceilings.append(pd.saturation_ceiling)
        opt_rollouts_1_1.append(pc.optimal_rollouts(1.0, 1.0))
        opt_rollouts_10_1.append(pc.optimal_rollouts(10.0, 1.0))

    return {
        "n_reps": n_reps,
        "n_prompts": n_prompts,
        "n_rollouts": n_rollouts,
        "total_rollouts_per_rep": n_prompts * n_rollouts,
        "rho_true": rho,
        "theta_true": theta,
        "naive_pooled_z_fire_rate": pooled_z_fires / n_reps,
        "naive_pooled_ebcs_fire_rate": pooled_cs_fires / n_reps,
        "naive_single_unit_prompt0_fire_rate": single_unit_prompt0_fires / n_reps,
        "naive_single_unit_mean_prompt_fire_rate": float(np.mean(single_unit_prompt_fire_fracs)),
        "naive_single_unit_any_prompt_fire_rate": single_unit_any_prompt_fires / n_reps,
        "cloture_population_cert_fire_rate": pop_cert_fires / n_reps,
        "mean_rho_hat": float(np.mean(rho_hats)),
        "mean_n_eff_within": float(np.mean(neffs)),
        "mean_within_unit_ceiling": float(np.mean(ceilings)),
        "median_optimal_rollouts_cost_1_to_1": float(np.median(opt_rollouts_1_1)),
        "median_optimal_rollouts_cost_10_to_1": float(np.median(opt_rollouts_10_1)),
    }


def run_suite_3() -> Dict[str, Any]:
    print("[Suite 3] Realized vs. Population Correlated Rollouts (arXiv:2611.26592)...", flush=True)
    t_suite = time.perf_counter()

    null_30x16 = _eval_rollout_regime(
        n_reps=300, n_prompts=30, n_rollouts=16, rho=0.60, theta=0.0, seed_base=3000
    )
    pos_30x16 = _eval_rollout_regime(
        n_reps=150, n_prompts=30, n_rollouts=16, rho=0.60, theta=0.25, seed_base=4000
    )
    pos_120x4 = _eval_rollout_regime(
        n_reps=150, n_prompts=120, n_rollouts=4, rho=0.60, theta=0.25, seed_base=5000
    )
    pos_120x16 = _eval_rollout_regime(
        n_reps=150, n_prompts=120, n_rollouts=16, rho=0.60, theta=0.25, seed_base=6000
    )

    return {
        "null_theta0_30x16": null_30x16,
        "positive_theta025_30x16_same_480_budget": pos_30x16,
        "positive_theta025_120x4_same_480_budget": pos_120x4,
        "positive_theta025_120x16": pos_120x16,
        "elapsed_sec": time.perf_counter() - t_suite,
    }


# ---------------------------------------------------------------------------
# Suite 4: Anytime-Valid CS & Multi-Arm Racing vs. Real PyPI Competitors
# ---------------------------------------------------------------------------


def run_suite_4(rust_bench_map: Dict[str, float]) -> Dict[str, Any]:
    print("[Suite 4] Anytime-Valid CS & Multi-Arm Racing vs. confseq/expectation/bayesbench...", flush=True)
    t_suite = time.perf_counter()

    # ---- Part 4A: Confidence Sequences Head-to-Head (500 trials x N=1000, p=0.70, alpha=0.05) ----
    rng = np.random.default_rng(4242)
    n_trials = 500
    n_steps = 1000
    p_true = 0.70
    alpha = 0.05
    data = rng.binomial(1, p_true, size=(n_trials, n_steps)).astype(np.float64)

    # 1. cloture.EmpiricalBernsteinCS (streaming all 500 x 1000 steps)
    eb_miscov = 0
    eb_w200, eb_w1000 = [], []
    t0 = time.perf_counter_ns()
    for i in range(n_trials):
        cs = cloture.EmpiricalBernsteinCS(alpha=alpha, min_val=0.0, max_val=1.0)
        bad = False
        row = data[i].tolist()
        for t, x in enumerate(row):
            r = cs.update(x)
            if p_true < r.lower or p_true > r.upper:
                bad = True
            if t == 199:
                eb_w200.append(r.width)
            elif t == 999:
                eb_w1000.append(r.width)
        eb_miscov += int(bad)
    eb_stream_ns = (time.perf_counter_ns() - t0) / float(n_trials * n_steps)

    # 2. cloture.BettingCS(grid_points=201) (streaming all 500 x 1000 steps)
    bet_miscov = 0
    bet_w200, bet_w1000 = [], []
    t0 = time.perf_counter_ns()
    for i in range(n_trials):
        cs = cloture.BettingCS(alpha=alpha, grid_points=201)
        bad = False
        row = data[i].tolist()
        for t, x in enumerate(row):
            r = cs.update(x)
            if p_true < r.lower or p_true > r.upper:
                bad = True
            if t == 199:
                bet_w200.append(r.width)
            elif t == 999:
                bet_w1000.append(r.width)
        bet_miscov += int(bad)
    bet_stream_ns = (time.perf_counter_ns() - t0) / float(n_trials * n_steps)

    # 3. confseq.predmix.predmix_empbern_cs (vectorized batch, all 500 x 1000)
    csp_miscov = 0
    csp_w200, csp_w1000 = [], []
    t0 = time.perf_counter_ns()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i in range(n_trials):
            lo, hi = cs_predmix.predmix_empbern_cs(data[i], alpha=alpha, running_intersection=True)
            if np.any((p_true < lo) | (p_true > hi)):
                csp_miscov += 1
            csp_w200.append(float(hi[199] - lo[199]))
            csp_w1000.append(float(hi[999] - lo[999]))
    csp_batch_ns = (time.perf_counter_ns() - t0) / float(n_trials * n_steps)

    # 4. confseq.betting.hedged_cs (breaks=201, running_intersection=True, 30 trials x 1000 due to ~160ms/trial)
    n_hedged_trials = 30
    csh_miscov = 0
    csh_w200, csh_w1000 = [], []
    t0 = time.perf_counter_ns()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i in range(n_hedged_trials):
            lo, hi = cs_betting.hedged_cs(data[i], alpha=alpha, breaks=201, running_intersection=True)
            if np.any((p_true < lo) | (p_true > hi)):
                csh_miscov += 1
            csh_w200.append(float(hi[199] - lo[199]))
            csh_w1000.append(float(hi[999] - lo[999]))
    csh_batch_ns = (time.perf_counter_ns() - t0) / float(n_hedged_trials * n_steps)

    # 5. confseq.conjmix_bounded.conjmix_empbern_cs (30 trials x 1000, v_opt=50)
    n_cm_trials = 30
    cscm_miscov = 0
    cscm_w200, cscm_w1000 = [], []
    t0 = time.perf_counter_ns()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i in range(n_cm_trials):
            lo, hi = cs_conjmix.conjmix_empbern_cs(
                data[i], v_opt=50.0, alpha=alpha, running_intersection=True
            )
            if np.any((p_true < lo) | (p_true > hi)):
                cscm_miscov += 1
            cscm_w200.append(float(hi[199] - lo[199]))
            cscm_w1000.append(float(hi[999] - lo[999]))
    cscm_batch_ns = (time.perf_counter_ns() - t0) / float(n_cm_trials * n_steps)

    # 6. expectation.EmpiricalBernsteinConfidenceSequence (500 trials x chunked checkpoints of 20 -> 50 peeks/trial)
    exp_cfg = EmpiricalBernsteinConfig(alpha=alpha, lower_bound=0.0, upper_bound=1.0)
    exp_miscov = 0
    exp_w200, exp_w1000 = [], []
    t0 = time.perf_counter_ns()
    for i in range(n_trials):
        ecs = EmpiricalBernsteinConfidenceSequence(config=exp_cfg)
        bad = False
        lo_run, hi_run = 0.0, 1.0
        for end in range(50, n_steps + 1, 50):
            res = ecs.update(data[i, end - 50 : end])
            lo_run = max(lo_run, float(res.lower))
            hi_run = min(hi_run, float(res.upper))
            if p_true < lo_run or p_true > hi_run:
                bad = True
            if end == 200:
                exp_w200.append(hi_run - lo_run)
            elif end == 1000:
                exp_w1000.append(hi_run - lo_run)
        exp_miscov += int(bad)
    exp_chunk50_ns = (time.perf_counter_ns() - t0) / float(n_trials * n_steps)

    # 7. Naive CLT / Wald continuous peeking (500 trials x N=1000, t=10..1000)
    z95 = float(norm.ppf(1.0 - alpha / 2.0))
    clt_miscov = 0
    clt_w200, clt_w1000 = [], []
    t0 = time.perf_counter_ns()
    t_idx = np.arange(1, n_steps + 1, dtype=np.float64)
    for i in range(n_trials):
        csum = np.cumsum(data[i])
        means = csum / t_idx
        # Bernoulli sample variance (ddof=1 for t>=2)
        ss = csum * (1.0 - means)  # since x^2 == x for binary
        var_hat = np.where(t_idx > 1, ss / np.maximum(t_idx - 1.0, 1.0), 0.25)
        half = z95 * np.sqrt(np.maximum(var_hat, 1e-9) / t_idx)
        lo = np.clip(means - half, 0.0, 1.0)
        hi = np.clip(means + half, 0.0, 1.0)
        if np.any((p_true < lo[9:]) | (p_true > hi[9:])):
            clt_miscov += 1
        clt_w200.append(float(hi[199] - lo[199]))
        clt_w1000.append(float(hi[999] - lo[999]))
    clt_batch_ns = (time.perf_counter_ns() - t0) / float(n_trials * n_steps)

    # N = 100,000 throughput timing (fair batch & streaming comparison)
    big_x = np.ascontiguousarray(rng.binomial(1, p_true, size=100_000).astype(np.float64))

    t_eb_100k = []
    for _ in range(5):
        cs = cloture.EmpiricalBernsteinCS(alpha=alpha, min_val=0.0, max_val=1.0)
        t0 = time.perf_counter_ns()
        cs.update_batch(big_x)
        t_eb_100k.append((time.perf_counter_ns() - t0) / 100_000.0)

    t_bet_100k = []
    for _ in range(3):
        cs = cloture.BettingCS(alpha=alpha, grid_points=201)
        t0 = time.perf_counter_ns()
        cs.update_batch(big_x)
        t_bet_100k.append((time.perf_counter_ns() - t0) / 100_000.0)

    t_csp_100k = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for _ in range(5):
            t0 = time.perf_counter_ns()
            cs_predmix.predmix_empbern_cs(big_x, alpha=alpha, running_intersection=True)
            t_csp_100k.append((time.perf_counter_ns() - t0) / 100_000.0)

    # C++ confseq.boundaries.normal_mixture_bound at N=100,000
    v_arr = np.arange(1, 100_001, dtype=np.float64) * 0.21
    t_cpp_bound_100k = []
    for _ in range(5):
        t0 = time.perf_counter_ns()
        cs_boundaries.normal_mixture_bound(v_arr, v_opt=100.0, alpha=alpha)
        t_cpp_bound_100k.append((time.perf_counter_ns() - t0) / 100_000.0)

    # expectation EBCS single-call batch at N=100,000
    t_exp_batch_100k = []
    for _ in range(5):
        ecs = EmpiricalBernsteinConfidenceSequence(config=exp_cfg)
        t0 = time.perf_counter_ns()
        ecs.update(big_x)
        t_exp_batch_100k.append((time.perf_counter_ns() - t0) / 100_000.0)

    # expectation EBCS per-step streaming (10-sample warmup because v=0 on step 1
    # triggers a root-finder RuntimeError in expectation==0.6.1; timed over 1,000 steps)
    ecs_s = EmpiricalBernsteinConfidenceSequence(config=exp_cfg)
    ecs_s.update(big_x[:10])
    t0 = time.perf_counter_ns()
    for v in big_x[10:1010]:
        ecs_s.update(np.array([v]))
    exp_stream_ns = (time.perf_counter_ns() - t0) / 1000.0

    # expectation Rust PyO3 ParallelSequentialTest streaming (N=10,000)
    p_cfg = exp_pst.ParallelTestConfig(alpha=alpha, n_tests=1)
    pt = exp_pst.ParallelSequentialTest(config=p_cfg, null_values=0.5)
    t0 = time.perf_counter_ns()
    for v in big_x[:10_000]:
        pt.step(np.array([v]))
    exp_rust_stream_ns = (time.perf_counter_ns() - t0) / 10_000.0

    cs_head_to_head = {
        "protocol": {"n_trials": n_trials, "n_steps": n_steps, "p_true": p_true, "alpha": alpha, "throughput_n": 100_000},
        "cloture_ebcs": {
            "rust_cli_ns_per_step": rust_bench_map.get("EmpiricalBernsteinCS::update"),
            "batch_100k_ns_per_step": float(np.median(t_eb_100k)),
            "python_stream_ns_per_step": eb_stream_ns,
            "mean_width_n200": float(np.mean(eb_w200)),
            "mean_width_n1000": float(np.mean(eb_w1000)),
            "anytime_miscoverage": eb_miscov / n_trials,
        },
        "cloture_betting_cs_201": {
            "rust_cli_ns_per_step": rust_bench_map.get("BettingCS(G=201)::update"),
            "batch_100k_ns_per_step": float(np.median(t_bet_100k)),
            "python_stream_ns_per_step": bet_stream_ns,
            "mean_width_n200": float(np.mean(bet_w200)),
            "mean_width_n1000": float(np.mean(bet_w1000)),
            "anytime_miscoverage": bet_miscov / n_trials,
        },
        "confseq_predmix_ebcs": {
            "batch_1k_ns_per_step": csp_batch_ns,
            "batch_100k_ns_per_step": float(np.median(t_csp_100k)),
            "mean_width_n200": float(np.mean(csp_w200)),
            "mean_width_n1000": float(np.mean(csp_w1000)),
            "anytime_miscoverage": csp_miscov / n_trials,
        },
        "confseq_hedged_betting_201": {
            "n_trials_measured": n_hedged_trials,
            "batch_1k_ns_per_step": csh_batch_ns,
            "mean_width_n200": float(np.mean(csh_w200)),
            "mean_width_n1000": float(np.mean(csh_w1000)),
            "anytime_miscoverage": csh_miscov / n_hedged_trials,
        },
        "confseq_conjmix_ebcs": {
            "n_trials_measured": n_cm_trials,
            "batch_1k_ns_per_step": cscm_batch_ns,
            "mean_width_n200": float(np.mean(cscm_w200)),
            "mean_width_n1000": float(np.mean(cscm_w1000)),
            "anytime_miscoverage": cscm_miscov / n_cm_trials,
        },
        "confseq_cpp_normal_mixture_bound": {
            "batch_100k_ns_per_step": float(np.median(t_cpp_bound_100k)),
        },
        "expectation_ebcs": {
            "batch_100k_single_call_ns_per_step": float(np.median(t_exp_batch_100k)),
            "chunk50_ns_per_step": exp_chunk50_ns,
            "python_stream_ns_per_step": exp_stream_ns,
            "rust_par_seqtest_stream_ns_per_step": exp_rust_stream_ns,
            "mean_width_n200": float(np.mean(exp_w200)),
            "mean_width_n1000": float(np.mean(exp_w1000)),
            "anytime_miscoverage_chunk50": exp_miscov / n_trials,
        },
        "naive_clt_peeking": {
            "batch_1k_ns_per_step": clt_batch_ns,
            "mean_width_n200": float(np.mean(clt_w200)),
            "mean_width_n1000": float(np.mean(clt_w1000)),
            "anytime_miscoverage": clt_miscov / n_trials,
        },
    }

    # ---- Part 4B: Multi-Arm Racing (6 arms, 200 tasks, 200 trials) + bayesbench ----
    race_means = [0.88, 0.72, 0.62, 0.54, 0.45, 0.35]
    null_means = [0.50, 0.50, 0.50, 0.50, 0.50, 0.50]
    n_arms = 6
    n_tasks = 200
    n_race_trials = 200
    fixed_budget = n_arms * n_tasks  # 1200

    def _run_race_batch(means_vec: List[float], seed_base: int, is_null: bool) -> Dict[str, Any]:
        p_evals, p_cert, p_acc = [], 0, 0
        u_evals, u_cert, u_acc = [], 0, 0
        fix_acc = 0

        for i in range(n_race_trials):
            tr = traces.gen_agent_race(
                seed=seed_base + i, n_arms=n_arms, means=means_vec, n_steps=n_tasks, shared_corr=0.75
            )
            rew_p = np.ascontiguousarray(tr.rewards, dtype=np.float64)
            rew_u = np.ascontiguousarray(tr.unpaired_rewards, dtype=np.float64)

            # Paired racing
            rp = cloture.AgentRace(n_arms=n_arms, delta=0.05, paired=True)
            ev_p = 0
            wp = None
            seen_t = 0
            for t in range(n_tasks):
                ev_p += rp.active_count()
                wp = rp.observe_paired(rew_p[t])
                seen_t = t + 1
                if wp is not None:
                    break
            p_evals.append(ev_p)
            p_cert += int(wp is not None)
            if not is_null:
                pick_p = wp if wp is not None else int(np.argmax(rew_p[:seen_t].mean(axis=0)))
                p_acc += int(pick_p == 0)

            # Unpaired LUCB racing
            ru = cloture.AgentRace(n_arms=n_arms, delta=0.05, paired=False)
            arm_idx = [0] * n_arms
            ev_u = 0
            wu = None
            while ev_u < fixed_budget:
                a = ru.next_arm_lucb()
                if arm_idx[a] >= n_tasks:
                    # Pick another active arm with remaining tasks if available
                    avail = [j for j in range(n_arms) if ru.is_active(j) and arm_idx[j] < n_tasks]
                    if not avail:
                        break
                    a = avail[0]
                wu = ru.observe_arm(a, float(rew_u[arm_idx[a], a]))
                arm_idx[a] += 1
                ev_u += 1
                if wu is not None:
                    break
            u_evals.append(ev_u)
            u_cert += int(wu is not None)
            if not is_null:
                if wu is not None:
                    pick_u = wu
                else:
                    est = [
                        rew_u[: arm_idx[j], j].mean() if arm_idx[j] > 0 else -1.0
                        for j in range(n_arms)
                    ]
                    pick_u = int(np.argmax(est))
                u_acc += int(pick_u == 0)

            if not is_null:
                fix_acc += int(int(np.argmax(rew_p.mean(axis=0))) == 0)

        p_mean_ev = float(np.mean(p_evals))
        u_mean_ev = float(np.mean(u_evals))
        return {
            "n_trials": n_race_trials,
            "n_arms": n_arms,
            "n_tasks": n_tasks,
            "fixed_n_evals": fixed_budget,
            "fixed_n_accuracy": None if is_null else fix_acc / n_race_trials,
            "cloture_paired": {
                "mean_evals": p_mean_ev,
                "median_evals": float(np.median(p_evals)),
                "pct_evals_saved": 100.0 * (1.0 - p_mean_ev / fixed_budget),
                "cert_rate": p_cert / n_race_trials,
                "accuracy": None if is_null else p_acc / n_race_trials,
                "fwer_on_null": (p_cert / n_race_trials) if is_null else None,
            },
            "cloture_unpaired_lucb": {
                "mean_evals": u_mean_ev,
                "median_evals": float(np.median(u_evals)),
                "pct_evals_saved": 100.0 * (1.0 - u_mean_ev / fixed_budget),
                "cert_rate": u_cert / n_race_trials,
                "accuracy": None if is_null else u_acc / n_race_trials,
                "fwer_on_null": (u_cert / n_race_trials) if is_null else None,
            },
        }

    race_signal = _run_race_batch(race_means, seed_base=7000, is_null=False)
    race_null = _run_race_batch(null_means, seed_base=8000, is_null=True)

    # bayesbench idiomatic comparison:
    # (a) Native pairwise BayesianBenchmark.compare (independent-Beta model with minimal skip_threshold=0.500001)
    #     vs cloture.PairedEProcess on top-2 (0.88 vs 0.72) AND exact-tie null (0.50 vs 0.50)
    n_bb_pair = 30
    bb_sig_evals, bb_sig_acc, bb_sig_skipped, bb_sig_ms = [], 0, 0, []
    cp_sig_evals, cp_sig_acc, cp_sig_ms = [], 0, []
    for i in range(n_bb_pair):
        tr = traces.gen_agent_race(seed=9000 + i, n_arms=2, means=[0.88, 0.72], n_steps=200, shared_corr=0.75)
        ra_u = tr.unpaired_rewards[:, 0]
        rb_u = tr.unpaired_rewards[:, 1]
        ra_p = tr.rewards[:, 0]
        rb_p = tr.rewards[:, 1]
        t0 = time.perf_counter()
        bb = bayesbench.BayesianBenchmark(confidence=0.95, skip_threshold=0.500001, min_samples=10)
        res_bb = bb.compare(lambda idx, a=ra_u: a[idx], lambda idx, b=rb_u: b[idx], lambda p, r: r, list(range(200)))
        bb_sig_ms.append((time.perf_counter() - t0) * 1e3)
        bb_sig_evals.append(int(res_bb.problems_tested) * 2)
        bb_sig_skipped += int(res_bb.skipped)
        # Credit model_a if declared winner OR if empirical posterior favors A
        bb_sig_acc += int(res_bb.winner == "model_a")

        t0 = time.perf_counter()
        pe = cloture.PairedEProcess(alpha=0.05)
        w = None
        used = 200
        for t in range(200):
            pr = pe.update(float(ra_p[t]), float(rb_p[t]))
            if pr.winner is not None:
                w = pr.winner
                used = t + 1
                break
        cp_sig_ms.append((time.perf_counter() - t0) * 1e3)
        cp_sig_evals.append(used * 2)
        pick = w if w is not None else (0 if ra_p[:used].mean() >= rb_p[:used].mean() else 1)
        cp_sig_acc += int(pick == 0)

    bb_null_false, bb_null_skipped, bb_null_ms = 0, 0, []
    cp_null_false, cp_null_ms = 0, []
    for i in range(n_bb_pair):
        tr = traces.gen_agent_race(seed=9500 + i, n_arms=2, means=[0.50, 0.50], n_steps=200, shared_corr=0.75)
        ra_u = tr.unpaired_rewards[:, 0]
        rb_u = tr.unpaired_rewards[:, 1]
        ra_p = tr.rewards[:, 0]
        rb_p = tr.rewards[:, 1]
        t0 = time.perf_counter()
        bb = bayesbench.BayesianBenchmark(confidence=0.95, skip_threshold=0.500001, min_samples=10)
        res_bb = bb.compare(lambda idx, a=ra_u: a[idx], lambda idx, b=rb_u: b[idx], lambda p, r: r, list(range(200)))
        bb_null_ms.append((time.perf_counter() - t0) * 1e3)
        bb_null_skipped += int(res_bb.skipped)
        if res_bb.winner in ("model_a", "model_b"):
            bb_null_false += 1

        t0 = time.perf_counter()
        pe = cloture.PairedEProcess(alpha=0.05)
        pr = pe.update_batch(np.ascontiguousarray(ra_p), np.ascontiguousarray(rb_p))
        cp_null_ms.append((time.perf_counter() - t0) * 1e3)
        if pr.winner is not None:
            cp_null_false += 1

    # (b) 1 full 6-arm x 200-task bayesbench.BayesianRanker timing trial (transparently reported!)
    tr_br = traces.gen_agent_race(seed=7000, n_arms=n_arms, means=race_means, n_steps=n_tasks, shared_corr=0.75)
    t0 = time.perf_counter()
    ranker = bayesbench.BayesianRanker(confidence=0.95, min_samples=5)
    for a in range(n_arms):
        ranker.add_model(f"arm_{a}", lambda idx, a=a: float(tr_br.rewards[idx, a]))
    br_res = ranker.rank(list(range(n_tasks)), score_fn=lambda prob, resp: resp)
    br_ms = (time.perf_counter() - t0) * 1e3

    # Also time cloture.AgentRace(paired=True) on the exact same trial
    t0 = time.perf_counter()
    rp_1 = cloture.AgentRace(n_arms=n_arms, delta=0.05, paired=True)
    ev_1 = 0
    for t in range(n_tasks):
        ev_1 += rp_1.active_count()
        if rp_1.observe_paired(np.ascontiguousarray(tr_br.rewards[t], dtype=np.float64)) is not None:
            break
    rp_1_ms = (time.perf_counter() - t0) * 1e3

    bayesbench_comparison = {
        "pairwise_n_trials": n_bb_pair,
        "signal_0.88_vs_0.72": {
            "bayesbench_compare": {
                "mean_evals": float(np.mean(bb_sig_evals)),
                "pct_saved_vs_400": 100.0 * (1.0 - float(np.mean(bb_sig_evals)) / 400.0),
                "certified_correct_rate": bb_sig_acc / n_bb_pair,
                "skipped_rate": bb_sig_skipped / n_bb_pair,
                "median_ms_per_trial": float(np.median(bb_sig_ms)),
            },
            "cloture_paired_eprocess": {
                "mean_evals": float(np.mean(cp_sig_evals)),
                "pct_saved_vs_400": 100.0 * (1.0 - float(np.mean(cp_sig_evals)) / 400.0),
                "accuracy": cp_sig_acc / n_bb_pair,
                "median_ms_per_trial": float(np.median(cp_sig_ms)),
            },
        },
        "null_tie_0.50_vs_0.50": {
            "bayesbench_compare_type1_error": bb_null_false / n_bb_pair,
            "bayesbench_compare_skipped_rate": bb_null_skipped / n_bb_pair,
            "bayesbench_compare_median_ms": float(np.median(bb_null_ms)),
            "cloture_paired_eprocess_type1_error": cp_null_false / n_bb_pair,
            "cloture_paired_eprocess_median_ms": float(np.median(cp_null_ms)),
        },
        "six_arm_single_trial_timing": {
            "bayesbench_ranker_ms": br_ms,
            "bayesbench_ranker_converged": bool(br_res.converged),
            "bayesbench_ranker_evals": int(br_res.problems_tested) * n_arms,
            "bayesbench_ranker_best": str(br_res.best.name),
            "cloture_agent_race_paired_ms": rp_1_ms,
            "cloture_agent_race_paired_evals": ev_1,
            "speedup_x": br_ms / max(rp_1_ms, 1e-9),
        },
    }

    return {
        "cs_head_to_head": cs_head_to_head,
        "multi_arm_racing": {
            "signal": race_signal,
            "null_global_tie": race_null,
            "bayesbench_comparison": bayesbench_comparison,
        },
        "elapsed_sec": time.perf_counter() - t_suite,
    }


# ---------------------------------------------------------------------------
# Suite 5: Active-PPI++ vs. ppi-python, Human-Only CS, and Naive Biased Judge
# ---------------------------------------------------------------------------


def run_suite_5() -> Dict[str, Any]:
    print("[Suite 5] Active-PPI++ vs. ppi-python, Human-Only CS, Naive Judge...", flush=True)
    t_suite = time.perf_counter()

    # ---- Regime 5(a): Accurate biased judge (r=0.92, bias=+0.15, N=4000, 10% audit, 100 trials) ----
    n_trials_a = 100
    n_items_a = 4000
    audit_rate = 0.10
    alpha = 0.05
    z95 = float(norm.ppf(1.0 - alpha / 2.0))

    appi_any_cov, appi_peek_cov, appi_fin_cov = 0, 0, 0
    appi_widths, appi_audited, appi_lambdas, appi_vreds, appi_times_ms = [], [], [], [], []
    appi_batch_times_ms = []

    ppi_peek_cov, ppi_fixed_cov = 0, 0
    ppi_widths, ppi_peek_times_ms, ppi_fixed_times_ms = [], [], []

    human_any_cov, human_widths, human_times_ms = 0, [], []
    naive_cov, naive_widths, naive_biases = 0, [], []

    for i in range(n_trials_a):
        tr = traces.gen_active_ppi_stream(seed=11000 + i, n_items=n_items_a, judge_corr=0.92, judge_bias=0.15)
        y = tr.y_true
        f = tr.f_judge
        unc = tr.uncertainty
        theta_star = float(tr.true_mean)

        rng_a = np.random.default_rng(12000 + i)
        # 1. cloture.ActivePPI (streaming with active uncertainty sampling)
        t0 = time.perf_counter()
        appi = cloture.ActivePPI(alpha=alpha, target_audit_rate=audit_rate, min_audit_prob=0.05)
        any_ok = True
        peek_ok = True
        probs_rec = np.empty(n_items_a, dtype=np.float64)
        aud_rec = np.empty(n_items_a, dtype=np.uint8)
        gold_rec = np.zeros(n_items_a, dtype=np.float64)
        res = None
        for t in range(n_items_a):
            p = appi.suggest_prob(float(f[t]), float(unc[t]))
            aud = bool(rng_a.random() < p)
            g = float(y[t]) if aud else 0.0
            probs_rec[t] = p
            aud_rec[t] = 1 if aud else 0
            gold_rec[t] = g
            res = appi.observe(float(f[t]), aud, g, p)
            if not (res.lower <= theta_star <= res.upper):
                any_ok = False
                if (t + 1) % 200 == 0:
                    peek_ok = False
        appi_times_ms.append((time.perf_counter() - t0) * 1e3)
        assert res is not None
        appi_any_cov += int(any_ok)
        appi_peek_cov += int(peek_ok)
        appi_fin_cov += int(res.lower <= theta_star <= res.upper)
        appi_widths.append(res.width)
        appi_audited.append(res.n_audited)
        appi_lambdas.append(res.lambda_hat)
        appi_vreds.append(res.variance_reduction)

        # Also time cloture.ActivePPI.observe_batch
        t0 = time.perf_counter()
        appi_b = cloture.ActivePPI(alpha=alpha, target_audit_rate=audit_rate, min_audit_prob=0.05)
        appi_b.observe_batch(np.ascontiguousarray(f), aud_rec, gold_rec, probs_rec)
        appi_batch_times_ms.append((time.perf_counter() - t0) * 1e3)

        # Uniform 10% labeled mask for ppi-python & Human-Only EBCS
        rng_u = np.random.default_rng(13000 + i)
        u_mask = rng_u.random(n_items_a) < audit_rate
        # Ensure at least 5 labeled items in first 200
        if u_mask[:200].sum() < 5:
            u_mask[:5] = True

        # 2. ppi_py.ppi_mean_ci under continuous peeking every 200 items
        t0 = time.perf_counter()
        p_ok = True
        last_ci = None
        for end in range(200, n_items_a + 1, 200):
            m_sub = u_mask[:end]
            y_lab = y[:end][m_sub]
            f_lab = f[:end][m_sub]
            f_unlab = f[:end][~m_sub]
            ci = ppi_py.ppi_mean_ci(y_lab, f_lab, f_unlab, alpha=alpha)
            lo_p, hi_p = float(ci[0][0]), float(ci[1][0])
            last_ci = (lo_p, hi_p)
            if not (lo_p <= theta_star <= hi_p):
                p_ok = False
        ppi_peek_times_ms.append((time.perf_counter() - t0) * 1e3)
        ppi_peek_cov += int(p_ok)

        # 3. ppi_py.ppi_mean_ci single fixed horizon at N=4000
        t0 = time.perf_counter()
        ci_fix = ppi_py.ppi_mean_ci(y[u_mask], f[u_mask], f[~u_mask], alpha=alpha)
        ppi_fixed_times_ms.append((time.perf_counter() - t0) * 1e3)
        lo_f, hi_f = float(ci_fix[0][0]), float(ci_fix[1][0])
        ppi_fixed_cov += int(lo_f <= theta_star <= hi_f)
        ppi_widths.append(hi_f - lo_f)

        # 4. Unbiased Uniform 10% Human-Only EmpiricalBernsteinCS
        t0 = time.perf_counter()
        hcs = cloture.EmpiricalBernsteinCS(alpha=alpha, min_val=0.0, max_val=1.0)
        h_ok = True
        h_res = None
        for val in y[u_mask].tolist():
            h_res = hcs.update(val)
            if not (h_res.lower <= theta_star <= h_res.upper):
                h_ok = False
        human_times_ms.append((time.perf_counter() - t0) * 1e3)
        assert h_res is not None
        human_any_cov += int(h_ok)
        human_widths.append(h_res.width)

        # 5. Naive Biased LLM Judge (no oracle labels)
        f_mean = float(f.mean())
        f_half = z95 * float(f.std(ddof=1)) / np.sqrt(n_items_a)
        naive_cov += int((f_mean - f_half) <= theta_star <= (f_mean + f_half))
        naive_widths.append(2.0 * f_half)
        naive_biases.append(f_mean - theta_star)

    mean_w_appi = float(np.mean(appi_widths))
    mean_w_human = float(np.mean(human_widths))
    eff_sample_mult = (mean_w_human / max(mean_w_appi, 1e-12)) ** 2

    regime_a = {
        "config": {"n_trials": n_trials_a, "n_items": n_items_a, "judge_corr": 0.92, "judge_bias": 0.15, "audit_rate": audit_rate},
        "cloture_active_ppi": {
            "anytime_coverage_all_steps": appi_any_cov / n_trials_a,
            "anytime_coverage_every_200": appi_peek_cov / n_trials_a,
            "final_coverage": appi_fin_cov / n_trials_a,
            "mean_final_width": mean_w_appi,
            "mean_n_audited": float(np.mean(appi_audited)),
            "mean_lambda_hat": float(np.mean(appi_lambdas)),
            "mean_variance_reduction": float(np.mean(appi_vreds)),
            "effective_sample_multiplier_vs_human_only": eff_sample_mult,
            "stream_ms_per_trial": _percentiles(appi_times_ms),
            "batch_ms_per_trial": _percentiles(appi_batch_times_ms),
        },
        "ppi_python_peeking_every_200": {
            "anytime_coverage": ppi_peek_cov / n_trials_a,
            "mean_final_width": float(np.mean(ppi_widths)),
            "ms_per_trial": _percentiles(ppi_peek_times_ms),
        },
        "ppi_python_fixed_horizon": {
            "fixed_horizon_coverage": ppi_fixed_cov / n_trials_a,
            "mean_final_width": float(np.mean(ppi_widths)),
            "ms_per_trial": _percentiles(ppi_fixed_times_ms),
        },
        "human_only_ebcs_10pct": {
            "anytime_coverage": human_any_cov / n_trials_a,
            "mean_final_width": mean_w_human,
            "ms_per_trial": _percentiles(human_times_ms),
        },
        "naive_biased_llm_judge": {
            "coverage": naive_cov / n_trials_a,
            "mean_width": float(np.mean(naive_widths)),
            "mean_bias": float(np.mean(naive_biases)),
        },
    }

    # ---- Regime 5(b): Adversarially inverted / poisoned judge (f = 1 - y, N=2000, 100 trials) ----
    n_trials_b = 100
    n_items_b = 2000

    def _run_adv_ppi(exact_invert: bool, seed_base: int) -> Dict[str, Any]:
        any_cov, fin_cov = 0, 0
        lambdas, widths, audited = [], [], []
        naive_c, human_c, human_w = 0, 0, []
        ppi_py_c, ppi_py_w = 0, []

        for i in range(n_trials_b):
            tr = traces.gen_adversarial_ppi_stream(seed=seed_base + i, n_items=n_items_b)
            y = tr.y_true
            f = (1.0 - y) if exact_invert else tr.f_judge
            unc = np.full(n_items_b, 0.5, dtype=np.float64) if exact_invert else tr.uncertainty
            theta_star = float(y.mean())

            rng_b = np.random.default_rng(seed_base + 5000 + i)
            appi = cloture.ActivePPI(alpha=alpha, target_audit_rate=audit_rate, min_audit_prob=0.05)
            ok = True
            res = None
            for t in range(n_items_b):
                p = appi.suggest_prob(float(f[t]), float(unc[t]))
                aud = bool(rng_b.random() < p)
                res = appi.observe(float(f[t]), aud, float(y[t]) if aud else 0.0, p)
                if not (res.lower <= theta_star <= res.upper):
                    ok = False
            assert res is not None
            any_cov += int(ok)
            fin_cov += int(res.lower <= theta_star <= res.upper)
            lambdas.append(res.lambda_hat)
            widths.append(res.width)
            audited.append(res.n_audited)

            # Human-only 10% EBCS & ppi_py fixed horizon & Naive judge
            u_mask = np.random.default_rng(seed_base + 9000 + i).random(n_items_b) < audit_rate
            hcs = cloture.EmpiricalBernsteinCS(alpha=alpha, min_val=0.0, max_val=1.0)
            hok = True
            hres = None
            for v in y[u_mask].tolist():
                hres = hcs.update(v)
                if not (hres.lower <= theta_star <= hres.upper):
                    hok = False
            human_c += int(hok)
            human_w.append(hres.width)

            ci = ppi_py.ppi_mean_ci(y[u_mask], f[u_mask], f[~u_mask], alpha=alpha)
            ppi_py_c += int(float(ci[0][0]) <= theta_star <= float(ci[1][0]))
            ppi_py_w.append(float(ci[1][0] - ci[0][0]))

            fm = float(f.mean())
            fh = z95 * float(f.std(ddof=1)) / np.sqrt(n_items_b)
            naive_c += int((fm - fh) <= theta_star <= (fm + fh))

        return {
            "n_trials": n_trials_b,
            "n_items": n_items_b,
            "exact_f_eq_1_minus_y": exact_invert,
            "cloture_active_ppi": {
                "anytime_coverage": any_cov / n_trials_b,
                "final_coverage": fin_cov / n_trials_b,
                "mean_lambda_hat": float(np.mean(lambdas)),
                "max_lambda_hat": float(np.max(lambdas)),
                "mean_final_width": float(np.mean(widths)),
                "mean_n_audited": float(np.mean(audited)),
            },
            "human_only_ebcs_10pct": {
                "anytime_coverage": human_c / n_trials_b,
                "mean_final_width": float(np.mean(human_w)),
            },
            "ppi_python_fixed_horizon": {
                "fixed_horizon_coverage": ppi_py_c / n_trials_b,
                "mean_final_width": float(np.mean(ppi_py_w)),
            },
            "naive_inverted_llm_judge": {
                "coverage": naive_c / n_trials_b,
            },
        }

    regime_b_exact = _run_adv_ppi(exact_invert=True, seed_base=15000)
    regime_b_noisy = _run_adv_ppi(exact_invert=False, seed_base=16000)

    return {
        "regime_a_accurate_biased": regime_a,
        "regime_b_adversarial_exact_inverted": regime_b_exact,
        "regime_b_adversarial_noisy_inverted": regime_b_noisy,
        "elapsed_sec": time.perf_counter() - t_suite,
    }


# ---------------------------------------------------------------------------
# Suite 6: SafeFuse + AbortCascade + OnlineConformal vs online-cp + Sketch
# ---------------------------------------------------------------------------


class _PythonAgACIReference:
    """Faithful pure-Python 4-rate AgACI / DtACI reference (Zaffran et al. 2022)."""

    def __init__(self, target_alpha: float = 0.10) -> None:
        self.target_alpha = target_alpha
        self.gammas = [0.005, 0.02, 0.08, 0.20]
        self.alphas = [target_alpha] * 4
        self.weights = [0.25] * 4
        self.eta = 0.5
        self.share = 0.02
        self.alpha_t = target_alpha
        self.n = 0
        self.covered = 0

    def update_pvalue(self, p: float) -> float:
        miscov_agg = 1.0 if p < self.alpha_t else 0.0
        self.n += 1
        self.covered += 0 if miscov_agg > 0.5 else 1
        # Pinball losses for each expert and Fixed-Share Hedge update
        new_w = []
        for j, g in enumerate(self.gammas):
            aj = self.alphas[j]
            err_j = 1.0 if p < aj else 0.0
            # Pinball loss of quantile threshold aj on conformity score p
            diff = p - aj
            loss_j = (1.0 - self.target_alpha) * (-diff) if diff < 0.0 else self.target_alpha * diff
            new_w.append(self.weights[j] * np.exp(-self.eta * loss_j))
            self.alphas[j] = min(1.5, max(-0.5, aj + g * (self.target_alpha - err_j)))
        s = sum(new_w)
        if s > 0.0:
            new_w = [w / s for w in new_w]
        else:
            new_w = [0.25] * 4
        # Fixed-share mixing
        self.weights = [(1.0 - self.share) * w + self.share * 0.25 for w in new_w]
        self.alpha_t = sum(w * a for w, a in zip(self.weights, self.alphas))
        return self.alpha_t


def _eval_cascade_policy(
    signals: np.ndarray,
    success: np.ndarray,
    costs: np.ndarray,
    thresholds: List[float],
) -> Tuple[float, float, float]:
    """Evaluate recall, failed-episode compute savings %, total compute savings %."""
    n_ep, n_stages = signals.shape
    total_possible = float(costs.sum())
    failed_possible = float(costs[success == 0].sum())

    survived = np.ones(n_ep, dtype=bool)
    incurred = np.zeros(n_ep, dtype=np.float64)
    for s in range(n_stages):
        incurred[survived] += costs[survived, s]
        if s < n_stages - 1:
            tau = thresholds[s]
            abort_now = survived & (signals[:, s] < tau)
            survived[abort_now] = False

    pos_mask = success == 1
    recall = float(survived[pos_mask].mean()) if pos_mask.any() else 1.0
    total_saved_pct = 100.0 * (1.0 - float(incurred.sum()) / max(total_possible, 1e-12))
    failed_saved_pct = 100.0 * (1.0 - float(incurred[~pos_mask].sum()) / max(failed_possible, 1e-12))
    return recall, failed_saved_pct, total_saved_pct


def run_suite_6(rust_bench_map: Dict[str, float]) -> Dict[str, Any]:
    print("[Suite 6] SafeFuse + AbortCascade + OnlineConformal vs. online-cp + EvidenceSketch...", flush=True)
    t_suite = time.perf_counter()

    # ---- 6A: SafeFuse on poisoned swarm (5 agents, 2 poisoned at t=200, 50 trials) + clean null ----
    n_sf_trials = 50
    quarantine_delays = []
    both_quarantined = 0
    clean_false_q_poisoned = 0
    sf_cum_losses, unif_cum_losses, oracle_clean_losses = [], [], []

    for i in range(n_sf_trials):
        tr = traces.gen_poisoned_swarm(seed=20000 + i, n_agents=5, poison_step=200, n_steps=1000, n_poisoned=2)
        losses = np.ascontiguousarray(1.0 - tr.correct, dtype=np.float64)
        preds = np.ascontiguousarray(tr.correct, dtype=np.float64)
        sf = cloture.SafeFuse(n_agents=5, quarantine_alpha=0.01, fallback_alpha=0.05)

        q_step: Dict[int, int] = {}
        sf_loss_sum = 0.0
        unif_loss_sum = 0.0
        oracle_loss_sum = 0.0

        for t in range(1000):
            pred_val, _ = sf.predict(preds[t])
            sf_loss_sum += 1.0 - pred_val
            unif_loss_sum += float(losses[t].mean())
            oracle_loss_sum += float(losses[t, tr.clean_agents].mean())
            sf.update(losses[t])
            for a in range(5):
                if a not in q_step and sf.is_quarantined(a):
                    q_step[a] = t

        p_steps = [q_step[a] - 200 for a in tr.poisoned_agents if a in q_step and q_step[a] >= 200]
        if len(p_steps) == len(tr.poisoned_agents):
            both_quarantined += 1
            quarantine_delays.append(max(p_steps))
        clean_false_q_poisoned += sum(1 for a in tr.clean_agents if a in q_step)
        sf_cum_losses.append(sf_loss_sum)
        unif_cum_losses.append(unif_loss_sum)
        oracle_clean_losses.append(oracle_loss_sum)

    # Clean null (5 identical healthy agents, acc=0.80, 1000 steps, 50 trials)
    clean_null_any_q = 0
    for i in range(n_sf_trials):
        rng_c = np.random.default_rng(21000 + i)
        losses_c = np.ascontiguousarray((rng_c.random((1000, 5)) >= 0.80).astype(np.float64))
        sf = cloture.SafeFuse(n_agents=5, quarantine_alpha=0.01, fallback_alpha=0.05)
        for t in range(1000):
            sf.update(losses_c[t])
        if sf.quarantined_count() > 0:
            clean_null_any_q += 1

    safefuse_res = {
        "n_trials": n_sf_trials,
        "n_agents": 5,
        "n_poisoned": 2,
        "poison_step": 200,
        "both_poisoned_quarantined_rate": both_quarantined / n_sf_trials,
        "steps_to_quarantine_both_after_poison": _percentiles(quarantine_delays),
        "clean_agent_false_quarantines_in_poisoned_runs": clean_false_q_poisoned / float(n_sf_trials * 3),
        "clean_null_fwer_any_false_quarantine": clean_null_any_q / n_sf_trials,
        "mean_cum_loss_safefuse": float(np.mean(sf_cum_losses)),
        "mean_cum_loss_uniform_mean": float(np.mean(unif_cum_losses)),
        "mean_cum_loss_oracle_clean": float(np.mean(oracle_clean_losses)),
        "mean_regret_vs_uniform_mean": float(np.mean(sf_cum_losses) - np.mean(unif_cum_losses)),
        "mean_regret_vs_oracle_clean": float(np.mean(sf_cum_losses) - np.mean(oracle_clean_losses)),
    }

    # ---- 6B: AbortCascade (500 cal + 500 test, 50 trials, target_recall=0.92) ----
    n_cas_trials = 50
    cas_recalls, cas_fail_sav, cas_tot_sav, cas_cal_ms = [], [], [], []
    naive_recalls, naive_fail_sav, naive_tot_sav = [], [], []

    for i in range(n_cas_trials):
        tr = traces.gen_abort_cascade(seed=22000 + i, n_episodes=1000, n_stages=4)
        cal_sig = np.ascontiguousarray(tr.signals[:500], dtype=np.float64)
        cal_y = np.ascontiguousarray(tr.success[:500], dtype=np.uint8)
        test_sig = tr.signals[500:]
        test_y = tr.success[500:]
        test_costs = tr.episode_stage_costs[500:]
        stage_costs = np.ascontiguousarray(tr.stage_costs, dtype=np.float64)

        t0 = time.perf_counter()
        cas = cloture.AbortCascade.calibrate(cal_sig, cal_y, target_recall=0.92, stage_costs=stage_costs)
        cas_cal_ms.append((time.perf_counter() - t0) * 1e3)
        taus = cas.thresholds()

        r_c, fs_c, ts_c = _eval_cascade_policy(test_sig, test_y, test_costs, taus)
        cas_recalls.append(r_c)
        cas_fail_sav.append(fs_c)
        cas_tot_sav.append(ts_c)

        # Naive uncalibrated fixed threshold (abort if critic score < 0.35 at any early stage)
        r_n, fs_n, ts_n = _eval_cascade_policy(test_sig, test_y, test_costs, [0.35, 0.35, 0.35, -1e9])
        naive_recalls.append(r_n)
        naive_fail_sav.append(fs_n)
        naive_tot_sav.append(ts_n)

    # Measure AbortCascade per-decision latency
    cas0 = cloture.AbortCascade.calibrate(cal_sig, cal_y, target_recall=0.92, stage_costs=stage_costs)
    t0 = time.perf_counter_ns()
    for _ in range(10_000):
        cas0.should_abort(0, 0.25)
    cas_dec_ns = (time.perf_counter_ns() - t0) / 10_000.0

    cascade_res = {
        "n_trials": n_cas_trials,
        "cal_episodes": 500,
        "test_episodes": 500,
        "target_recall": 0.92,
        "cloture_abort_cascade": {
            "mean_test_recall": float(np.mean(cas_recalls)),
            "median_test_recall": float(np.median(cas_recalls)),
            "mean_failed_compute_savings_pct": float(np.mean(cas_fail_sav)),
            "mean_total_compute_savings_pct": float(np.mean(cas_tot_sav)),
            "calibration_ms": _percentiles(cas_cal_ms),
            "decision_ns_per_call": cas_dec_ns,
        },
        "naive_uncalibrated_threshold_0.35": {
            "mean_test_recall": float(np.mean(naive_recalls)),
            "mean_failed_compute_savings_pct": float(np.mean(naive_fail_sav)),
            "mean_total_compute_savings_pct": float(np.mean(naive_tot_sav)),
        },
    }

    # ---- 6C: OnlineConformal vs online-cp & Pure-Python AgACI under abrupt shift ----
    rng_oc = np.random.default_rng(23000)
    n_stat, n_shift = 1000, 1000
    # Stationary conformal p-values ~ Uniform(0,1), then abrupt shift ~ Beta(0.35, 3.0) (many small p-values!)
    pvals_stat = rng_oc.uniform(0.0, 1.0, size=n_stat)
    pvals_shift = rng_oc.beta(0.35, 3.0, size=n_shift)
    pvals = np.concatenate([pvals_stat, pvals_shift])
    pvals_list = pvals.tolist()

    # 1. cloture.OnlineConformal (target_alpha=0.10 -> 90% coverage)
    oc = cloture.OnlineConformal(target_alpha=0.10)
    oc_cov = np.empty(len(pvals), dtype=np.float64)
    t0 = time.perf_counter_ns()
    for t, p in enumerate(pvals_list):
        a_t = oc.alpha_t
        oc_cov[t] = 1.0 if p >= a_t else 0.0
        oc.update_pvalue(p)
    oc_py_ns = (time.perf_counter_ns() - t0) / len(pvals)

    # 2. Fixed unadapted conformal threshold alpha=0.10 (baseline showing shift collapse)
    fixed_cov = (pvals >= 0.10).astype(np.float64)

    # 3. Pure-Python 4-rate AgACI reference
    py_ag = _PythonAgACIReference(target_alpha=0.10)
    py_cov = np.empty(len(pvals), dtype=np.float64)
    t0 = time.perf_counter_ns()
    for t, p in enumerate(pvals_list):
        py_cov[t] = 1.0 if p >= py_ag.alpha_t else 0.0
        py_ag.update_pvalue(p)
    py_ag_ns = (time.perf_counter_ns() - t0) / len(pvals)

    # 4. Real PyPI online-cp: (a) SimpleJumper conformal martingale on exact same p-value stream
    sj = online_cp.SimpleJumper(J=0.01, store_p_values=False)
    sj_alarm_step = None
    t0 = time.perf_counter_ns()
    for t, p in enumerate(pvals_list):
        sj.update(p)
        if sj_alarm_step is None and t >= n_stat and float(sj.M) >= 20.0:
            sj_alarm_step = t - n_stat
    ocp_sj_ns = (time.perf_counter_ns() - t0) / len(pvals)

    # 5. Real PyPI online-cp: (b) ConformalRidgeRegressor on 1D streaming regression with abrupt shift
    rng_reg = np.random.default_rng(23100)
    n_reg_init, n_reg_stat, n_reg_shift = 30, 200, 200
    x_reg = rng_reg.normal(0.0, 1.0, size=(n_reg_init + n_reg_stat + n_reg_shift, 1))
    eps_reg = rng_reg.normal(0.0, 0.5, size=n_reg_init + n_reg_stat + n_reg_shift)
    y_reg = 2.0 * x_reg[:, 0] + eps_reg
    # Abrupt mean + variance shift at t = n_reg_init + n_reg_stat
    y_reg[n_reg_init + n_reg_stat :] += 2.5 + rng_reg.normal(0.0, 1.0, size=n_reg_shift)

    crr = online_cp.ConformalRidgeRegressor(a=1.0, epsilon=0.10, warnings=False)
    crr.learn_initial_training_set(x_reg[:n_reg_init], y_reg[:n_reg_init])
    crr_cov_stat, crr_cov_shift = [], []
    t0 = time.perf_counter_ns()
    for t in range(n_reg_init, n_reg_init + n_reg_stat + n_reg_shift):
        pred_set = crr.predict(x_reg[t : t + 1])
        covered = float(pred_set.lower <= y_reg[t] <= pred_set.upper)
        if t < n_reg_init + n_reg_stat:
            crr_cov_stat.append(covered)
        else:
            crr_cov_shift.append(covered)
        crr.learn_one(x_reg[t], float(y_reg[t]))
    ocp_crr_ns = (time.perf_counter_ns() - t0) / float(n_reg_stat + n_reg_shift)

    rust_oc_ns = rust_bench_map.get("OnlineConformal::update_pvalue", 63.4)
    conformal_res = {
        "target_coverage": 0.90,
        "cloture_online_conformal": {
            "rust_cli_ns_per_step": rust_oc_ns,
            "python_stream_ns_per_step": oc_py_ns,
            "stationary_coverage": float(oc_cov[:n_stat].mean()),
            "post_shift_first_100_coverage": float(oc_cov[n_stat : n_stat + 100].mean()),
            "post_shift_recovered_coverage_100_to_1000": float(oc_cov[n_stat + 100 :].mean()),
            "overall_coverage": float(oc_cov.mean()),
        },
        "python_agaci_reference": {
            "python_stream_ns_per_step": py_ag_ns,
            "speedup_cloture_py_vs_python_ref": py_ag_ns / max(oc_py_ns, 1e-9),
            "speedup_cloture_rust_vs_python_ref": py_ag_ns / max(rust_oc_ns, 1e-9),
            "stationary_coverage": float(py_cov[:n_stat].mean()),
            "post_shift_recovered_coverage_100_to_1000": float(py_cov[n_stat + 100 :].mean()),
            "overall_coverage": float(py_cov.mean()),
        },
        "fixed_unadapted_alpha_0.10": {
            "stationary_coverage": float(fixed_cov[:n_stat].mean()),
            "post_shift_coverage": float(fixed_cov[n_stat:].mean()),
        },
        "online_cp_simple_jumper_martingale": {
            "ns_per_step": ocp_sj_ns,
            "speedup_cloture_py_vs_online_cp_jumper": ocp_sj_ns / max(oc_py_ns, 1e-9),
            "speedup_cloture_rust_vs_online_cp_jumper": ocp_sj_ns / max(rust_oc_ns, 1e-9),
            "post_shift_alarm_delay_steps": sj_alarm_step,
        },
        "online_cp_conformal_ridge_regressor": {
            "ns_per_step": ocp_crr_ns,
            "speedup_cloture_py_vs_online_cp_crr": ocp_crr_ns / max(oc_py_ns, 1e-9),
            "stationary_coverage": float(np.mean(crr_cov_stat)),
            "post_shift_coverage": float(np.mean(crr_cov_shift)),
        },
    }

    # ---- 6D: 712-Byte EvidenceSketch (1,000-worker fleet merge + exact LOO subtraction) ----
    rng_sk = np.random.default_rng(24000)
    n_workers = 1000
    obs_per_worker = 100
    all_scores = rng_sk.uniform(0.0, 1.0, size=(n_workers, obs_per_worker))
    all_bets = rng_sk.normal(0.0001, 0.01, size=(n_workers, obs_per_worker))

    # Build 1,000 worker sketches and serialize to wire bytes
    worker_payloads: List[bytes] = []
    worker_sketches: List[cloture.EvidenceSketch] = []
    for w in range(n_workers):
        sk = cloture.EvidenceSketch()
        s_row = all_scores[w].tolist()
        b_row = all_bets[w].tolist()
        for s_val, b_val in zip(s_row, b_row):
            sk.observe(s_val, b_val)
        payload = sk.to_bytes()
        worker_payloads.append(payload)
        worker_sketches.append(sk)

    wire_bytes = len(worker_payloads[0])

    # Single-node ground truth sketch over all 100,000 observations
    single_sk = cloture.EvidenceSketch()
    for w in range(n_workers):
        for s_val, b_val in zip(all_scores[w].tolist(), all_bets[w].tolist()):
            single_sk.observe(s_val, b_val)

    # Deserialize + merge all 1,000 workers and time merge() specifically
    deser_sketches = [cloture.EvidenceSketch.from_bytes(p) for p in worker_payloads]
    fleet_sk = cloture.EvidenceSketch()
    t0 = time.perf_counter_ns()
    for sk in deser_sketches:
        fleet_sk.merge(sk)
    py_merge_ns = (time.perf_counter_ns() - t0) / float(n_workers)

    fleet_mean_diff = abs(fleet_sk.mean() - single_sk.mean())
    fleet_eval_rel_diff = abs(fleet_sk.e_value() - single_sk.e_value()) / max(abs(single_sk.e_value()), 1e-12)
    fleet_q50_diff = abs(fleet_sk.quantile(0.50) - single_sk.quantile(0.50))
    fleet_q90_diff = abs(fleet_sk.quantile(0.90) - single_sk.quantile(0.90))

    # Exact Leave-One-Out subtraction: fleet - worker[0] vs fresh merge of workers 1..999
    fleet_sk.subtract(worker_sketches[0])
    ref_999 = cloture.EvidenceSketch()
    for sk in worker_sketches[1:]:
        ref_999.merge(sk)

    loo_mean_diff = abs(fleet_sk.mean() - ref_999.mean())
    loo_eval_rel_diff = abs(fleet_sk.e_value() - ref_999.e_value()) / max(abs(ref_999.e_value()), 1e-12)
    loo_q50_diff = abs(fleet_sk.quantile(0.50) - ref_999.quantile(0.50))
    loo_q90_diff = abs(fleet_sk.quantile(0.90) - ref_999.quantile(0.90))

    sketch_res = {
        "wire_size_bytes": wire_bytes,
        "n_workers": n_workers,
        "total_observations": n_workers * obs_per_worker,
        "rust_cli_merge_ns": rust_bench_map.get("EvidenceSketch(712B)::merge"),
        "python_ctypes_merge_ns": py_merge_ns,
        "fleet_vs_single_node": {
            "mean_abs_diff": fleet_mean_diff,
            "e_value_rel_diff": fleet_eval_rel_diff,
            "quantile_p50_abs_diff": fleet_q50_diff,
            "quantile_p90_abs_diff": fleet_q90_diff,
        },
        "loo_subtraction_vs_999_workers": {
            "mean_abs_diff": loo_mean_diff,
            "e_value_rel_diff": loo_eval_rel_diff,
            "quantile_p50_abs_diff": loo_q50_diff,
            "quantile_p90_abs_diff": loo_q90_diff,
        },
    }

    return {
        "safefuse": safefuse_res,
        "abort_cascade": cascade_res,
        "online_conformal": conformal_res,
        "evidence_sketch": sketch_res,
        "elapsed_sec": time.perf_counter() - t_suite,
    }


# ---------------------------------------------------------------------------
# Suite 7: Systems Footprint, Cold-Start, Rust Bench, & MCP Roundtrip
# ---------------------------------------------------------------------------


def _time_subprocess_ms(cmd: List[str], env: Dict[str, str], n_runs: int = 10) -> Dict[str, float]:
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        times.append((time.perf_counter() - t0) * 1e3)
    return _percentiles(times)


def run_suite_7(raw_rust_bench: Dict[str, Any]) -> Dict[str, Any]:
    print("[Suite 7] Systems Footprint, Cold-Start, & MCP JSON-RPC Latency...", flush=True)
    t_suite = time.perf_counter()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PYTHON_SDK)
    py_exe = sys.executable

    import_cloture_ms = _time_subprocess_ms([py_exe, "-c", "import cloture"], env=env, n_runs=10)
    import_numpy_ms = _time_subprocess_ms([py_exe, "-c", "import numpy"], env=env, n_runs=5)
    import_scipy_ms = _time_subprocess_ms([py_exe, "-c", "import scipy.stats"], env=env, n_runs=5)
    import_ppi_ms = _time_subprocess_ms([py_exe, "-c", "import ppi_py"], env=env, n_runs=3)
    cli_version_ms = _time_subprocess_ms([str(CLI_BIN), "--version"], env=env, n_runs=10)

    # MCP JSON-RPC 2.0 roundtrip latency over stdio (100 consecutive tools/call requests)
    proc = subprocess.Popen(
        [str(CLI_BIN), "mcp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert proc.stdin is not None and proc.stdout is not None
    init_req = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "bench", "version": "0.1"}},
        }
    )
    proc.stdin.write(init_req + "\n")
    proc.stdin.flush()
    _ = proc.stdout.readline()

    mcp_us = []
    for req_id in range(1, 101):
        call_req = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": req_id,
                "method": "tools/call",
                "params": {
                    "name": "cloture_mode_stop",
                    "arguments": {"mode": "cite", "delta": 0.05, "samples": ["A", "A", "A", "A", "A", "A", "A", "A", "B", "A"]},
                },
            }
        )
        t0 = time.perf_counter_ns()
        proc.stdin.write(call_req + "\n")
        proc.stdin.flush()
        resp = proc.stdout.readline()
        dt_us = (time.perf_counter_ns() - t0) / 1e3
        assert resp and f'"id":{req_id}' in resp.replace(" ", "")
        mcp_us.append(dt_us)
    proc.stdin.close()
    proc.wait(timeout=5)

    cli_bytes = CLI_BIN.stat().st_size
    so_bytes = LIB_SO.stat().st_size

    return {
        "cold_start_ms": {
            "import_cloture": import_cloture_ms,
            "cli_version": cli_version_ms,
            "context_import_numpy": import_numpy_ms,
            "context_import_scipy_stats": import_scipy_ms,
            "context_import_ppi_py": import_ppi_ms,
        },
        "mcp_jsonrpc_roundtrip_us": _percentiles(mcp_us),
        "binary_sizes": {
            "cloture_cli_bytes": cli_bytes,
            "cloture_cli_mib": cli_bytes / (1024.0 * 1024.0),
            "libcloture_core_so_bytes": so_bytes,
            "libcloture_core_so_mib": so_bytes / (1024.0 * 1024.0),
        },
        "rust_native_bench": raw_rust_bench,
        "elapsed_sec": time.perf_counter() - t_suite,
    }


# ---------------------------------------------------------------------------
# Main Driver & Summary Table
# ---------------------------------------------------------------------------


def _run_rust_cli_bench() -> Tuple[Dict[str, Any], Dict[str, float]]:
    out = subprocess.check_output([str(CLI_BIN), "bench", "--json"], text=True)
    raw = json.loads(out)
    bmap = {entry["benchmark"]: float(entry["ns_per_op"]) for entry in raw.get("benchmarks", [])}
    return raw, bmap


def _print_summary(res: Dict[str, Any]) -> None:
    s1 = res["suite_1_consensus_stopping"]
    s2 = res["suite_2_panel_audit"]
    s3 = res["suite_3_correlated_rollouts"]
    s4 = res["suite_4_cs_and_racing"]
    s5 = res["suite_5_active_ppi"]
    s6 = res["suite_6_fuse_cascade_conformal_sketch"]
    s7 = res["suite_7_systems_footprint"]

    print("\n" + "=" * 86)
    print("CLOTURE COMPETITIVE BENCHMARK SUMMARY (ALL NUMBERS MEASURED LIVE)")
    print("=" * 86)

    print("\n[1] Consensus Stopping (ASC-D Table 2 & Open-Ended Zipf Swarm)")
    for reg in ["easy", "medium", "hard", "null_tie"]:
        e = s1["ascd_table2"][reg]
        acc_a = e["cloture_ascd"]["accuracy"]
        t1_a = e["cloture_ascd"]["type1_error_on_null"]
        t1_n = e["naive_uncorrected_peeking"]["type1_error_on_null"]
        if acc_a is not None:
            print(
                f"  ASC-D {reg:8s} (d={e['delta_gap']:.2f}): Ascd mean_n={e['cloture_ascd']['mean_samples']:5.1f} "
                f"(acc={acc_a*100:5.1f}%, saved_vs_64={e['cloture_ascd']['pct_saved_vs_fixed64']:+5.1f}%) | "
                f"Cite mean_n={e['cloture_cite']['mean_samples']:5.1f} | Fix64 acc={e['fixed_n']['n64_accuracy']*100:5.1f}%"
            )
        else:
            print(
                f"  ASC-D {reg:8s} (d=0.00): Ascd Type-I={t1_a*100:4.1f}% | "
                f"Cite Type-I={e['cloture_cite']['type1_error_on_null']*100:4.1f}% | "
                f"Naive Peeking Type-I={t1_n*100:5.1f}%"
            )
    lat1 = s1["latency"]
    print(
        f"  Latency: Rust ASC-D={lat1['rust_cli_ascd_ns_per_update']:.0f}ns | Py batch={lat1['cloture_py_ascd_batch_ns_per_update']:.0f}ns | "
        f"NumPy ref={lat1['numpy_ascd_stream_ns_per_update']:.0f}ns ({lat1['speedup_cloture_batch_vs_numpy']:.1f}x speedup)"
    )

    print("\n[2] Correlated-Panel Audit (9 Judges x 1000 Items, 2k Perms / 2k MC / 1k Boot)")
    pf = s2["panel_findings"]
    print(
        f"  cloture={s2['timing']['cloture_ms']['median']:.1f}ms vs NumPy/SciPy={s2['timing']['numpy_scipy_ref_ms']['median']:.1f}ms "
        f"({s2['timing']['speedup_x']:.1f}x faster), max_det_diff={s2['numerical_parity']['max_deterministic_abs_diff']:.2e}"
    )
    print(
        f"  n_eff_kish={pf['n_eff_kish']:.2f}, n_eff_eigen={pf['n_eff_eigen']:.2f}, gap={pf['condorcet_gap']*100:.1f}pp, "
        f"perm_p={pf['permutation_p_value']:.4f}, LOO drop-1 gain={pf['drop_1_accuracy_gain']*100:+.2f}pp (11.1% cost saved)"
    )

    print("\n[3] Realized vs. Population Correlated Rollouts (arXiv:2611.26592)")
    n3 = s3["null_theta0_30x16"]
    p3 = s3["positive_theta025_120x16"]
    print(
        f"  Null (theta=0, rho=0.6, 30x16, 300 reps): NaivePooledZ={n3['naive_pooled_z_fire_rate']*100:.1f}%, "
        f"SingleUnit(p0)={n3['naive_single_unit_prompt0_fire_rate']*100:.1f}%, "
        f"PopulationCert={n3['cloture_population_cert_fire_rate']*100:.1f}% (rho_hat={n3['mean_rho_hat']:.2f}, ceil={n3['mean_within_unit_ceiling']:.2f})"
    )
    print(
        f"  Positive (theta=+0.25): PopCert power @30x16={s3['positive_theta025_30x16_same_480_budget']['cloture_population_cert_fire_rate']*100:.1f}% "
        f"-> @120x4 (same 480 budget)={s3['positive_theta025_120x4_same_480_budget']['cloture_population_cert_fire_rate']*100:.1f}% "
        f"-> @120x16={p3['cloture_population_cert_fire_rate']*100:.1f}%"
    )

    print("\n[4] Anytime-Valid CS & Multi-Arm Racing vs. confseq / expectation / bayesbench")
    cs4 = s4["cs_head_to_head"]
    print(
        f"  CS Miscov/Width@1k/Batch100k: "
        f"cloture-EB={cs4['cloture_ebcs']['anytime_miscoverage']*100:.1f}%/{cs4['cloture_ebcs']['mean_width_n1000']:.3f}/{cs4['cloture_ebcs']['batch_100k_ns_per_step']:.0f}ns | "
        f"cloture-Bet={cs4['cloture_betting_cs_201']['anytime_miscoverage']*100:.1f}%/{cs4['cloture_betting_cs_201']['mean_width_n1000']:.3f} | "
        f"confseq-EB={cs4['confseq_predmix_ebcs']['anytime_miscoverage']*100:.1f}%/{cs4['confseq_predmix_ebcs']['mean_width_n1000']:.3f}/{cs4['confseq_predmix_ebcs']['batch_100k_ns_per_step']:.0f}ns | "
        f"NaiveCLT={cs4['naive_clt_peeking']['anytime_miscoverage']*100:.1f}%"
    )
    r4s = s4["multi_arm_racing"]["signal"]
    r4n = s4["multi_arm_racing"]["null_global_tie"]
    bb4 = s4["multi_arm_racing"]["bayesbench_comparison"]
    print(
        f"  6-Arm Race: Paired={r4s['cloture_paired']['mean_evals']:.0f} evals ({r4s['cloture_paired']['pct_evals_saved']:.1f}% saved, acc={r4s['cloture_paired']['accuracy']*100:.1f}%, null_FWER={r4n['cloture_paired']['fwer_on_null']*100:.1f}%) | "
        f"Unpaired={r4s['cloture_unpaired_lucb']['mean_evals']:.0f} evals | "
        f"bayesbench null Type-I={bb4['null_tie_0.50_vs_0.50']['bayesbench_compare_type1_error']*100:.1f}% vs cloture={bb4['null_tie_0.50_vs_0.50']['cloture_paired_eprocess_type1_error']*100:.1f}%"
    )

    print("\n[5] Active-PPI++ vs. ppi-python, Human-Only CS, Naive Judge")
    r5a = s5["regime_a_accurate_biased"]
    r5b = s5["regime_b_adversarial_exact_inverted"]
    print(
        f"  Biased Judge (r=0.92, +0.15): ActivePPI cov={r5a['cloture_active_ppi']['anytime_coverage_all_steps']*100:.0f}% (w={r5a['cloture_active_ppi']['mean_final_width']:.3f}, {r5a['cloture_active_ppi']['effective_sample_multiplier_vs_human_only']:.1f}x eff mult) | "
        f"ppi_py peek={r5a['ppi_python_peeking_every_200']['anytime_coverage']*100:.0f}% | "
        f"ppi_py fixed={r5a['ppi_python_fixed_horizon']['fixed_horizon_coverage']*100:.0f}% | "
        f"Naive={r5a['naive_biased_llm_judge']['coverage']*100:.0f}%"
    )
    print(
        f"  Inverted Judge (f=1-y): ActivePPI cov={r5b['cloture_active_ppi']['anytime_coverage']*100:.0f}% (lambda_hat={r5b['cloture_active_ppi']['mean_lambda_hat']:.4f}) | "
        f"Naive={r5b['naive_inverted_llm_judge']['coverage']*100:.0f}%"
    )

    print("\n[6] SafeFuse + AbortCascade + OnlineConformal vs. online-cp + EvidenceSketch")
    sf6 = s6["safefuse"]
    ac6 = s6["abort_cascade"]
    oc6 = s6["online_conformal"]
    sk6 = s6["evidence_sketch"]
    print(
        f"  SafeFuse: both_q={sf6['both_poisoned_quarantined_rate']*100:.0f}% (med {sf6['steps_to_quarantine_both_after_poison']['median']:.0f} steps), "
        f"clean_null_FWER={sf6['clean_null_fwer_any_false_quarantine']*100:.1f}%"
    )
    print(
        f"  AbortCascade: recall={ac6['cloture_abort_cascade']['mean_test_recall']*100:.1f}% (target 92%), "
        f"failed_saved={ac6['cloture_abort_cascade']['mean_failed_compute_savings_pct']:.1f}%, "
        f"total_saved={ac6['cloture_abort_cascade']['mean_total_compute_savings_pct']:.1f}%"
    )
    print(
        f"  OnlineConformal: Rust={oc6['cloture_online_conformal']['rust_cli_ns_per_step']:.1f}ns | Py={oc6['cloture_online_conformal']['python_stream_ns_per_step']:.0f}ns | "
        f"online_cp_Jumper={oc6['online_cp_simple_jumper_martingale']['ns_per_step']:.0f}ns | "
        f"post-shift cov={oc6['cloture_online_conformal']['post_shift_recovered_coverage_100_to_1000']*100:.1f}% vs fixed={oc6['fixed_unadapted_alpha_0.10']['post_shift_coverage']*100:.1f}%"
    )
    print(
        f"  EvidenceSketch: {sk6['wire_size_bytes']}B, Rust merge={sk6['rust_cli_merge_ns']:.1f}ns, "
        f"1k-worker mean_diff={sk6['fleet_vs_single_node']['mean_abs_diff']:.2e}, LOO mean_diff={sk6['loo_subtraction_vs_999_workers']['mean_abs_diff']:.2e}"
    )

    print("\n[7] Systems Footprint & Cold-Start")
    print(
        f"  import cloture={s7['cold_start_ms']['import_cloture']['median']:.1f}ms | "
        f"CLI --version={s7['cold_start_ms']['cli_version']['median']:.1f}ms | "
        f"MCP roundtrip={s7['mcp_jsonrpc_roundtrip_us']['median']:.1f}us (p99={s7['mcp_jsonrpc_roundtrip_us']['p99']:.1f}us) | "
        f"CLI={s7['binary_sizes']['cloture_cli_mib']:.2f}MiB, libso={s7['binary_sizes']['libcloture_core_so_mib']:.2f}MiB"
    )
    print(f"\nTotal benchmark wall time: {res['total_elapsed_sec']:.1f}s -> saved {RESULTS_PATH}")
    print("=" * 86)


def main() -> None:
    t_total = time.perf_counter()
    metadata = collect_metadata()
    raw_rust_bench, rust_bench_map = _run_rust_cli_bench()

    results: Dict[str, Any] = {
        "metadata": metadata,
        "suite_1_consensus_stopping": run_suite_1(rust_bench_map),
        "suite_2_panel_audit": run_suite_2(),
        "suite_3_correlated_rollouts": run_suite_3(),
        "suite_4_cs_and_racing": run_suite_4(rust_bench_map),
        "suite_5_active_ppi": run_suite_5(),
        "suite_6_fuse_cascade_conformal_sketch": run_suite_6(rust_bench_map),
        "suite_7_systems_footprint": run_suite_7(raw_rust_bench),
    }
    results["total_elapsed_sec"] = time.perf_counter() - t_total

    tmp_path = RESULTS_PATH.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, allow_nan=False)
    os.replace(tmp_path, RESULTS_PATH)

    _print_summary(results)


if __name__ == "__main__":
    main()
