"""Adversarial edge-case & invariant breaker tests for PanelAudit and EvidenceSketch."""
from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from cloture import EvidenceSketch, OnlinePanelTracker, audit_panel


def _fnv1a_64(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def _check_panel_no_nans(rep) -> None:
    scalars = [
        rep.panel_accuracy,
        rep.best_single_accuracy,
        rep.panel_lift,
        rep.mean_phi,
        rep.std_phi,
        rep.n_eff_kish,
        rep.n_eff_eigen,
        rep.lambda_max,
        rep.effective_rank,
        rep.independence_ratio,
        rep.krippendorff_alpha,
        rep.unanimous_rate,
        rep.unanimous_accuracy,
        rep.all_wrong_rate,
        rep.condorcet_predicted_accuracy,
        rep.condorcet_gap,
        rep.permutation_p_value,
        rep.n_eff_ci_low,
        rep.n_eff_ci_high,
    ]
    for val in scalars:
        assert not math.isnan(val), f"NaN scalar in PanelReport: {rep}"
    assert not math.isnan(rep.asymptotic_ceiling)
    for row in rep.phi_matrix:
        for v in row:
            assert math.isfinite(v), f"Non-finite phi entry: {v}"
    for v in rep.judge_accuracies + rep.loo_delta_acc + rep.loo_delta_neff:
        assert math.isfinite(v), f"Non-finite vector entry: {v}"


# ============================================================================
# 1. PanelAudit adversarial edge cases
# ============================================================================


def test_panel_audit_constant_judges_zero_variance_phi():
    """All-0s, all-1s, and mixed constant judges (0/0 phi!) must produce zero NaNs."""
    # Case A: All judges always correct (0 errors)
    votes_all_ok = [[1, 1, 1, 1] for _ in range(20)]
    gold = [1] * 20
    rep = audit_panel(votes_all_ok, gold, n_condorcet_sims=100, n_perms=200, n_boot=100)
    _check_panel_no_nans(rep)
    assert rep.panel_accuracy == 1.0
    assert rep.mean_phi == 0.0
    assert rep.krippendorff_alpha == 1.0

    # Case B: All judges always wrong (100% errors)
    votes_all_wrong = [[0, 0, 0, 0] for _ in range(20)]
    rep_w = audit_panel(votes_all_wrong, gold, n_condorcet_sims=100, n_perms=200, n_boot=100)
    _check_panel_no_nans(rep_w)
    assert rep_w.panel_accuracy == 0.0
    assert rep_w.all_wrong_rate == 1.0
    assert rep_w.mean_phi == 0.0

    # Case C: Mixed constant (always right, always wrong) + active judges
    votes_mix = [[1, 0, i % 2, (i + 1) % 2] for i in range(20)]
    rep_m = audit_panel(votes_mix, gold, n_condorcet_sims=100, n_perms=200, n_boot=100)
    _check_panel_no_nans(rep_m)

    # OnlinePanelTracker with all-0s and all-1s
    tr = OnlinePanelTracker(4)
    for _ in range(15):
        tr.observe([0, 0, 0, 0])
    assert math.isfinite(tr.n_eff_kish()) and math.isfinite(tr.n_eff_eigen())
    tr2 = OnlinePanelTracker(4)
    for _ in range(15):
        tr2.observe([1, 1, 1, 1])
    assert math.isfinite(tr2.n_eff_kish()) and math.isfinite(tr2.n_eff_eigen())


def test_panel_audit_anticorrelated_judges():
    """Anti-correlated judges (phi = -1 for K=2, phi = -0.5 for K=3)."""
    # K=2 perfect anti-correlation (phi = -1.0)
    votes2 = [[1, 0], [0, 1]] * 25
    gold2 = [1] * 50
    rep2 = audit_panel(votes2, gold2, n_condorcet_sims=200, n_perms=300, n_boot=200)
    _check_panel_no_nans(rep2)
    assert abs(rep2.mean_phi - (-1.0)) < 1e-12
    assert abs(rep2.n_eff_kish - 2.0) < 1e-12
    assert math.isinf(rep2.asymptotic_ceiling) and rep2.asymptotic_ceiling > 0
    assert 0.0 < rep2.permutation_p_value <= 1.0

    # K=3 mutually anti-correlated (exactly 1 error per item rotating -> phi = -0.5)
    votes3 = [[0, 1, 1], [1, 0, 1], [1, 1, 0]] * 20
    gold3 = [1] * 60
    rep3 = audit_panel(votes3, gold3, n_condorcet_sims=200, n_perms=300, n_boot=200)
    _check_panel_no_nans(rep3)
    assert rep3.mean_phi < -0.45
    assert abs(rep3.n_eff_kish - 3.0) < 1e-12
    assert abs(rep3.n_eff_eigen - 2.0) < 1e-6  # lambda_max = 1.5 -> 3 / 1.5 = 2.0


def test_panel_audit_minimal_dimensions_k1_k2_n1_n5():
    """Minimal dimensions: K=1, K=2, N=1, N=5."""
    for k in [1, 2]:
        for n in [1, 5]:
            votes = [[(i + j) % 2 for j in range(k)] for i in range(n)]
            gold = [i % 2 for i in range(n)]
            rep = audit_panel(votes, gold, n_condorcet_sims=50, n_perms=100, n_boot=50)
            _check_panel_no_nans(rep)
            assert rep.k_judges == k
            assert rep.n_items == n
            assert 1.0 <= rep.n_eff_kish <= k + 1e-9
            assert rep.n_eff_ci_low <= rep.n_eff_ci_high + 1e-12


def test_panel_audit_multiclass_arbitrary_label_ids_and_singleton_strata():
    """Multiclass votes with arbitrary large u32 label IDs (1000, 2000, 99999, 2^32-1) + size-1 strata."""
    labels = [1000, 2000, 99999, 0xFFFFFFFF]
    rng = np.random.default_rng(12345)
    n_items, k_judges = 40, 5
    gold = [labels[i % len(labels)] for i in range(n_items)]
    votes = []
    for i in range(n_items):
        row = []
        for _ in range(k_judges):
            if rng.random() < 0.7:
                row.append(gold[i])
            else:
                row.append(labels[int(rng.integers(0, len(labels)))])
        votes.append(row)

    # Every single item in its own size-1 stratum!
    singleton_strata = [10_000 + i * 7 for i in range(n_items)]
    rep = audit_panel(
        votes,
        gold,
        strata=singleton_strata,
        n_condorcet_sims=100,
        n_perms=200,
        n_boot=100,
        seed=99,
    )
    _check_panel_no_nans(rep)
    # With all size-1 strata, permutation distribution is degenerate at observed -> p == 1.0
    assert abs(rep.permutation_p_value - 1.0) < 1e-12

    # Mixed strata: some size-1, some size-10
    mixed_strata = [0 if i < 10 else (1 if i < 20 else 1000 + i) for i in range(n_items)]
    rep2 = audit_panel(
        votes,
        gold,
        strata=mixed_strata,
        n_condorcet_sims=100,
        n_perms=200,
        n_boot=100,
        seed=99,
    )
    _check_panel_no_nans(rep2)
    assert 0.0 < rep2.permutation_p_value <= 1.0


# ============================================================================
# 2. EvidenceSketch: Monoid invariants, LOO, extreme values, wire corruption
# ============================================================================


def _make_sketch(scores: list[float], bets: list[float]) -> EvidenceSketch:
    s = EvidenceSketch()
    for x, b in zip(scores, bets):
        s.observe(x, b)
    return s


def test_evidence_sketch_associativity_commutativity_and_identity():
    """Verify (A+B)+C == A+(B+C), A+B == B+A, and A+0 == 0+A == A."""
    rng = np.random.default_rng(90001)
    a = _make_sketch(rng.normal(0.5, 2.0, 200).tolist(), rng.uniform(-0.8, 0.8, 200).tolist())
    b = _make_sketch(rng.normal(-1.0, 5.0, 150).tolist(), rng.uniform(-0.8, 0.8, 150).tolist())
    c = _make_sketch(rng.normal(3.0, 10.0, 120).tolist(), rng.uniform(-0.8, 0.8, 120).tolist())

    def clone(sk: EvidenceSketch) -> EvidenceSketch:
        return EvidenceSketch.from_bytes(sk.to_bytes())

    # Commutativity: A + B == B + A
    ab = clone(a)
    ab.merge(b)
    ba = clone(b)
    ba.merge(a)
    assert abs(ab.mean() - ba.mean()) < 1e-12
    assert abs(ab.e_value() - ba.e_value()) / max(ab.e_value(), 1.0) < 1e-12
    for q in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
        assert abs(ab.quantile(q) - ba.quantile(q)) < 1e-12

    # Associativity: (A + B) + C == A + (B + C)
    ab_c = clone(a)
    ab_c.merge(b)
    ab_c.merge(c)

    bc = clone(b)
    bc.merge(c)
    a_bc = clone(a)
    a_bc.merge(bc)

    assert abs(ab_c.mean() - a_bc.mean()) < 1e-12
    assert abs(ab_c.e_value() - a_bc.e_value()) / max(ab_c.e_value(), 1.0) < 1e-12
    for q in [0.0, 0.1, 0.5, 0.9, 1.0]:
        assert abs(ab_c.quantile(q) - a_bc.quantile(q)) < 1e-12

    # Identity: A + empty == empty + A == A (semantically exact across mean, e_value, quantiles)
    empty = EvidenceSketch()
    ae = clone(a)
    ae.merge(empty)
    ea = EvidenceSketch()
    ea.merge(a)
    assert ae.to_bytes() == a.to_bytes()
    assert abs(ea.mean() - a.mean()) < 1e-12
    assert abs(ea.e_value() - a.e_value()) / max(a.e_value(), 1.0) < 1e-12
    for q in [0.0, 0.25, 0.5, 0.75, 1.0]:
        assert abs(ea.quantile(q) - a.quantile(q)) < 1e-12


def test_evidence_sketch_exact_loo_and_underflow_rejection():
    """Exact LOO (A+B)-A == B, and underflow A - (A+B) MUST fail cleanly."""
    rng = np.random.default_rng(90002)
    a = _make_sketch(rng.uniform(0.2, 0.8, 120).tolist(), rng.uniform(-0.5, 0.5, 120).tolist())
    b = _make_sketch(rng.uniform(0.0, 1.0, 300).tolist(), rng.uniform(-0.5, 0.5, 300).tolist())

    ab = EvidenceSketch.from_bytes(a.to_bytes())
    ab.merge(b)

    # Underflow A - (A+B) MUST raise ValueError (C ABI returns -1) and leave A intact!
    a_before = a.to_bytes()
    with pytest.raises(ValueError):
        a.subtract(ab)
    assert a.to_bytes() == a_before, "Failed subtract mutated dst sketch!"

    # LOO (A+B) - A == B (since B contains global min/max here, exact match!)
    loo = EvidenceSketch.from_bytes(ab.to_bytes())
    loo.subtract(a)
    assert abs(loo.mean() - b.mean()) < 1e-12
    assert abs(loo.e_value() - b.e_value()) / max(b.e_value(), 1.0) < 1e-12
    for q in [0.0, 0.25, 0.5, 0.75, 1.0]:
        assert abs(loo.quantile(q) - b.quantile(q)) < 1e-12

    # Disjoint single-element underflow (same n_samples=1, different bins!) MUST reject!
    x1 = _make_sketch([0.1], [0.0])
    x2 = _make_sketch([50.0], [0.0])
    with pytest.raises(ValueError):
        x1.subtract(x2)


def test_evidence_sketch_extreme_scores_and_compsum_overflow():
    """Extreme scores (-1e300, +1e300, 0.0, -0.0, NaN, Inf) and 1e154 CompSum overflow attack."""
    s = EvidenceSketch()
    # Non-finite and square-overflowing scores ignored cleanly
    for bad in [float("nan"), float("inf"), float("-inf"), 1e300, -1e300]:
        s.observe(bad, 0.0)
    assert s.mean() == 0.0
    assert s.e_value() == 1.0

    # +0.0 and -0.0 must work cleanly and roundtrip
    s.observe(0.0, 0.0)
    s.observe(-0.0, 0.0)
    assert s.mean() == 0.0
    assert s.quantile(0.5) == 0.0
    rt = EvidenceSketch.from_bytes(s.to_bytes())
    assert rt.mean() == 0.0

    # TORTURE: 1e154 passes score*score = 1e308 < f64::MAX, so it is ACCEPTED.
    # Observing 1e154 twice overflows CompSum(sum_x2) into sum=+Inf, comp=-Inf => NaN!
    # Check if wire serialization silently corrupts sum_x2 to 0.0!
    s_big = EvidenceSketch()
    s_big.observe(1e154, 0.0)
    s_big.observe(1e154, 0.0)
    wire = s_big.to_bytes()
    # Inspect sum_x2.sum (offset 32..40) and sum_x2.comp (offset 40..48) in wire payload
    sx2_sum, sx2_comp = struct.unpack_from("<dd", wire, 32)
    assert sx2_sum + sx2_comp > 0.0 and math.isfinite(sx2_sum + sx2_comp), (
        f"EvidenceSketch CompSum overflow on 2x 1e154 silently corrupted wire sum_x2 to "
        f"(sum={sx2_sum}, comp={sx2_comp}) (expected ~2e308, got 0.0 from NaN/Inf clamping!)"
    )


def test_evidence_sketch_wire_corruption_rejection():
    """Bit-flipped, truncated, oversized, bad-magic, bad-version, NaN-injected wire payloads MUST fail."""
    s = _make_sketch([0.1, 0.5, 0.9, -2.5], [0.2, -0.1, 0.3, 0.0])
    valid = s.to_bytes()
    assert len(valid) == 712

    # 1. Empty, truncated, and oversized payloads
    for bad_len in [0, 1, 4, 16, 711, 713, 1024, 100_000]:
        payload = valid[:bad_len] if bad_len < len(valid) else valid + b"\x00" * (bad_len - len(valid))
        with pytest.raises(ValueError):
            EvidenceSketch.from_bytes(payload)

    # 2. Every single byte position bit-flipped must be rejected by FNV-1a checksum
    for pos in [0, 3, 4, 7, 8, 16, 32, 64, 300, 576, 703, 704, 711]:
        mut = bytearray(valid)
        mut[pos] ^= 0x80
        with pytest.raises(ValueError):
            EvidenceSketch.from_bytes(bytes(mut))

    # 3. Crafted valid-checksum payloads with semantic violations
    def resign(buf: bytearray) -> bytes:
        chk = _fnv1a_64(bytes(buf[:-8]))
        struct.pack_into("<Q", buf, len(buf) - 8, chk)
        return bytes(buf)

    # Bad version with valid checksum
    b_ver = bytearray(valid)
    struct.pack_into("<H", b_ver, 4, 2)
    with pytest.raises(ValueError):
        EvidenceSketch.from_bytes(resign(b_ver))

    # NaN in bet_log_wealth with valid checksum
    b_nan = bytearray(valid)
    struct.pack_into("<d", b_nan, 16 + 48 + 512, float("nan"))
    with pytest.raises(ValueError):
        EvidenceSketch.from_bytes(resign(b_nan))

    # +Inf in min_val with valid checksum
    b_inf = bytearray(valid)
    struct.pack_into("<d", b_inf, 16 + 32, float("inf"))
    with pytest.raises(ValueError):
        EvidenceSketch.from_bytes(resign(b_inf))

    # min_val > max_val with valid checksum
    b_ord = bytearray(valid)
    struct.pack_into("<dd", b_ord, 16 + 32, 10.0, -10.0)
    with pytest.raises(ValueError):
        EvidenceSketch.from_bytes(resign(b_ord))

    # Histogram bin sum != n_samples with valid checksum
    b_hist = bytearray(valid)
    struct.pack_into("<Q", b_hist, 8, 999)
    with pytest.raises(ValueError):
        EvidenceSketch.from_bytes(resign(b_hist))
