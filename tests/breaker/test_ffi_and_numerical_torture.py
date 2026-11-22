"""Adversarial C ABI / FFI memory safety, crash, UB, and numerical torture suite."""
from __future__ import annotations

import ctypes
import math
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from cloture._native import (
    _CCsResult,
    _CModeResult,
    _CPairedResult,
    _CPanelSummary,
    _CPopResult,
    _CPpiResult,
    _get_lib,
)
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
    e_bh_discoveries,
)

USIZE_MAX = (1 << (ctypes.sizeof(ctypes.c_size_t) * 8)) - 1


def _run_py_isolated(code: str, timeout: float = 10.0) -> subprocess.CompletedProcess[str]:
    """Run potentially crashing ctypes/FFI code in a child Python process."""
    full_code = f"import sys; sys.path.insert(0, {str(ROOT / 'python')!r})\n{code}"
    return subprocess.run(
        [sys.executable, "-c", full_code],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ============================================================================
# 1. NULL pointer attacks on EVERY cloture_* C ABI function
# ============================================================================


def test_ffi_null_pointers_every_exported_function():
    """Passing NULL pointers to EVERY cloture_* function must NEVER segfault or panic."""
    lib = _get_lib()
    null_vp = ctypes.c_void_p(0)
    null_dp = ctypes.POINTER(ctypes.c_double)()
    null_u64p = ctypes.POINTER(ctypes.c_uint64)()
    null_u32p = ctypes.POINTER(ctypes.c_uint32)()
    null_u8p = ctypes.POINTER(ctypes.c_uint8)()
    null_i32p = ctypes.POINTER(ctypes.c_int32)()
    null_szp = ctypes.POINTER(ctypes.c_size_t)()
    null_panel = ctypes.POINTER(_CPanelSummary)()

    # ASC-D
    lib.cloture_ascd_free(null_vp)
    lib.cloture_ascd_reset(null_vp)
    r = lib.cloture_ascd_update(null_vp, null_dp, 0)
    assert r.certified == 0 and r.has_winner == 0 and r.has_leader == 0
    r = lib.cloture_ascd_update(null_vp, null_dp, 10)
    assert r.certified == 0
    r = lib.cloture_ascd_update_batch(null_vp, null_dp, 10, 4)
    assert r.certified == 0
    r = lib.cloture_ascd_update_one_hot(null_vp, 0)
    assert r.certified == 0
    r = lib.cloture_ascd_update_weighted(null_vp, 0, 0.5)
    assert r.certified == 0

    # Valid handle + NULL data pointers
    h_ascd = lib.cloture_ascd_new(3, 0.05, 0)
    assert h_ascd
    r = lib.cloture_ascd_update(h_ascd, null_dp, 3)
    assert r.n_samples == 0
    r = lib.cloture_ascd_update_batch(h_ascd, null_dp, 5, 3)
    assert r.n_samples == 0
    lib.cloture_ascd_free(h_ascd)

    # CITE
    lib.cloture_cite_free(null_vp)
    lib.cloture_cite_reset(null_vp)
    assert lib.cloture_cite_update(null_vp, 1).certified == 0
    assert lib.cloture_cite_update_batch(null_vp, null_u64p, 5).certified == 0
    assert lib.cloture_cite_update_weighted(null_vp, 1, 0.5).certified == 0
    h_cite = lib.cloture_cite_new(0.05, 0)
    assert lib.cloture_cite_update_batch(h_cite, null_u64p, 5).n_samples == 0
    lib.cloture_cite_free(h_cite)

    # PopulationCert
    lib.cloture_pop_free(null_vp)
    assert lib.cloture_pop_add_unit(null_vp, null_dp, 5).n_units == 0
    assert lib.cloture_pop_optimal_rollouts(null_vp, 1.0, 1.0) == 1
    h_pop = lib.cloture_pop_new(0.05)
    assert lib.cloture_pop_add_unit(h_pop, null_dp, 5).n_units == 0
    lib.cloture_pop_free(h_pop)

    # PanelAudit & OnlinePanelTracker
    summary = _CPanelSummary()
    assert lib.cloture_panel_audit(
        null_u32p, null_u32p, null_u32p, 10, 3, 10, 10, 10, 42,
        null_panel, null_dp, null_dp, null_dp, null_dp
    ) < 0
    assert lib.cloture_panel_audit(
        null_u32p, null_u32p, null_u32p, 10, 3, 10, 10, 10, 42,
        ctypes.byref(summary), null_dp, null_dp, null_dp, null_dp
    ) < 0
    lib.cloture_online_panel_free(null_vp)
    lib.cloture_online_panel_observe(null_vp, null_u8p, 5)
    assert lib.cloture_online_panel_neff_kish(null_vp) == 0.0
    assert lib.cloture_online_panel_neff_eigen(null_vp) == 0.0
    h_op = lib.cloture_online_panel_new(3)
    lib.cloture_online_panel_observe(h_op, null_u8p, 3)
    lib.cloture_online_panel_free(h_op)

    # EBCS, BettingCS, PairedE
    lib.cloture_ebcs_free(null_vp)
    assert lib.cloture_ebcs_update(null_vp, 0.5).n == 0
    assert lib.cloture_ebcs_update_batch(null_vp, null_dp, 5).n == 0
    h_eb = lib.cloture_ebcs_new(0.05, 0.0, 1.0)
    assert lib.cloture_ebcs_update_batch(h_eb, null_dp, 5).n == 0
    lib.cloture_ebcs_free(h_eb)

    lib.cloture_betcs_free(null_vp)
    assert lib.cloture_betcs_update(null_vp, 0.5).n == 0
    assert lib.cloture_betcs_update_batch(null_vp, null_dp, 5).n == 0
    h_bet = lib.cloture_betcs_new(0.05, 51)
    assert lib.cloture_betcs_update_batch(h_bet, null_dp, 5).n == 0
    lib.cloture_betcs_free(h_bet)

    lib.cloture_paired_free(null_vp)
    assert lib.cloture_paired_update(null_vp, 0.5, 0.5).n_pairs == 0
    assert lib.cloture_paired_update_batch(null_vp, null_dp, null_dp, 5).n_pairs == 0
    h_pe = lib.cloture_paired_new(0.05, 1.0)
    assert lib.cloture_paired_update_batch(h_pe, null_dp, null_dp, 5).n_pairs == 0
    lib.cloture_paired_free(h_pe)

    # Race
    lib.cloture_race_free(null_vp)
    assert lib.cloture_race_observe_arm(null_vp, 0, 0.5) == -1
    assert lib.cloture_race_observe_paired(null_vp, null_dp, 3) == -1
    assert lib.cloture_race_next_arm_lucb(null_vp) == 0
    assert lib.cloture_race_winner(null_vp) == -1
    assert lib.cloture_race_active_count(null_vp) == 0
    assert lib.cloture_race_arm_active(null_vp, 0) == 0
    h_race = lib.cloture_race_new(3, 0.05, 1)
    assert lib.cloture_race_observe_paired(h_race, null_dp, 3) == -1
    lib.cloture_race_free(h_race)

    # ActivePPI
    lib.cloture_ppi_free(null_vp)
    assert lib.cloture_ppi_suggest_prob(null_vp, 0.5, 0.5) == 1.0
    assert lib.cloture_ppi_observe(null_vp, 0.5, 1, 1.0, 0.1).n_total == 0
    assert lib.cloture_ppi_observe_batch(null_vp, null_dp, null_u8p, null_dp, null_dp, 5).n_total == 0
    h_ppi = lib.cloture_ppi_new(0.05, 0.1, 0.05)
    assert lib.cloture_ppi_observe_batch(h_ppi, null_dp, null_u8p, null_dp, null_dp, 5).n_total == 0
    lib.cloture_ppi_free(h_ppi)

    # SafeFuse
    lib.cloture_fuse_free(null_vp)
    fb = ctypes.c_int32(99)
    assert math.isnan(lib.cloture_fuse_predict(null_vp, null_dp, 3, ctypes.byref(fb)))
    assert fb.value == 0
    assert math.isnan(lib.cloture_fuse_predict(null_vp, null_dp, 3, null_i32p))
    assert lib.cloture_fuse_update(null_vp, null_dp, 3, 0.0, 0) == -1
    assert lib.cloture_fuse_quarantined_count(null_vp) == 0
    assert lib.cloture_fuse_is_quarantined(null_vp, 0) == 0
    assert lib.cloture_fuse_weights(null_vp, null_dp, 5) == 0
    h_fuse = lib.cloture_fuse_new(3, 0.05, 0.05)
    assert math.isnan(lib.cloture_fuse_predict(h_fuse, null_dp, 3, null_i32p))
    assert lib.cloture_fuse_update(h_fuse, null_dp, 3, 0.0, 0) == -1
    assert lib.cloture_fuse_weights(h_fuse, null_dp, 3) == 3
    lib.cloture_fuse_free(h_fuse)

    # AbortCascade & OnlineConformal
    assert not lib.cloture_cascade_calibrate(null_dp, null_u8p, 10, 3, 0.9, null_dp, 0)
    lib.cloture_cascade_free(null_vp)
    assert lib.cloture_cascade_should_abort(null_vp, 0, 0.1) == 0
    assert lib.cloture_cascade_thresholds(null_vp, null_dp, 5) == 0

    lib.cloture_conformal_free(null_vp)
    assert math.isnan(lib.cloture_conformal_update_pvalue(null_vp, 0.5))
    assert math.isnan(lib.cloture_conformal_update_miscovered(null_vp, 1))
    assert math.isnan(lib.cloture_conformal_alpha_t(null_vp))
    assert math.isnan(lib.cloture_conformal_empirical_coverage(null_vp))

    # e-BH
    assert lib.cloture_e_bh_discoveries(null_dp, 5, 0.05, null_szp, 5) == 0

    # EvidenceSketch
    lib.cloture_sketch_free(null_vp)
    lib.cloture_sketch_observe(null_vp, 1.0, 0.5)
    assert lib.cloture_sketch_merge(null_vp, null_vp) == -1
    assert lib.cloture_sketch_subtract(null_vp, null_vp) == -1
    assert math.isnan(lib.cloture_sketch_quantile(null_vp, 0.5))
    assert lib.cloture_sketch_e_value(null_vp) == 1.0
    assert math.isnan(lib.cloture_sketch_mean(null_vp))
    assert lib.cloture_sketch_serialized_len(null_vp) == 0
    assert lib.cloture_sketch_to_bytes(null_vp, null_u8p, 1024) == 0
    assert not lib.cloture_sketch_from_bytes(null_u8p, 712)

    h_sk = lib.cloture_sketch_new()
    assert lib.cloture_sketch_merge(h_sk, null_vp) == -1
    assert lib.cloture_sketch_merge(null_vp, h_sk) == -1
    assert lib.cloture_sketch_subtract(h_sk, null_vp) == -1
    assert lib.cloture_sketch_subtract(null_vp, h_sk) == -1
    assert lib.cloture_sketch_to_bytes(h_sk, null_u8p, 1024) == 0
    # Insufficient capacity buffer
    small_buf = (ctypes.c_uint8 * 100)()
    assert lib.cloture_sketch_to_bytes(h_sk, small_buf, 100) == 0
    lib.cloture_sketch_free(h_sk)


# ============================================================================
# 2. Invalid constructor arguments (standard + usize::MAX SIGABRT crash attacks)
# ============================================================================


def test_ffi_invalid_constructor_args_return_null():
    """Invalid scalar constructor arguments must return NULL cleanly without crashing."""
    lib = _get_lib()
    bad_probs = [0.0, 1.0, -0.5, 1.5, float("nan"), float("inf"), float("-inf")]

    # AscdStopper
    for k in [0, 1, USIZE_MAX]:
        assert not lib.cloture_ascd_new(k, 0.05, 0)
    for d in bad_probs:
        assert not lib.cloture_ascd_new(3, d, 0)

    # CiteStopper & PopulationCert
    for d in bad_probs:
        assert not lib.cloture_cite_new(d, 0)
        assert not lib.cloture_pop_new(d)

    # OnlinePanelTracker
    assert not lib.cloture_online_panel_new(0)

    # EmpiricalBernsteinCS
    for a in bad_probs:
        assert not lib.cloture_ebcs_new(a, 0.0, 1.0)
    assert not lib.cloture_ebcs_new(0.05, 1.0, 1.0)
    assert not lib.cloture_ebcs_new(0.05, 2.0, 1.0)
    assert not lib.cloture_ebcs_new(0.05, float("nan"), 1.0)
    assert not lib.cloture_ebcs_new(0.05, 0.0, float("inf"))

    # BettingCS
    for a in bad_probs:
        assert not lib.cloture_betcs_new(a, 51)
    for g in [0, 1, 2]:
        assert not lib.cloture_betcs_new(0.05, g)

    # PairedEProcess
    for a in bad_probs:
        assert not lib.cloture_paired_new(a, 1.0)
    for sc in [0.0, -1.0, float("nan"), float("inf")]:
        assert not lib.cloture_paired_new(0.05, sc)

    # AgentRace
    for m in [0, 1]:
        assert not lib.cloture_race_new(m, 0.05, 0)
        assert not lib.cloture_race_new(m, 0.05, 1)
    for d in bad_probs:
        assert not lib.cloture_race_new(3, d, 1)

    # ActivePPI
    for a in bad_probs:
        assert not lib.cloture_ppi_new(a, 0.2, 0.05)
    assert not lib.cloture_ppi_new(0.05, 0.0, 0.05)
    assert not lib.cloture_ppi_new(0.05, 1.1, 0.05)
    assert not lib.cloture_ppi_new(0.05, 0.1, 0.2)  # min_audit_prob > target_audit_rate
    assert not lib.cloture_ppi_new(0.05, 0.1, 0.0)
    assert not lib.cloture_ppi_new(0.05, float("nan"), 0.05)

    # SafeFuse
    assert not lib.cloture_fuse_new(0, 0.05, 0.05)
    for a in bad_probs:
        assert not lib.cloture_fuse_new(3, a, 0.05)
        assert not lib.cloture_fuse_new(3, 0.05, a)

    # OnlineConformal
    for a in bad_probs:
        assert not lib.cloture_conformal_new(a)


@pytest.mark.parametrize(
    "label,call_expr",
    [
        ("cloture_betcs_new(0.05, usize::MAX)", "lib.cloture_betcs_new(0.05, USIZE_MAX)"),
        ("cloture_online_panel_new(usize::MAX)", "lib.cloture_online_panel_new(USIZE_MAX)"),
        ("cloture_race_new(usize::MAX, 0.05, 1)", "lib.cloture_race_new(USIZE_MAX, 0.05, 1)"),
        ("cloture_race_new(usize::MAX, 0.05, 0)", "lib.cloture_race_new(USIZE_MAX, 0.05, 0)"),
        ("cloture_fuse_new(usize::MAX, 0.05, 0.05)", "lib.cloture_fuse_new(USIZE_MAX, 0.05, 0.05)"),
        ("cloture_ascd_new(100_000_000, 0.05, 0)", "lib.cloture_ascd_new(100_000_000, 0.05, 0)"),
    ],
)
def test_ffi_huge_size_constructor_must_return_null_not_sigabrt(label: str, call_expr: str):
    """Passing usize::MAX / huge counts to C ABI constructors must return NULL, NEVER SIGABRT!

    Cargo.toml sets `panic = "abort"`, neutralizing `catch_unwind` across the FFI boundary.
    """
    code = f"""
import ctypes
from cloture._native import _get_lib
USIZE_MAX = (1 << (ctypes.sizeof(ctypes.c_size_t) * 8)) - 1
lib = _get_lib()
ptr = {call_expr}
assert not ptr, f"Expected NULL pointer, got {{ptr}}"
print("OK_NULL")
"""
    proc = _run_py_isolated(code)
    assert proc.returncode == 0 and "OK_NULL" in proc.stdout, (
        f"C ABI CRASH on {label}: exit_code={proc.returncode} "
        f"(signal {-proc.returncode if proc.returncode < 0 else 'none'}), "
        f"stderr={proc.stderr.strip()!r}"
    )


# ============================================================================
# 3. Self-aliasing pointer UB checks (sketch_merge(p,p), sketch_subtract(p,p))
# ============================================================================


def test_ffi_sketch_self_aliasing_merge_and_subtract():
    """cloture_sketch_merge(p, p) and cloture_sketch_subtract(p, p) must never UB or crash."""
    lib = _get_lib()
    p = lib.cloture_sketch_new()
    assert p
    for i in range(50):
        lib.cloture_sketch_observe(p, float(i) * 0.1, 0.2)
    mean_before = lib.cloture_sketch_mean(p)
    eval_before = lib.cloture_sketch_e_value(p)

    # Self-merge: p += p (doubles counts and log-wealth, mean unchanged, e_value changes cleanly)
    rc = lib.cloture_sketch_merge(p, p)
    assert rc == 0
    assert abs(lib.cloture_sketch_mean(p) - mean_before) < 1e-12
    assert math.isfinite(lib.cloture_sketch_e_value(p))
    assert lib.cloture_sketch_e_value(p) > eval_before

    # Self-subtract: p -= p (must cleanly produce empty identity sketch)
    rc2 = lib.cloture_sketch_subtract(p, p)
    assert rc2 == 0
    assert lib.cloture_sketch_mean(p) == 0.0
    assert abs(lib.cloture_sketch_e_value(p) - 1.0) < 1e-12
    lib.cloture_sketch_free(p)


# ============================================================================
# 4. Numerical torture: NaN, Inf, subnormals, negative/zero probs, OOB labels, 10^6 updates
# ============================================================================


def test_numerical_torture_nan_inf_subnormal_and_oob_labels():
    """Feed NaN, ±Inf, 1e-320, negative probs, all-zero probs, non-normalized, OOB labels."""
    # AscdStopper
    s = AscdStopper(k=3, delta=0.05)
    for bad in [
        [float("nan"), 0.5, 0.5],
        [float("inf"), 0.0, 0.0],
        [float("-inf"), 0.5, 0.5],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -3.0],
        [0.5, 0.5],  # wrong length
        [0.2, 0.2, 0.2, 0.4],  # wrong length
    ]:
        d = s.update(bad)
        assert d.n_samples == 0
        assert math.isfinite(d.log_evidence)

    # Out-of-range label indices
    assert s.update_one_hot(9_999_999).n_samples == 0
    assert s.update_weighted(9_999_999, 0.8).n_samples == 0
    assert s.update_weighted(0, float("nan")).n_samples == 0
    assert s.update_weighted(0, float("inf")).n_samples == 0

    # Subnormals & non-normalized positive vectors must normalize cleanly with no NaN
    d_sub = s.update([1e-320, 1e-320, 0.0])
    assert d_sub.n_samples == 1
    assert math.isfinite(d_sub.log_evidence) and math.isfinite(d_sub.margin_estimate)
    d_unnorm = s.update([1000.0, 2000.0, 7000.0])
    assert d_unnorm.n_samples == 2
    assert math.isfinite(d_unnorm.log_evidence)
    d_neg_mix = s.update([-5.0, 3.0, 1.0])
    assert d_neg_mix.n_samples == 3
    assert math.isfinite(d_neg_mix.log_evidence)

    # Tiny delta = 1e-12 and huge delta = 0.99
    for dval in [1e-12, 0.99]:
        st = AscdStopper(k=3, delta=dval)
        res = st.update([0.9, 0.05, 0.05])
        assert math.isfinite(res.log_evidence) and math.isfinite(res.log_threshold)

        ct = CiteStopper(delta=dval)
        cres = ct.update(42)
        assert math.isfinite(cres.log_evidence) and math.isfinite(cres.log_threshold)

        eb = EmpiricalBernsteinCS(alpha=dval)
        ebr = eb.update(0.5)
        assert math.isfinite(ebr.lower) and math.isfinite(ebr.upper) and ebr.lower <= ebr.upper

        bt = BettingCS(alpha=dval, grid_points=51)
        btr = bt.update(0.5)
        assert math.isfinite(btr.lower) and math.isfinite(btr.upper) and btr.lower <= btr.upper

    # EBCS, BettingCS, PairedE, ActivePPI, SafeFuse, OnlineConformal NaN/Inf/subnormal
    eb = EmpiricalBernsteinCS(alpha=0.05)
    for bad in [float("nan"), float("inf"), float("-inf")]:
        b = eb.update(bad)
        assert b.n == 0 and b.lower == 0.0 and b.upper == 1.0
    b = eb.update(1e-320)
    assert b.n == 1 and math.isfinite(b.lower) and math.isfinite(b.upper)

    bet = BettingCS(alpha=0.05, grid_points=51)
    for bad in [float("nan"), float("inf"), float("-inf")]:
        b = bet.update(bad)
        assert b.n == 0
    b = bet.update(1e-320)
    assert b.n == 1 and math.isfinite(b.lower) and math.isfinite(b.upper)

    pe = PairedEProcess(alpha=0.05, scale=1.0)
    for bad_a, bad_b in [(float("nan"), 0.5), (0.5, float("inf")), (float("-inf"), 0.0)]:
        st = pe.update(bad_a, bad_b)
        assert st.n_pairs == 0 and st.anytime_p_value == 1.0

    race = AgentRace(n_arms=3, delta=0.05, paired=True)
    assert race.observe_arm(9_999_999, 0.5) is None
    assert race.observe_arm(0, float("nan")) is None
    assert race.observe_paired([float("nan"), float("inf"), float("-inf")]) is None
    assert race.active_count() == 3

    ppi = ActivePPI(alpha=0.05, target_audit_rate=0.2, min_audit_prob=0.05)
    assert ppi.observe(float("nan"), False, 0.0, 0.2).n_total == 0
    assert ppi.observe(0.5, True, float("nan"), 0.2).n_total == 0
    assert ppi.observe(0.5, False, 0.0, float("nan")).n_total == 0
    assert math.isfinite(ppi.suggest_prob(float("nan"), float("nan")))

    oc = OnlineConformal(target_alpha=0.10)
    a0 = oc.alpha_t
    assert oc.update_pvalue(float("nan")) == a0
    assert oc.update_pvalue(float("inf")) == a0

    fuse = SafeFuse(n_agents=3, quarantine_alpha=0.05, fallback_alpha=0.05)
    fuse.update([float("nan"), float("inf"), float("-inf")], baseline_loss=float("nan"))
    val, _ = fuse.predict([float("nan"), 1.0, float("inf")])
    assert math.isfinite(val)
    assert all(math.isfinite(w) and w >= 0.0 for w in fuse.weights())


def test_numerical_torture_one_million_sequential_updates():
    """10^6 sequential updates on EBCS, PairedE, AscdStopper, CiteStopper without NaN/Inf/drift."""
    n = 1_000_000
    rng = np.random.default_rng(123456)
    xs = rng.binomial(1, 0.42, size=n).astype(np.float64)

    eb = EmpiricalBernsteinCS(alpha=0.05)
    b = eb.update_batch(xs)
    assert b.n == n
    assert math.isfinite(b.lower) and math.isfinite(b.upper) and math.isfinite(b.mean)
    assert b.lower <= 0.42 <= b.upper
    assert b.width < 0.01

    pe = PairedEProcess(alpha=0.05, scale=1.0)
    a = xs
    b_arr = rng.binomial(1, 0.42, size=n).astype(np.float64)
    pr = pe.update_batch(a, b_arr)
    assert pr.n_pairs == n
    assert math.isfinite(pr.mean_diff) and math.isfinite(pr.anytime_p_value)

    # 10^6 updates to CiteStopper under exact alternating tie (never certifies, runs full 10^6!)
    alt = np.tile(np.array([1, 2], dtype=np.uint64), n // 2)
    cite = CiteStopper(delta=0.05)
    cd = cite.update_batch(alt)
    assert cd.n_samples == n
    assert not cd.certified
    assert math.isfinite(cd.log_evidence)


# ============================================================================
# 5. Targeted bug reproducers found during code audit!
# ============================================================================


def test_cite_u64_high_bit_label_id_not_corrupted_to_none():
    """CiteStopper accepts uint64_t label_id; values >= 2^63 (e.g. 50% of 64-bit hashes or u64::MAX)
    MUST NOT silently become winner=None / negative i64 when certified!
    """
    for label_id in [1 << 63, 0xCAFEBABE_DEADBEEF, (1 << 64) - 1]:
        cite = CiteStopper(delta=0.05)
        dec = None
        for _ in range(30):
            dec = cite.update(label_id)
            if dec.certified:
                break
        assert dec is not None and dec.certified
        assert dec.winner == label_id, (
            f"C ABI int64_t truncation bug: CiteStopper certified uint64 label_id={label_id:#x}, "
            f"but returned winner={dec.winner!r}, leader={dec.leader!r}!"
        )
        assert dec.leader == label_id


def test_ebcs_extreme_finite_bounds_span_overflow_must_not_produce_nan():
    """EmpiricalBernsteinCS(-1e308, +1e308): span = max_val - min_val overflows to +Inf -> NaN bounds!"""
    try:
        eb = EmpiricalBernsteinCS(alpha=0.05, min_val=-1e308, max_val=1e308)
    except ValueError:
        # Rejecting overflowing span at construction is valid!
        return
    res = eb.update(0.0)
    assert math.isfinite(res.lower) and math.isfinite(res.upper) and math.isfinite(res.mean), (
        f"EmpiricalBernsteinCS(-1e308, 1e308) produced non-finite bounds after valid update(0.0): {res}"
    )


@pytest.mark.parametrize(
    "label,snippet",
    [
        (
            "AscdStopper.update_batch ragged rows",
            "from cloture import AscdStopper; s = AscdStopper(k=4); s.update_batch([[0.5], [0.5]])",
        ),
        (
            "ActivePPI.observe_batch mismatched array lengths",
            "from cloture import ActivePPI; p = ActivePPI(); "
            "p.observe_batch([0.5]*50, [1]*50, [1.0], [0.1])",
        ),
        (
            "audit_panel votes shorter than gold",
            "from cloture import audit_panel; audit_panel([[0, 1]], [0, 1, 0, 1])",
        ),
        (
            "audit_panel strata numpy array shorter than gold",
            "import numpy as np; from cloture import audit_panel; "
            "audit_panel([[0, 1]]*10, [0]*10, strata=np.array([0], dtype=np.uint32))",
        ),
        (
            "AbortCascade.calibrate scores shorter than succeeded",
            "from cloture import AbortCascade; "
            "AbortCascade.calibrate([[0.5, 0.5]], [1, 0, 1, 0])",
        ),
    ],
)
def test_python_wrapper_rejects_mismatched_buffer_lengths_without_overread(label: str, snippet: str):
    """Python FFI wrappers MUST validate sequence lengths to prevent C heap buffer overreads!"""
    code = f"""
try:
    {snippet}
    print("NO_ERROR_OVERREAD")
except (ValueError, IndexError, TypeError) as e:
    print("CLEAN_EXC:", type(e).__name__, e)
"""
    proc = _run_py_isolated(code)
    assert proc.returncode == 0, f"CRASHED with exit={proc.returncode} on {label}: {proc.stderr}"
    assert "CLEAN_EXC:" in proc.stdout, (
        f"HEAP BUFFER OVERREAD in Python wrapper [{label}]: "
        f"accepted mismatched buffer sizes without raising ValueError! stdout={proc.stdout!r}"
    )


# ============================================================================
# 6. Multi-threaded FFI stress test
# ============================================================================


def test_ffi_multithreaded_isolated_handles():
    """Concurrent multi-threaded FFI calls on independent handles must remain thread-safe."""
    errors: list[Exception] = []

    def worker(tid: int) -> None:
        try:
            rng = np.random.default_rng(77700 + tid)
            ascd = AscdStopper(k=4, delta=0.05)
            cite = CiteStopper(delta=0.05)
            eb = EmpiricalBernsteinCS(alpha=0.05)
            sk = EvidenceSketch()
            for _ in range(500):
                p = rng.dirichlet([4.0, 1.0, 1.0, 1.0])
                ascd.update(p)
                cite.update(int(rng.integers(0, 5)))
                x = float(rng.random())
                eb.update(x)
                sk.observe(x, x - 0.5)
            wire = sk.to_bytes()
            sk2 = EvidenceSketch.from_bytes(wire)
            assert abs(sk2.mean() - sk.mean()) < 1e-12
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, f"Thread-safety failures: {errors}"
