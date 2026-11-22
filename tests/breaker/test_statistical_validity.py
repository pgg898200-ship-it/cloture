"""Adversarial statistical validity & anytime Type-I error attacks on cloture engines."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from cloture import (
    AbortCascade,
    ActivePPI,
    AgentRace,
    AscdStopper,
    BettingCS,
    CiteStopper,
    EmpiricalBernsteinCS,
    PairedEProcess,
    PopulationCert,
    SafeFuse,
)


def _assert_rate_within_alpha(
    k_events: int,
    n_trials: int,
    alpha: float,
    label: str,
    p_crit: float = 0.005,
) -> None:
    """Assert observed false-positive/miscoverage rate does not statistically exceed alpha.

    Uses exact one-sided binomial test H0: p <= alpha vs H1: p > alpha.
    Fails loudly if p-value < p_crit (strong evidence of genuine validity violation).
    """
    rate = k_events / n_trials
    res = binomtest(k_events, n_trials, p=alpha, alternative="greater")
    ci = res.proportion_ci(confidence_level=0.99, method="exact")
    assert res.pvalue >= p_crit, (
        f"STATISTICAL VALIDITY VIOLATION [{label}]: "
        f"observed rate = {k_events}/{n_trials} = {rate:.4f} > nominal alpha={alpha:.4f} "
        f"(one-sided exact binomial p={res.pvalue:.3e}, 99% lower bound={ci.low:.4f})"
    )


# ============================================================================
# 1. AscdStopper: Tie nulls, K=6 uniform, bimodal, adversarial non-stationary
# ============================================================================


def test_ascd_exact_tie_null_k2_onehot():
    """Exact tie null p=[0.5, 0.5] with one-hot draws (extreme bimodal [1,0] vs [0,1])."""
    delta = 0.05
    n_trials = 2500
    horizon = 600
    rng = np.random.default_rng(10001)
    stopper = AscdStopper(k=2, delta=delta)
    false_certs = 0

    for _ in range(n_trials):
        stopper.reset()
        draws = rng.integers(0, 2, size=horizon)
        for y in draws:
            d = stopper.update_one_hot(int(y))
            if d.certified:
                false_certs += 1
                break

    _assert_rate_within_alpha(false_certs, n_trials, delta, "AscdStopper K=2 tie one-hot")


def test_ascd_exact_tie_null_k2_small_step_greybox():
    """Adversarial exact tie null p=[0.5, 0.5] with grey-box steps [0.75, 0.25] vs [0.25, 0.75].

    Smaller step size reduces martingale overshoot at boundary 1/delta, exposing the missing
    factor-of-2 two-sided correction in ln((K-1)/delta) = ln(1/delta) when K=2.
    """
    delta = 0.05
    n_trials = 2000
    horizon = 4000
    rng = np.random.default_rng(10002)
    stopper = AscdStopper(k=2, delta=delta)
    p_pos = np.array([0.75, 0.25], dtype=np.float64)
    p_neg = np.array([0.25, 0.75], dtype=np.float64)
    false_certs = 0

    for _ in range(n_trials):
        stopper.reset()
        signs = rng.integers(0, 2, size=horizon)
        batch = np.where(signs[:, None] == 1, p_pos, p_neg)
        d = stopper.update_batch(batch)
        if d.certified:
            false_certs += 1

    _assert_rate_within_alpha(
        false_certs, n_trials, delta, "AscdStopper K=2 exact tie grey-box [0.75,0.25]"
    )


def test_ascd_exact_tie_null_k6_uniform():
    """Uniform tie null K=6 with p=[1/6]*6 (both one-hot and small-step grey-box)."""
    delta = 0.05
    n_trials = 1500
    horizon = 600
    rng = np.random.default_rng(10003)
    stopper = AscdStopper(k=6, delta=delta)
    false_certs = 0

    for _ in range(n_trials):
        stopper.reset()
        draws = rng.integers(0, 6, size=horizon)
        for y in draws:
            d = stopper.update_one_hot(int(y))
            if d.certified:
                false_certs += 1
                break

    _assert_rate_within_alpha(false_certs, n_trials, delta, "AscdStopper K=6 uniform tie")


def test_ascd_adversarial_nonstationary_martingale_null():
    """Adversarial non-stationary grey-box martingale whose conditional mean stays tied at [0.5, 0.5].

    Step amplitude varies adaptively with past sign/time (large swings early, fine steps near
    threshold to eliminate overshoot) while E[theta_t | F_{t-1}] = [0.5, 0.5] identically.
    """
    delta = 0.05
    n_trials = 2500
    horizon = 1200
    rng = np.random.default_rng(10004)
    stopper = AscdStopper(k=2, delta=delta)
    false_certs = 0

    for _ in range(n_trials):
        stopper.reset()
        log_ev = 0.0
        for t in range(horizon):
            # Strictly F_{t-1}-measurable amplitude: shrink step near threshold to minimize overshoot
            if log_ev > 1.8:
                eps = 0.03
            elif t % 17 < 5:
                eps = 0.45
            else:
                eps = 0.12
            sgn = 1.0 if rng.random() < 0.5 else -1.0
            p0 = 0.5 + sgn * eps
            d = stopper.update([p0, 1.0 - p0])
            log_ev = d.log_evidence
            if d.certified:
                false_certs += 1
                break

    _assert_rate_within_alpha(
        false_certs, n_trials, delta, "AscdStopper K=2 adversarial non-stationary martingale"
    )


def test_ascd_extreme_bimodal_alternating_vs_bernoulli():
    """Extreme bimodal [1,0] vs [0,1] deterministic alternating + random Bernoulli tie."""
    delta = 0.05
    stopper = AscdStopper(k=2, delta=delta)
    # Deterministic exact alternating [1,0], [0,1] must NEVER certify
    for t in range(2000):
        d = stopper.update([1.0, 0.0] if t % 2 == 0 else [0.0, 1.0])
        assert not d.certified, f"Alternating tie certified at step {t}: {d}"


# ============================================================================
# 2. CiteStopper: Tie nulls (K=2, 10, 100), extreme weights, deterministic lock
# ============================================================================


@pytest.mark.parametrize("k_cats", [2, 10, 100])
def test_cite_exact_tie_null_multi_k(k_cats: int):
    """Exact tie null across 2, 10, and 100 categories: FPR must stay <= delta."""
    delta = 0.05
    n_trials = 1200
    horizon = 400
    rng = np.random.default_rng(20000 + k_cats)
    stopper = CiteStopper(delta=delta)
    false_certs = 0

    for _ in range(n_trials):
        stopper.reset()
        draws = rng.integers(0, k_cats, size=horizon, dtype=np.uint64)
        d = stopper.update_batch(draws)
        if d.certified:
            false_certs += 1

    _assert_rate_within_alpha(false_certs, n_trials, delta, f"CiteStopper exact tie K={k_cats}")


def test_cite_weighted_extreme_weights_and_validity():
    """Weighted CITE with extreme weights (0, 1e-12, 1.0, 100.0, negative) and tie null."""
    delta = 0.05
    stopper = CiteStopper(delta=delta)

    # Invalid/zero/negative weights must be ignored without mutating n_samples
    for bad_w in [0.0, -0.0, -1.0, -1e100, float("nan"), float("inf"), float("-inf")]:
        d = stopper.update_weighted(1, bad_w)
        assert d.n_samples == 0
        assert not d.certified

    # Tiny positive weight 1e-12 must not crash or produce NaN
    d_tiny = stopper.update_weighted(1, 1e-12)
    assert d_tiny.n_samples == 1
    assert math.isfinite(d_tiny.log_evidence)

    # Huge weight 100.0 clamped to 1.0 (must NOT certify in 1 step!)
    stopper.reset()
    d_huge = stopper.update_weighted(1, 100.0)
    assert d_huge.n_samples == 1
    assert not d_huge.certified
    assert d_huge.log_evidence == 0.0  # post-discovery step has 0 post-discovery evidence!

    # Monte Carlo weighted tie null with mixed weights including 1e-12, 0.5, 1.0, 100.0
    rng = np.random.default_rng(20202)
    n_trials = 1000
    horizon = 300
    weights_pool = np.array([1e-12, 0.05, 0.5, 1.0, 100.0])
    false_certs = 0
    for _ in range(n_trials):
        stopper.reset()
        for _ in range(horizon):
            lbl = int(rng.integers(0, 3))
            w = float(weights_pool[int(rng.integers(0, len(weights_pool)))])
            d = stopper.update_weighted(lbl, w)
            if d.certified:
                false_certs += 1
                break

    _assert_rate_within_alpha(false_certs, n_trials, delta, "CiteStopper extreme-weights tie")


def test_cite_deterministic_budget_lock_exactness():
    """Deterministic budget lock must NEVER fire if any challenger can mathematically tie or pass."""
    # Unweighted: sweep budgets N=1..20 and all possible score splits
    for budget in [1, 2, 3, 4, 5, 10, 15, 20]:
        # Case A: leader gets exactly ceil(budget/2) votes (if budget even, challenger can tie!)
        s = CiteStopper(delta=1e-12, max_budget=budget)
        half = budget // 2
        for step in range(1, half + 1):
            d = s.update(1)
            # Remaining = budget - step >= step, so challenger could reach >= step -> MUST NOT LOCK
            assert not d.deterministic_lock, (
                f"Premature lock at step={step}, budget={budget}: leader={step}, rem={budget - step}"
            )
            assert not d.certified
        # Next vote gives leader = half + 1, rem = budget - half - 1 <= half < leader -> MUST LOCK
        d_next = s.update(1)
        assert d_next.deterministic_lock, (
            f"Missed lock at step={half + 1}, budget={budget}"
        )
        assert d_next.winner == 1

    # Case B: weighted budget lock with weight < 1.0 and weight > 1.0 (clamped to 1.0)
    # Budget = 10: 6 votes of weight 0.5 give leader_w = 3.0; 4 remaining votes of weight 1.0 could
    # give challenger 4.0 > 3.0 -> MUST NOT LOCK!
    sw = CiteStopper(delta=1e-12, max_budget=10)
    for step in range(1, 7):
        d = sw.update_weighted(1, 0.5)
        assert not d.deterministic_lock, f"Premature weighted lock at step {step}"
    # At step 7: leader_w = 3.5, rem = 3 -> 3.5 > 3.0 -> challenger max is 3.0 -> LOCKS!
    d7 = sw.update_weighted(1, 0.5)
    assert d7.deterministic_lock
    assert d7.winner == 1

    # Case C: exact boundary tie in weighted mode: budget = 6, leader gets 3 * 1.0, rem = 3 -> NO LOCK
    sw2 = CiteStopper(delta=1e-12, max_budget=6)
    sw2.update_weighted(1, 0.5)
    sw2.update_weighted(1, 0.5)
    sw2.update_weighted(1, 1.0)
    # leader_w = 2.0, rem = 3 -> no lock
    d = sw2.update_weighted(1, 1.0)
    # leader_w = 3.0, rem = 2 -> LOCK!
    assert d.deterministic_lock


# ============================================================================
# 3. PopulationCert: High ICC (rho=0.90) + unbalanced rollout counts
# ============================================================================


def test_population_cert_high_icc_and_unbalanced_null():
    """High-ICC (rho=0.90) null with n=50 rollouts/prompt AND unbalanced [1, 2, 50, 1, 100]."""
    delta = 0.05
    rng = np.random.default_rng(30001)

    # Regime A: Balanced high-ICC (rho ~ 0.90), n=50 rollouts/unit, theta = 0
    n_trials = 1000
    n_units = 60
    false_certs = 0
    rho_estimates = []

    for trial in range(n_trials):
        pop = PopulationCert(delta=delta)
        for _ in range(n_units):
            # Between-unit mean +/-0.6 (var = 0.36), within-unit noise +/-0.2 (var = 0.04) => rho = 0.90
            unit_mu = 0.6 if rng.random() < 0.5 else -0.6
            noise = np.where(rng.random(50) < 0.5, 0.2, -0.2)
            dec = pop.add_unit(unit_mu + noise)
            if dec.certified:
                false_certs += 1
                break
        if trial < 20:
            rho_estimates.append(dec.rho_hat)

    _assert_rate_within_alpha(false_certs, n_trials, delta, "PopulationCert rho=0.90 balanced n=50")
    assert np.mean(rho_estimates) > 0.80, f"ICC estimate unexpectedly low: {np.mean(rho_estimates)}"

    # Regime B: Strongly unbalanced rollout counts [1, 2, 50, 1, 100] cycling across units
    pattern = [1, 2, 50, 1, 100]
    false_certs_unbal = 0
    for _ in range(n_trials):
        pop = PopulationCert(delta=delta)
        for u in range(n_units):
            m = pattern[u % len(pattern)]
            unit_mu = 0.7 if rng.random() < 0.5 else -0.7
            noise = rng.uniform(-0.2, 0.2, size=m)
            dec = pop.add_unit(np.clip(unit_mu + noise, -1.0, 1.0))
            assert math.isfinite(dec.rho_hat)
            assert math.isfinite(dec.sigma_b2) and math.isfinite(dec.sigma_w2)
            if dec.certified:
                false_certs_unbal += 1
                break

    _assert_rate_within_alpha(
        false_certs_unbal, n_trials, delta, "PopulationCert unbalanced [1,2,50,1,100] null"
    )


# ============================================================================
# 4. EmpiricalBernsteinCS & BettingCS: Boundaries, off-grid, skew, variance-switching
# ============================================================================


@pytest.mark.parametrize("true_mu", [0.0, 1.0, 0.005, 0.995, 0.50234567, 0.01])
def test_ebcs_and_bettingcs_boundary_and_skew_anytime_coverage(true_mu: float):
    """Anytime miscoverage across boundary, off-grid, and extreme-skew Bernoulli means."""
    alpha = 0.05
    n_trials = 600
    horizon = 400
    rng = np.random.default_rng(40000 + int(true_mu * 1_000_000))

    eb_viol = 0
    bet_viol = 0

    for _ in range(n_trials):
        eb = EmpiricalBernsteinCS(alpha=alpha)
        bet = BettingCS(alpha=alpha, grid_points=201)
        xs = (rng.random(horizon) < true_mu).astype(np.float64)
        eb_bad = False
        bet_bad = False
        for x in xs:
            xf = float(x)
            if not eb_bad:
                b = eb.update(xf)
                if true_mu < b.lower - 1e-15 or true_mu > b.upper + 1e-15:
                    eb_bad = True
            if not bet_bad:
                b = bet.update(xf)
                if true_mu < b.lower - 1e-15 or true_mu > b.upper + 1e-15:
                    bet_bad = True
            if eb_bad and bet_bad:
                break
        eb_viol += int(eb_bad)
        bet_viol += int(bet_bad)

    _assert_rate_within_alpha(eb_viol, n_trials, alpha, f"EmpiricalBernsteinCS mu={true_mu}")
    _assert_rate_within_alpha(bet_viol, n_trials, alpha, f"BettingCS mu={true_mu}")


def test_ebcs_and_bettingcs_adversarial_variance_switching_martingale():
    """Adversarial martingale with constant E[X_t | F_{t-1}] = 0.37 but wild variance switches.

    Feeds 60 constant observations X_t = mu to lull variance estimates into near-zero
    (driving betting fractions to their ceiling), then slams with maximal-variance {0, 1}
    Bernoulli bursts, repeating across cycles.
    """
    alpha = 0.05
    true_mu = 0.37
    n_trials = 800
    horizon = 400
    rng = np.random.default_rng(40999)

    eb_viol = 0
    bet_viol = 0

    for _ in range(n_trials):
        eb = EmpiricalBernsteinCS(alpha=alpha)
        bet = BettingCS(alpha=alpha, grid_points=201)
        eb_bad = False
        bet_bad = False
        for t in range(horizon):
            phase = t % 80
            if phase < 60:
                x = true_mu  # zero conditional variance lull
            else:
                x = 1.0 if rng.random() < true_mu else 0.0  # maximal conditional variance burst
            if not eb_bad:
                b = eb.update(x)
                if true_mu < b.lower or true_mu > b.upper:
                    eb_bad = True
            if not bet_bad:
                b = bet.update(x)
                if true_mu < b.lower or true_mu > b.upper:
                    bet_bad = True
            if eb_bad and bet_bad:
                break
        eb_viol += int(eb_bad)
        bet_viol += int(bet_bad)

    _assert_rate_within_alpha(
        eb_viol, n_trials, alpha, "EmpiricalBernsteinCS adversarial variance-switching"
    )
    _assert_rate_within_alpha(
        bet_viol, n_trials, alpha, "BettingCS adversarial variance-switching"
    )


# ============================================================================
# 5. PairedEProcess & AgentRace: Extreme skewed zero-mean differences
# ============================================================================


def test_paired_eprocess_extreme_skewed_null():
    """PairedEProcess under extreme skewed zero-mean diffs: +1 w.p. 0.05, -0.05263... w.p. 0.95."""
    alpha = 0.05
    n_trials = 1500
    horizon = 600
    p_hi = 0.05
    v_lo = -p_hi / (1.0 - p_hi)  # exact mean 0: 0.05 * 1.0 + 0.95 * (-1/19) = 0
    rng = np.random.default_rng(50001)
    false_pos = 0

    for _ in range(n_trials):
        pe = PairedEProcess(alpha=alpha, scale=1.0)
        u = rng.random(horizon)
        for ui in u:
            if ui < p_hi:
                a, b = 1.0, 0.0  # diff = +1.0
            else:
                a, b = 0.0, -v_lo  # diff = -1/19
            st = pe.update(a, b)
            if st.winner is not None or st.anytime_p_value <= alpha:
                false_pos += 1
                break

    _assert_rate_within_alpha(false_pos, n_trials, alpha, "PairedEProcess extreme skewed null")


def test_agent_race_global_tie_skewed_paired_and_unpaired_lucb():
    """AgentRace FWER under global tie null with extreme skewed paired diffs + unpaired LUCB."""
    delta = 0.05
    n_arms = 4
    n_trials = 800
    horizon = 300
    rng = np.random.default_rng(50002)

    # Paired mode: arm 0 vs peers has extreme skewed difference (+1 w.p. 0.05, -1/19 w.p. 0.95)
    # Construct scores in [0, 1] with identical mean mu = 0.10 across all arms:
    # Arm 0: 1.0 w.p. 0.10, 0.0 w.p. 0.90; Arms 1..3: constant 0.10 (extreme skew!)
    false_paired = 0
    for _ in range(n_trials):
        race = AgentRace(n_arms=n_arms, delta=delta, paired=True)
        for _ in range(horizon):
            s0 = 1.0 if rng.random() < 0.05 else 0.0
            # Shift so E[s0] = 0.05 and E[s1..3] = 0.05 (diff = +0.95 w.p. 0.05, -0.05 w.p. 0.95)
            scores = [s0, 0.05, 0.05, 0.05]
            w = race.observe_paired(scores)
            if w is not None:
                false_paired += 1
                break

    _assert_rate_within_alpha(
        false_paired, n_trials, delta, "AgentRace paired skewed global-tie FWER"
    )

    # Unpaired LUCB mode: identical Bernoulli(0.05) extreme-skew arms
    false_unpaired = 0
    for _ in range(n_trials):
        race = AgentRace(n_arms=n_arms, delta=delta, paired=False)
        for _ in range(horizon * 2):
            arm = race.next_arm_lucb()
            x = 1.0 if rng.random() < 0.05 else 0.0
            w = race.observe_arm(arm, x)
            if w is not None:
                false_unpaired += 1
                break

    _assert_rate_within_alpha(
        false_unpaired, n_trials, delta, "AgentRace unpaired LUCB skewed global-tie FWER"
    )


# ============================================================================
# 6. ActivePPI: Inverted judge, extreme bias, tiny min_audit_prob, correlated pi_t
# ============================================================================


def test_active_ppi_adversarial_judges_and_correlated_adaptive_pi():
    """ActivePPI anytime coverage under inverted judge, biased judge, min_pi=0.01, correlated pi_t."""
    alpha = 0.05
    n_trials = 500
    horizon = 400
    true_mu = 0.35
    rng = np.random.default_rng(60001)

    regimes = [
        ("inverted_judge_f=1-y", 0.15, 0.05),
        ("extreme_bias_f=0.99", 0.15, 0.05),
        ("tiny_min_pi=0.01", 0.05, 0.01),
        ("extreme_adaptive_pi_correlated_with_y", 0.20, 0.02),
    ]

    for name, target_rate, min_pi in regimes:
        violations = 0
        for _ in range(n_trials):
            ppi = ActivePPI(alpha=alpha, target_audit_rate=target_rate, min_audit_prob=min_pi)
            violated = False
            for _ in range(horizon):
                if name == "extreme_adaptive_pi_correlated_with_y":
                    # Subpopulation Z observable to judge & sampler -> pi_t strongly correlated with Y_t!
                    # Z=1 (prob 0.35) => Y=1 w.p. 0.90, Z=0 (prob 0.65) => Y=1 w.p. 0.053846...
                    # Exact E[Y] = 0.35 * 0.90 + 0.65 * (0.035 / 0.65) = 0.35
                    z = 1 if rng.random() < 0.35 else 0
                    p_y = 0.90 if z == 1 else (0.035 / 0.65)
                    y = 1.0 if rng.random() < p_y else 0.0
                    f = 0.95 if z == 1 else 0.05
                    # Extreme adaptive pi_t: 0.90 when z=1 (high y), min_pi=0.02 when z=0 (low y)!
                    pi = 0.90 if z == 1 else min_pi
                elif name == "inverted_judge_f=1-y":
                    y = 1.0 if rng.random() < true_mu else 0.0
                    f = 1.0 - y
                    pi = ppi.suggest_prob(f, 0.5)
                elif name == "extreme_bias_f=0.99":
                    y = 1.0 if rng.random() < true_mu else 0.0
                    f = 0.99
                    pi = ppi.suggest_prob(f, 0.1)
                else:  # tiny_min_pi=0.01
                    y = 1.0 if rng.random() < true_mu else 0.0
                    f = float(np.clip(0.8 * y + 0.1 + rng.normal(0.0, 0.15), 0.0, 1.0))
                    pi = min_pi if rng.random() < 0.7 else 0.15

                aud = bool(rng.random() < pi)
                res = ppi.observe(f, aud, y if aud else 0.0, pi)
                if true_mu < res.lower - 1e-12 or true_mu > res.upper + 1e-12:
                    violated = True
                    break
            violations += int(violated)

        _assert_rate_within_alpha(violations, n_trials, alpha, f"ActivePPI [{name}]")


# ============================================================================
# 7. SafeFuse: Skewed heavy-tailed null losses -> quarantine FWER attack!
# ============================================================================


def test_safe_fuse_skewed_heavy_tailed_null_fwer_right_skewed_agent0():
    """All 4 agents have identical mean loss E[L_j] = 1.0: agent 0 is skewed Bernoulli(0.05)*20
    while peers 1..3 are Gaussian(1.0, 0.02). Quarantine FWER MUST stay <= quarantine_alpha!
    """
    q_alpha = 0.05
    n_trials = 500
    horizon = 250
    rng = np.random.default_rng(70001)
    false_quarantines = 0

    for _ in range(n_trials):
        fuse = SafeFuse(n_agents=4, quarantine_alpha=q_alpha, fallback_alpha=0.05)
        for _ in range(horizon):
            l0 = 20.0 if rng.random() < 0.05 else 0.0  # E[l0] = 1.0
            peers = 1.0 + rng.normal(0.0, 0.02, size=3)  # E[peers] = 1.0
            n_new = fuse.update([l0, float(peers[0]), float(peers[1]), float(peers[2])])
            if n_new > 0:
                false_quarantines += 1
                break

    _assert_rate_within_alpha(
        false_quarantines,
        n_trials,
        q_alpha,
        "SafeFuse null: right-skewed Bernoulli(0.05)*20 agent0 vs Gaussian(1,0.02) peers",
    )


def test_safe_fuse_skewed_heavy_tailed_null_fwer_exp_agent0():
    """All 4 agents have identical mean loss E[L_j] = 1.0: agent 0 is Exp(1) while peers are N(1, 0.02)."""
    q_alpha = 0.05
    n_trials = 400
    horizon = 800
    rng = np.random.default_rng(70002)
    false_quarantines = 0

    for _ in range(n_trials):
        fuse = SafeFuse(n_agents=4, quarantine_alpha=q_alpha, fallback_alpha=0.05)
        for _ in range(horizon):
            l0 = float(rng.exponential(1.0))  # E[l0] = 1.0
            peers = 1.0 + rng.normal(0.0, 0.02, size=3)  # E[peers] = 1.0
            n_new = fuse.update([l0, float(peers[0]), float(peers[1]), float(peers[2])])
            if n_new > 0:
                false_quarantines += 1
                break

    _assert_rate_within_alpha(
        false_quarantines,
        n_trials,
        q_alpha,
        "SafeFuse null: Exp(1) agent0 vs Gaussian(1,0.02) peers",
    )


def test_safe_fuse_left_skewed_null_against_baseline():
    """Left-skewed null agent0 (1.05 w.p. 20/21, 0.0 w.p. 1/21 => exact mean 1.0) vs baseline=1.0."""
    q_alpha = 0.05
    n_trials = 400
    horizon = 300
    rng = np.random.default_rng(70003)
    false_quarantines = 0

    for _ in range(n_trials):
        fuse = SafeFuse(n_agents=3, quarantine_alpha=q_alpha, fallback_alpha=0.05)
        for _ in range(horizon):
            l0 = 0.0 if rng.random() < (1.0 / 21.0) else 1.05  # exact mean 1.0
            l1 = 1.0 + float(rng.normal(0.0, 0.01))
            l2 = 1.0 + float(rng.normal(0.0, 0.01))
            n_new = fuse.update([l0, l1, l2], baseline_loss=1.0)
            if n_new > 0:
                false_quarantines += 1
                break

    _assert_rate_within_alpha(
        false_quarantines,
        n_trials,
        q_alpha,
        "SafeFuse null: left-skewed agent0 vs explicit baseline=1.0",
    )


def test_safe_fuse_global_fallback_false_trip_skewed_null():
    """Global fallback gate must maintain false-trip rate <= fallback_alpha under skewed null."""
    fb_alpha = 0.05
    n_trials = 400
    horizon = 300
    rng = np.random.default_rng(70004)
    false_trips = 0

    for _ in range(n_trials):
        fuse = SafeFuse(n_agents=3, quarantine_alpha=0.001, fallback_alpha=fb_alpha)
        for _ in range(horizon):
            l0 = 0.0 if rng.random() < (1.0 / 21.0) else 1.05  # exact mean 1.0
            fuse.update([l0, l0, l0], baseline_loss=1.0)
            _, fb = fuse.predict([0.0, 0.0, 0.0])
            if fb:
                false_trips += 1
                break

    _assert_rate_within_alpha(
        false_trips,
        n_trials,
        fb_alpha,
        "SafeFuse global fallback false-trip: left-skewed null vs baseline=1.0",
    )


# ============================================================================
# 8. AbortCascade: Finite-sample recall across 200 splits + small n_pos + ties + overflow
# ============================================================================


@pytest.mark.parametrize("n_pos", [10, 30, 50, 200])
def test_abort_cascade_finite_sample_recall_200_splits(n_pos: int):
    """Verify finite-sample expected test recall >= target_recall across 200 independent splits."""
    target_recall = 0.90
    n_splits = 200
    n_neg = n_pos
    n_test_pos = 200
    rng = np.random.default_rng(80000 + n_pos)
    recalls = []

    for _ in range(n_splits):
        pos_cal = rng.normal(loc=[0.6, 0.7, 0.8], scale=[0.2, 0.2, 0.2], size=(n_pos, 3))
        neg_cal = rng.normal(loc=[0.2, 0.2, 0.1], scale=[0.2, 0.2, 0.2], size=(n_neg, 3))
        scores_cal = np.ascontiguousarray(np.vstack([pos_cal, neg_cal]), dtype=np.float64)
        succ_cal = np.ascontiguousarray(
            np.concatenate([np.ones(n_pos, dtype=np.uint8), np.zeros(n_neg, dtype=np.uint8)])
        )
        cascade = AbortCascade.calibrate(
            scores_cal, succ_cal, target_recall=target_recall, stage_costs=[1.0, 2.0, 4.0]
        )
        pos_test = rng.normal(loc=[0.6, 0.7, 0.8], scale=[0.2, 0.2, 0.2], size=(n_test_pos, 3))
        survived = 0
        for row in pos_test:
            aborted = False
            for s in range(3):
                if cascade.should_abort(s, float(row[s])):
                    aborted = True
                    break
            if not aborted:
                survived += 1
        recalls.append(survived / n_test_pos)

    mean_recall = float(np.mean(recalls))
    se = float(np.std(recalls, ddof=1) / math.sqrt(n_splits))
    assert mean_recall >= target_recall - 3.0 * se, (
        f"AbortCascade (n_pos={n_pos}) finite-sample recall {mean_recall:.4f} "
        f"(SE={se:.4f}) below target_recall={target_recall}"
    )


def test_abort_cascade_tied_scores_recall():
    """Heavily discrete / tied scores must still guarantee test recall >= target_recall."""
    target_recall = 0.90
    n_splits = 200
    rng = np.random.default_rng(80999)
    recalls = []

    for _ in range(n_splits):
        # Only 3 discrete score levels {0.0, 0.5, 1.0} => massive ties!
        pos_cal = rng.choice([0.0, 0.5, 1.0], size=(60, 3), p=[0.1, 0.4, 0.5])
        neg_cal = rng.choice([0.0, 0.5, 1.0], size=(60, 3), p=[0.6, 0.3, 0.1])
        scores_cal = np.ascontiguousarray(np.vstack([pos_cal, neg_cal]), dtype=np.float64)
        succ_cal = np.ascontiguousarray(
            np.concatenate([np.ones(60, dtype=np.uint8), np.zeros(60, dtype=np.uint8)])
        )
        cascade = AbortCascade.calibrate(scores_cal, succ_cal, target_recall=target_recall)
        pos_test = rng.choice([0.0, 0.5, 1.0], size=(150, 3), p=[0.1, 0.4, 0.5])
        survived = sum(
            not any(cascade.should_abort(s, float(row[s])) for s in range(3)) for row in pos_test
        )
        recalls.append(survived / len(pos_test))

    mean_recall = float(np.mean(recalls))
    assert mean_recall >= target_recall - 0.01, (
        f"AbortCascade tied-scores recall {mean_recall:.4f} < {target_recall}"
    )


def test_abort_cascade_extreme_finite_scores_no_inf_threshold():
    """Extreme finite scores (+-1e308) must NOT overflow into threshold = +Inf (0% recall)."""
    n_pos = 80
    n_neg = 40
    # Positives at +1e308, negatives at -1e308 => triggers split-conformal CRC path!
    pos = np.full((n_pos, 2), 1e308, dtype=np.float64)
    neg = np.full((n_neg, 2), -1e308, dtype=np.float64)
    # Give one design positive a -1e308 so tau_design can sit at -1e308 while pos_cal has +1e308
    pos[:5, 0] = -1e308
    scores = np.ascontiguousarray(np.vstack([pos, neg]), dtype=np.float64)
    succ = np.ascontiguousarray(
        np.concatenate([np.ones(n_pos, dtype=np.uint8), np.zeros(n_neg, dtype=np.uint8)])
    )
    cascade = AbortCascade.calibrate(scores, succ, target_recall=0.90)
    thresh = cascade.thresholds()
    assert all(t != float("inf") and not math.isnan(t) for t in thresh), (
        f"AbortCascade overflowed to +Inf/NaN threshold on finite scores: {thresh}"
    )
    # A positive at +1e308 MUST NOT be aborted!
    assert not cascade.should_abort(0, 1e308), f"Aborted +1e308 positive! thresholds={thresh}"
