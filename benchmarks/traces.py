"""Deterministic, calibrated empirical trace generators for sequential inference benchmarks.

All generators are fully deterministic given ``seed`` (using ``numpy.random.Generator``
with ``PCG64``) and calibrated to published empirical regimes:

1. ``gen_ascd_benchmark`` — Table 2 mixture-categorical soft/hard self-consistency regime
   from Huang et al. (arXiv:2609.38931, Sep 2026).
2. ``gen_open_ended_swarm`` — Open-ended Zipfian long-tail plurality distribution for
   category-agnostic mode stopping (e.g., code patches or free-form math answers).
3. ``gen_correlated_judge_panel`` — Correlated 9-judge, 3-class LLM evaluation panel
   calibrated to Kohli (arXiv:2605.29800, May 2026).
4. ``gen_correlated_rollouts_aime`` — Correlated multi-rollout AIME benchmark regime
   calibrated to Agbeyegbe & Agbeyegbe (arXiv:2611.26592, Nov 2026).
5. ``gen_active_ppi_stream`` & ``gen_adversarial_ppi_stream`` — Heteroskedastic biased
   and adversarially inverted LLM-judge streams for Prediction-Powered Inference
   (Angelopoulos et al. 2023/2024; Zrnic & Candes 2024).
6. ``gen_agent_race`` — Multi-arm agent evaluation with shared prompt difficulty for
   paired vs. unpaired sequential racing.
7. ``gen_poisoned_swarm`` — Multi-agent consensus stream with mid-stream silent agent
   poisoning / degradation.
8. ``gen_abort_cascade`` — Multi-turn agent trajectories with causal early-round failure
   signals and growing per-stage context costs.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
from scipy.stats import multivariate_normal, norm


class TraceDict(dict):
    """Dictionary supporting both key (``d['x']``) and attribute (``d.x``) access."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


# ---------------------------------------------------------------------------
# 1. ASCD Benchmark (Huang et al., arXiv:2609.38931, Sep 2026, Table 2)
# ---------------------------------------------------------------------------


def gen_ascd_benchmark(
    seed: int = 42,
    delta_gap: float = 0.10,
    n_trials: int = 500,
    max_n: int = 500,
) -> TraceDict:
    """Reproduce Table 2 of Huang et al. (arXiv:2609.38931, Sep 2026).

    Data-generating process (K = 4):
        pi_Delta = (0.35 + Delta / 2, 0.35 - Delta / 2, 0.15, 0.15)
        Z_t ~ Cat(pi_Delta)
        theta_t = 0.5 * e_{Z_t} + 0.5 * pi_Delta
        Y_t | theta_t ~ Cat(theta_t)

    Both ``theta_t`` (soft probability vector) and ``e_{Y_t}`` (hard vote) have exact
    expectation ``pi_Delta``, while ``theta_t`` enjoys a 4x variance reduction over
    hard one-hot draws, enabling paired evaluation of soft-probability vs. hard-vote
    sequential consensus stoppers.
    """
    if not (0.0 <= delta_gap <= 0.70):
        raise ValueError(f"delta_gap must lie in [0.0, 0.70], got {delta_gap}")
    if n_trials <= 0 or max_n <= 0:
        raise ValueError("n_trials and max_n must be positive")

    rng = np.random.default_rng(seed)
    k = 4
    pi = np.array(
        [0.35 + 0.5 * delta_gap, 0.35 - 0.5 * delta_gap, 0.15, 0.15],
        dtype=np.float64,
    )
    pi /= pi.sum()

    # Sample latent reasoning class Z_t ~ Cat(pi_Delta)
    cdf_pi = np.cumsum(pi)
    cdf_pi[-1] = 1.0
    u_z = rng.random((n_trials, max_n))
    zs = np.searchsorted(cdf_pi, u_z).astype(np.int32)

    # theta_t = 0.5 * e_{Z_t} + 0.5 * pi_Delta
    eye_k = np.eye(k, dtype=np.float64)
    thetas = 0.5 * eye_k[zs] + 0.5 * pi[None, None, :]

    # Matched hard draw Y_t | theta_t ~ Cat(theta_t)
    cdf_theta = np.cumsum(thetas, axis=-1)
    cdf_theta[..., -1] = 1.0
    u_y = rng.random((n_trials, max_n, 1))
    ys = np.argmax(u_y < cdf_theta, axis=-1).astype(np.int32)

    true_winner = 0 if delta_gap > 0.0 else -1

    emp_theta_mean = thetas.mean(axis=(0, 1))
    emp_y_counts = np.bincount(ys.ravel(), minlength=k) / float(n_trials * max_n)
    empirical_stats = {
        "theta_mean": emp_theta_mean.tolist(),
        "y_empirical_probs": emp_y_counts.tolist(),
        "theta_var_0": float(thetas[..., 0].var()),
        "y_var_0": float((ys == 0).astype(np.float64).var()),
        "variance_ratio_y_over_theta": float(
            (ys == 0).astype(np.float64).var() / max(thetas[..., 0].var(), 1e-12)
        ),
    }

    return TraceDict(
        thetas=thetas,
        ys=ys,
        zs=zs,
        pi=pi,
        k=k,
        delta_gap=float(delta_gap),
        true_winner=true_winner,
        n_trials=int(n_trials),
        max_n=int(max_n),
        seed=int(seed),
        empirical_stats=empirical_stats,
    )


# ---------------------------------------------------------------------------
# 2. Open-Ended Swarm (Unbounded / Zipfian Long-Tail Answer Distribution)
# ---------------------------------------------------------------------------


def gen_open_ended_swarm(
    seed: int = 42,
    k_true: int = 200,
    zipf_s: float = 1.25,
    top_gap: float = 0.12,
    n_trials: int = 500,
    max_n: int = 256,
) -> TraceDict:
    """Generate an open-ended long-tail answer stream for category-agnostic stopping.

    Constructs a strictly ordered Zipfian distribution over ``k_true`` candidate answers
    with exact plurality margin ``probs[0] - probs[1] == top_gap`` and a diffuse tail
    ``probs[j] proportional to (j + 1)^(-zipf_s)`` for ``j >= 2``. Typically ``probs[0] < 0.5``
    (plurality mode without absolute majority), modeling code patches or free-form math answers.
    """
    if k_true < 2:
        raise ValueError(f"k_true must be >= 2, got {k_true}")
    if zipf_s <= 0.0:
        raise ValueError(f"zipf_s must be > 0, got {zipf_s}")
    if not (0.0 <= top_gap < 1.0):
        raise ValueError(f"top_gap must lie in [0.0, 1.0), got {top_gap}")
    if n_trials <= 0 or max_n <= 0:
        raise ValueError("n_trials and max_n must be positive")

    ranks = np.arange(1, k_true + 1, dtype=np.float64)
    raw = ranks ** (-zipf_s)
    q = raw / raw.sum()

    if k_true == 2:
        probs = np.array([0.5 * (1.0 + top_gap), 0.5 * (1.0 - top_gap)], dtype=np.float64)
    else:
        tail_raw = raw[2:]
        tail_norm = tail_raw / tail_raw.sum()
        t2 = float(tail_norm[0])
        # Guarantee p1 > p2 = M_tail * t2 while preserving natural Zipfian tail mass
        max_tail_mass = (1.0 - top_gap) / (1.0 + 2.0 * t2)
        natural_tail_mass = float(q[2:].sum())
        tail_mass = min(natural_tail_mass, 0.85 * max_tail_mass)
        tail_mass = max(tail_mass, 0.0)
        p0 = 0.5 * (1.0 - tail_mass + top_gap)
        p1 = 0.5 * (1.0 - tail_mass - top_gap)
        probs = np.empty(k_true, dtype=np.float64)
        probs[0] = p0
        probs[1] = p1
        probs[2:] = tail_mass * tail_norm
        probs /= probs.sum()

    rng = np.random.default_rng(seed)
    cdf = np.cumsum(probs)
    cdf[-1] = 1.0
    u = rng.random((n_trials, max_n))
    ys = np.searchsorted(cdf, u).astype(np.int32)

    true_winner = 0 if top_gap > 0.0 else -1
    unique_per_trial = np.array([np.unique(row).size for row in ys], dtype=np.float64)
    empirical_stats = {
        "p0": float(probs[0]),
        "p1": float(probs[1]),
        "p2": float(probs[2]) if k_true >= 3 else 0.0,
        "exact_top_gap": float(probs[0] - probs[1]),
        "tail_mass": float(probs[2:].sum()) if k_true >= 3 else 0.0,
        "mean_unique_categories_observed": float(unique_per_trial.mean()),
        "empirical_p0": float((ys == 0).mean()),
        "empirical_p1": float((ys == 1).mean()),
    }

    return TraceDict(
        ys=ys,
        probs=probs,
        k_true=int(k_true),
        zipf_s=float(zipf_s),
        top_gap=float(top_gap),
        true_winner=true_winner,
        n_trials=int(n_trials),
        max_n=int(max_n),
        seed=int(seed),
        empirical_stats=empirical_stats,
    )


# ---------------------------------------------------------------------------
# 3. Correlated 9-Judge Panel (Kohli, arXiv:2605.29800, May 2026)
# ---------------------------------------------------------------------------


def _bivariate_error_phi(e_i: float, e_j: float, rho_ij: float) -> float:
    """Exact Pearson/Matthews phi correlation between thresholded bivariate normals."""
    rho_clamped = float(np.clip(rho_ij, -0.9995, 0.9995))
    t_i = float(norm.ppf(1.0 - e_i))
    t_j = float(norm.ppf(1.0 - e_j))
    cov = np.array([[1.0, rho_clamped], [rho_clamped, 1.0]], dtype=np.float64)
    p_both = float(multivariate_normal.cdf([-t_i, -t_j], mean=[0.0, 0.0], cov=cov))
    denom = np.sqrt(e_i * (1.0 - e_i) * e_j * (1.0 - e_j))
    return (p_both - e_i * e_j) / max(denom, 1e-12)


def _plurality_accuracy_expected_tie(
    votes: np.ndarray, true_labels: np.ndarray, n_classes: int
) -> float:
    """Plurality accuracy with exact expected uniform tie-breaking."""
    n_items = votes.shape[0]
    counts = np.zeros((n_items, n_classes), dtype=np.int32)
    for c in range(n_classes):
        counts[:, c] = (votes == c).sum(axis=1)
    max_counts = counts.max(axis=1, keepdims=True)
    is_max = counts == max_counts
    n_tied = is_max.sum(axis=1)
    true_is_max = is_max[np.arange(n_items), true_labels]
    return float(np.mean(true_is_max / n_tied))


def _condorcet_independent_accuracy(
    error_rates: np.ndarray, n_classes: int = 3, mc_samples: int = 40000, seed: int = 0
) -> float:
    """Independent Condorcet plurality accuracy for judges with error_rates (uniform wrong class)."""
    rng = np.random.default_rng(seed)
    k = error_rates.shape[0]
    errs = rng.random((mc_samples, k)) < error_rates[None, :]
    # Without loss of generality, true class = 0; wrong classes uniform in {1, ..., n_classes - 1}
    wrong_choice = rng.integers(1, n_classes, size=(mc_samples, k), dtype=np.int32)
    votes = np.where(errs, wrong_choice, 0)
    true_labels = np.zeros(mc_samples, dtype=np.int32)
    return _plurality_accuracy_expected_tie(votes, true_labels, n_classes)


def gen_correlated_judge_panel(
    seed: int = 42,
    n_items: int = 1000,
    k_judges: int = 9,
    target_phi: float = 0.391,
) -> TraceDict:
    """Reproduce the correlated 9-judge, 3-class panel of Kohli (arXiv:2605.29800, May 2026).

    Calibrated target statistics (for ``k_judges=9, target_phi=0.391``):
    - 9 judges, 3 classes, individual base error rates in ~0.28--0.36.
    - Mean pairwise error correlation phi ~ 0.391 (calibrated via exact bivariate probit root-finding).
    - Effective panel size ``n_eff = K / (1 + (K - 1) * mean_phi)`` ~ 2.1--2.3.
    - Condorcet gap (independent Condorcet accuracy minus actual correlated majority accuracy) ~ 15--22pp.
    - One redundant, high-correlation judge (index ``k_judges - 1``) belonging to the dominant
      error cluster whose removal improves unweighted majority-vote accuracy.
    """
    if n_items <= 0 or k_judges < 3:
        raise ValueError("n_items must be > 0 and k_judges >= 3")
    if not (0.0 <= target_phi < 0.85):
        raise ValueError(f"target_phi must lie in [0.0, 0.85), got {target_phi}")

    n_classes = 3

    # Individual base error rates in [0.285, 0.352]
    if k_judges == 9:
        base_errors = np.array(
            [0.285, 0.292, 0.298, 0.306, 0.312, 0.322, 0.332, 0.342, 0.352],
            dtype=np.float64,
        )
        # 3 model families:
        #   Family 0 (judges 0..2): lower-error, diverse judges
        #   Family 1 (judges 3..4): mid-tier judges
        #   Family 2 (judges 5..8): tightly correlated dominant cluster, with judge 8 redundant
        clusters = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2], dtype=np.int32)
        raw_global = np.array([0.65, 0.67, 0.69, 0.75, 0.77, 0.81, 0.82, 0.83, 0.84], dtype=np.float64)
        raw_cluster = np.array([0.08, 0.08, 0.08, 0.22, 0.22, 0.55, 0.56, 0.55, 0.54], dtype=np.float64)
        distractor_probs = np.array([0.64, 0.66, 0.68, 0.72, 0.74, 0.84, 0.86, 0.88, 0.94], dtype=np.float64)
    else:
        base_errors = np.linspace(0.285, 0.352, k_judges, dtype=np.float64)
        n_dom = max(2, k_judges // 2)
        clusters = np.zeros(k_judges, dtype=np.int32)
        clusters[-n_dom:] = 1
        raw_global = np.linspace(0.66, 0.84, k_judges, dtype=np.float64)
        raw_cluster = np.where(clusters == 1, 0.54, 0.12)
        distractor_probs = np.linspace(0.64, 0.94, k_judges, dtype=np.float64)

    # Normalize raw loadings so max(a_j^2 + b_j^2) = 0.985 at scale s = 1.0
    norms = np.sqrt(raw_global**2 + raw_cluster**2)
    max_norm = float(norms.max())
    if max_norm > np.sqrt(0.985):
        rescale = np.sqrt(0.985) / max_norm
        raw_global = raw_global * rescale
        raw_cluster = raw_cluster * rescale

    def _mean_theoretical_phi(scale: float) -> float:
        a = scale * raw_global
        b = scale * raw_cluster
        phis = []
        for i in range(k_judges):
            for j in range(i + 1, k_judges):
                rho_ij = float(a[i] * a[j] + (b[i] * b[j] if clusters[i] == clusters[j] else 0.0))
                phis.append(_bivariate_error_phi(float(base_errors[i]), float(base_errors[j]), rho_ij))
        return float(np.mean(phis))

    max_phi = _mean_theoretical_phi(1.0)
    if target_phi <= 0.0:
        scale_star = 0.0
    elif target_phi >= max_phi:
        scale_star = 1.0
    else:
        scale_star = float(brentq(lambda s: _mean_theoretical_phi(s) - target_phi, 0.0, 1.0, xtol=1e-5))

    alpha = scale_star * raw_global
    beta = scale_star * raw_cluster
    gamma = np.sqrt(np.maximum(1.0 - alpha**2 - beta**2, 1e-6))

    rng = np.random.default_rng(seed)
    true_labels = rng.integers(0, n_classes, size=n_items, dtype=np.int32)

    # Shared item difficulty U_t, family factors C_{t,g}, and idiosyncratic noise eps_{t,j}
    n_clusters = int(clusters.max()) + 1
    u_item = rng.standard_normal(n_items)
    c_cluster = rng.standard_normal((n_items, n_clusters))
    eps = rng.standard_normal((n_items, k_judges))

    z = (
        u_item[:, None] * alpha[None, :]
        + c_cluster[:, clusters] * beta[None, :]
        + eps * gamma[None, :]
    )
    thresholds = norm.ppf(1.0 - base_errors)
    errors = (z > thresholds[None, :]).astype(np.int32)

    # Primary confusable distractor per item (different from true_label)
    offset = rng.integers(1, n_classes, size=n_items, dtype=np.int32)
    primary_distractor = (true_labels + offset) % n_classes
    secondary_distractor = (true_labels + (3 - offset)) % n_classes

    pick_primary = rng.random((n_items, k_judges)) < distractor_probs[None, :]
    wrong_vote = np.where(pick_primary, primary_distractor[:, None], secondary_distractor[:, None])
    votes = np.where(errors == 1, wrong_vote, true_labels[:, None]).astype(np.int32)

    # Empirical diagnostics
    err_f = errors.astype(np.float64)
    emp_error_rates = err_f.mean(axis=0)
    phi_matrix = np.corrcoef(err_f, rowvar=False)
    triu_idx = np.triu_indices(k_judges, k=1)
    mean_phi = float(phi_matrix[triu_idx].mean())
    kish_n_eff = float(k_judges / (1.0 + (k_judges - 1.0) * mean_phi))

    maj_acc = _plurality_accuracy_expected_tie(votes, true_labels, n_classes)
    condorcet_acc = _condorcet_independent_accuracy(emp_error_rates, n_classes=n_classes, seed=seed + 10007)
    condorcet_gap = float(condorcet_acc - maj_acc)

    redundant_idx = k_judges - 1
    loo_accs = []
    for j in range(k_judges):
        keep = [idx for idx in range(k_judges) if idx != j]
        loo_accs.append(_plurality_accuracy_expected_tie(votes[:, keep], true_labels, n_classes))
    loo_acc_without_redundant = float(loo_accs[redundant_idx])

    empirical_stats = {
        "error_rates": emp_error_rates.tolist(),
        "mean_error_rate": float(emp_error_rates.mean()),
        "mean_phi": mean_phi,
        "redundant_mean_phi": float(
            np.mean([phi_matrix[redundant_idx, j] for j in range(k_judges) if j != redundant_idx])
        ),
        "kish_n_eff": kish_n_eff,
        "majority_acc": maj_acc,
        "condorcet_independent_acc": condorcet_acc,
        "condorcet_gap": condorcet_gap,
        "loo_acc_without_redundant": loo_acc_without_redundant,
        "redundant_removal_gain": float(loo_acc_without_redundant - maj_acc),
        "loo_accs": [float(x) for x in loo_accs],
    }

    return TraceDict(
        votes=votes,
        true_labels=true_labels,
        errors=errors,
        n_classes=n_classes,
        k_judges=int(k_judges),
        n_items=int(n_items),
        base_error_rates=base_errors,
        redundant_judge_idx=int(redundant_idx),
        target_phi=float(target_phi),
        phi_matrix=phi_matrix,
        clusters=clusters,
        seed=int(seed),
        empirical_stats=empirical_stats,
    )


# ---------------------------------------------------------------------------
# 4. Correlated Rollouts AIME Regime (Agbeyegbe & Agbeyegbe, arXiv:2611.26592)
# ---------------------------------------------------------------------------


def _anova_icc(matrix: np.ndarray) -> float:
    """One-way random-effects ANOVA intraclass correlation ICC(1)."""
    n_units, m = matrix.shape
    unit_means = matrix.mean(axis=1)
    grand_mean = float(matrix.mean())
    ss_between = m * float(np.sum((unit_means - grand_mean) ** 2))
    ss_within = float(np.sum((matrix - unit_means[:, None]) ** 2))
    ms_between = ss_between / max(n_units - 1, 1)
    ms_within = ss_within / max(n_units * (m - 1), 1)
    denom = ms_between + (m - 1) * ms_within
    return float((ms_between - ms_within) / denom) if denom > 0 else 0.0


def _single_unit_eb_cs_rejects(unit_scores_01: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Check if an anytime-valid empirical-Bernstein / betting CS on [0,1] rejects mu <= 0.5.

    Uses a Waudby-Smith & Ramdas (2024) predictable plug-in betting martingale on each prompt's
    rollout sequence independently (ignoring between-prompt variation).
    """
    n_prompts, n_rollouts = unit_scores_01.shape
    mu0 = 0.5
    threshold = 1.0 / alpha
    rejected = np.zeros(n_prompts, dtype=bool)
    for i in range(n_prompts):
        x = unit_scores_01[i]
        wealth = 1.0
        sum_x = 0.5
        sum_sq = 0.25
        for t in range(n_rollouts):
            # Predictable mean and variance estimates from history
            n_hist = t + 1.0
            mu_hat = sum_x / n_hist
            var_hat = max(sum_sq / n_hist - mu_hat * mu_hat, 1e-4)
            # Kelly-style predictable bet clipped safely into [0, 0.75 / mu0]
            lam = np.clip((mu_hat - mu0) / (var_hat + (mu_hat - mu0) ** 2 + 1e-6), 0.0, 1.5)
            factor = 1.0 + lam * (x[t] - mu0)
            wealth *= max(factor, 1e-12)
            if wealth >= threshold:
                rejected[i] = True
                break
            sum_x += x[t]
            sum_sq += x[t] * x[t]
    return rejected


def gen_correlated_rollouts_aime(
    seed: int = 42,
    n_prompts: int = 200,
    n_rollouts: int = 32,
    rho: float = 0.48,
    theta: float = -0.15,
) -> TraceDict:
    """Reproduce the Agbeyegbe & Agbeyegbe (arXiv:2611.26592, Nov 2026) AIME regime.

    Models paired multi-rollout evaluation across ``n_prompts`` AIME problems with
    ``n_rollouts`` repeated attempts per problem for two models (A vs. B):
        D_{i,r} = A_{i,r} - B_{i,r} in {-1, 0, +1}
        X_{i,r} = (D_{i,r} + 1) / 2 in {0.0, 0.5, 1.0}

    Prompt-level solve probabilities ``(p_A, p_B)`` follow a calibrated bivariate
    logit-normal distribution solved via Gauss-Hermite quadrature so that:
    - True population mean ``E[D_{i,r}] = theta`` (default ``-0.15 < 0``, so population
      null ``H_0: theta <= 0`` is TRUE).
    - Exact within-prompt intraclass correlation ``ICC(D) = rho`` (default ``0.48``).
    - Single-unit (per-prompt) certificates falsely fire on ~20--25% of prompts due to
      prompt-level heterogeneity, whereas cluster-level inference controls Type-I error.
    """
    if n_prompts <= 1 or n_rollouts <= 1:
        raise ValueError("n_prompts and n_rollouts must be > 1")
    if not (0.01 <= rho <= 0.92):
        raise ValueError(f"rho must lie in [0.01, 0.92], got {rho}")
    if not (-0.85 <= theta <= 0.85):
        raise ValueError(f"theta must lie in [-0.85, 0.85], got {theta}")

    # 21-point Gauss-Hermite quadrature for exact population moments of logit-normal (p_A, p_B)
    nodes_1d, weights_1d = np.polynomial.hermite.hermgauss(21)
    z_pts = np.sqrt(2.0) * nodes_1d
    w_pts = weights_1d / np.sqrt(np.pi)
    zz, ww = np.meshgrid(z_pts, z_pts, indexing="ij")
    weights_2d = np.outer(w_pts, w_pts)

    # Shared prompt difficulty scale s_0; model-advantage heterogeneity scale s_d solved for rho
    s_0 = 1.65
    alpha_mid = 0.0

    def _solve_alpha_diff(s_d: float) -> tuple[float, float]:
        """For given s_d, find shift d_a so E[p_A - p_B] == theta, and return (d_a, icc)."""
        def _mean_diff(d_a: float) -> float:
            pa = expit(alpha_mid + d_a + s_0 * zz + s_d * ww)
            pb = expit(alpha_mid - d_a + s_0 * zz - s_d * ww)
            return float(np.sum(weights_2d * (pa - pb)))

        d_a = float(brentq(lambda d: _mean_diff(d) - theta, -10.0, 10.0, xtol=1e-6))
        pa = expit(alpha_mid + d_a + s_0 * zz + s_d * ww)
        pb = expit(alpha_mid - d_a + s_0 * zz - s_d * ww)
        mu = pa - pb
        mean_mu = float(np.sum(weights_2d * mu))
        var_between = float(np.sum(weights_2d * (mu - mean_mu) ** 2))
        var_within = float(np.sum(weights_2d * (pa * (1.0 - pa) + pb * (1.0 - pb))))
        icc = var_between / max(var_between + var_within, 1e-12)
        return d_a, icc

    s_d_star = float(brentq(lambda sd: _solve_alpha_diff(sd)[1] - rho, 0.05, 8.0, xtol=1e-5))
    d_a_star, theo_icc = _solve_alpha_diff(s_d_star)

    rng = np.random.default_rng(seed)
    z_prompt = rng.standard_normal(n_prompts)
    w_prompt = rng.standard_normal(n_prompts)

    p_a = expit(alpha_mid + d_a_star + s_0 * z_prompt + s_d_star * w_prompt)
    p_b = expit(alpha_mid - d_a_star + s_0 * z_prompt - s_d_star * w_prompt)
    prompt_means_true = p_a - p_b

    pass_a = (rng.random((n_prompts, n_rollouts)) < p_a[:, None]).astype(np.int32)
    pass_b = (rng.random((n_prompts, n_rollouts)) < p_b[:, None]).astype(np.int32)
    diffs = (pass_a - pass_b).astype(np.float64)
    unit_scores = 0.5 * (diffs + 1.0)

    # Diagnostics
    emp_mean = float(diffs.mean())
    emp_icc = _anova_icc(diffs)
    # Single-prompt one-sided t-test / z-test at alpha=0.05 (without divide-by-zero warning)
    p_means = diffs.mean(axis=1)
    p_stds = diffs.std(axis=1, ddof=1)
    se = p_stds / np.sqrt(n_rollouts)
    z_stats = np.where(p_means > 0.0, np.inf, -np.inf)
    np.divide(p_means, se, out=z_stats, where=se > 0.0)
    single_unit_z_fire = float(np.mean(z_stats > 1.6448536269514722))
    single_unit_cs_fire = float(np.mean(_single_unit_eb_cs_rejects(unit_scores, alpha=0.05)))

    empirical_stats = {
        "mean_diff": emp_mean,
        "true_theta": float(theta),
        "theoretical_icc": float(theo_icc),
        "empirical_icc": emp_icc,
        "frac_prompts_true_positive": float(np.mean(prompt_means_true > 0.0)),
        "single_unit_z_false_fire_rate": single_unit_z_fire,
        "single_unit_cs_false_fire_rate": single_unit_cs_fire,
    }

    return TraceDict(
        diffs=diffs,
        unit_scores=unit_scores,
        pass_a=pass_a,
        pass_b=pass_b,
        p_a=p_a,
        p_b=p_b,
        prompt_means_true=prompt_means_true,
        theta_true=float(theta),
        rho_true=float(rho),
        n_prompts=int(n_prompts),
        n_rollouts=int(n_rollouts),
        seed=int(seed),
        empirical_stats=empirical_stats,
    )


# ---------------------------------------------------------------------------
# 5. Active & Adversarial PPI Streams
# ---------------------------------------------------------------------------


def gen_active_ppi_stream(
    seed: int = 42,
    n_items: int = 5000,
    judge_corr: float = 0.85,
    judge_bias: float = 0.12,
) -> TraceDict:
    """Generate a heteroskedastic LLM-judge stream for Active Prediction-Powered Inference.

    Ground-truth labels ``y_true in {0, 1}`` have population mean ~0.45. The cheap LLM judge
    outputs ``f_judge in [0, 1]`` with systematic upward leniency bias ``E[f_judge] - E[y_true]
    approx judge_bias`` and correlation ``Corr(y_true, f_judge) approx judge_corr``. Judge error
    variance concentrates on borderline items where observable ``uncertainty`` is high, enabling
    active labeling to reduce variance relative to uniform PPI.
    """
    if n_items <= 10:
        raise ValueError("n_items must be > 10")
    if not (0.0 <= judge_corr <= 0.96):
        raise ValueError(f"judge_corr must lie in [0.0, 0.96], got {judge_corr}")
    if not (-0.40 <= judge_bias <= 0.45):
        raise ValueError(f"judge_bias must lie in [-0.40, 0.45], got {judge_bias}")

    rng = np.random.default_rng(seed)

    # Latent score S_t; steep slope yields clear easy/hard items + borderline uncertain band
    s = rng.standard_normal(n_items)
    # Shift c so E[sigmoid(slope * (S - c))] ~ 0.45
    slope = 7.0
    c_shift = 0.13
    p_true = expit(slope * (s - c_shift))
    y_true = (rng.random(n_items) < p_true).astype(np.float64)

    # Heteroskedastic judge noise: small on clear items, larger on borderline items
    borderline = 4.0 * p_true * (1.0 - p_true)  # in [0, 1], peaks at p_true = 0.5
    z_noise = rng.standard_normal(n_items)
    noise_weight = 0.35 + 1.65 * borderline

    # Calibrate noise scale sigma and logit bias b so Corr(y, f) == judge_corr and bias == judge_bias
    # Construct f_judge = clip(a_scale * p_true + b_shift + sigma * noise_weight * z_noise, 0, 1)
    # Use a_scale = 0.82 so there is headroom in [0, 1] for upward leniency bias.
    a_scale = 0.84 if judge_bias >= 0 else 0.84

    def _make_f(sigma: float) -> np.ndarray:
        base = a_scale * p_true + sigma * noise_weight * z_noise
        target_mean = float(np.clip(y_true.mean() + judge_bias, 0.02, 0.98))
        b_star = float(
            brentq(lambda b: float(np.clip(base + b, 0.0, 1.0).mean()) - target_mean, -2.0, 2.0, xtol=1e-6)
        )
        return np.clip(base + b_star, 0.0, 1.0)

    def _corr_at_sigma(sigma: float) -> float:
        f = _make_f(sigma)
        return float(np.corrcoef(y_true, f)[0, 1])

    max_corr = _corr_at_sigma(0.0)
    min_corr = _corr_at_sigma(2.5)
    if judge_corr >= max_corr:
        sigma_star = 0.0
    elif judge_corr <= min_corr:
        sigma_star = 2.5
    else:
        sigma_star = float(brentq(lambda sg: _corr_at_sigma(sg) - judge_corr, 0.0, 2.5, xtol=1e-5))

    f_judge = _make_f(sigma_star)

    # Observable pre-query uncertainty score (known from judge score/entropy without seeing y_true)
    # Strongly tracks residual variance |y_true - f_judge|
    judge_margin_unc = 4.0 * f_judge * (1.0 - f_judge)
    unc_noise = rng.uniform(0.0, 0.05, size=n_items)
    uncertainty = np.clip(0.55 * borderline + 0.40 * judge_margin_unc + unc_noise, 0.02, 1.0)

    emp_bias = float(f_judge.mean() - y_true.mean())
    emp_corr = float(np.corrcoef(y_true, f_judge)[0, 1])
    rectifier = y_true - f_judge
    empirical_stats = {
        "mean_y": float(y_true.mean()),
        "mean_f": float(f_judge.mean()),
        "bias": emp_bias,
        "corr": emp_corr,
        "y_std": float(y_true.std(ddof=1)),
        "rectifier_std": float(rectifier.std(ddof=1)),
        "uncertainty_abs_err_corr": float(np.corrcoef(uncertainty, np.abs(rectifier))[0, 1]),
    }

    return TraceDict(
        y_true=y_true,
        f_judge=f_judge,
        uncertainty=uncertainty,
        true_mean=float(y_true.mean()),
        target_corr=float(judge_corr),
        target_bias=float(judge_bias),
        n_items=int(n_items),
        seed=int(seed),
        empirical_stats=empirical_stats,
    )


def gen_adversarial_ppi_stream(
    seed: int = 42,
    n_items: int = 5000,
    judge_bias: float = 0.35,
    inverted_corr: float = -0.75,
) -> TraceDict:
    """Generate an adversarially biased and inverted LLM-judge PPI stream.

    The judge outputs scores strongly anti-correlated with ground truth (``Corr approx -0.75``)
    with large positive bias (``E[f_judge] - E[y_true] approx +0.35``) and miscalibrated
    (overconfident-when-wrong) uncertainty. Tests that PPI++/power-tuned anytime-valid PPI
    maintains valid coverage and avoids catastrophic variance inflation.
    """
    if n_items <= 10:
        raise ValueError("n_items must be > 10")
    if not (-0.95 <= inverted_corr <= 0.0):
        raise ValueError(f"inverted_corr must lie in [-0.95, 0.0], got {inverted_corr}")

    rng = np.random.default_rng(seed)
    s = rng.standard_normal(n_items)
    p_true = expit(6.0 * (s - 0.38))  # E[y_true] ~ 0.36
    y_true = (rng.random(n_items) < p_true).astype(np.float64)

    z_noise = rng.standard_normal(n_items)
    a_inv = 0.85

    def _make_f(sigma: float) -> np.ndarray:
        base = a_inv * (1.0 - p_true) + sigma * z_noise
        target_mean = float(np.clip(y_true.mean() + judge_bias, 0.05, 0.95))
        b_star = float(
            brentq(lambda b: float(np.clip(base + b, 0.0, 1.0).mean()) - target_mean, -2.0, 2.0, xtol=1e-6)
        )
        return np.clip(base + b_star, 0.0, 1.0)

    def _corr_at_sigma(sigma: float) -> float:
        return float(np.corrcoef(y_true, _make_f(sigma))[0, 1])

    most_neg = _corr_at_sigma(0.0)
    near_zero = _corr_at_sigma(3.0)
    if inverted_corr <= most_neg:
        sigma_star = 0.0
    elif inverted_corr >= near_zero:
        sigma_star = 3.0
    else:
        sigma_star = float(brentq(lambda sg: _corr_at_sigma(sg) - inverted_corr, 0.0, 3.0, xtol=1e-5))

    f_judge = _make_f(sigma_star)
    # Overconfident when wrong: low reported uncertainty on inverted predictions
    uncertainty = np.clip(4.0 * f_judge * (1.0 - f_judge), 0.05, 1.0)

    rectifier = y_true - f_judge
    empirical_stats = {
        "mean_y": float(y_true.mean()),
        "mean_f": float(f_judge.mean()),
        "bias": float(f_judge.mean() - y_true.mean()),
        "corr": float(np.corrcoef(y_true, f_judge)[0, 1]),
        "y_std": float(y_true.std(ddof=1)),
        "rectifier_std": float(rectifier.std(ddof=1)),
    }

    return TraceDict(
        y_true=y_true,
        f_judge=f_judge,
        uncertainty=uncertainty,
        true_mean=float(y_true.mean()),
        target_corr=float(inverted_corr),
        target_bias=float(judge_bias),
        n_items=int(n_items),
        seed=int(seed),
        empirical_stats=empirical_stats,
    )


# ---------------------------------------------------------------------------
# 6. Multi-Arm Agent Race with Shared Prompt Difficulty
# ---------------------------------------------------------------------------


def gen_agent_race(
    seed: int = 42,
    n_arms: int = 8,
    means: Sequence[float] | None = None,
    n_steps: int = 2000,
    shared_corr: float = 0.75,
) -> TraceDict:
    """Generate paired and unpaired binary evaluation streams for multi-arm agent racing.

    Uses an exact probit Gaussian copula with shared latent prompt difficulty ``D_t ~ N(0, 1)``:
        Z_{t,a} = sqrt(shared_corr) * D_t + sqrt(1 - shared_corr) * eps_{t,a}
        X_{t,a} = 1[Z_{t,a} <= Phi^{-1}(means[a])]

    Marginally ``P(X_{t,a} = 1) == means[a]`` exactly for both ``rewards`` (paired across
    arms on each prompt ``t``) and ``unpaired_rewards`` (independent prompts per arm),
    while paired differences enjoy substantial variance reduction.
    """
    if means is None:
        if n_arms == 8:
            arm_means = np.array([0.82, 0.74, 0.68, 0.62, 0.55, 0.50, 0.45, 0.35], dtype=np.float64)
        else:
            arm_means = np.linspace(0.82, 0.35, n_arms, dtype=np.float64)
    else:
        arm_means = np.asarray(means, dtype=np.float64)
        if arm_means.ndim != 1 or arm_means.size != n_arms:
            raise ValueError(f"means must be a 1D sequence of length n_arms={n_arms}")
    if np.any((arm_means <= 0.0) | (arm_means >= 1.0)):
        raise ValueError("All arm means must lie strictly in (0, 1)")
    if not (0.0 <= shared_corr < 1.0):
        raise ValueError(f"shared_corr must lie in [0.0, 1.0), got {shared_corr}")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")

    rng = np.random.default_rng(seed)
    thresholds = norm.ppf(arm_means)

    # Paired rewards: shared prompt difficulty across all arms on prompt t
    prompt_difficulty = rng.standard_normal(n_steps)
    eps_paired = rng.standard_normal((n_steps, n_arms))
    z_paired = np.sqrt(shared_corr) * prompt_difficulty[:, None] + np.sqrt(1.0 - shared_corr) * eps_paired
    rewards = (z_paired <= thresholds[None, :]).astype(np.float64)

    # Matched unpaired rewards: independent prompts per arm with identical marginal means
    z_unpaired = rng.standard_normal((n_steps, n_arms))
    unpaired_rewards = (z_unpaired <= thresholds[None, :]).astype(np.float64)

    order = np.argsort(-arm_means)
    best_arm = int(order[0])
    second_arm = int(order[1]) if n_arms >= 2 else best_arm
    top_gap = float(arm_means[best_arm] - arm_means[second_arm])

    paired_diff_var = float(np.var(rewards[:, best_arm] - rewards[:, second_arm], ddof=1))
    unpaired_diff_var = float(np.var(unpaired_rewards[:, best_arm] - unpaired_rewards[:, second_arm], ddof=1))

    empirical_stats = {
        "arm_means_paired": rewards.mean(axis=0).tolist(),
        "arm_means_unpaired": unpaired_rewards.mean(axis=0).tolist(),
        "top2_paired_corr": float(np.corrcoef(rewards[:, best_arm], rewards[:, second_arm])[0, 1]),
        "paired_diff_var_top2": paired_diff_var,
        "unpaired_diff_var_top2": unpaired_diff_var,
        "variance_reduction_ratio": float(unpaired_diff_var / max(paired_diff_var, 1e-12)),
    }

    return TraceDict(
        rewards=rewards,
        unpaired_rewards=unpaired_rewards,
        prompt_difficulty=prompt_difficulty,
        means=arm_means,
        n_arms=int(n_arms),
        n_steps=int(n_steps),
        shared_corr=float(shared_corr),
        best_arm=best_arm,
        top_gap=top_gap,
        seed=int(seed),
        empirical_stats=empirical_stats,
    )


# ---------------------------------------------------------------------------
# 7. Poisoned Swarm (Mid-Stream Silent Agent Degradation)
# ---------------------------------------------------------------------------


def gen_poisoned_swarm(
    seed: int = 42,
    n_agents: int = 8,
    poison_step: int = 200,
    n_steps: int = 1000,
    n_poisoned: int = 2,
) -> TraceDict:
    """Generate a multi-agent consensus stream where ``n_poisoned`` agents degrade mid-stream.

    For ``t < poison_step``, all ``n_agents`` agents answer 4-class tasks accurately (~0.78--0.84).
    Starting at ``t == poison_step``, the last ``n_poisoned`` agents (default 2 of 8) silently
    degrade to ~0.22 accuracy and coordinate on a shared adversarial distractor answer.
    """
    if n_agents < 2:
        raise ValueError("n_agents must be >= 2")
    if not (0 < poison_step < n_steps):
        raise ValueError("poison_step must satisfy 0 < poison_step < n_steps")
    if not (1 <= n_poisoned < n_agents):
        raise ValueError("n_poisoned must satisfy 1 <= n_poisoned < n_agents")

    rng = np.random.default_rng(seed)
    n_classes = 4
    true_labels = rng.integers(0, n_classes, size=n_steps, dtype=np.int32)
    adv_offset = rng.integers(1, n_classes, size=n_steps, dtype=np.int32)
    adv_target = (true_labels + adv_offset) % n_classes

    # Per-agent pre-change accuracies (~0.78 - 0.84) and post-change accuracies
    pre_acc = np.linspace(0.84, 0.78, n_agents, dtype=np.float64)
    post_acc = pre_acc.copy()
    poisoned_agents = list(range(n_agents - n_poisoned, n_agents))
    clean_agents = list(range(0, n_agents - n_poisoned))
    post_acc[poisoned_agents] = 0.22

    # Mild shared task difficulty (rho_task = 0.20) for realistic agent behavior
    rho_task = 0.20
    task_diff = rng.standard_normal(n_steps)
    eps = rng.standard_normal((n_steps, n_agents))
    z = np.sqrt(rho_task) * task_diff[:, None] + np.sqrt(1.0 - rho_task) * eps

    acc_schedule = np.broadcast_to(pre_acc, (n_steps, n_agents)).copy()
    acc_schedule[poison_step:, :] = post_acc[None, :]
    thresholds = norm.ppf(acc_schedule)
    correct = (z <= thresholds).astype(np.float64)

    # Generate multi-class votes: clean mistakes are diffuse; poisoned mistakes coordinate on adv_target
    rand_wrong_offset = rng.integers(1, n_classes, size=(n_steps, n_agents), dtype=np.int32)
    diffuse_wrong = (true_labels[:, None] + rand_wrong_offset) % n_classes
    votes = np.where(correct == 1.0, true_labels[:, None], diffuse_wrong).astype(np.int32)

    # Coordinated adversarial wrong votes for poisoned agents at t >= poison_step
    for idx in poisoned_agents:
        is_wrong_post = correct[poison_step:, idx] == 0.0
        coord = rng.random(n_steps - poison_step) < 0.90
        use_adv = is_wrong_post & coord
        votes[poison_step:, idx] = np.where(use_adv, adv_target[poison_step:], votes[poison_step:, idx])

    pre_maj_acc = _plurality_accuracy_expected_tie(votes[:poison_step], true_labels[:poison_step], n_classes)
    post_maj_acc = _plurality_accuracy_expected_tie(votes[poison_step:], true_labels[poison_step:], n_classes)
    post_clean_maj_acc = _plurality_accuracy_expected_tie(
        votes[poison_step:, clean_agents], true_labels[poison_step:], n_classes
    )

    empirical_stats = {
        "clean_pre_acc": float(correct[:poison_step, clean_agents].mean()),
        "clean_post_acc": float(correct[poison_step:, clean_agents].mean()),
        "poisoned_pre_acc": float(correct[:poison_step, poisoned_agents].mean()),
        "poisoned_post_acc": float(correct[poison_step:, poisoned_agents].mean()),
        "pre_majority_acc": pre_maj_acc,
        "post_majority_acc_all": post_maj_acc,
        "post_majority_acc_clean_only": post_clean_maj_acc,
    }

    return TraceDict(
        correct=correct,
        votes=votes,
        true_labels=true_labels,
        poisoned_agents=poisoned_agents,
        clean_agents=clean_agents,
        poison_step=int(poison_step),
        n_agents=int(n_agents),
        n_steps=int(n_steps),
        n_classes=n_classes,
        pre_acc_true=pre_acc,
        post_acc_true=post_acc,
        seed=int(seed),
        empirical_stats=empirical_stats,
    )


# ---------------------------------------------------------------------------
# 8. Multi-Turn Agent Abort Cascade
# ---------------------------------------------------------------------------


def _roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Exact rank-based ROC-AUC (no external ML dependency required)."""
    y = y_true.astype(np.int32)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    # Average ranks for ties
    sorted_scores = scores[order]
    i = 0
    n = scores.size
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    sum_pos_ranks = float(ranks[y == 1].sum())
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def gen_abort_cascade(
    seed: int = 42,
    n_episodes: int = 2000,
    n_stages: int = 4,
) -> TraceDict:
    """Generate multi-turn agent trajectories with causal early-round failure signals.

    Each episode evolves a latent health trajectory ``H_{i, s}`` across ``n_stages`` turns
    with realistic early-turn catastrophic drops (e.g., tool loops, syntax collapse),
    progressively sharper verifier signals ``signals[i, s] in (0, 1)``, growing per-stage
    context token costs, and a strictly causal final outcome ``success[i] in {0, 1}``.
    """
    if n_episodes <= 10 or n_stages < 2:
        raise ValueError("n_episodes must be > 10 and n_stages >= 2")

    rng = np.random.default_rng(seed)

    # Latent episode aptitude + Markov progress state with early catastrophic failures
    aptitude = rng.normal(loc=-0.10, scale=0.85, size=n_episodes)
    health = np.empty((n_episodes, n_stages), dtype=np.float64)

    # Stage 0: initial state plus ~22% early catastrophic derailments
    catastrophic_0 = rng.random(n_episodes) < 0.22
    health[:, 0] = aptitude + rng.normal(0.0, 0.55, size=n_episodes) - np.where(catastrophic_0, 2.2, 0.0)

    for s in range(1, n_stages):
        # Subsequent stages accumulate progress or additional derailments (~14% per stage)
        derail = rng.random(n_episodes) < 0.14
        drift = 0.35 * aptitude + rng.normal(0.0, 0.65, size=n_episodes) - np.where(derail, 1.9, 0.0)
        health[:, s] = health[:, s - 1] + drift

    # Causal final success determined strictly by final latent trajectory state
    final_prob = expit(1.45 * health[:, -1] + 0.15)
    success = (rng.random(n_episodes) < final_prob).astype(np.int32)

    # Stage verifier / critic signals: noisy observations of current health, sharpening over turns
    obs_noise_scales = np.linspace(0.95, 0.35, n_stages, dtype=np.float64)
    signals = np.empty((n_episodes, n_stages), dtype=np.float64)
    for s in range(n_stages):
        obs = health[:, s] + rng.normal(0.0, obs_noise_scales[s], size=n_episodes)
        signals[:, s] = expit(0.85 * obs)
    failure_signals = 1.0 - signals

    # Growing per-stage compute costs (reflecting expanding multi-turn context window)
    base_stage_costs = np.array([1.0 * (1.48**s) for s in range(n_stages)], dtype=np.float64)
    cost_jitter = rng.lognormal(mean=0.0, sigma=0.12, size=(n_episodes, n_stages))
    episode_stage_costs = base_stage_costs[None, :] * cost_jitter
    stage_costs = episode_stage_costs.mean(axis=0)

    stage_aucs = [_roc_auc(success, signals[:, s]) for s in range(n_stages)]
    empirical_stats = {
        "success_rate": float(success.mean()),
        "stage_aucs": [float(a) for a in stage_aucs],
        "stage_mean_costs": stage_costs.tolist(),
        "mean_total_episode_cost": float(episode_stage_costs.sum(axis=1).mean()),
    }

    return TraceDict(
        signals=signals,
        failure_signals=failure_signals,
        success=success,
        stage_costs=stage_costs,
        episode_stage_costs=episode_stage_costs,
        n_episodes=int(n_episodes),
        n_stages=int(n_stages),
        seed=int(seed),
        empirical_stats=empirical_stats,
    )


# ---------------------------------------------------------------------------
# Self-Verification CLI
# ---------------------------------------------------------------------------


def verify_all(seed: int = 42) -> dict[str, Any]:
    """Run all 8 generators and return a summary dictionary of calibrated empirical stats."""
    ascd = gen_ascd_benchmark(seed=seed, delta_gap=0.10, n_trials=500, max_n=500)
    open_ended = gen_open_ended_swarm(seed=seed, k_true=200, zipf_s=1.25, top_gap=0.12, n_trials=500, max_n=256)
    panel = gen_correlated_judge_panel(seed=seed, n_items=1000, k_judges=9, target_phi=0.391)
    aime = gen_correlated_rollouts_aime(seed=seed, n_prompts=200, n_rollouts=32, rho=0.48, theta=-0.15)
    ppi_active = gen_active_ppi_stream(seed=seed, n_items=5000, judge_corr=0.85, judge_bias=0.12)
    ppi_adv = gen_adversarial_ppi_stream(seed=seed, n_items=5000)
    race = gen_agent_race(seed=seed)
    poison = gen_poisoned_swarm(seed=seed, n_agents=8, poison_step=200, n_steps=1000)
    cascade = gen_abort_cascade(seed=seed, n_episodes=2000, n_stages=4)

    return {
        "ascd": ascd.empirical_stats,
        "open_ended_swarm": open_ended.empirical_stats,
        "correlated_judge_panel": panel.empirical_stats,
        "correlated_rollouts_aime": aime.empirical_stats,
        "active_ppi": ppi_active.empirical_stats,
        "adversarial_ppi": ppi_adv.empirical_stats,
        "agent_race": race.empirical_stats,
        "poisoned_swarm": poison.empirical_stats,
        "abort_cascade": cascade.empirical_stats,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(verify_all(seed=42), indent=2))
