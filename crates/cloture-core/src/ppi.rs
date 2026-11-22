//! Anytime-valid online Active Prediction-Powered Inference (`ActivePPI`).
//!
//! Combines 100% cheap proxy / LLM-judge scores `f_t in [0, 1]` with sparse, adaptively-sampled
//! ground-truth / human labels `y_t in [0, 1]` to construct an **exact finite-sample anytime-valid**
//! confidence sequence for `mu = E[Y_t]`:
//!
//! - Uses the strictly predictable power-tuned AIPW / PPI++ score
//!   `z_t = lam_t * f_t + (R_t / pi_t) * (y_t - lam_t * f_t)`
//!   (Angelopoulos et al., Science 2023; Zrnic & Candes 2024), which satisfies
//!   `E[z_t | F_{t-1}] = mu` **exactly** even when the judge `f_t` is arbitrarily biased, noisy,
//!   or adversarially poisoned!
//! - Automatically drives `lam_t -> 0` for useless or inverted judges and `lam_t -> 1` for accurate
//!   judges using Horvitz-Thompson weighted past statistics (strictly `F_{t-1}`-measurable).
//! - Exploits an exact predictable plug-in Empirical-Bernstein supermartingale via Fan's
//!   inequality using the strictly `F_{t-1}`-measurable deviation radius
//!   `S_t = max(mu_prev - z_min_t, z_max_t - mu_prev) <= 1 / min_audit_prob`, achieving
//!   O(1)-time, zero-allocation updates with zero asymptotic approximations.

use crate::betting::{log_mean_exp_2, psi_e};
use crate::math::CompSum;

/// Default burn-in audited count before activating non-zero `lambda_hat`.
const DEFAULT_BURN_IN: u64 = 10;

/// Default sub-exponential betting cap `c in (0, 1)`.
/// Combined with exact Fan-inequality damping `lam = q / (1 + q)`, `0.75`
/// sharpens finite-sample width when the proxy judge has low residual variance
/// without inflating `psi_E(lam)` during high-variance steps.
const DEFAULT_CAP: f64 = 0.75;

/// Snapshot of [`ActivePPI`] estimation and confidence sequence state.
#[derive(Clone, Copy, Debug)]
pub struct PpiStatus {
    /// Total tasks observed (with or without ground-truth audit).
    pub n_total: u64,
    /// Number of tasks audited with ground-truth labels.
    pub n_audited: u64,
    /// Realized empirical audit rate `n_audited / n_total`.
    pub audit_rate: f64,
    /// Unbiased PPI++ point estimate of `E[Y]`, clamped to `[0, 1]`.
    pub mean_estimate: f64,
    /// Anytime-valid lower confidence bound in `[0, 1]`.
    pub lower: f64,
    /// Anytime-valid upper confidence bound in `[0, 1]`.
    pub upper: f64,
    /// Confidence sequence width `upper - lower`.
    pub width: f64,
    /// Current strictly predictable power-tuning weight `lambda_hat in [0, 1]`.
    pub lambda_hat: f64,
    /// Ratio of naive Horvitz-Thompson (`lambda = 0`) variance to PPI variance.
    pub variance_reduction_factor: f64,
    /// Effective sample size multiplier relative to an audited-only estimator.
    pub effective_sample_multiplier: f64,
}

/// Anytime-valid online Active Prediction-Powered Inference estimator.
#[derive(Clone, Debug)]
pub struct ActivePPI {
    alpha: f64,
    log_two_over_alpha: f64,
    target_audit_rate: f64,
    min_audit_prob: f64,
    max_inv_pi: f64,
    burn_in: u64,
    c: f64,

    n_total: u64,
    n_audited: u64,

    // Running statistics of PPI score z_t and naive HT baseline z0_t.
    sum_z: CompSum,
    sum_sq_z: CompSum,
    sum_sq_dev_z: CompSum,
    mu_prev_z: f64,
    sum_z0: CompSum,
    sum_sq_z0: CompSum,
    sum_audit_prob: CompSum,

    // Horvitz-Thompson weighted sufficient statistics from audited samples (strictly F_{t-1}).
    ht_w: CompSum,
    ht_wf: CompSum,
    ht_wy: CompSum,
    ht_wff: CompSum,
    ht_wfy: CompSum,
    ht_wyy: CompSum,

    // Exact Empirical-Bernstein supermartingale accumulators with symmetric predictable scale S_t.
    sum_beta_z: CompSum,
    sum_beta: CompSum,
    sum_v_psi: CompSum,

    lower: f64,
    upper: f64,
}

impl ActivePPI {
    /// Creates an [`ActivePPI`] estimator with miscoverage level `alpha in (0, 1)`,
    /// target average audit rate `target_audit_rate in (0, 1]`, and strict positivity floor
    /// `min_audit_prob in (0, target_audit_rate]`.
    pub fn new(
        alpha: f64,
        target_audit_rate: f64,
        min_audit_prob: f64,
    ) -> Result<Self, &'static str> {
        if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
            return Err("alpha must be finite and strictly in (0, 1)");
        }
        if !target_audit_rate.is_finite()
            || target_audit_rate <= 0.0
            || target_audit_rate > 1.0
        {
            return Err("target_audit_rate must be finite and in (0, 1]");
        }
        if !min_audit_prob.is_finite()
            || min_audit_prob <= 0.0
            || min_audit_prob > target_audit_rate
        {
            return Err("min_audit_prob must be finite and in (0, target_audit_rate]");
        }

        Ok(Self {
            alpha,
            log_two_over_alpha: (2.0 / alpha).ln(),
            target_audit_rate,
            min_audit_prob,
            max_inv_pi: 1.0 / min_audit_prob,
            burn_in: DEFAULT_BURN_IN,
            c: DEFAULT_CAP,
            n_total: 0,
            n_audited: 0,
            sum_z: CompSum::default(),
            sum_sq_z: CompSum::default(),
            sum_sq_dev_z: CompSum::default(),
            mu_prev_z: 0.5,
            sum_z0: CompSum::default(),
            sum_sq_z0: CompSum::default(),
            sum_audit_prob: CompSum::default(),
            ht_w: CompSum::default(),
            ht_wf: CompSum::default(),
            ht_wy: CompSum::default(),
            ht_wff: CompSum::default(),
            ht_wfy: CompSum::default(),
            ht_wyy: CompSum::default(),
            sum_beta_z: CompSum::default(),
            sum_beta: CompSum::default(),
            sum_v_psi: CompSum::default(),
            lower: 0.0,
            upper: 1.0,
        })
    }

    /// Computes the strictly predictable power-tuning weight `lambda_hat in [0, 1]`
    /// from Horvitz-Thompson weighted past audited pairs `(f_s, y_s)` (`s < t`).
    ///
    /// Returns `0.0` during burn-in (`n_audited < 10`) and smoothly ramps over the next
    /// 10 audited samples, clamping negative or zero correlations to `0.0` so noisy or
    /// adversarially inverted judges never inflate variance.
    #[inline]
    pub fn lambda_hat(&self) -> f64 {
        if self.n_audited < self.burn_in {
            return 0.0;
        }
        let w_sum = self.ht_w.value();
        if w_sum <= 0.0 {
            return 0.0;
        }
        let mean_f = self.ht_wf.value() / w_sum;
        let mean_y = self.ht_wy.value() / w_sum;
        let var_f = (self.ht_wff.value() / w_sum - mean_f * mean_f).max(0.0);
        let cov_fy = self.ht_wfy.value() / w_sum - mean_f * mean_y;
        let raw = (cov_fy / (var_f + 1e-6)).clamp(0.0, 1.0);
        let ramp = (((self.n_audited - self.burn_in + 1) as f64) / 10.0).min(1.0);
        (raw * ramp).clamp(0.0, 1.0)
    }

    /// Suggests a strictly predictable adaptive audit probability `pi_t in [min_audit_prob, 1.0]`
    /// allocating higher audit budget to high-uncertainty items while tracking `target_audit_rate`.
    ///
    /// - `judge_pred`: proxy score in `[0, 1]`.
    /// - `uncertainty_hint`: optional uncertainty proxy in `[0, 1]` (e.g. judge ensemble
    ///   disagreement or entropy). If `None`, uses the Bernoulli std proxy `2 * sqrt(f * (1 - f))`.
    #[inline]
    pub fn suggest_audit_prob(&self, judge_pred: f64, uncertainty_hint: Option<f64>) -> f64 {
        let excess = (self.target_audit_rate - self.min_audit_prob).max(0.0);
        if excess == 0.0 {
            return self.min_audit_prob;
        }

        let f = if judge_pred.is_finite() {
            judge_pred.clamp(0.0, 1.0)
        } else {
            0.5
        };

        let u = match uncertainty_hint {
            Some(h) if h.is_finite() => h.clamp(0.0, 1.0),
            _ => 2.0 * (f * (1.0 - f)).sqrt(),
        };

        // Monotone Neyman-style uncertainty multiplier in [0.25, 1.75] (centered at 1.0 when u=0.5).
        let rel = 0.25 + 1.5 * u;

        // Closed-loop strictly F_{t-1}-measurable budget controller tracking target_audit_rate.
        let avg_pi = (self.target_audit_rate * 10.0 + self.sum_audit_prob.value())
            / (10.0 + self.n_total as f64);
        let budget_gain = (self.target_audit_rate / avg_pi.max(1e-6)).clamp(0.5, 2.0);

        (self.min_audit_prob + excess * rel * budget_gain).clamp(self.min_audit_prob, 1.0)
    }

    /// Ingests one task observation in O(1) time with zero heap allocations.
    ///
    /// - `judge_pred`: cheap proxy score in `[0, 1]`.
    /// - `audited`: whether ground-truth auditing was triggered (`R_t = 1`).
    /// - `gold_label`: ground-truth label `y_t in [0, 1]` (required when `audited == true`;
    ///   completely ignored when `audited == false`).
    /// - `audit_prob`: the exact sampling probability `pi_t` used to draw `audited`,
    ///   clamped to `[min_audit_prob, 1.0]`.
    ///
    /// Non-finite inputs (or `audited == true` with missing/non-finite `gold_label`) are safely
    /// ignored without mutating state.
    pub fn observe(
        &mut self,
        judge_pred: f64,
        audited: bool,
        gold_label: Option<f64>,
        audit_prob: f64,
    ) -> PpiStatus {
        if !judge_pred.is_finite() || !audit_prob.is_finite() {
            return self.status();
        }
        let y_opt = if audited {
            match gold_label {
                Some(y) if y.is_finite() => Some(y.clamp(0.0, 1.0)),
                _ => return self.status(),
            }
        } else {
            None
        };

        let f = judge_pred.clamp(0.0, 1.0);
        let pi = audit_prob.clamp(self.min_audit_prob, 1.0);

        // Strictly predictable power-tuning weight and affine debiasing intercept from steps 1..t-1.
        let lam = self.lambda_hat();
        let w_sum = self.ht_w.value();
        let (m_y, m_f) = if self.n_audited >= 2 && w_sum > 0.0 {
            (
                (self.ht_wy.value() / w_sum).clamp(0.0, 1.0),
                (self.ht_wf.value() / w_sum).clamp(0.0, 1.0),
            )
        } else {
            (self.mu_prev_z, 0.5)
        };

        // Monotone bounded affine proxy y_hat(f) in [a_t, b_t] subseteq [0, 1].
        // Debiases systematically optimistic/pessimistic judges and eliminates the
        // ((1 - pi)/pi) * mu^2 Horvitz-Thompson variance spike during burn-in or noisy regimes.
        let y_hat = (m_y + lam * (f - m_f)).clamp(0.0, 1.0);
        let a_t = (m_y - lam * m_f).clamp(0.0, 1.0);
        let b_t = (m_y + lam * (1.0 - m_f)).clamp(0.0, 1.0);

        // Exact unbiased PPI++ / AIPW increment: E[z_t | F_{t-1}] = E[Y_t].
        let (z_t, z0_t) = match y_opt {
            Some(y) => (y_hat + (y - y_hat) / pi, y / pi),
            None => (y_hat, 0.0),
        };

        // Exact strictly F_{t-1}-measurable range of z_t for all f,y in [0,1], pi in [1/M, 1]:
        //   z_t >= -b_t * (M - 1)   and   z_t <= M - a_t * (M - 1)
        // Notice: when lam = 0 and m_y = 0.5, scale halves from M to 0.5 * M!
        let m_minus_1 = self.max_inv_pi - 1.0;
        let z_min_t = -b_t * m_minus_1;
        let z_max_t = self.max_inv_pi - a_t * m_minus_1;
        let mu_prev = self.mu_prev_z;
        let scale = (mu_prev - z_min_t).max(z_max_t - mu_prev).max(0.5);

        // Predictable variance and Fan-damped step size on the original z_t scale.
        let t_f = (self.n_total + 1) as f64;
        let v_prior = 0.25 / self.target_audit_rate;
        let v_hat_prev = (v_prior + self.sum_sq_dev_z.value()) / t_f;
        let denom = v_hat_prev * t_f * (t_f + 1.0).ln();
        let lam_eb = if denom > 0.0 {
            let beta_raw = (2.0 * self.log_two_over_alpha / denom).sqrt();
            let q = beta_raw * scale;
            // Exact optimal solution to minimizing (A + B * lam^2 / (2(1-lam))) / lam is q / (1 + q).
            (q / (1.0 + q)).min(self.c)
        } else {
            self.c
        };

        // Exact Fan-inequality supermartingale update valid on both tails simultaneously.
        let dz_prev = z_t - mu_prev;
        let v_t = dz_prev * dz_prev;

        let beta = lam_eb / scale;
        let psi_term = v_t * psi_e(lam_eb) / (scale * scale);

        self.sum_beta_z.add(beta * z_t);
        self.sum_beta.add(beta);
        self.sum_v_psi.add(psi_term);

        // Running-intersected exact CS bounds on mu in [0, 1].
        let s_beta = self.sum_beta.value();
        if s_beta > 0.0 {
            let center = self.sum_beta_z.value() / s_beta;
            let margin = (self.log_two_over_alpha + self.sum_v_psi.value()) / s_beta;
            let cand_lo = (center - margin).clamp(0.0, 1.0);
            let cand_hi = (center + margin).clamp(0.0, 1.0);
            let new_lo = self.lower.max(cand_lo);
            let new_hi = self.upper.min(cand_hi);
            if new_lo <= new_hi {
                self.lower = new_lo;
                self.upper = new_hi;
            } else {
                // Only reachable on the <= alpha miscoverage event or non-stationary shift.
                let mid = (0.5 * (new_lo + new_hi)).clamp(0.0, 1.0);
                self.lower = mid;
                self.upper = mid;
            }
        }

        // Advance predictable state for step t+1.
        self.n_total += 1;
        self.sum_z.add(z_t);
        self.sum_sq_z.add(z_t * z_t);
        let mu_curr = ((0.5 + self.sum_z.value()) / ((self.n_total + 1) as f64))
            .clamp(0.0, 1.0);
        let dz_curr = z_t - mu_curr;
        self.sum_sq_dev_z.add(dz_curr * dz_curr);
        self.mu_prev_z = mu_curr;

        self.sum_z0.add(z0_t);
        self.sum_sq_z0.add(z0_t * z0_t);
        self.sum_audit_prob.add(pi);

        if let Some(y) = y_opt {
            self.n_audited += 1;
            let w = 1.0 / pi;
            self.ht_w.add(w);
            self.ht_wf.add(w * f);
            self.ht_wy.add(w * y);
            self.ht_wff.add(w * f * f);
            self.ht_wfy.add(w * f * y);
            self.ht_wyy.add(w * y * y);
        }

        self.status()
    }

    /// Exact two-sided Empirical-Bernstein log e-value at candidate mean `m in [0, 1]`.
    #[inline]
    pub fn log_e_value_at(&self, m: f64) -> f64 {
        if self.n_total == 0 || !m.is_finite() {
            return 0.0;
        }
        let m01 = m.clamp(0.0, 1.0);
        let sbz = self.sum_beta_z.value();
        let sb = self.sum_beta.value();
        let svp = self.sum_v_psi.value();
        let log_lo = sbz - m01 * sb - svp;
        let log_hi = m01 * sb - sbz - svp;
        log_mean_exp_2(log_lo, log_hi)
    }

    /// Exact two-sided Empirical-Bernstein e-value at candidate mean `m in [0, 1]`.
    #[inline]
    pub fn e_value_at(&self, m: f64) -> f64 {
        self.log_e_value_at(m).exp()
    }

    #[inline]
    pub fn lower(&self) -> f64 {
        self.lower
    }

    #[inline]
    pub fn upper(&self) -> f64 {
        self.upper
    }

    #[inline]
    pub fn mean_estimate(&self) -> f64 {
        if self.n_total == 0 {
            0.5
        } else {
            (self.sum_z.value() / (self.n_total as f64)).clamp(0.0, 1.0)
        }
    }

    #[inline]
    pub fn n_total(&self) -> u64 {
        self.n_total
    }

    #[inline]
    pub fn n_audited(&self) -> u64 {
        self.n_audited
    }

    #[inline]
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Computes diagnostics and current confidence sequence snapshot.
    pub fn status(&self) -> PpiStatus {
        let n_f = self.n_total as f64;
        let na_f = self.n_audited as f64;
        let audit_rate = if self.n_total == 0 { 0.0 } else { na_f / n_f };
        let mean_estimate = self.mean_estimate();
        let lam = self.lambda_hat();

        let (variance_reduction_factor, effective_sample_multiplier) =
            if self.n_total < 2 || self.n_audited < 2 {
                (1.0, 1.0)
            } else {
                let mean_z = self.sum_z.value() / n_f;
                let var_z = (self.sum_sq_z.value() / n_f - mean_z * mean_z).max(0.0);

                let mean_z0 = self.sum_z0.value() / n_f;
                let var_z0 = (self.sum_sq_z0.value() / n_f - mean_z0 * mean_z0).max(0.0);

                let vrf = if var_z > 1e-12 && var_z0 > 1e-12 {
                    (var_z0 / var_z).max(0.0)
                } else {
                    1.0
                };

                // HT-estimated Var(Y) / n_audited vs Var(z) / n_total.
                let w_sum = self.ht_w.value();
                let esm = if w_sum > 0.0 && var_z > 1e-12 {
                    let mean_y = self.ht_wy.value() / w_sum;
                    let var_y = (self.ht_wyy.value() / w_sum - mean_y * mean_y).max(0.0);
                    if var_y > 1e-12 {
                        ((var_y / na_f) / (var_z / n_f)).max(0.0)
                    } else {
                        1.0
                    }
                } else {
                    1.0
                };

                (vrf, esm)
            };

        PpiStatus {
            n_total: self.n_total,
            n_audited: self.n_audited,
            audit_rate,
            mean_estimate,
            lower: self.lower,
            upper: self.upper,
            width: (self.upper - self.lower).max(0.0),
            lambda_hat: lam,
            variance_reduction_factor,
            effective_sample_multiplier,
        }
    }

    /// Resets all accumulated state while preserving configuration.
    pub fn reset(&mut self) {
        self.n_total = 0;
        self.n_audited = 0;
        self.sum_z = CompSum::default();
        self.sum_sq_z = CompSum::default();
        self.sum_sq_dev_z = CompSum::default();
        self.mu_prev_z = 0.5;
        self.sum_z0 = CompSum::default();
        self.sum_sq_z0 = CompSum::default();
        self.sum_audit_prob = CompSum::default();
        self.ht_w = CompSum::default();
        self.ht_wf = CompSum::default();
        self.ht_wy = CompSum::default();
        self.ht_wff = CompSum::default();
        self.ht_wfy = CompSum::default();
        self.ht_wyy = CompSum::default();
        self.sum_beta_z = CompSum::default();
        self.sum_beta = CompSum::default();
        self.sum_v_psi = CompSum::default();
        self.lower = 0.0;
        self.upper = 1.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Rng;

    #[test]
    fn test_ppi_validation_and_nan_safety() {
        assert!(ActivePPI::new(0.0, 0.1, 0.05).is_err());
        assert!(ActivePPI::new(1.0, 0.1, 0.05).is_err());
        assert!(ActivePPI::new(0.05, 0.0, 0.05).is_err());
        assert!(ActivePPI::new(0.05, 1.1, 0.05).is_err());
        assert!(ActivePPI::new(0.05, 0.1, 0.2).is_err());
        assert!(ActivePPI::new(0.05, 0.1, 0.0).is_err());

        let mut ppi = ActivePPI::new(0.05, 0.2, 0.05).unwrap();
        ppi.observe(f64::NAN, false, None, 0.2);
        ppi.observe(0.5, false, None, f64::NAN);
        ppi.observe(0.5, true, None, 0.2); // missing gold label when audited -> ignored
        ppi.observe(0.5, true, Some(f64::INFINITY), 0.2);
        assert_eq!(ppi.n_total(), 0);
        assert_eq!(ppi.n_audited(), 0);
        assert_eq!(ppi.lower(), 0.0);
        assert_eq!(ppi.upper(), 1.0);
    }

    #[test]
    fn test_ppi_accurate_judge_gives_tight_cs_and_high_multiplier() {
        let mut ppi = ActivePPI::new(0.05, 0.15, 0.05).unwrap();
        let mut rng = Rng::from_seed(1337);
        let true_mu = 0.72;

        for _ in 0..4000 {
            let y = if rng.next_f64() < true_mu { 1.0 } else { 0.0 };
            // Accurate judge with 94% agreement and calibrated confidence scores.
            let accurate = rng.next_f64() < 0.94;
            let f = if accurate {
                if y > 0.5 { 0.92 } else { 0.08 }
            } else if y > 0.5 {
                0.20
            } else {
                0.80
            };
            let pi = ppi.suggest_audit_prob(f, None);
            assert!(pi >= 0.05 && pi <= 1.0);
            let audited = rng.next_f64() < pi;
            ppi.observe(f, audited, if audited { Some(y) } else { None }, pi);
        }

        let st = ppi.status();
        assert!(
            st.lower <= true_mu && true_mu <= st.upper,
            "true_mu={} not in [{}, {}]",
            true_mu,
            st.lower,
            st.upper
        );
        assert!(st.lambda_hat > 0.75, "expected high lambda_hat, got {}", st.lambda_hat);
        assert!(st.width < 0.15, "expected tight CS width, got {}", st.width);
        assert!(
            st.variance_reduction_factor > 3.0,
            "expected >3x variance reduction, got {}",
            st.variance_reduction_factor
        );
        assert!(
            st.effective_sample_multiplier > 2.5,
            "expected >2.5x effective sample multiplier, got {}",
            st.effective_sample_multiplier
        );
    }

    #[test]
    fn test_ppi_adversarial_poisoned_and_biased_judges_stay_exact() {
        let alpha = 0.05;
        let trials = 250;
        let horizon = 600;
        let mut rng = Rng::from_seed(9001);

        // Regime 0: Adversarially inverted judge (f = 1 - y) -> lambda_hat must collapse to 0!
        // Regime 1: Heavily positively biased judge (f = clamp(y + 0.45, 0, 1)) -> unbiased CS!
        // Regime 2: Pure random noise judge independent of y -> lambda_hat stays near 0!
        for regime in 0..3 {
            let true_mu = 0.35;
            let mut violations = 0usize;
            let mut avg_final_lam = 0.0;

            for _ in 0..trials {
                let mut ppi = ActivePPI::new(alpha, 0.25, 0.10).unwrap();
                let mut violated = false;
                for _ in 0..horizon {
                    let y: f64 = if rng.next_f64() < true_mu { 1.0 } else { 0.0 };
                    let (f, hint) = match regime {
                        0 => (1.0 - y, Some(0.5)), // poisoned inverted judge
                        1 => ((0.5 * y + 0.45).clamp(0.0, 1.0), Some(0.3 + 0.5 * rng.next_f64())),
                        _ => (rng.next_f64(), None), // pure noise judge
                    };
                    let pi = ppi.suggest_audit_prob(f, hint);
                    let audited = rng.next_f64() < pi;
                    let st = ppi.observe(f, audited, if audited { Some(y) } else { None }, pi);
                    if true_mu < st.lower || true_mu > st.upper {
                        violated = true;
                        break;
                    }
                }
                if violated {
                    violations += 1;
                }
                avg_final_lam += ppi.lambda_hat();
            }

            let rate = (violations as f64) / (trials as f64);
            avg_final_lam /= trials as f64;
            assert!(
                rate <= alpha + 0.02,
                "ActivePPI violated coverage in regime {}: rate={} (alpha={})",
                regime,
                rate,
                alpha
            );
            if regime == 0 {
                assert!(
                    avg_final_lam < 0.05,
                    "inverted judge should drive lambda_hat -> 0, got {}",
                    avg_final_lam
                );
            }
        }
    }

    #[test]
    fn test_ppi_adaptive_sampling_coverage_and_budget_tracking() {
        let alpha = 0.05;
        let target = 0.20;
        let min_pi = 0.05;
        let mut rng = Rng::from_seed(20261122);

        // Latent difficulty determines both uncertainty and label probability.
        let mut ppi = ActivePPI::new(alpha, target, min_pi).unwrap();
        let mut sum_y_true = 0.0;
        let n = 5000;

        for _ in 0..n {
            // Two subpopulations: easy (p=0.95, low uncertainty) and hard (p=0.5, high uncertainty).
            let hard = rng.next_f64() < 0.4;
            let (p_y, f, hint) = if hard {
                (0.50, 0.50, Some(0.95))
            } else {
                (0.95, 0.94, Some(0.08))
            };
            let y = if rng.next_f64() < p_y { 1.0 } else { 0.0 };
            sum_y_true += y;

            let pi = ppi.suggest_audit_prob(f, hint);
            assert!(pi >= min_pi && pi <= 1.0);
            let audited = rng.next_f64() < pi;
            ppi.observe(f, audited, if audited { Some(y) } else { None }, pi);
        }

        let true_mu = 0.4 * 0.50 + 0.6 * 0.95; // = 0.77
        let st = ppi.status();
        assert!(
            st.lower <= true_mu && true_mu <= st.upper,
            "adaptive PPI missed true_mu={} (empirical={}): [{}, {}]",
            true_mu,
            sum_y_true / (n as f64),
            st.lower,
            st.upper
        );
        // Realized audit rate should stay reasonably close to target_audit_rate = 0.20.
        assert!(
            st.audit_rate > 0.10 && st.audit_rate < 0.35,
            "realized audit rate {} far from target {}",
            st.audit_rate,
            target
        );
    }

    #[test]
    fn test_ppi_full_auditing_recovers_tight_eb_bounds() {
        let mut ppi = ActivePPI::new(0.05, 1.0, 1.0).unwrap();
        let mut rng = Rng::from_seed(77);
        let true_mu = 0.6;
        for _ in 0..2000 {
            let y = if rng.next_f64() < true_mu { 1.0 } else { 0.0 };
            let pi = ppi.suggest_audit_prob(0.5, None);
            assert_eq!(pi, 1.0);
            ppi.observe(0.5, true, Some(y), pi);
        }
        let st = ppi.status();
        assert_eq!(st.n_audited, 2000);
        assert!(st.lower <= true_mu && true_mu <= st.upper);
        assert!(st.width < 0.12);
    }
}
