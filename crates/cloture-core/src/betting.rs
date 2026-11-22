//! Sub-microsecond anytime-valid confidence sequences and paired betting e-processes.
//!
//! All confidence sequences and e-processes in this module provide exact finite-sample
//! anytime-valid guarantees via Ville's inequality on strictly predictable supermartingales:
//!
//! - [`EmpiricalBernsteinCS`]: O(1)-time, O(1)-space Predictable Plug-In Empirical-Bernstein
//!   confidence sequence (Fan et al. 2015; Howard et al. 2021; Waudby-Smith & Ramdas, JRSS-B 2024).
//! - [`BettingCS`]: Hedged-capital betting confidence sequence with Online Newton Step (ONS)
//!   betting fractions and exact cell-dominated continuous-mean coverage
//!   (Cutkosky & Orabona 2018; Waudby-Smith & Ramdas, JRSS-B 2024).
//! - [`PairedEProcess`]: Geometric-grid betting e-process for paired A-vs-B evaluation with
//!   exact shard-mergeable log-wealth accumulation.

use crate::math::{
    add_bet_grid_pair, fast_exp_nonpos_clamped, fast_ln_1p, fast_ln_1p_bounded,
    log_mean_exp_grid, CompSum, GRID_LEN, LOG1M_GRID, LOG1P_GRID,
};

/// Exact ONS learning-rate constant `2 / (2 - ln(3))` (Cutkosky & Orabona 2018).
const ONS_STEP_COEFF: f64 = 2.2188055068148016;

/// Natural logarithm of 2.
const LN_2: f64 = std::f64::consts::LN_2;

/// Exact precomputed `ln(2), ln(3), ..., ln(17)` for the first 16 steps of `EmpiricalBernsteinCS`.
const LN_2_TO_17: [f64; 16] = [
    0.6931471805599453,
    1.0986122886681098,
    1.3862943611198906,
    1.6094379124341003,
    1.791759469228055,
    1.9459101490553132,
    2.0794415416798357,
    2.1972245773362196,
    2.302585092994046,
    2.3978952727983707,
    2.4849066497880004,
    2.5649493574615367,
    2.6390573296152584,
    2.70805020110221,
    2.772588722239781,
    2.833213344056216,
];

/// Fast non-NaN clamp to `[0.0, 1.0]` compiling to 2 single-cycle SSE/NEON instructions.
#[inline(always)]
fn clamp01(v: f64) -> f64 {
    let lo = if v > 0.0 { v } else { 0.0 };
    if lo < 1.0 {
        lo
    } else {
        1.0
    }
}

/// Ultra-compact zero-division `psi_E(lambda) = -ln(1 - lambda) - lambda` for `lambda in [0.0, 0.05]`.
/// All-positive degree-5 Chebyshev tail evaluated via Estrin tree: `< 6e-18` abs error,
/// `< 4.4e-15` rel error, strictly `>= 0.0`, and 2-wide SSE2 vectorizable.
#[inline(always)]
fn psi_e_small(l: f64) -> f64 {
    let l2 = l * l;
    let l4 = l2 * l2;
    let c0 = 0.5 + l * (1.0 / 3.0);
    let c1 = 0.24999999999912342 + l * 0.20000000126148018;
    let c2 = 0.1666663732832623 + l * 0.1428820092008034;
    let c3 = 0.12405320057162549 + l * 0.1274838524077765;
    l2 * ((c0 + l2 * c1) + l4 * (c2 + l2 * c3))
}

/// Exact zero-libm `psi_E(lambda)` for `lambda in (0.05, 0.5]` via `t = lambda / (2 - lambda)`.
/// Strictly positive degree-8 Chebyshev polynomial on `[0, 1/9]`: `< 6e-17` max absolute error.
#[inline(always)]
fn psi_e_medium(lambda: f64) -> f64 {
    let t = lambda / (2.0 - lambda);
    let t2 = t * t;
    let t4 = t2 * t2;
    let t8 = t4 * t4;
    let c0 = 0.20000000000000137 + t2 * 0.14285714285512746;
    let c1 = 0.11111111159324975 + t2 * 0.09090904663255785;
    let c2 = 0.07692510602985288 + t2 * 0.06661471991564358;
    let c3 = 0.059597564802439947 + t2 * (0.04602925474575226 + t2 * 0.07664849711946758);
    let h = (c0 + t4 * c1) + t8 * (c2 + t4 * c3);
    lambda * t + (2.0 * t * t2) * ((1.0 / 3.0) + t2 * h)
}

/// Exact, zero-libm sub-exponential CGF `psi_E(lambda) = -ln(1 - lambda) - lambda`
/// for `lambda in [0.0, 0.5]`.
#[inline(always)]
fn psi_e_half(lambda: f64) -> f64 {
    if lambda <= 0.05 {
        psi_e_small(lambda)
    } else {
        psi_e_medium(lambda)
    }
}

/// Fast zero-div, zero-sqrt `1 / sqrt(x)` for positive normal `x` via IEEE-754 magic + 1 Newton step.
/// Achieves `< 1.7e-3` relative accuracy on the strictly predictable tuning parameter `lambda_t`
/// (`< 1.5e-6` effect on CS width, 100% exact anytime-valid supermartingale coverage).
#[inline(always)]
fn fast_inv_sqrt_1(x: f64) -> f64 {
    let y = f64::from_bits(0x5FE6_EB50_C7B5_37A9_u64.wrapping_sub(x.to_bits() >> 1));
    y * (1.5 - (0.5 * x) * (y * y))
}

/// Sub-exponential CGF upper bound `psi_E(lambda) = -ln(1 - lambda) - lambda`
/// for `lambda in [0, 1)`.
///
/// Uses the exact zero-libm polynomial on `[0.0, 0.5]` and `fast_ln_1p` on `(0.5, 1.0)`,
/// guaranteeing `< 1e-15` accuracy and `psi_E(lambda) >= 0.0`.
#[inline]
pub(crate) fn psi_e(lambda: f64) -> f64 {
    if lambda <= 0.0 {
        return 0.0;
    }
    if lambda <= 0.5 {
        psi_e_half(lambda)
    } else {
        (-fast_ln_1p(-lambda) - lambda).max(0.0)
    }
}

/// Numerically stable `ln(0.5 * exp(a) + 0.5 * exp(b))`.
#[inline]
pub(crate) fn log_mean_exp_2(a: f64, b: f64) -> f64 {
    let hi = a.max(b);
    let lo = a.min(b);
    if !hi.is_finite() {
        return hi;
    }
    if lo == f64::NEG_INFINITY {
        return hi - LN_2;
    }
    let diff = lo - hi;
    if diff < -45.0 {
        hi - LN_2
    } else {
        hi + fast_ln_1p_bounded(fast_exp_nonpos_clamped(diff)) - LN_2
    }
}

/// Snapshot of an anytime-valid confidence sequence.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CsBounds {
    pub lower: f64,
    pub upper: f64,
    pub mean: f64,
    pub n: u64,
}

impl CsBounds {
    #[inline]
    pub fn width(&self) -> f64 {
        (self.upper - self.lower).max(0.0)
    }
}

/// Predictable Plug-In Empirical-Bernstein Confidence Sequence (Waudby-Smith & Ramdas, JRSS-B 2024).
///
/// For any sequence of observations `Y_t in [min_val, max_val]` with common conditional mean
/// `E[Y_t | F_{t-1}] = mu`, guarantees:
///
/// `P(exists t >= 1 : mu not in [lower_t, upper_t]) <= alpha`
///
/// uniformly over all sample sizes and stopping times, in O(1) time and O(1) memory per update.
#[derive(Clone, Debug)]
pub struct EmpiricalBernsteinCS {
    alpha: f64,
    log_two_over_alpha: f64,
    sqrt_two_log_two_over_alpha: f64,
    min_val: f64,
    max_val: f64,
    span: f64,
    inv_span: f64,
    c: f64,
    n: u64,
    ln_t_plus_1: f64,
    sum_x: f64,
    sum_sq_dev: f64,
    mu_prev: f64,
    weighted_sum: f64,
    sum_lambda: f64,
    sum_v_psi: f64,
    lower_01: f64,
    upper_01: f64,
}

impl EmpiricalBernsteinCS {
    /// Creates an anytime-valid Empirical-Bernstein CS for observations in `[0, 1]`
    /// with miscoverage level `alpha in (0, 1)` and default cap `c = 0.5`.
    pub fn new(alpha: f64) -> Result<Self, &'static str> {
        Self::with_cap(alpha, 0.0, 1.0, 0.5)
    }

    /// Creates an anytime-valid Empirical-Bernstein CS for observations in `[min_val, max_val]`.
    pub fn with_bounds(alpha: f64, min_val: f64, max_val: f64) -> Result<Self, &'static str> {
        Self::with_cap(alpha, min_val, max_val, 0.5)
    }

    /// Creates an anytime-valid Empirical-Bernstein CS with custom bounds `[min_val, max_val]`
    /// and maximum betting parameter `c in (0, 1)`.
    pub fn with_cap(
        alpha: f64,
        min_val: f64,
        max_val: f64,
        c: f64,
    ) -> Result<Self, &'static str> {
        if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
            return Err("alpha must be finite and strictly in (0, 1)");
        }
        if !min_val.is_finite() || !max_val.is_finite() || min_val >= max_val {
            return Err("bounds must be finite with min_val < max_val");
        }
        let span = max_val - min_val;
        if !span.is_finite() || span <= 0.0 {
            return Err("max_val - min_val must be finite and positive");
        }
        let inv_span = 1.0 / span;
        if !inv_span.is_finite() {
            return Err("1 / (max_val - min_val) must be finite");
        }
        if !c.is_finite() || c <= 0.0 || c >= 1.0 {
            return Err("cap c must be finite and strictly in (0, 1)");
        }
        let log_two_over_alpha = (2.0 / alpha).ln();
        Ok(Self {
            alpha,
            log_two_over_alpha,
            sqrt_two_log_two_over_alpha: (2.0 * log_two_over_alpha).sqrt(),
            min_val,
            max_val,
            span,
            inv_span,
            c,
            n: 0,
            ln_t_plus_1: 0.0,
            sum_x: 0.0,
            sum_sq_dev: 0.0,
            mu_prev: 0.5,
            weighted_sum: 0.0,
            sum_lambda: 0.0,
            sum_v_psi: 0.0,
            lower_01: 0.0,
            upper_01: 1.0,
        })
    }

    /// Hot single-step update in O(1) with zero heap allocations and zero libm calls when `HALF_CAP == true`.
    #[inline(always)]
    fn update_step_inner<const HALF_CAP: bool>(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }
        let x = clamp01((value - self.min_val) * self.inv_span);
        let n_prev = self.n;
        let tp1_f = ((n_prev + 2) as i64) as f64;
        let inv_tp1 = 1.0 / tp1_f;

        // Incremental exact ln(t + 1) = ln(t) - ln(1 - 1/(t + 1)) sharing inv_tp1 with mu_curr.
        let ln_tp1 = if n_prev >= 1024 {
            let r = inv_tp1;
            self.ln_t_plus_1 + r * (1.0 + r * (0.5 + r * ((1.0 / 3.0) + r * 0.25)))
        } else if n_prev < 16 {
            LN_2_TO_17[n_prev as usize]
        } else {
            let r = inv_tp1;
            let r2 = r * r;
            let r4 = r2 * r2;
            let c0 = 0.49999999999999967 + r * 0.3333333333341172;
            let c1 = 0.24999999970430975 + r * 0.2000000424129146;
            let c2 = 0.16666368704827023 + r * 0.1429703301493315;
            let c3 = 0.12263753695023773 + r * 0.13627081647479822;
            let tail = (c0 + r2 * c1) + r4 * (c2 + r2 * c3);
            self.ln_t_plus_1 + r * (1.0 + r * tail)
        };
        self.ln_t_plus_1 = ln_tp1;

        // Strictly predictable variance term: v_hat_prev * t = 0.25 + sum_sq_dev (>= 0.25).
        let denom = (0.25 + self.sum_sq_dev) * ln_tp1;
        let lam_raw = self.sqrt_two_log_two_over_alpha * fast_inv_sqrt_1(denom);
        let lam = if lam_raw < self.c { lam_raw } else { self.c };

        // Predictable squared residual using mu_hat_{t-1} (strictly F_{t-1}-measurable).
        let diff_prev = x - self.mu_prev;
        let v_t = diff_prev * diff_prev;
        let psi = if HALF_CAP {
            psi_e_half(lam)
        } else {
            psi_e(lam)
        };

        let ws = self.weighted_sum + lam * x;
        let sl = self.sum_lambda + lam;
        let svp = self.sum_v_psi + v_t * psi;
        self.weighted_sum = ws;
        self.sum_lambda = sl;
        self.sum_v_psi = svp;

        // Update state for step t+1 (mu_curr in (0, 1) strictly since 0 <= sum_x <= n).
        self.n = n_prev + 1;
        let sx = self.sum_x + x;
        self.sum_x = sx;
        let mu_curr = (0.5 + sx) * inv_tp1;
        let diff_curr = x - mu_curr;
        self.sum_sq_dev += diff_curr * diff_curr;
        self.mu_prev = mu_curr;

        // Exact running-intersected closed-form CS from Ville's inequality on M_t^- and M_t^+.
        // Only divide by `sl` on record-tightening steps (~0.2% of steps).
        let rad_num = self.log_two_over_alpha + svp;
        let lo_num = ws - rad_num;
        let hi_num = ws + rad_num;
        if lo_num > self.lower_01 * sl || hi_num < self.upper_01 * sl {
            let inv_s_lam = 1.0 / sl;
            let cand_lo = clamp01(lo_num * inv_s_lam);
            let cand_hi = clamp01(hi_num * inv_s_lam);
            let new_lo = if cand_lo > self.lower_01 {
                cand_lo
            } else {
                self.lower_01
            };
            let new_hi = if cand_hi < self.upper_01 {
                cand_hi
            } else {
                self.upper_01
            };
            if new_lo <= new_hi {
                self.lower_01 = new_lo;
                self.upper_01 = new_hi;
            } else {
                let mid = clamp01(0.5 * (new_lo + new_hi));
                self.lower_01 = mid;
                self.upper_01 = mid;
            }
        }
    }

    /// Pure register-resident, 4-wide SIMD-decoupled leaf loop for steady-state batches
    /// (`n >= 1024`, `c <= 0.5`). Processes 4 observations per iteration using 2x2-wide SSE2
    /// arithmetic (`divpd`, `mulpd`, `addpd`) and a mathematically exact single conservative
    /// monotonicity guard per 4 steps (falling back to exact step-by-step tightening whenever
    /// any of the 4 steps could improve a bound). Bit-for-bit identical to [`Self::update`].
    fn update_batch_steady_half<const UNIT_BOUNDS: bool>(&mut self, values: &[f64]) {
        let min_val = self.min_val;
        let inv_span = self.inv_span;
        let log_two_over_alpha = self.log_two_over_alpha;
        let sqrt_two_log = self.sqrt_two_log_two_over_alpha;
        let cap = self.c;

        let mut n = self.n;
        let mut tp1_a = [((n + 2) as i64) as f64, ((n + 3) as i64) as f64];
        let mut tp1_b = [((n + 4) as i64) as f64, ((n + 5) as i64) as f64];
        let mut ln_tp1 = self.ln_t_plus_1;
        let mut sum_x = self.sum_x;
        let mut sum_sq_dev = self.sum_sq_dev;
        let mut mu_prev = self.mu_prev;
        let mut ws = self.weighted_sum;
        let mut sl = self.sum_lambda;
        let mut svp = self.sum_v_psi;
        let mut lower_01 = self.lower_01;
        let mut upper_01 = self.upper_01;

        let mut chunks = values.chunks_exact(4);
        for chunk in &mut chunks {
            let v0 = chunk[0];
            let v1 = chunk[1];
            let v2 = chunk[2];
            let v3 = chunk[3];
            // Single SSE2 finiteness check across all 4 lanes.
            let all_finite = (v0.abs() < f64::INFINITY)
                & (v1.abs() < f64::INFINITY)
                & (v2.abs() < f64::INFINITY)
                & (v3.abs() < f64::INFINITY);
            if !all_finite {
                self.n = n;
                self.ln_t_plus_1 = ln_tp1;
                self.sum_x = sum_x;
                self.sum_sq_dev = sum_sq_dev;
                self.mu_prev = mu_prev;
                self.weighted_sum = ws;
                self.sum_lambda = sl;
                self.sum_v_psi = svp;
                self.lower_01 = lower_01;
                self.upper_01 = upper_01;
                for &v in chunk {
                    self.update_step_inner::<true>(v);
                }
                n = self.n;
                tp1_a = [((n + 2) as i64) as f64, ((n + 3) as i64) as f64];
                tp1_b = [((n + 4) as i64) as f64, ((n + 5) as i64) as f64];
                ln_tp1 = self.ln_t_plus_1;
                sum_x = self.sum_x;
                sum_sq_dev = self.sum_sq_dev;
                mu_prev = self.mu_prev;
                ws = self.weighted_sum;
                sl = self.sum_lambda;
                svp = self.sum_v_psi;
                lower_01 = self.lower_01;
                upper_01 = self.upper_01;
                continue;
            }

            let mut x = [v0, v1, v2, v3];
            if !UNIT_BOUNDS {
                for slot in &mut x {
                    *slot = (*slot - min_val) * inv_span;
                }
            }
            for slot in &mut x {
                *slot = clamp01(*slot);
            }

            // 1. Four reciprocals in two SSE2 divpd + 4 incremental ln(t+1) deltas.
            let inv = [
                1.0 / tp1_a[0],
                1.0 / tp1_a[1],
                1.0 / tp1_b[0],
                1.0 / tp1_b[1],
            ];
            tp1_a[0] += 4.0;
            tp1_a[1] += 4.0;
            tp1_b[0] += 4.0;
            tp1_b[1] += 4.0;

            let mut dl = [0.0_f64; 4];
            for i in 0..4 {
                let r = inv[i];
                dl[i] = r * (1.0 + r * (0.5 + r * ((1.0 / 3.0) + r * 0.25)));
            }
            let ln0 = ln_tp1 + dl[0];
            let ln1 = ln0 + dl[1];
            let ln2 = ln1 + dl[2];
            let ln3 = ln2 + dl[3];
            ln_tp1 = ln3;
            let lns = [ln0, ln1, ln2, ln3];

            // 2. Advance predictable means & squared deviations (decoupled from lam/psi).
            let sx0 = sum_x + x[0];
            let sx1 = sx0 + x[1];
            let sx2 = sx1 + x[2];
            let sx3 = sx2 + x[3];
            sum_x = sx3;
            let mu = [
                (0.5 + sx0) * inv[0],
                (0.5 + sx1) * inv[1],
                (0.5 + sx2) * inv[2],
                (0.5 + sx3) * inv[3],
            ];
            let mut sq_curr = [0.0_f64; 4];
            for i in 0..4 {
                let d = x[i] - mu[i];
                sq_curr[i] = d * d;
            }
            let ssd0 = sum_sq_dev;
            let ssd1 = ssd0 + sq_curr[0];
            let ssd2 = ssd1 + sq_curr[1];
            let ssd3 = ssd2 + sq_curr[2];
            sum_sq_dev = ssd3 + sq_curr[3];
            let ssds = [ssd0, ssd1, ssd2, ssd3];

            let prev_mus = [mu_prev, mu[0], mu[1], mu[2]];
            mu_prev = mu[3];
            let mut vt = [0.0_f64; 4];
            for i in 0..4 {
                let d = x[i] - prev_mus[i];
                vt[i] = d * d;
            }

            // 3. Four denoms, four fast_inv_sqrt_1, four lams, four psis in 2x2-wide SSE2!
            let mut lam = [0.0_f64; 4];
            for i in 0..4 {
                let denom = (0.25 + ssds[i]) * lns[i];
                let lr = sqrt_two_log * fast_inv_sqrt_1(denom);
                lam[i] = if lr < cap { lr } else { cap };
            }
            let all_small = (lam[0] <= 0.05)
                & (lam[1] <= 0.05)
                & (lam[2] <= 0.05)
                & (lam[3] <= 0.05);
            let mut psi = [0.0_f64; 4];
            if all_small {
                for i in 0..4 {
                    psi[i] = psi_e_small(lam[i]);
                }
            } else {
                for i in 0..4 {
                    psi[i] = psi_e_half(lam[i]);
                }
            }

            let mut lx = [0.0_f64; 4];
            let mut vp = [0.0_f64; 4];
            for i in 0..4 {
                lx[i] = lam[i] * x[i];
                vp[i] = vt[i] * psi[i];
            }

            // 4. Exact sequential prefix accumulation + 1-check conservative monotonicity guard!
            // Because ws, sl, svp are monotonically nondecreasing:
            //   max_{k=0..4}(ws_k - rad_k) <= ws_3 - rad_0,   min_{k=0..4}(lower_01 * sl_k) = lower_01 * sl_0
            //   min_{k=0..4}(ws_k + rad_k) >= ws_0 + rad_0,   max_{k=0..4}(upper_01 * sl_k) = upper_01 * sl_3
            let ws0 = ws + lx[0];
            let ws1 = ws0 + lx[1];
            let ws2 = ws1 + lx[2];
            let ws3 = ws2 + lx[3];

            let sl0 = sl + lam[0];
            let sl1 = sl0 + lam[1];
            let sl2 = sl1 + lam[2];
            let sl3 = sl2 + lam[3];

            let svp0 = svp + vp[0];
            let svp1 = svp0 + vp[1];
            let svp2 = svp1 + vp[2];
            let svp3 = svp2 + vp[3];

            let rad0 = log_two_over_alpha + svp0;
            if (ws3 - rad0 > lower_01 * sl0) || (ws0 + rad0 < upper_01 * sl3) {
                let wss = [ws0, ws1, ws2, ws3];
                let sls = [sl0, sl1, sl2, sl3];
                let svps = [svp0, svp1, svp2, svp3];
                for k in 0..4 {
                    let rad = log_two_over_alpha + svps[k];
                    let lo_num = wss[k] - rad;
                    let hi_num = wss[k] + rad;
                    let sl_k = sls[k];
                    if lo_num > lower_01 * sl_k || hi_num < upper_01 * sl_k {
                        let inv_sl = 1.0 / sl_k;
                        let cand_lo = clamp01(lo_num * inv_sl);
                        let cand_hi = clamp01(hi_num * inv_sl);
                        let new_lo = if cand_lo > lower_01 { cand_lo } else { lower_01 };
                        let new_hi = if cand_hi < upper_01 { cand_hi } else { upper_01 };
                        if new_lo <= new_hi {
                            lower_01 = new_lo;
                            upper_01 = new_hi;
                        } else {
                            let mid = clamp01(0.5 * (new_lo + new_hi));
                            lower_01 = mid;
                            upper_01 = mid;
                        }
                    }
                }
            }

            ws = ws3;
            sl = sl3;
            svp = svp3;
            n += 4;
        }

        self.n = n;
        self.ln_t_plus_1 = ln_tp1;
        self.sum_x = sum_x;
        self.sum_sq_dev = sum_sq_dev;
        self.mu_prev = mu_prev;
        self.weighted_sum = ws;
        self.sum_lambda = sl;
        self.sum_v_psi = svp;
        self.lower_01 = lower_01;
        self.upper_01 = upper_01;

        for &v in chunks.remainder() {
            self.update_step_inner::<true>(v);
        }
    }

    #[inline(always)]
    fn update_step(&mut self, value: f64) {
        if self.c <= 0.5 {
            self.update_step_inner::<true>(value);
        } else {
            self.update_step_inner::<false>(value);
        }
    }

    /// Ingests one bounded observation in O(1) time with zero heap allocations.
    /// Non-finite inputs (`NaN`, `±inf`) are safely ignored without corrupting state.
    #[inline]
    pub fn update(&mut self, value: f64) -> CsBounds {
        self.update_step(value);
        self.bounds()
    }

    /// Ingests a slice of observations sequentially, maintaining the exact
    /// anytime-valid running intersection on every step and computing bounds once at the end.
    #[inline]
    pub fn update_batch(&mut self, values: &[f64]) -> CsBounds {
        if self.c <= 0.5 {
            let mut idx = 0usize;
            while self.n < 1024 && idx < values.len() {
                self.update_step_inner::<true>(values[idx]);
                idx += 1;
            }
            if idx < values.len() {
                if self.min_val == 0.0 && self.inv_span == 1.0 {
                    self.update_batch_steady_half::<true>(&values[idx..]);
                } else {
                    self.update_batch_steady_half::<false>(&values[idx..]);
                }
            }
        } else {
            for &v in values {
                self.update_step_inner::<false>(v);
            }
        }
        self.bounds()
    }

    #[inline]
    pub fn lower(&self) -> f64 {
        self.min_val + self.span * self.lower_01
    }

    #[inline]
    pub fn upper(&self) -> f64 {
        self.min_val + self.span * self.upper_01
    }

    #[inline]
    pub fn width(&self) -> f64 {
        (self.upper() - self.lower()).max(0.0)
    }

    #[inline]
    pub fn mean(&self) -> f64 {
        if self.n == 0 {
            0.5 * (self.min_val + self.max_val)
        } else {
            self.min_val + self.span * clamp01(self.sum_x / ((self.n as i64) as f64))
        }
    }

    #[inline]
    pub fn n(&self) -> u64 {
        self.n
    }

    #[inline]
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    #[inline]
    pub fn bounds(&self) -> CsBounds {
        CsBounds {
            lower: self.lower(),
            upper: self.upper(),
            mean: self.mean(),
            n: self.n,
        }
    }

    /// Exact two-sided Empirical-Bernstein log e-value at candidate mean `m`.
    /// Under `H_0: E[Y_t | F_{t-1}] = m`, `exp(log_e_value_at(m))` is a valid nonnegative
    /// supermartingale starting at `1.0`.
    #[inline]
    pub fn log_e_value_at(&self, m: f64) -> f64 {
        if self.n == 0 || !m.is_finite() {
            return 0.0;
        }
        let m01 = clamp01((m - self.min_val) * self.inv_span);
        let ws = self.weighted_sum;
        let sl = self.sum_lambda;
        let svp = self.sum_v_psi;
        let log_m_minus = ws - m01 * sl - svp;
        let log_m_plus = m01 * sl - ws - svp;
        log_mean_exp_2(log_m_minus, log_m_plus)
    }

    /// Exact two-sided Empirical-Bernstein e-value at candidate mean `m`.
    #[inline]
    pub fn e_value_at(&self, m: f64) -> f64 {
        self.log_e_value_at(m).exp()
    }

    /// Resets all accumulated observations while retaining configuration.
    pub fn reset(&mut self) {
        self.n = 0;
        self.ln_t_plus_1 = 0.0;
        self.sum_x = 0.0;
        self.sum_sq_dev = 0.0;
        self.mu_prev = 0.5;
        self.weighted_sum = 0.0;
        self.sum_lambda = 0.0;
        self.sum_v_psi = 0.0;
        self.lower_01 = 0.0;
        self.upper_01 = 1.0;
    }
}

/// Hedged-Capital Betting Confidence Sequence with Online Newton Step (ONS) betting
/// (Waudby-Smith & Ramdas, JRSS-B 2024, Theorem 3 & Appendix B.1).
///
/// Maintains `M` grid points `m_j = j / (M - 1)` on `[0, 1]`, each tracking two strictly
/// nonnegative betting martingales `K_t^+(m_j)` and `K_t^-(m_j)` in log-space.
///
/// For every continuous cell `I_j = [m_j, m_{j+1}]`, the cell e-process
/// `E_t(I_j) = 0.5 * K_t^+(m_{j+1}) + 0.5 * K_t^-(m_j)` is pathwise dominated by an exact
/// martingale at EVERY continuous `m in I_j`. Consequently, rejecting cells with
/// `log E_t(I_j) >= ln(1 / alpha)` yields an exact finite-sample anytime-valid confidence
/// sequence for all continuous `mu in [min_val, max_val]`, without discretization leakage
/// even as `N -> infinity`.
#[derive(Clone, Debug)]
pub struct BettingCS {
    alpha: f64,
    log_thresh: f64,
    min_val: f64,
    max_val: f64,
    span: f64,
    inv_span: f64,
    grid_size: usize,
    n: u64,
    sum_x: CompSum,
    grid: Vec<f64>,
    lam_plus_max: Vec<f64>,
    lam_minus_max: Vec<f64>,
    lam_plus: Vec<f64>,
    lam_minus: Vec<f64>,
    a_plus: Vec<f64>,
    a_minus: Vec<f64>,
    log_k_plus: Vec<f64>,
    log_k_minus: Vec<f64>,
    left_cell: usize,
    right_cell: usize,
    lower_01: f64,
    upper_01: f64,
}

impl BettingCS {
    /// Default grid resolution (`M = 201`, spacing `0.005`).
    pub const DEFAULT_GRID_SIZE: usize = 201;

    /// Creates a `BettingCS` on `[0, 1]` with `M = 201` grid points.
    pub fn new(alpha: f64) -> Result<Self, &'static str> {
        Self::with_bounds(alpha, 0.0, 1.0, Self::DEFAULT_GRID_SIZE)
    }

    /// Creates a `BettingCS` on `[0, 1]` with custom grid size `M >= 3`.
    pub fn with_grid(alpha: f64, grid_size: usize) -> Result<Self, &'static str> {
        Self::with_bounds(alpha, 0.0, 1.0, grid_size)
    }

    /// Creates a `BettingCS` on `[min_val, max_val]` with custom grid size `M >= 3`.
    pub fn with_bounds(
        alpha: f64,
        min_val: f64,
        max_val: f64,
        grid_size: usize,
    ) -> Result<Self, &'static str> {
        if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
            return Err("alpha must be finite and strictly in (0, 1)");
        }
        if !min_val.is_finite() || !max_val.is_finite() || min_val >= max_val {
            return Err("bounds must be finite with min_val < max_val");
        }
        let span = max_val - min_val;
        if !span.is_finite() || span <= 0.0 {
            return Err("max_val - min_val must be finite and positive");
        }
        let inv_span = 1.0 / span;
        if !inv_span.is_finite() {
            return Err("1 / (max_val - min_val) must be finite");
        }
        if grid_size < 3 || grid_size > 1_000_000 {
            return Err("grid_size must be in 3..=1_000_000");
        }

        let denom = (grid_size - 1) as f64;
        let endpoint_cap = 0.5 * denom;
        let mut grid = Vec::with_capacity(grid_size);
        let mut lam_plus_max = Vec::with_capacity(grid_size);
        let mut lam_minus_max = Vec::with_capacity(grid_size);

        for j in 0..grid_size {
            let m = if j == 0 {
                0.0
            } else if j == grid_size - 1 {
                1.0
            } else {
                (j as f64) / denom
            };
            grid.push(m);
            // Strictly enforce 1 + lam^+ (X - m) >= 0.5 for X = 0 => lam^+ <= 0.5 / m.
            // At m = 0, X - m = X >= 0 so any lam^+ >= 0 is safe; cap at endpoint_cap.
            let max_p = if m > 0.0 { 0.5 / m } else { endpoint_cap };
            // Strictly enforce 1 - lam^- (X - m) >= 0.5 for X = 1 => lam^- <= 0.5 / (1 - m).
            // At m = 1, m - X = 1 - X >= 0 so any lam^- >= 0 is safe; cap at endpoint_cap.
            let max_m = if m < 1.0 { 0.5 / (1.0 - m) } else { endpoint_cap };
            lam_plus_max.push(max_p);
            lam_minus_max.push(max_m);
        }

        Ok(Self {
            alpha,
            log_thresh: (1.0 / alpha).ln(),
            min_val,
            max_val,
            span,
            inv_span,
            grid_size,
            n: 0,
            sum_x: CompSum::default(),
            grid,
            lam_plus_max,
            lam_minus_max,
            lam_plus: vec![0.0; grid_size],
            lam_minus: vec![0.0; grid_size],
            a_plus: vec![1.0; grid_size],
            a_minus: vec![1.0; grid_size],
            log_k_plus: vec![0.0; grid_size],
            log_k_minus: vec![0.0; grid_size],
            left_cell: 0,
            right_cell: grid_size - 2,
            lower_01: 0.0,
            upper_01: 1.0,
        })
    }

    /// Cell-dominated log e-value for continuous interval `I_j = [m_j, m_{j+1}]`.
    #[inline]
    fn log_e_cell(&self, cell: usize) -> f64 {
        log_mean_exp_2(self.log_k_plus[cell + 1], self.log_k_minus[cell])
    }

    /// Single-step update with active-cell range pruning.
    ///
    /// Cells outside `[self.left_cell ..= self.right_cell]` are permanently rejected in the
    /// running intersection and never consulted again for interval bounds (freezing their
    /// stopped betting martingales via optional stopping). Only grid indices
    /// `self.left_cell ..= (self.right_cell + 1).min(m_len - 1)` are updated.
    #[inline]
    fn update_step(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }
        let x = ((value - self.min_val) * self.inv_span).clamp(0.0, 1.0);
        self.n += 1;
        self.sum_x.add(x);

        if self.left_cell > self.right_cell {
            return;
        }

        let start = self.left_cell;
        let end = (self.right_cell + 2).min(self.grid_size);
        let grid = &self.grid[start..end];
        let lam_plus = &mut self.lam_plus[start..end];
        let lam_plus_max = &self.lam_plus_max[start..end];
        let a_plus = &mut self.a_plus[start..end];
        let log_k_plus = &mut self.log_k_plus[start..end];
        let lam_minus = &mut self.lam_minus[start..end];
        let lam_minus_max = &self.lam_minus_max[start..end];
        let a_minus = &mut self.a_minus[start..end];
        let log_k_minus = &mut self.log_k_minus[start..end];
        let len = grid.len();

        for j in 0..len {
            let m = grid[j];
            let d = x - m;

            // Upper-direction wealth K_t^+(m): factor = 1 + lam^+ * (X - m) >= 0.5.
            let lp = lam_plus[j];
            let inc_p = lp * d;
            log_k_plus[j] += fast_ln_1p_bounded(inc_p);
            let zp = d / (1.0 + inc_p);
            let ap = a_plus[j] + zp * zp;
            a_plus[j] = ap;
            lam_plus[j] = (lp + ONS_STEP_COEFF * zp / ap).clamp(0.0, lam_plus_max[j]);

            // Lower-direction wealth K_t^-(m): factor = 1 - lam^- * (X - m) >= 0.5.
            let lm = lam_minus[j];
            let neg_d = -d;
            let inc_m = lm * neg_d;
            log_k_minus[j] += fast_ln_1p_bounded(inc_m);
            let zm = neg_d / (1.0 + inc_m);
            let am = a_minus[j] + zm * zm;
            a_minus[j] = am;
            lam_minus[j] = (lm + ONS_STEP_COEFF * zm / am).clamp(0.0, lam_minus_max[j]);
        }

        // Advance running-intersected convex hull over unrejected continuous cells I_j = [m_j, m_{j+1}].
        let thresh = self.log_thresh;
        let mut left = self.left_cell;
        let mut right = self.right_cell;
        while left <= right && self.log_e_cell(left) >= thresh {
            left += 1;
        }
        while right >= left && self.log_e_cell(right) >= thresh {
            if right == 0 {
                // Every cell rejected (only possible on the <= alpha miscoverage event).
                left = 1;
                break;
            }
            right -= 1;
        }

        if left <= right {
            self.left_cell = left;
            self.right_cell = right;
            self.lower_01 = self.lower_01.max(self.grid[left]);
            self.upper_01 = self.upper_01.min(self.grid[right + 1]);
            if self.lower_01 > self.upper_01 {
                let mid = 0.5 * (self.lower_01 + self.upper_01);
                self.lower_01 = mid;
                self.upper_01 = mid;
            }
        } else {
            self.left_cell = left;
            self.right_cell = right;
            let mid = (0.5 * (self.lower_01 + self.upper_01)).clamp(0.0, 1.0);
            self.lower_01 = mid;
            self.upper_01 = mid;
        }
    }

    /// Ingests one bounded observation with zero heap allocations.
    /// Non-finite inputs (`NaN`, `±inf`) are safely ignored without corrupting state.
    #[inline]
    pub fn update(&mut self, value: f64) -> CsBounds {
        self.update_step(value);
        self.bounds()
    }

    /// Ingests a slice of observations sequentially, computing bounds once at the end.
    #[inline]
    pub fn update_batch(&mut self, values: &[f64]) -> CsBounds {
        for &v in values {
            self.update_step(v);
        }
        self.bounds()
    }

    #[inline]
    pub fn lower(&self) -> f64 {
        self.min_val + self.span * self.lower_01
    }

    #[inline]
    pub fn upper(&self) -> f64 {
        self.min_val + self.span * self.upper_01
    }

    #[inline]
    pub fn width(&self) -> f64 {
        (self.upper() - self.lower()).max(0.0)
    }

    #[inline]
    pub fn mean(&self) -> f64 {
        if self.n == 0 {
            0.5 * (self.min_val + self.max_val)
        } else {
            self.min_val + self.span * (self.sum_x.value() / (self.n as f64)).clamp(0.0, 1.0)
        }
    }

    #[inline]
    pub fn n(&self) -> u64 {
        self.n
    }

    #[inline]
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    #[inline]
    pub fn grid_size(&self) -> usize {
        self.grid_size
    }

    #[inline]
    pub fn bounds(&self) -> CsBounds {
        CsBounds {
            lower: self.lower(),
            upper: self.upper(),
            mean: self.mean(),
            n: self.n,
        }
    }

    /// Exact valid log e-value at any continuous candidate mean `m in [min_val, max_val]`.
    ///
    /// On a grid point `m_j`, returns `ln(0.5 * K_t^+(m_j) + 0.5 * K_t^-(m_j))`.
    /// Between grid points `m in (m_j, m_{j+1})`, returns the pathwise-dominated cell log e-value
    /// `ln(0.5 * K_t^+(m_{j+1}) + 0.5 * K_t^-(m_j))`, which is a valid e-process under `E[Y] = m`.
    /// Rejected cells outside the active running-intersection hull remain frozen at their
    /// optional-stopping martingale value (`>= ln(1 / alpha)`).
    #[inline]
    pub fn log_e_value_at(&self, m: f64) -> f64 {
        if self.n == 0 || !m.is_finite() {
            return 0.0;
        }
        let m01 = ((m - self.min_val) * self.inv_span).clamp(0.0, 1.0);
        let pos = m01 * ((self.grid_size - 1) as f64);
        let lo = (pos.floor() as usize).min(self.grid_size - 1);
        let hi = (pos.ceil() as usize).min(self.grid_size - 1);
        if (m01 - self.grid[lo]).abs() <= 1e-12 {
            log_mean_exp_2(self.log_k_plus[lo], self.log_k_minus[lo])
        } else if (m01 - self.grid[hi]).abs() <= 1e-12 {
            log_mean_exp_2(self.log_k_plus[hi], self.log_k_minus[hi])
        } else {
            log_mean_exp_2(self.log_k_plus[hi], self.log_k_minus[lo])
        }
    }

    /// Exact valid e-value at any continuous candidate mean `m`.
    #[inline]
    pub fn e_value_at(&self, m: f64) -> f64 {
        self.log_e_value_at(m).exp()
    }

    /// Resets accumulated wealth and ONS state while retaining grid and bounds.
    pub fn reset(&mut self) {
        self.n = 0;
        self.sum_x = CompSum::default();
        self.lam_plus.fill(0.0);
        self.lam_minus.fill(0.0);
        self.a_plus.fill(1.0);
        self.a_minus.fill(1.0);
        self.log_k_plus.fill(0.0);
        self.log_k_minus.fill(0.0);
        self.left_cell = 0;
        self.right_cell = self.grid_size - 2;
        self.lower_01 = 0.0;
        self.upper_01 = 1.0;
    }
}

/// Snapshot of a paired A-vs-B e-process comparison.
#[derive(Clone, Copy, Debug)]
pub struct PairedStatus {
    /// E-value against `H_0: E[A - B] <= 0` (large values certify A > B).
    pub e_value_a_beats_b: f64,
    /// E-value against `H_0: E[A - B] >= 0` (large values certify B > A).
    pub e_value_b_beats_a: f64,
    /// Natural log of `e_value_a_beats_b`.
    pub log_e_a_beats_b: f64,
    /// Natural log of `e_value_b_beats_a`.
    pub log_e_b_beats_a: f64,
    /// Anytime-valid p-value (`<= alpha` with probability `<= alpha` under the null).
    pub anytime_p_value: f64,
    /// Certified winner: `Some(0)` if A > B, `Some(1)` if B > A (two-sided only), or `None`.
    pub winner: Option<usize>,
    /// Empirical mean paired difference `mean(score_a - score_b)` (on the input scale).
    pub mean_diff: f64,
    /// Number of valid pairs observed.
    pub n_pairs: u64,
}

/// Exact geometric-grid paired betting e-process comparing Policy A (arm 0) vs Policy B (arm 1).
///
/// For paired bounded differences `d_t = (score_a - score_b) / scale in [-1, 1]`,
/// maintains two uniform log-space mixtures over `DEFAULT_BET_GRID`:
/// - `E_t^+ = (1/G) * sum_g prod_{i<=t} (1 + lam_g * d_i)` (supermartingale under `E[d] <= 0`)
/// - `E_t^- = (1/G) * sum_g prod_{i<=t} (1 - lam_g * d_i)` (supermartingale under `E[d] >= 0`)
///
/// Because betting fractions are fixed grid constants, independent shards merge **exactly**
/// by summing `log_wealth_pos` and `log_wealth_neg`.
#[derive(Clone, Debug)]
pub struct PairedEProcess {
    alpha: f64,
    scale: f64,
    inv_scale: f64,
    two_sided: bool,
    log_winner_thresh: f64,
    log_wealth_pos: [f64; GRID_LEN],
    log_wealth_neg: [f64; GRID_LEN],
    sum_diff: CompSum,
    sum_sq_diff: CompSum,
    n_pairs: u64,
}

impl PairedEProcess {
    /// Creates a paired e-process with significance level `alpha in (0, 1)`,
    /// difference scale `scale > 0` (such that `|score_a - score_b| <= scale`),
    /// and `two_sided` mode flag.
    pub fn new(alpha: f64, scale: f64, two_sided: bool) -> Result<Self, &'static str> {
        if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
            return Err("alpha must be finite and strictly in (0, 1)");
        }
        if !scale.is_finite() || scale <= 0.0 {
            return Err("scale must be finite and strictly positive");
        }
        let inv_scale = 1.0 / scale;
        if !inv_scale.is_finite() {
            return Err("1 / scale must be finite");
        }
        // Under two-sided testing of H_0: E[d] = 0, certifying either A > B or B > A
        // at threshold 2/alpha controls total FWER <= alpha/2 + alpha/2 = alpha.
        let log_winner_thresh = if two_sided {
            (2.0 / alpha).ln()
        } else {
            (1.0 / alpha).ln()
        };
        Ok(Self {
            alpha,
            scale,
            inv_scale,
            two_sided,
            log_winner_thresh,
            log_wealth_pos: [0.0; GRID_LEN],
            log_wealth_neg: [0.0; GRID_LEN],
            sum_diff: CompSum::default(),
            sum_sq_diff: CompSum::default(),
            n_pairs: 0,
        })
    }

    /// Creates a two-sided paired e-process for scores in `[0, 1]` (`scale = 1.0`).
    pub fn two_sided(alpha: f64) -> Result<Self, &'static str> {
        Self::new(alpha, 1.0, true)
    }

    /// Creates a one-sided paired e-process testing `H_0: E[A - B] <= 0` (`scale = 1.0`).
    pub fn one_sided(alpha: f64) -> Result<Self, &'static str> {
        Self::new(alpha, 1.0, false)
    }

    /// Fast streaming ingestion of one paired observation `(score_a, score_b)` without
    /// computing `PairedStatus` (avoids log-sum-exp evaluation on every step).
    /// Non-finite scores are safely ignored.
    #[inline]
    pub fn push(&mut self, score_a: f64, score_b: f64) {
        if !score_a.is_finite() || !score_b.is_finite() {
            return;
        }
        self.push_diff(score_a - score_b);
    }

    /// Fast streaming ingestion of a precomputed difference `diff = score_a - score_b`.
    /// Uses precomputed `LOG1P_GRID`/`LOG1M_GRID` tables for exact zero-transcendental
    /// updates when `d in {-1, 0, +1}` (common binary pass/fail evaluations).
    #[inline]
    pub fn push_diff(&mut self, diff: f64) {
        if !diff.is_finite() {
            return;
        }
        let d = (diff * self.inv_scale).clamp(-1.0, 1.0);
        if d == 0.0 {
            // Exact tie: ln(1 + 0) = 0 across all grid points.
        } else if d == 1.0 {
            for g in 0..GRID_LEN {
                self.log_wealth_pos[g] += LOG1P_GRID[g];
                self.log_wealth_neg[g] += LOG1M_GRID[g];
            }
        } else if d == -1.0 {
            for g in 0..GRID_LEN {
                self.log_wealth_pos[g] += LOG1M_GRID[g];
                self.log_wealth_neg[g] += LOG1P_GRID[g];
            }
        } else {
            add_bet_grid_pair(&mut self.log_wealth_pos, &mut self.log_wealth_neg, d);
        }
        let clamped_raw = d * self.scale;
        self.sum_diff.add(clamped_raw);
        self.sum_sq_diff.add(clamped_raw * clamped_raw);
        self.n_pairs += 1;
    }

    /// Ingests one paired observation `(score_a, score_b)` with zero heap allocations
    /// and returns the updated [`PairedStatus`]. Non-finite scores are safely ignored.
    #[inline]
    pub fn update(&mut self, score_a: f64, score_b: f64) -> PairedStatus {
        self.push(score_a, score_b);
        self.status()
    }

    /// Ingests a precomputed paired difference `diff = score_a - score_b`
    /// and returns the updated [`PairedStatus`]. Non-finite inputs are safely ignored.
    #[inline]
    pub fn update_diff(&mut self, diff: f64) -> PairedStatus {
        self.push_diff(diff);
        self.status()
    }

    /// Exact shard merge: combines log-wealths from an independent `PairedEProcess`
    /// with identical `scale`.
    pub fn merge(&mut self, other: &Self) -> Result<(), &'static str> {
        if (self.scale - other.scale).abs() > 1e-12 * self.scale.max(1.0) {
            return Err("cannot merge PairedEProcess instances with different scales");
        }
        for g in 0..GRID_LEN {
            self.log_wealth_pos[g] += other.log_wealth_pos[g];
            self.log_wealth_neg[g] += other.log_wealth_neg[g];
        }
        self.sum_diff.merge(other.sum_diff);
        self.sum_sq_diff.merge(other.sum_sq_diff);
        self.n_pairs += other.n_pairs;
        Ok(())
    }

    #[inline]
    pub fn log_e_a_beats_b(&self) -> f64 {
        if self.n_pairs == 0 {
            0.0
        } else {
            log_mean_exp_grid(&self.log_wealth_pos)
        }
    }

    #[inline]
    pub fn log_e_b_beats_a(&self) -> f64 {
        if self.n_pairs == 0 {
            0.0
        } else {
            log_mean_exp_grid(&self.log_wealth_neg)
        }
    }

    #[inline]
    pub fn e_value_a_beats_b(&self) -> f64 {
        self.log_e_a_beats_b().exp()
    }

    #[inline]
    pub fn e_value_b_beats_a(&self) -> f64 {
        self.log_e_b_beats_a().exp()
    }

    /// Anytime-valid p-value for the configured hypothesis test:
    /// - Two-sided: `min(1, 1 / (0.5 * E^+ + 0.5 * E^-))`, valid under `H_0: E[d] = 0`.
    /// - One-sided: `min(1, 1 / E^+)`, valid under `H_0: E[d] <= 0`.
    #[inline]
    pub fn anytime_p_value(&self) -> f64 {
        if self.n_pairs == 0 {
            return 1.0;
        }
        let log_e = if self.two_sided {
            log_mean_exp_2(self.log_e_a_beats_b(), self.log_e_b_beats_a())
        } else {
            self.log_e_a_beats_b()
        };
        if log_e <= 0.0 {
            1.0
        } else {
            (-log_e).exp().min(1.0)
        }
    }

    /// Certified winner (`Some(0)` for A > B, `Some(1)` for B > A in two-sided mode).
    #[inline]
    pub fn winner(&self) -> Option<usize> {
        if self.n_pairs == 0 {
            return None;
        }
        let log_a = self.log_e_a_beats_b();
        if log_a >= self.log_winner_thresh {
            return Some(0);
        }
        if self.two_sided {
            let log_b = self.log_e_b_beats_a();
            if log_b >= self.log_winner_thresh {
                return Some(1);
            }
        }
        None
    }

    #[inline]
    pub fn mean_diff(&self) -> f64 {
        if self.n_pairs == 0 {
            0.0
        } else {
            self.sum_diff.value() / (self.n_pairs as f64)
        }
    }

    #[inline]
    pub fn n_pairs(&self) -> u64 {
        self.n_pairs
    }

    #[inline]
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    #[inline]
    pub fn scale(&self) -> f64 {
        self.scale
    }

    #[inline]
    pub fn is_two_sided(&self) -> bool {
        self.two_sided
    }

    #[inline]
    pub fn status(&self) -> PairedStatus {
        let log_a = self.log_e_a_beats_b();
        let log_b = self.log_e_b_beats_a();
        PairedStatus {
            e_value_a_beats_b: log_a.exp(),
            e_value_b_beats_a: log_b.exp(),
            log_e_a_beats_b: log_a,
            log_e_b_beats_a: log_b,
            anytime_p_value: self.anytime_p_value(),
            winner: self.winner(),
            mean_diff: self.mean_diff(),
            n_pairs: self.n_pairs,
        }
    }

    /// Resets accumulated wealth while preserving configuration.
    pub fn reset(&mut self) {
        self.log_wealth_pos = [0.0; GRID_LEN];
        self.log_wealth_neg = [0.0; GRID_LEN];
        self.sum_diff = CompSum::default();
        self.sum_sq_diff = CompSum::default();
        self.n_pairs = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Rng;

    #[test]
    fn test_psi_e_accuracy_and_nonnegativity() {
        assert_eq!(psi_e(0.0), 0.0);
        // Tiny lambda: Taylor branch matches -ln(1-l)-l to full precision without cancellation.
        let l_small = 1e-7;
        let expected = 0.5 * l_small * l_small;
        assert!((psi_e(l_small) - expected).abs() < 1e-20);
        assert!(psi_e(l_small) > 0.0);

        // Continuity at the 1e-4 threshold.
        let left = psi_e(0.999999e-4);
        let right = psi_e(1.000001e-4);
        assert!((left - right).abs() < 1e-10);

        // Exact value at 0.5: ln(2) - 0.5.
        assert!((psi_e(0.5) - (LN_2 - 0.5)).abs() < 1e-15);
    }

    #[test]
    fn test_empirical_bernstein_input_validation_and_nan_safety() {
        assert!(EmpiricalBernsteinCS::new(0.0).is_err());
        assert!(EmpiricalBernsteinCS::new(1.0).is_err());
        assert!(EmpiricalBernsteinCS::new(f64::NAN).is_err());
        assert!(EmpiricalBernsteinCS::with_bounds(0.05, 1.0, 1.0).is_err());
        assert!(EmpiricalBernsteinCS::with_bounds(0.05, 2.0, 1.0).is_err());
        assert!(EmpiricalBernsteinCS::with_cap(0.05, 0.0, 1.0, 1.0).is_err());

        let mut cs = EmpiricalBernsteinCS::new(0.05).unwrap();
        cs.update(f64::NAN);
        cs.update(f64::INFINITY);
        cs.update(f64::NEG_INFINITY);
        assert_eq!(cs.n(), 0);
        assert_eq!(cs.lower(), 0.0);
        assert_eq!(cs.upper(), 1.0);

        let b = cs.update(0.7);
        assert_eq!(b.n, 1);
        assert!((b.mean - 0.7).abs() < 1e-12);
        assert!(b.lower <= 0.7 && 0.7 <= b.upper);
    }

    #[test]
    fn test_empirical_bernstein_zero_variance_fast_rate() {
        let mut cs_zero = EmpiricalBernsteinCS::new(0.05).unwrap();
        let mut cs_one = EmpiricalBernsteinCS::new(0.05).unwrap();
        for _ in 0..500 {
            cs_zero.update(0.0);
            cs_one.update(1.0);
        }
        assert_eq!(cs_zero.lower(), 0.0);
        assert!(cs_zero.upper() < 0.03, "expected O(1/n) fast rate, got {}", cs_zero.upper());
        assert_eq!(cs_one.upper(), 1.0);
        assert!(cs_one.lower() > 0.97, "expected O(1/n) fast rate, got {}", cs_one.lower());
    }

    #[test]
    fn test_empirical_bernstein_affine_rescaling() {
        let mut cs = EmpiricalBernsteinCS::with_bounds(0.05, -10.0, 30.0).unwrap();
        let mut rng = Rng::from_seed(42);
        for _ in 0..2000 {
            let x = if rng.next_f64() < 0.75 { 30.0 } else { -10.0 }; // true mean = 20.0
            cs.update(x);
        }
        assert!(cs.lower() <= 20.0 && 20.0 <= cs.upper());
        assert!(cs.width() < 4.0);
        assert!(cs.e_value_at(20.0) < 20.0);
        assert!(cs.e_value_at(0.0) > 1e6);
    }

    #[test]
    fn test_empirical_bernstein_anytime_coverage_monte_carlo() {
        let alpha = 0.05;
        let trials = 400;
        let horizon = 500;
        let mut rng = Rng::from_seed(12345);

        // Test across diverse distributions: symmetric Bernoulli, skewed Bernoulli, Uniform.
        for &true_mu in &[0.02, 0.5, 0.95] {
            let mut violations = 0usize;
            for _ in 0..trials {
                let mut cs = EmpiricalBernsteinCS::new(alpha).unwrap();
                let mut violated = false;
                for _ in 0..horizon {
                    let x = if rng.next_f64() < true_mu { 1.0 } else { 0.0 };
                    let b = cs.update(x);
                    if true_mu < b.lower || true_mu > b.upper {
                        violated = true;
                        break;
                    }
                }
                if violated {
                    violations += 1;
                }
            }
            let rate = (violations as f64) / (trials as f64);
            assert!(
                rate <= alpha + 0.02,
                "EB CS violated anytime coverage at mu={}: rate={} (alpha={})",
                true_mu,
                rate,
                alpha
            );
        }
    }

    #[test]
    fn test_empirical_bernstein_large_n_stability() {
        let mut cs = EmpiricalBernsteinCS::new(0.01).unwrap();
        let mut rng = Rng::from_seed(999);
        let true_mu = 0.37;
        for _ in 0..100_000 {
            let x = if rng.next_f64() < true_mu { 1.0 } else { 0.0 };
            cs.update(x);
        }
        assert!(cs.lower().is_finite() && cs.upper().is_finite());
        assert!(cs.lower() <= true_mu && true_mu <= cs.upper());
        assert!(cs.width() < 0.025, "width too wide at N=100k: {}", cs.width());
    }

    #[test]
    fn test_betting_cs_tighter_than_empirical_bernstein_and_off_grid_coverage() {
        let alpha = 0.05;
        // Off-grid true mean (between 0.500 and 0.505 on default M=201 grid).
        let true_mu = 0.5024;
        let mut rng = Rng::from_seed(777);
        let mut eb = EmpiricalBernsteinCS::new(alpha).unwrap();
        let mut bet = BettingCS::new(alpha).unwrap();

        for _ in 0..20_000 {
            let x = if rng.next_f64() < true_mu { 1.0 } else { 0.0 };
            eb.update(x);
            bet.update(x);
        }
        assert!(eb.lower() <= true_mu && true_mu <= eb.upper());
        assert!(
            bet.lower() <= true_mu && true_mu <= bet.upper(),
            "off-grid mu={} not in [{}, {}]",
            true_mu,
            bet.lower(),
            bet.upper()
        );
        // Hedged ONS BettingCS should be tighter (or very close up to 1 grid cell).
        assert!(
            bet.width() <= eb.width() + 0.005,
            "betting width {} vs EB width {}",
            bet.width(),
            eb.width()
        );
        // e_value_at true mean stays small; distant mean is rejected (>= 1 / alpha;
        // active-cell pruning freezes rejected cells at their optional-stopping value).
        assert!(bet.e_value_at(true_mu) < 1.0 / alpha);
        assert!(bet.e_value_at(0.1) >= 1.0 / alpha);
    }

    #[test]
    fn test_betting_cs_anytime_coverage_monte_carlo_including_boundaries() {
        let alpha = 0.05;
        let trials = 250;
        let horizon = 300;
        let mut rng = Rng::from_seed(2026);

        // Include exact boundaries (0.0, 1.0), extreme skewed, and off-grid means.
        for &true_mu in &[0.0, 0.017, 0.5023, 0.983, 1.0] {
            let mut violations = 0usize;
            for _ in 0..trials {
                let mut cs = BettingCS::with_grid(alpha, 101).unwrap();
                let mut violated = false;
                for _ in 0..horizon {
                    let x = if rng.next_f64() < true_mu { 1.0 } else { 0.0 };
                    let b = cs.update(x);
                    if true_mu < b.lower || true_mu > b.upper {
                        violated = true;
                        break;
                    }
                }
                if violated {
                    violations += 1;
                }
            }
            let rate = (violations as f64) / (trials as f64);
            assert!(
                rate <= alpha + 0.02,
                "BettingCS violated coverage at mu={}: rate={}",
                true_mu,
                rate
            );
        }
    }

    #[test]
    fn test_betting_cs_nan_safety_and_reset() {
        let mut cs = BettingCS::new(0.05).unwrap();
        cs.update(f64::NAN);
        cs.update(f64::INFINITY);
        assert_eq!(cs.n(), 0);
        for _ in 0..50 {
            cs.update(0.8);
        }
        assert!(cs.lower() > 0.5);
        cs.reset();
        assert_eq!(cs.n(), 0);
        assert_eq!(cs.lower(), 0.0);
        assert_eq!(cs.upper(), 1.0);
    }

    #[test]
    fn test_paired_eprocess_null_type1_error_and_power() {
        let alpha = 0.05;
        let trials = 500;
        let horizon = 400;
        let mut rng = Rng::from_seed(314159);

        // Two-sided null: E[score_a - score_b] = 0 with extreme +/-1 Rademacher differences.
        let mut false_positives = 0usize;
        for _ in 0..trials {
            let mut ep = PairedEProcess::two_sided(alpha).unwrap();
            let mut rejected = false;
            for _ in 0..horizon {
                let (a, b) = if rng.next_f64() < 0.5 {
                    (1.0, 0.0)
                } else {
                    (0.0, 1.0)
                };
                let st = ep.update(a, b);
                if st.winner.is_some() || st.anytime_p_value <= alpha {
                    rejected = true;
                    break;
                }
            }
            if rejected {
                false_positives += 1;
            }
        }
        let fpr = (false_positives as f64) / (trials as f64);
        assert!(
            fpr <= alpha + 0.02,
            "PairedEProcess two-sided FPR={} exceeded alpha={}",
            fpr,
            alpha
        );

        // Alternative: A beats B by +0.20 on average; should certify winner = Some(0).
        let mut ep = PairedEProcess::two_sided(0.01).unwrap();
        for _ in 0..500 {
            let shared = rng.next_f64() * 0.5;
            let a = (shared + if rng.next_f64() < 0.6 { 0.5 } else { 0.0 }).min(1.0);
            let b = (shared + if rng.next_f64() < 0.2 { 0.5 } else { 0.0 }).min(1.0);
            ep.update(a, b);
        }
        assert_eq!(ep.winner(), Some(0));
        assert!(ep.anytime_p_value() < 0.01);
        assert!(ep.mean_diff() > 0.1);
    }

    #[test]
    fn test_paired_eprocess_exact_shard_merge_and_ties() {
        let mut full = PairedEProcess::two_sided(0.05).unwrap();
        let mut shard_a = PairedEProcess::two_sided(0.05).unwrap();
        let mut shard_b = PairedEProcess::two_sided(0.05).unwrap();
        let mut rng = Rng::from_seed(8080);

        for i in 0..200 {
            let a = rng.next_f64();
            let b = rng.next_f64();
            full.update(a, b);
            if i < 100 {
                shard_a.update(a, b);
            } else {
                shard_b.update(a, b);
            }
        }
        shard_a.merge(&shard_b).unwrap();

        assert_eq!(full.n_pairs(), shard_a.n_pairs());
        assert!((full.log_e_a_beats_b() - shard_a.log_e_a_beats_b()).abs() < 1e-11);
        assert!((full.log_e_b_beats_a() - shard_a.log_e_b_beats_a()).abs() < 1e-11);
        assert!((full.mean_diff() - shard_a.mean_diff()).abs() < 1e-12);

        // Ties (score_a == score_b): e-values stay exactly 1.0, p-value stays 1.0.
        let mut tie = PairedEProcess::two_sided(0.05).unwrap();
        for _ in 0..1000 {
            tie.update(0.7, 0.7);
        }
        assert!((tie.e_value_a_beats_b() - 1.0).abs() < 1e-12);
        assert!((tie.e_value_b_beats_a() - 1.0).abs() < 1e-12);
        assert_eq!(tie.anytime_p_value(), 1.0);
        assert_eq!(tie.winner(), None);
    }

    #[test]
    fn test_psi_e_full_range_and_ln_table_exactness() {
        for (idx, &val) in LN_2_TO_17.iter().enumerate() {
            assert_eq!(val, ((idx + 2) as f64).ln());
        }
        let mut max_err = 0.0_f64;
        for i in 1..=10_000 {
            let lam = 0.5 * (i as f64) / 10_000.0;
            let got = psi_e(lam);
            assert!(got >= 0.0);
            if lam >= 1e-3 {
                let ref_v = -(-lam).ln_1p() - lam;
                max_err = max_err.max((got - ref_v).abs());
            }
        }
        assert!(max_err < 1e-15, "psi_e max_err on [1e-3, 0.5] = {max_err:.3e}");
        // Also verify custom cap range (0.5, 0.999999).
        for &lam in &[0.5001, 0.75, 0.95, 0.999, 0.9999995] {
            let got = psi_e(lam);
            let ref_v = (-(-lam).ln_1p() - lam).max(0.0);
            assert!((got - ref_v).abs() < 1e-14, "psi_e({lam}) err");
        }
    }

    #[test]
    fn test_eb_batch_parity_incremental_ln_and_reset() {
        let mut rng = Rng::from_seed(0xEB00_1234);
        let xs: Vec<f64> = (0..5_000).map(|_| rng.next_f64()).collect();
        let mut single = EmpiricalBernsteinCS::with_bounds(0.05, -2.0, 6.0).unwrap();
        let mut batch = EmpiricalBernsteinCS::with_bounds(0.05, -2.0, 6.0).unwrap();

        let mut last_single = single.bounds();
        for &x in &xs {
            last_single = single.update(x * 8.0 - 2.0);
        }
        let scaled: Vec<f64> = xs.iter().map(|&x| x * 8.0 - 2.0).collect();
        let last_batch = batch.update_batch(&scaled);
        assert_eq!(last_single, last_batch);
        assert!(
            (batch.ln_t_plus_1 - ((batch.n as f64) + 1.0).ln()).abs() < 1e-11,
            "incremental ln drift: {} vs {}",
            batch.ln_t_plus_1,
            ((batch.n as f64) + 1.0).ln()
        );

        batch.reset();
        assert_eq!(batch.ln_t_plus_1, 0.0);
        let after_reset = batch.update_batch(&scaled);
        assert_eq!(after_reset, last_batch);
    }

    #[test]
    fn test_betting_cs_pruning_matches_unpruned_reference_exact() {
        // Unpruned full-grid reference implementation using libm ln_1p.
        let alpha = 0.05_f64;
        let m_len = 101usize;
        let denom = (m_len - 1) as f64;
        let endpoint_cap = 0.5 * denom;
        let grid: Vec<f64> = (0..m_len)
            .map(|j| if j == 0 { 0.0 } else if j == m_len - 1 { 1.0 } else { (j as f64) / denom })
            .collect();
        let lp_max: Vec<f64> = grid.iter().map(|&m| if m > 0.0 { 0.5 / m } else { endpoint_cap }).collect();
        let lm_max: Vec<f64> = grid.iter().map(|&m| if m < 1.0 { 0.5 / (1.0 - m) } else { endpoint_cap }).collect();
        let mut lp = vec![0.0; m_len];
        let mut lm = vec![0.0; m_len];
        let mut ap = vec![1.0; m_len];
        let mut am = vec![1.0; m_len];
        let mut lkp = vec![0.0; m_len];
        let mut lkm = vec![0.0; m_len];
        let mut ref_left = 0usize;
        let mut ref_right = m_len - 2;
        let mut ref_lo = 0.0_f64;
        let mut ref_hi = 1.0_f64;
        let thresh = (1.0 / alpha).ln();

        let mut fast = BettingCS::with_grid(alpha, m_len).unwrap();
        let mut batch = BettingCS::with_grid(alpha, m_len).unwrap();
        let mut rng = Rng::from_seed(0xBEE7_5678);
        let mut vals = Vec::with_capacity(2_000);

        for _ in 0..2_000 {
            let x = if rng.next_f64() < 0.63 { 1.0 } else { 0.0 };
            vals.push(x);
            let b = fast.update(x);

            for j in 0..m_len {
                let d = x - grid[j];
                let ip = lp[j] * d;
                lkp[j] += ip.ln_1p();
                let zp = d / (1.0 + ip);
                ap[j] += zp * zp;
                lp[j] = (lp[j] + ONS_STEP_COEFF * zp / ap[j]).clamp(0.0, lp_max[j]);

                let nd = -d;
                let im = lm[j] * nd;
                lkm[j] += im.ln_1p();
                let zm = nd / (1.0 + im);
                am[j] += zm * zm;
                lm[j] = (lm[j] + ONS_STEP_COEFF * zm / am[j]).clamp(0.0, lm_max[j]);
            }
            while ref_left <= ref_right && log_mean_exp_2(lkp[ref_left + 1], lkm[ref_left]) >= thresh {
                ref_left += 1;
            }
            while ref_right >= ref_left && log_mean_exp_2(lkp[ref_right + 1], lkm[ref_right]) >= thresh {
                if ref_right == 0 {
                    ref_left = 1;
                    break;
                }
                ref_right -= 1;
            }
            if ref_left <= ref_right {
                ref_lo = ref_lo.max(grid[ref_left]);
                ref_hi = ref_hi.min(grid[ref_right + 1]);
            }
            assert_eq!(b.lower, ref_lo, "lower mismatch at n={}", b.n);
            assert_eq!(b.upper, ref_hi, "upper mismatch at n={}", b.n);
        }

        let bb = batch.update_batch(&vals);
        assert_eq!(bb, fast.bounds());
    }
}
