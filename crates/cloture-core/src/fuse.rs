//! Scale-Free Online Multi-Agent Fusion with Anytime-Valid Poisoning Quarantine (`SafeFuse`).
//!
//! Combines:
//! 1. **Scale-Free Bernstein Online Aggregation (BOA)** (Wintenberger 2017) with
//!    strictly `F_{t-1}`-predictable scale normalization and Sleeping-Hedge weight masking.
//! 2. **Per-Agent Ville E-Process Quarantine Gates** testing `H0: E[loss_j - loss_ref | F_{t-1}] <= 0`
//!    in log-space over `DEFAULT_BET_GRID`, with Bonferroni family-wise threshold `K / quarantine_alpha`
//!    and a hard invariant that at least one best surviving agent always remains active.
//! 3. **Global Safety Fallback & Recovery Gate** comparing the pre-update fused ensemble loss
//!    against a baseline (explicit `baseline_loss` or agent 0), tripping automatic baseline fallback
//!    at `1 / fallback_alpha` and recovering only when a fresh recovery E-process certifies that
//!    the fused ensemble outperforms the baseline at `1 / fallback_alpha`.

use crate::math::{log_mean_exp_grid, GRID_LEN};

/// 16-point geometric betting grid on (0, 0.80] for exact unclamped Fan-inequality
/// exponential supermartingales: `E[exp(lam * x - psi_E(lam) * x^2)] <= 1` for all `x >= -1`
/// with `E[x] <= 0` (and for all zero-mean Gaussians regardless of variance).
const FAN_GRID: [f64; GRID_LEN] = [
    0.005, 0.010, 0.020, 0.035, 0.060, 0.100, 0.150, 0.220,
    0.300, 0.380, 0.460, 0.540, 0.620, 0.680, 0.740, 0.800,
];

/// Exact precomputed `psi_E(lam) = -ln(1 - lam) - lam` for `FAN_GRID`.
const PSI_E_FAN_GRID: [f64; GRID_LEN] = [
    0.00001254182354428629,
    0.00005033585350145318,
    0.00020270731751944838,
    0.0006270922668145184,
    0.0018754415686792513,
    0.005360515657826301,
    0.012518929497774945,
    0.028461359261649358,
    0.05667494393873238,
    0.09803580094299982,
    0.15618613942381696,
    0.23652878949899635,
    0.3475840262617056,
    0.4594342831883648,
    0.6070736479666062,
    0.8094379124341003,
];

/// Output of [`SafeFuse::predict`].
#[derive(Clone, Debug)]
pub struct FusePrediction {
    /// Fused prediction (or baseline prediction when `used_fallback` is true).
    pub fused: f64,
    /// Whether the global safety gate routed to the baseline fallback.
    pub used_fallback: bool,
    /// Number of currently active (non-quarantined) agents.
    pub active_count: usize,
    /// Number of currently quarantined agents.
    pub quarantined_count: usize,
    /// Current normalized convex combination weights (zero for quarantined agents).
    pub weights: Vec<f64>,
}

/// Summary returned by [`SafeFuse::update`].
#[derive(Clone, Debug)]
pub struct FuseUpdate {
    /// Total update steps completed (1-indexed after first update).
    pub step: u64,
    /// Realized fused loss evaluated with strictly pre-update (`F_{t-1}`) weights.
    pub fused_loss: f64,
    /// Realized baseline loss used by the global safety gate.
    pub baseline_loss: f64,
    /// Strictly `F_{t-1}`-predictable scale used to normalize excess losses on this step.
    pub scale_used: f64,
    /// Indices of agents newly quarantined on this step.
    pub newly_quarantined: Vec<usize>,
    /// Indices of active agents whose reported loss was NaN or infinite on this step.
    pub flagged_nonfinite: Vec<usize>,
    /// Whether global baseline fallback is currently active after this step.
    pub fallback_active: bool,
    /// True if the global fallback gate tripped on this exact step.
    pub fallback_tripped: bool,
    /// True if the global fallback gate recovered on this exact step.
    pub fallback_recovered: bool,
    /// Current log E-value of the global degradation monitor (`log_mean_exp_grid`).
    pub global_deg_log_e: f64,
    /// Current log E-value of the global recovery monitor (`log_mean_exp_grid`).
    pub global_rec_log_e: f64,
}

/// Scale-free online multi-agent aggregator with anytime-valid poisoning quarantine
/// and global fallback/recovery safety gating.
#[derive(Clone, Debug)]
pub struct SafeFuse {
    n_agents: usize,
    quarantine_alpha: f64,
    fallback_alpha: f64,
    quarantine_log_thresh: f64,
    fallback_log_thresh: f64,
    initial_eta: f64,

    // Per-agent BOA + Sleeping-Hedge state
    weights: Vec<f64>,
    eta: Vec<f64>,
    cum_regret: Vec<f64>,
    cum_sq_regret: Vec<f64>,
    quarantined: Vec<bool>,
    agent_e_logw: Vec<[f64; GRID_LEN]>,

    // Strictly F_{t-1}-predictable scale normalization state (BOA EMA + monotone E-process scale)
    scale_ema: f64,
    scale_initialized: bool,
    e_scale: f64,
    e_scale_initialized: bool,
    ref_loss_ema: f64,
    has_ref_ema: bool,

    // Global degradation / recovery E-process safety gate
    fallback_active: bool,
    global_deg_logw: [f64; GRID_LEN],
    global_rec_logw: [f64; GRID_LEN],
    steps: u64,

    // Preallocated scratch buffers to avoid hot-path allocations
    scratch_sorted: Vec<f64>,
    scratch_peers: Vec<f64>,
    scratch_candidates: Vec<(usize, f64)>,
}

impl SafeFuse {
    /// Construct a new `SafeFuse` ensemble over `1 <= n_agents <= 100_000` agents.
    ///
    /// - `quarantine_alpha`: family-wise anytime Type-I error bound in `(0, 1)` for
    ///   per-agent poisoning quarantine (uses Bonferroni threshold `K / quarantine_alpha`).
    /// - `fallback_alpha`: anytime Type-I error bound in `(0, 1)` for global fallback
    ///   degradation tripping and subsequent recovery certification (`1 / fallback_alpha`).
    pub fn new(
        n_agents: usize,
        quarantine_alpha: f64,
        fallback_alpha: f64,
    ) -> Result<Self, &'static str> {
        if n_agents == 0 || n_agents > 100_000 {
            return Err("n_agents must be in 1..=100_000");
        }
        if !quarantine_alpha.is_finite() || quarantine_alpha <= 0.0 || quarantine_alpha >= 1.0 {
            return Err("quarantine_alpha must be finite and in (0, 1)");
        }
        if !fallback_alpha.is_finite() || fallback_alpha <= 0.0 || fallback_alpha >= 1.0 {
            return Err("fallback_alpha must be finite and in (0, 1)");
        }

        let k = n_agents as f64;
        let ln_k = (n_agents.max(2) as f64).ln();
        let initial_eta = (ln_k.sqrt()).min(0.5);

        Ok(Self {
            n_agents,
            quarantine_alpha,
            fallback_alpha,
            quarantine_log_thresh: (k / quarantine_alpha).ln(),
            fallback_log_thresh: (1.0 / fallback_alpha).ln(),
            initial_eta,

            weights: vec![1.0 / k; n_agents],
            eta: vec![initial_eta; n_agents],
            cum_regret: vec![0.0; n_agents],
            cum_sq_regret: vec![0.0; n_agents],
            quarantined: vec![false; n_agents],
            agent_e_logw: vec![[0.0; GRID_LEN]; n_agents],

            scale_ema: 1.0,
            scale_initialized: false,
            e_scale: 1.0,
            e_scale_initialized: false,
            ref_loss_ema: 0.0,
            has_ref_ema: false,

            fallback_active: false,
            global_deg_logw: [0.0; GRID_LEN],
            global_rec_logw: [0.0; GRID_LEN],
            steps: 0,

            scratch_sorted: Vec::with_capacity(n_agents + 1),
            scratch_peers: Vec::with_capacity(n_agents),
            scratch_candidates: Vec::with_capacity(n_agents),
        })
    }

    /// Number of agents in the ensemble.
    #[inline]
    pub fn n_agents(&self) -> usize {
        self.n_agents
    }

    /// Current normalized convex combination weights.
    #[inline]
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Current quarantine status flags per agent.
    #[inline]
    pub fn quarantined(&self) -> &[bool] {
        &self.quarantined
    }

    /// Boolean mask indicating which agents are currently active (not quarantined).
    pub fn active_mask(&self) -> Vec<bool> {
        self.quarantined.iter().map(|&q| !q).collect()
    }

    /// Whether global baseline fallback is currently active.
    #[inline]
    pub fn fallback_active(&self) -> bool {
        self.fallback_active
    }

    /// Number of updates performed so far.
    #[inline]
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// Current strictly `F_{t-1}`-predictable scale estimate.
    #[inline]
    pub fn scale_ema(&self) -> f64 {
        self.scale_ema
    }

    /// Current log E-value for agent `j`'s quarantine monitor (or `-inf` if out of bounds).
    #[inline]
    pub fn agent_log_e(&self, agent: usize) -> f64 {
        self.agent_e_logw
            .get(agent)
            .map(log_mean_exp_grid)
            .unwrap_or(f64::NEG_INFINITY)
    }

    /// Current log E-value of the global degradation monitor.
    #[inline]
    pub fn global_deg_log_e(&self) -> f64 {
        log_mean_exp_grid(&self.global_deg_logw)
    }

    /// Current log E-value of the global recovery monitor.
    #[inline]
    pub fn global_rec_log_e(&self) -> f64 {
        log_mean_exp_grid(&self.global_rec_logw)
    }

    /// Configured quarantine alpha.
    #[inline]
    pub fn quarantine_alpha(&self) -> f64 {
        self.quarantine_alpha
    }

    /// Configured fallback alpha.
    #[inline]
    pub fn fallback_alpha(&self) -> f64 {
        self.fallback_alpha
    }

    /// Compute the fused prediction from `agent_preds`.
    ///
    /// - Quarantined agents contribute zero weight.
    /// - Non-finite agent predictions (`NaN`/`Inf`) are safely ignored and the remaining
    ///   active weights are renormalized so a faulty agent can never corrupt `fused`.
    /// - When `fallback_active` is true, returns baseline agent 0's prediction (if finite)
    ///   with `used_fallback = true`.
    pub fn predict(&self, agent_preds: &[f64]) -> FusePrediction {
        let active_count = self.quarantined.iter().filter(|&&q| !q).count();
        let quarantined_count = self.n_agents - active_count;

        let n = self.n_agents.min(agent_preds.len());
        let mut weighted_sum = 0.0_f64;
        let mut weight_sum = 0.0_f64;
        for j in 0..n {
            if !self.quarantined[j] && agent_preds[j].is_finite() {
                let w = self.weights[j];
                weighted_sum += w * agent_preds[j];
                weight_sum += w;
            }
        }

        let ensemble_pred = if weight_sum > 0.0 {
            weighted_sum / weight_sum
        } else {
            // Fall back to first finite prediction if active finite weight collapsed.
            agent_preds
                .iter()
                .take(n)
                .copied()
                .find(|v| v.is_finite())
                .unwrap_or(0.0)
        };

        let (fused, used_fallback) = if self.fallback_active {
            if n > 0 && agent_preds[0].is_finite() {
                (agent_preds[0], true)
            } else {
                (ensemble_pred, true)
            }
        } else {
            (ensemble_pred, false)
        };

        FusePrediction {
            fused,
            used_fallback,
            active_count,
            quarantined_count,
            weights: self.weights.clone(),
        }
    }

    /// Release a quarantined agent back into the active pool on fresh probation.
    ///
    /// Resets its quarantine E-process and variance accumulator and assigns it a neutral
    /// re-entry log-potential equal to the mean active log-potential so it re-enters smoothly
    /// without disrupting existing weights.
    pub fn unquarantine(&mut self, agent: usize) {
        if agent >= self.n_agents || !self.quarantined[agent] {
            return;
        }
        // Compute mean active log-potential before unquarantining.
        let mut lw_sum = 0.0_f64;
        let mut lw_cnt = 0usize;
        for j in 0..self.n_agents {
            if !self.quarantined[j] {
                let lw = self.eta[j].ln() + self.eta[j] * self.cum_regret[j];
                if lw.is_finite() {
                    lw_sum += lw;
                    lw_cnt += 1;
                }
            }
        }

        self.quarantined[agent] = false;
        self.agent_e_logw[agent] = [0.0; GRID_LEN];
        self.cum_sq_regret[agent] = 0.0;
        self.eta[agent] = self.initial_eta;

        if lw_cnt > 0 {
            let target_lw = lw_sum / (lw_cnt as f64);
            self.cum_regret[agent] = (target_lw - self.eta[agent].ln()) / self.eta[agent];
        } else {
            self.cum_regret[agent] = 0.0;
        }

        self.recompute_weights();
    }

    /// Update BOA weights, per-agent quarantine E-processes, global safety gate,
    /// and the strictly `F_{t-1}`-predictable scale estimate.
    ///
    /// - Any agent with a non-finite (`NaN`/`Inf`) loss is flagged in `flagged_nonfinite`,
    ///   receives maximum normalized excess loss `+1.0` (regret `-1.0`), and NEVER poisons
    ///   other agents' weights or the ensemble loss calculation.
    /// - All E-process updates at step `t` use strictly `F_{t-1}`-predictable weights and
    ///   `scale_ema`; `scale_ema` is updated only at the end of the step for `t + 1`.
    pub fn update(&mut self, agent_losses: &[f64], baseline_loss: Option<f64>) -> FuseUpdate {
        self.steps += 1;
        let k = self.n_agents;
        let n = k.min(agent_losses.len());
        let explicit_base = baseline_loss.filter(|v| v.is_finite());

        // 1. Evaluate pre-update ensemble loss over active agents with finite losses.
        let mut active_finite_wsum = 0.0_f64;
        let mut active_finite_lnum = 0.0_f64;
        let mut active_finite_count = 0usize;
        let mut flagged_nonfinite = Vec::new();

        for j in 0..k {
            if self.quarantined[j] {
                continue;
            }
            let loss_opt = if j < n {
                let v = agent_losses[j];
                if v.is_finite() {
                    Some(v)
                } else {
                    None
                }
            } else {
                None
            };
            match loss_opt {
                Some(loss) => {
                    active_finite_wsum += self.weights[j];
                    active_finite_lnum += self.weights[j] * loss;
                    active_finite_count += 1;
                }
                None => {
                    flagged_nonfinite.push(j);
                }
            }
        }

        let fused_loss = if active_finite_wsum > 0.0 {
            active_finite_lnum / active_finite_wsum
        } else if active_finite_count > 0 {
            // Equal average over active finite agents if weights were all zero on them.
            let mut s = 0.0_f64;
            for j in 0..n {
                if !self.quarantined[j] && agent_losses[j].is_finite() {
                    s += agent_losses[j];
                }
            }
            s / (active_finite_count as f64)
        } else {
            explicit_base.unwrap_or(0.0)
        };

        let global_base_opt = explicit_base.or_else(|| {
            if n > 0 && agent_losses[0].is_finite() {
                Some(agent_losses[0])
            } else {
                None
            }
        });
        let baseline_used = global_base_opt.unwrap_or(fused_loss);

        // 2. Strictly F_{t-1}-predictable scales for this step.
        let scale_used = self.scale_ema.max(1e-8);
        let e_scale_used = self.e_scale.max(1e-12);
        let can_bet_finite = self.e_scale_initialized;

        // Helper for exact unclamped Fan-inequality exponential increment:
        //   Delta log W(lam) = lam * x - psi_E(lam) * x^2
        // Satisfies E[exp(Delta log W)] <= 1 for ALL x >= -1 with E[x] <= 0
        // (and for all zero-mean Gaussians regardless of variance) with ZERO clamping!
        #[inline]
        fn apply_fan_step(logw: &mut [f64; GRID_LEN], x: f64) {
            if x == 0.0 || !x.is_finite() {
                return;
            }
            let x2 = x * x;
            for g in 0..GRID_LEN {
                let inc = FAN_GRID[g] * x - PSI_E_FAN_GRID[g] * x2;
                if inc.is_finite() {
                    logw[g] += inc;
                } else {
                    logw[g] = f64::NEG_INFINITY;
                }
            }
        }

        // 3. Per-agent Ville E-process updates (strictly F_{t-1}-predictable weights & scale).
        for j in 0..k {
            if self.quarantined[j] {
                continue;
            }
            let is_finite = j < n && agent_losses[j].is_finite();
            if !is_finite {
                // Non-finite loss: positive unit excess x = +1.0 regardless of scale.
                apply_fan_step(&mut self.agent_e_logw[j], 1.0);
                continue;
            }
            if !can_bet_finite {
                // First non-degenerate step establishes e_scale for strict scale-free validity.
                continue;
            }

            let loss_j = agent_losses[j];
            let loss_ref_opt = if let Some(b) = explicit_base {
                Some(b)
            } else if active_finite_count >= 2 {
                // Leave-one-out weighted peer mean: E[loss_j - mean_peer] <= 0 for ANY equal-mean null.
                let rem_w = active_finite_wsum - self.weights[j];
                let loo_mean = if rem_w > 1e-15 {
                    (active_finite_lnum - self.weights[j] * loss_j) / rem_w
                } else {
                    let mut sum_other = 0.0_f64;
                    let mut cnt_other = 0usize;
                    for m in 0..n {
                        if m != j && !self.quarantined[m] && agent_losses[m].is_finite() {
                            sum_other += agent_losses[m];
                            cnt_other += 1;
                        }
                    }
                    if cnt_other > 0 {
                        sum_other / (cnt_other as f64)
                    } else {
                        loss_j
                    }
                };
                // Take max(loo_weighted_mean, loo_median) when >= 3 active finite agents:
                // Pointwise >= loo_mean preserves E[loss_j - loss_ref] <= 0 under ALL equal-mean
                // nulls (including skewed discrete Bern(0.1)) while preventing a lone spiky peer
                // (e.g. 20*Bern(0.05)) from dragging clean peers' reference downward.
                if active_finite_count >= 3 {
                    self.scratch_peers.clear();
                    for m in 0..n {
                        if m != j && !self.quarantined[m] && agent_losses[m].is_finite() {
                            self.scratch_peers.push(agent_losses[m]);
                        }
                    }
                    let pm = self.scratch_peers.len();
                    if pm >= 2 {
                        self.scratch_peers
                            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                        let loo_med = if pm % 2 == 1 {
                            self.scratch_peers[pm / 2]
                        } else {
                            0.5 * (self.scratch_peers[pm / 2 - 1] + self.scratch_peers[pm / 2])
                        };
                        Some(loo_mean.max(loo_med))
                    } else {
                        Some(loo_mean)
                    }
                } else {
                    Some(loo_mean)
                }
            } else {
                None
            };

            if let Some(loss_ref) = loss_ref_opt {
                let x = (loss_j - loss_ref) / e_scale_used;
                apply_fan_step(&mut self.agent_e_logw[j], x);
            }
        }

        // 4. Global safety gate E-process (degradation vs recovery, unclamped Fan supermartingale).
        let mut fallback_tripped = false;
        let mut fallback_recovered = false;
        if can_bet_finite && active_finite_count > 0 {
            if let Some(base_loss) = global_base_opt {
                let x_deg = (fused_loss - base_loss) / e_scale_used;
                if !self.fallback_active {
                    apply_fan_step(&mut self.global_deg_logw, x_deg);
                    if log_mean_exp_grid(&self.global_deg_logw) >= self.fallback_log_thresh {
                        self.fallback_active = true;
                        fallback_tripped = true;
                        self.global_rec_logw = [0.0; GRID_LEN];
                    }
                } else {
                    let x_rec = -x_deg;
                    apply_fan_step(&mut self.global_rec_logw, x_rec);
                    if log_mean_exp_grid(&self.global_rec_logw) >= self.fallback_log_thresh {
                        self.fallback_active = false;
                        fallback_recovered = true;
                        self.global_deg_logw = [0.0; GRID_LEN];
                    }
                }
            }
        }

        // 5. Quarantine check: quarantine agents crossing ln(K / quarantine_alpha),
        //    ordered from strongest degradation evidence first, ALWAYS keeping >= 1 active agent.
        self.scratch_candidates.clear();
        let mut active_count = 0usize;
        for j in 0..k {
            if !self.quarantined[j] {
                active_count += 1;
                let log_e = log_mean_exp_grid(&self.agent_e_logw[j]);
                if log_e >= self.quarantine_log_thresh {
                    self.scratch_candidates.push((j, log_e));
                }
            }
        }

        let mut newly_quarantined = Vec::new();
        if !self.scratch_candidates.is_empty() && active_count > 1 {
            self.scratch_candidates
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            for &(j, _) in &self.scratch_candidates {
                if active_count <= 1 {
                    break;
                }
                self.quarantined[j] = true;
                self.weights[j] = 0.0;
                active_count -= 1;
                newly_quarantined.push(j);
            }
        }

        // 6. Scale-Free BOA + Sleeping-Hedge weight update for next step (t + 1).
        //    Determine BOA scale (uses scale_used when initialized, else current finite spread).
        let obs_scale = self.compute_obs_scale(agent_losses, explicit_base, fused_loss);
        let boa_scale = if self.scale_initialized {
            scale_used
        } else if obs_scale > 1e-12 {
            obs_scale
        } else {
            1.0
        };

        let ln_k = (k.max(2) as f64).ln();
        if active_finite_count > 0 || !flagged_nonfinite.is_empty() {
            for j in 0..k {
                if self.quarantined[j] {
                    // Sleeping expert: zero regret drift while quarantined.
                    continue;
                }
                let r_norm = if j < n && agent_losses[j].is_finite() {
                    if active_finite_count > 0 {
                        ((fused_loss - agent_losses[j]) / boa_scale).clamp(-1.0, 1.0)
                    } else {
                        0.0
                    }
                } else {
                    // Non-finite loss: cap normalized excess loss at +1.0 => regret = -1.0.
                    -1.0
                };

                let eta_prev = self.eta[j];
                // Wintenberger (2017) BOA second-order penalized regret increment.
                self.cum_regret[j] += r_norm - eta_prev * r_norm * r_norm;
                self.cum_sq_regret[j] += r_norm * r_norm;
                self.eta[j] = (ln_k / (1.0 + self.cum_sq_regret[j])).sqrt().min(0.5);
            }
        }

        self.recompute_weights();

        // 7. Update strictly F_{t-1}-predictable scales AFTER step t's inference is complete.
        self.update_scale_ema(obs_scale, baseline_used, active_finite_count > 0);
        let mut step_bound = 0.0_f64;
        let mut min_v = f64::INFINITY;
        let mut max_v = f64::NEG_INFINITY;
        for j in 0..n {
            if !self.quarantined[j] && agent_losses[j].is_finite() {
                let v = agent_losses[j];
                if v.abs() > step_bound {
                    step_bound = v.abs();
                }
                if v < min_v {
                    min_v = v;
                }
                if v > max_v {
                    max_v = v;
                }
            }
        }
        if let Some(b) = explicit_base {
            if b.abs() > step_bound {
                step_bound = b.abs();
            }
            if b < min_v {
                min_v = b;
            }
            if b > max_v {
                max_v = b;
            }
        }
        if min_v.is_finite() && max_v.is_finite() {
            let spread = max_v - min_v;
            if spread > step_bound {
                step_bound = spread;
            }
        }
        if step_bound > 1e-12 && step_bound.is_finite() {
            if !self.e_scale_initialized {
                self.e_scale = step_bound;
                self.e_scale_initialized = true;
            } else if step_bound > self.e_scale {
                self.e_scale = step_bound;
            }
        }

        FuseUpdate {
            step: self.steps,
            fused_loss,
            baseline_loss: baseline_used,
            scale_used: if can_bet_finite { e_scale_used } else { scale_used },
            newly_quarantined,
            flagged_nonfinite,
            fallback_active: self.fallback_active,
            fallback_tripped,
            fallback_recovered,
            global_deg_log_e: log_mean_exp_grid(&self.global_deg_logw),
            global_rec_log_e: log_mean_exp_grid(&self.global_rec_logw),
        }
    }

    /// Recompute normalized convex weights over active agents in log-space (overflow-free).
    fn recompute_weights(&mut self) {
        let mut max_lw = f64::NEG_INFINITY;
        let mut active_cnt = 0usize;
        for j in 0..self.n_agents {
            if self.quarantined[j] {
                self.weights[j] = 0.0;
                continue;
            }
            active_cnt += 1;
            let lw = self.eta[j].ln() + self.eta[j] * self.cum_regret[j];
            if lw > max_lw {
                max_lw = lw;
            }
        }

        if active_cnt == 0 {
            return;
        }
        if !max_lw.is_finite() {
            let uniform = 1.0 / (active_cnt as f64);
            for j in 0..self.n_agents {
                self.weights[j] = if self.quarantined[j] { 0.0 } else { uniform };
            }
            return;
        }

        let mut sum_exp = 0.0_f64;
        for j in 0..self.n_agents {
            if self.quarantined[j] {
                self.weights[j] = 0.0;
                continue;
            }
            let lw = self.eta[j].ln() + self.eta[j] * self.cum_regret[j];
            let d = lw - max_lw;
            let w = if d > -745.0 { d.exp() } else { 0.0 };
            self.weights[j] = w;
            sum_exp += w;
        }

        if sum_exp > 0.0 && sum_exp.is_finite() {
            let inv = 1.0 / sum_exp;
            for j in 0..self.n_agents {
                self.weights[j] *= inv;
            }
        } else {
            let uniform = 1.0 / (active_cnt as f64);
            for j in 0..self.n_agents {
                self.weights[j] = if self.quarantined[j] { 0.0 } else { uniform };
            }
        }
    }

    /// Compute a robust observed loss spread for updating `scale_ema` for future steps.
    fn compute_obs_scale(
        &mut self,
        agent_losses: &[f64],
        explicit_base: Option<f64>,
        ref_loss: f64,
    ) -> f64 {
        self.scratch_sorted.clear();
        let n = self.n_agents.min(agent_losses.len());
        for j in 0..n {
            if !self.quarantined[j] && agent_losses[j].is_finite() {
                self.scratch_sorted.push(agent_losses[j]);
            }
        }
        if let Some(b) = explicit_base {
            self.scratch_sorted.push(b);
        }

        let mut obs_scale = 0.0_f64;
        let m = self.scratch_sorted.len();
        if m >= 2 {
            self.scratch_sorted
                .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let full_spread = self.scratch_sorted[m - 1] - self.scratch_sorted[0];
            // When >= 3 finite values exist, trim the single largest value so a lone
            // poisoned outlier spike cannot blow up scale_ema; fall back to full_spread
            // only if the clean agents have zero noise (trimmed_spread == 0).
            let trimmed_spread = if m >= 3 {
                self.scratch_sorted[m - 2] - self.scratch_sorted[0]
            } else {
                full_spread
            };
            obs_scale = if trimmed_spread > 1e-12 {
                trimmed_spread
            } else {
                full_spread
            };
        }

        if self.has_ref_ema && ref_loss.is_finite() {
            let temp_diff = (ref_loss - self.ref_loss_ema).abs();
            if temp_diff > obs_scale {
                obs_scale = temp_diff;
            }
        }

        obs_scale
    }

    /// Update `scale_ema` strictly after step `t` has completed.
    fn update_scale_ema(&mut self, obs_scale: f64, ref_loss: f64, has_finite: bool) {
        if has_finite && ref_loss.is_finite() {
            if !self.has_ref_ema {
                self.ref_loss_ema = ref_loss;
                self.has_ref_ema = true;
            } else {
                self.ref_loss_ema = 0.9 * self.ref_loss_ema + 0.1 * ref_loss;
            }
        }

        if !obs_scale.is_finite() || obs_scale <= 1e-12 {
            return;
        }

        if !self.scale_initialized {
            self.scale_ema = obs_scale;
            self.scale_initialized = true;
        } else if obs_scale > self.scale_ema {
            // Fast upward tracking with a per-step cap to resist sudden outlier spikes.
            let capped = obs_scale.min(self.scale_ema * 4.0);
            self.scale_ema = 0.5 * self.scale_ema + 0.5 * capped;
        } else {
            // Gentle downward decay so scale_ema acts as a stable envelope for bounded losses.
            self.scale_ema = 0.98 * self.scale_ema + 0.02 * obs_scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Rng;

    fn normal(rng: &mut Rng) -> f64 {
        let u1 = rng.next_f64().max(1e-12);
        let u2 = rng.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn exp_sample(rng: &mut Rng, rate: f64) -> f64 {
        let u = (1.0 - rng.next_f64()).max(1e-12);
        -u.ln() / rate
    }

    #[test]
    fn constructor_validation() {
        assert!(SafeFuse::new(0, 0.05, 0.05).is_err());
        assert!(SafeFuse::new(3, 0.0, 0.05).is_err());
        assert!(SafeFuse::new(3, 1.0, 0.05).is_err());
        assert!(SafeFuse::new(3, f64::NAN, 0.05).is_err());
        assert!(SafeFuse::new(3, 0.05, 0.0).is_err());
        assert!(SafeFuse::new(3, 0.05, 1.5).is_err());

        let fuse = SafeFuse::new(4, 0.05, 0.05).unwrap();
        assert_eq!(fuse.n_agents(), 4);
        assert_eq!(fuse.active_mask(), vec![true; 4]);
        let sum_w: f64 = fuse.weights().iter().sum();
        assert!((sum_w - 1.0).abs() < 1e-12);
    }

    #[test]
    fn poisoning_quarantines_bad_agent_and_preserves_clean() {
        let mut rng = Rng::from_seed(42);
        let mut fuse = SafeFuse::new(5, 0.01, 0.05).unwrap();

        let mut quarantined_at = None;
        for t in 0..200 {
            let mut losses = [0.0; 5];
            for j in 0..4 {
                losses[j] = (0.2 + 0.05 * normal(&mut rng)).clamp(0.0, 1.0);
            }
            // Agent 4 is persistently poisoned.
            losses[4] = (0.85 + 0.05 * normal(&mut rng)).clamp(0.0, 1.0);

            let upd = fuse.update(&losses, None);
            if upd.newly_quarantined.contains(&4) {
                quarantined_at = Some(t);
            }
        }

        assert!(
            quarantined_at.is_some(),
            "poisoned agent 4 should be quarantined"
        );
        assert!(
            quarantined_at.unwrap() < 60,
            "poisoned agent should be quarantined quickly, got step {:?}",
            quarantined_at
        );
        assert!(fuse.quarantined()[4]);
        assert_eq!(fuse.weights()[4], 0.0);
        for j in 0..4 {
            assert!(!fuse.quarantined()[j], "clean agent {j} should stay active");
        }
        let wsum: f64 = fuse.weights().iter().sum();
        assert!((wsum - 1.0).abs() < 1e-12);

        // Prediction must completely ignore quarantined agent 4 even with adversarial output.
        let pred = fuse.predict(&[1.0, 1.0, 1.0, 1.0, 1e9]);
        assert!((pred.fused - 1.0).abs() < 1e-12);
        assert_eq!(pred.active_count, 4);
        assert_eq!(pred.quarantined_count, 1);
    }

    #[test]
    fn nan_and_inf_losses_never_poison_clean_weights() {
        let mut fuse = SafeFuse::new(4, 0.05, 0.05).unwrap();

        for step in 0..40 {
            let bad = if step % 2 == 0 {
                f64::NAN
            } else {
                f64::INFINITY
            };
            let upd = fuse.update(&[0.2, 0.25, 0.22, bad], None);
            if !fuse.quarantined()[3] {
                assert!(upd.flagged_nonfinite.contains(&3));
            }
            for &w in fuse.weights() {
                assert!(w.is_finite() && w >= 0.0);
            }
            let sum_w: f64 = fuse.weights().iter().sum();
            assert!((sum_w - 1.0).abs() < 1e-12);
        }

        assert!(
            fuse.quarantined()[3],
            "NaN/Inf agent must be quarantined automatically"
        );
        assert_eq!(fuse.weights()[3], 0.0);

        // Even with NaN in predictions, predict() must return a clean finite result.
        let pred = fuse.predict(&[10.0, 10.0, f64::NAN, f64::INFINITY]);
        assert!(pred.fused.is_finite());
        assert!((pred.fused - 10.0).abs() < 1e-12);
    }

    #[test]
    fn keeps_at_least_one_best_agent_alive_when_all_degraded() {
        let mut fuse = SafeFuse::new(4, 0.05, 0.05).unwrap();
        for _ in 0..120 {
            // All agents worse than explicit baseline 0.1, with agent 1 least bad.
            fuse.update(&[0.9, 0.4, 0.85, 0.95], Some(0.1));
        }

        let active: Vec<usize> = fuse
            .active_mask()
            .into_iter()
            .enumerate()
            .filter_map(|(i, a)| a.then_some(i))
            .collect();
        assert_eq!(active.len(), 1, "must keep exactly 1 surviving agent");
        assert_eq!(
            active[0], 1,
            "best surviving agent (agent 1) should be the one kept alive"
        );
        assert!((fuse.weights()[1] - 1.0).abs() < 1e-12);
        assert!(
            fuse.fallback_active(),
            "global fallback must be tripped when all agents are worse than baseline"
        );
    }

    #[test]
    fn global_fallback_trips_and_recovers() {
        let mut fuse = SafeFuse::new(3, 0.01, 0.05).unwrap();

        // Phase 1: all agents worse than explicit baseline => global fallback trips.
        let mut tripped = false;
        for _ in 0..80 {
            let upd = fuse.update(&[0.8, 0.85, 0.82], Some(0.2));
            if upd.fallback_tripped {
                tripped = true;
            }
        }
        assert!(tripped, "global fallback should trip during degradation");
        assert!(fuse.fallback_active());

        let pred_fb = fuse.predict(&[42.0, 100.0, 100.0]);
        assert!(pred_fb.used_fallback);
        assert_eq!(pred_fb.fused, 42.0);

        // Phase 2: ensemble agents improve and strictly beat baseline => recovers!
        let mut recovered = false;
        for _ in 0..120 {
            let upd = fuse.update(&[0.1, 0.1, 0.1], Some(0.8));
            if upd.fallback_recovered {
                recovered = true;
            }
        }
        assert!(recovered, "global fallback should recover once ensemble beats baseline");
        assert!(!fuse.fallback_active());

        let pred_ok = fuse.predict(&[42.0, 10.0, 10.0]);
        assert!(!pred_ok.used_fallback);
        assert!(pred_ok.fused.is_finite());
    }

    #[test]
    fn unquarantine_restores_agent_smoothly() {
        let mut fuse = SafeFuse::new(3, 0.05, 0.05).unwrap();
        for _ in 0..60 {
            fuse.update(&[0.1, 0.1, 0.9], None);
        }
        assert!(fuse.quarantined()[2]);
        assert_eq!(fuse.weights()[2], 0.0);

        fuse.unquarantine(2);
        assert!(!fuse.quarantined()[2]);
        assert!(fuse.weights()[2] > 0.05, "unquarantined agent should receive positive weight");
        let sum_w: f64 = fuse.weights().iter().sum();
        assert!((sum_w - 1.0).abs() < 1e-12);

        // If agent 2 now behaves cleanly, it stays active.
        for _ in 0..40 {
            fuse.update(&[0.2, 0.2, 0.15], None);
        }
        assert!(!fuse.quarantined()[2]);
    }

    #[test]
    fn scale_invariance_across_six_orders_of_magnitude() {
        let mut rng = Rng::from_seed(777);
        let mut base_losses = Vec::new();
        for _ in 0..80 {
            base_losses.push([
                0.3 + 0.1 * normal(&mut rng),
                0.3 + 0.1 * normal(&mut rng),
                0.6 + 0.1 * normal(&mut rng),
            ]);
        }

        let mut f_unit = SafeFuse::new(3, 0.05, 0.05).unwrap();
        let mut f_big = SafeFuse::new(3, 0.05, 0.05).unwrap();
        let mut f_small = SafeFuse::new(3, 0.05, 0.05).unwrap();

        for row in &base_losses {
            f_unit.update(row, None);
            let big = [row[0] * 1e4, row[1] * 1e4, row[2] * 1e4];
            let small = [row[0] * 1e-2, row[1] * 1e-2, row[2] * 1e-2];
            f_big.update(&big, None);
            f_small.update(&small, None);
        }

        for j in 0..3 {
            assert!(
                (f_unit.weights()[j] - f_big.weights()[j]).abs() < 1e-9,
                "weights must be scale-invariant"
            );
            assert!(
                (f_unit.weights()[j] - f_small.weights()[j]).abs() < 1e-9,
                "weights must be scale-invariant"
            );
            assert_eq!(f_unit.quarantined()[j], f_big.quarantined()[j]);
            assert_eq!(f_unit.quarantined()[j], f_small.quarantined()[j]);
        }
    }

    #[test]
    fn monte_carlo_type1_error_under_symmetric_skewed_and_discrete_nulls() {
        let alpha = 0.05;
        let trials = 300;
        let horizon = 120;

        // 1. Symmetric bounded null: Uniform(0, 1)
        let mut rng = Rng::from_seed(1001);
        let mut false_quarantines = 0usize;
        for _ in 0..trials {
            let mut fuse = SafeFuse::new(4, alpha, alpha).unwrap();
            let mut any_q = false;
            for _ in 0..horizon {
                let losses = [
                    rng.next_f64(),
                    rng.next_f64(),
                    rng.next_f64(),
                    rng.next_f64(),
                ];
                let upd = fuse.update(&losses, None);
                if !upd.newly_quarantined.is_empty() {
                    any_q = true;
                    break;
                }
            }
            if any_q {
                false_quarantines += 1;
            }
        }
        let rate_unif = false_quarantines as f64 / trials as f64;
        assert!(
            rate_unif <= alpha + 0.03,
            "Uniform null false-quarantine rate {rate_unif:.4} exceeded tolerance for alpha={alpha}"
        );

        // 2. Right-skewed null: Exponential(1.0) (tests unbiasedness of LOO peer reference!)
        let mut rng_exp = Rng::from_seed(2002);
        let mut false_q_exp = 0usize;
        for _ in 0..trials {
            let mut fuse = SafeFuse::new(4, alpha, alpha).unwrap();
            let mut any_q = false;
            for _ in 0..horizon {
                let losses = [
                    exp_sample(&mut rng_exp, 1.0),
                    exp_sample(&mut rng_exp, 1.0),
                    exp_sample(&mut rng_exp, 1.0),
                    exp_sample(&mut rng_exp, 1.0),
                ];
                let upd = fuse.update(&losses, None);
                if !upd.newly_quarantined.is_empty() {
                    any_q = true;
                    break;
                }
            }
            if any_q {
                false_q_exp += 1;
            }
        }
        let rate_exp = false_q_exp as f64 / trials as f64;
        assert!(
            rate_exp <= alpha + 0.03,
            "Exponential skewed null false-quarantine rate {rate_exp:.4} exceeded tolerance"
        );

        // 3. Discrete asymmetric null: Bernoulli(0.1)
        let mut rng_bern = Rng::from_seed(3003);
        let mut false_q_bern = 0usize;
        for _ in 0..trials {
            let mut fuse = SafeFuse::new(4, alpha, alpha).unwrap();
            let mut any_q = false;
            for _ in 0..horizon {
                let losses = [
                    if rng_bern.next_f64() < 0.1 { 1.0 } else { 0.0 },
                    if rng_bern.next_f64() < 0.1 { 1.0 } else { 0.0 },
                    if rng_bern.next_f64() < 0.1 { 1.0 } else { 0.0 },
                    if rng_bern.next_f64() < 0.1 { 1.0 } else { 0.0 },
                ];
                let upd = fuse.update(&losses, None);
                if !upd.newly_quarantined.is_empty() {
                    any_q = true;
                    break;
                }
            }
            if any_q {
                false_q_bern += 1;
            }
        }
        let rate_bern = false_q_bern as f64 / trials as f64;
        assert!(
            rate_bern <= alpha + 0.03,
            "Bernoulli(0.1) null false-quarantine rate {rate_bern:.4} exceeded tolerance"
        );
    }

    #[test]
    fn edge_cases_single_agent_zero_variance_and_long_horizon() {
        // K = 1
        let mut f1 = SafeFuse::new(1, 0.05, 0.05).unwrap();
        let u = f1.update(&[0.5], Some(0.5));
        assert_eq!(u.step, 1);
        assert!(!f1.quarantined()[0]);
        assert!((f1.predict(&[7.5]).fused - 7.5).abs() < 1e-12);

        // Zero-variance constant losses
        let mut fz = SafeFuse::new(3, 0.05, 0.05).unwrap();
        for _ in 0..50 {
            fz.update(&[1.0, 1.0, 1.0], None);
        }
        assert!(fz.active_mask().iter().all(|&a| a));
        for &w in fz.weights() {
            assert!((w - 1.0 / 3.0).abs() < 1e-12);
        }

        // Long horizon (N = 100_000): verify numerical stability (no NaN/Inf/overflow).
        let mut fl = SafeFuse::new(3, 0.01, 0.01).unwrap();
        for t in 0..100_000u64 {
            let jitter = ((t % 7) as f64) * 0.01;
            fl.update(&[0.1 + jitter, 0.12 + jitter, 0.15 + jitter], Some(0.12 + jitter));
        }
        for &w in fl.weights() {
            assert!(w.is_finite() && w >= 0.0);
        }
        assert!(fl.global_deg_log_e().is_finite());
        assert!(fl.predict(&[1.0, 2.0, 3.0]).fused.is_finite());
    }
}
