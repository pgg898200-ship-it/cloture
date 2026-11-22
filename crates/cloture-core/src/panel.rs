//! Correlated-panel diagnostics, spectral effective panel size, stratified
//! permutation testing, Condorcet null auditing, and greedy judge pruning.
//!
//! Implements batch auditing ([`PanelAudit`] -> [`PanelReport`]) and zero-copy
//! streaming health monitoring ([`OnlinePanelTracker`]) for multi-judge AI
//! evaluation panels.

use std::collections::HashMap;

use crate::math::{symmetric_eigenvalues, Rng};

/// Configuration parameters for [`PanelAudit::run_with_config`].
#[derive(Clone, Debug)]
pub struct PanelAuditConfig {
    /// Number of Monte Carlo draws for the item-aware stratified Condorcet null model.
    pub n_condorcet_sims: usize,
    /// Number of permutations for the stratified omnibus error-correlation test.
    pub n_permutations: usize,
    /// Number of item-level bootstrap resamples for the 95% CI on `n_eff_kish`.
    pub n_bootstrap: usize,
    /// Fraction of full-panel `n_eff_kish` targeted by greedy subset pruning (default `0.90`).
    pub subset_target_ratio: f64,
    /// Deterministic PRNG seed for Monte Carlo, permutation, and bootstrap routines.
    pub seed: u64,
}

impl Default for PanelAuditConfig {
    fn default() -> Self {
        Self {
            n_condorcet_sims: 2_000,
            n_permutations: 2_000,
            n_bootstrap: 1_000,
            subset_target_ratio: 0.90,
            seed: 0x5EED_C0DE_2026_0529,
        }
    }
}

/// Leave-one-out impact of dropping a single judge from the panel.
#[derive(Clone, Debug)]
pub struct LooJudgeImpact {
    /// Zero-based index of the dropped judge.
    pub judge_idx: usize,
    /// Majority/plurality panel accuracy after dropping `judge_idx`.
    pub accuracy_without: f64,
    /// Change in panel accuracy on removal: `accuracy_without - panel_accuracy`.
    pub delta_accuracy: f64,
    /// Kish effective panel size after dropping `judge_idx`.
    pub n_eff_without: f64,
    /// Change in Kish effective panel size on removal: `n_eff_without - n_eff_kish`.
    pub delta_n_eff: f64,
    /// True if dropping `judge_idx` improves accuracy, improves `n_eff`, or preserves both.
    pub redundant: bool,
}

/// Full diagnostic report produced by [`PanelAudit`].
#[derive(Clone, Debug)]
pub struct PanelReport {
    /// Number of judges `K`.
    pub k_judges: usize,
    /// Number of evaluated items `N`.
    pub n_items: usize,
    /// Individual accuracy of each judge.
    pub judge_accuracies: Vec<f64>,
    /// Plurality-vote accuracy of the full panel (uniform expected tie-breaking).
    pub panel_accuracy: f64,
    /// Highest single-judge accuracy.
    pub best_single_accuracy: f64,
    /// Index of the best single judge.
    pub best_single_index: usize,
    /// Panel lift over best single judge (`panel_accuracy - best_single_accuracy`).
    pub panel_lift: f64,
    /// Row-major `k_judges x k_judges` pairwise Pearson/phi error correlation matrix.
    pub phi_matrix: Vec<f64>,
    /// Mean pairwise off-diagonal error correlation.
    pub mean_phi: f64,
    /// Standard deviation of off-diagonal pairwise error correlations.
    pub std_phi: f64,
    /// Kish effective panel size `K / (1 + (K - 1) * max(mean_phi, 0))`.
    pub n_eff_kish: f64,
    /// Unclamped Kish effective panel size `K / (1 + (K - 1) * mean_phi)` (may exceed `K` when `mean_phi < 0`).
    pub n_eff_kish_raw: f64,
    /// Spectral effective panel size `K / lambda_max`.
    pub n_eff_eigen: f64,
    /// Largest eigenvalue of `phi_matrix`.
    pub lambda_max: f64,
    /// Spectral entropy effective rank `exp(-sum_i (lam_i / S) ln(lam_i / S))`.
    pub effective_rank: f64,
    /// Independence ratio `n_eff_kish / K` in `(0, 1]`.
    pub independence_ratio: f64,
    /// Asymptotic Kish effective-size ceiling `1 / mean_phi` (or `INFINITY` if `mean_phi <= 1e-9`).
    pub asymptotic_ceiling: f64,
    /// Exact nominal Krippendorff's alpha across judges' discrete votes.
    pub krippendorff_alpha: f64,
    /// Fraction of items on which all judges cast identical votes.
    pub unanimous_rate: f64,
    /// Accuracy conditioned on unanimous items (`0.0` if no items are unanimous).
    pub unanimous_accuracy: f64,
    /// Fraction of items on which every judge erred.
    pub all_wrong_rate: f64,
    /// Predicted panel accuracy under the item-aware stratified independent Condorcet null.
    pub condorcet_predicted_accuracy: f64,
    /// Accuracy shortfall relative to Condorcet independence (`condorcet_predicted_accuracy - panel_accuracy`).
    pub condorcet_gap: f64,
    /// One-sided stratified permutation omnibus p-value for excess positive error correlation.
    pub permutation_p_value: f64,
    /// Lower bound of the item-bootstrap 95% CI on `n_eff_kish`.
    pub n_eff_ci_low: f64,
    /// Upper bound of the item-bootstrap 95% CI on `n_eff_kish`.
    pub n_eff_ci_high: f64,
    /// Per-judge leave-one-out impact and redundancy diagnostics.
    pub loo_analysis: Vec<LooJudgeImpact>,
    /// Greedy subset of judge indices capturing `>= subset_target_ratio` of full-panel `n_eff_kish`.
    pub optimal_subset: Vec<usize>,
}

/// Batch correlated-panel auditor.
#[derive(Clone, Debug)]
pub struct PanelAudit {
    n_items: usize,
    k_judges: usize,
    /// Row-major `n_items x k_judges` discrete judge votes.
    votes: Vec<u32>,
    /// Length `n_items` gold labels.
    gold: Vec<u32>,
    /// Row-major `n_items x k_judges` binary error indicators (`1` = error, `0` = correct).
    errors: Vec<u8>,
    /// Compacted zero-indexed stratum ID per item (`0..n_strata`).
    strata: Vec<usize>,
    n_strata: usize,
}

impl PanelAudit {
    /// Constructs a `PanelAudit` from a row-major `n_items x k_judges` discrete vote matrix,
    /// `gold` labels of length `n_items`, and optional difficulty `strata` of length `n_items`.
    pub fn from_votes(
        votes: &[u32],
        gold: &[u32],
        n_items: usize,
        k_judges: usize,
        strata: Option<&[usize]>,
    ) -> Result<Self, &'static str> {
        if n_items == 0 {
            return Err("n_items must be >= 1");
        }
        if k_judges == 0 {
            return Err("k_judges must be >= 1");
        }
        let expected_len = n_items
            .checked_mul(k_judges)
            .ok_or("n_items * k_judges overflows usize")?;
        if votes.len() != expected_len {
            return Err("votes length must equal n_items * k_judges");
        }
        if gold.len() != n_items {
            return Err("gold length must equal n_items");
        }
        let (compacted_strata, n_strata) = compact_strata(strata, n_items)?;

        let mut errors = vec![0u8; expected_len];
        for i in 0..n_items {
            let g = gold[i];
            let row = i * k_judges;
            for j in 0..k_judges {
                if votes[row + j] != g {
                    errors[row + j] = 1;
                }
            }
        }

        Ok(Self {
            n_items,
            k_judges,
            votes: votes.to_vec(),
            gold: gold.to_vec(),
            errors,
            strata: compacted_strata,
            n_strata,
        })
    }

    /// Constructs a `PanelAudit` directly from a row-major `n_items x k_judges` binary error matrix
    /// (`0` = correct, non-zero = error).
    pub fn from_binary_errors(
        errors: &[u8],
        n_items: usize,
        k_judges: usize,
        strata: Option<&[usize]>,
    ) -> Result<Self, &'static str> {
        if n_items == 0 {
            return Err("n_items must be >= 1");
        }
        if k_judges == 0 {
            return Err("k_judges must be >= 1");
        }
        let expected_len = n_items
            .checked_mul(k_judges)
            .ok_or("n_items * k_judges overflows usize")?;
        if errors.len() != expected_len {
            return Err("errors length must equal n_items * k_judges");
        }
        let (compacted_strata, n_strata) = compact_strata(strata, n_items)?;

        let mut norm_errors = vec![0u8; expected_len];
        let mut votes = vec![1u32; expected_len];
        for idx in 0..expected_len {
            if errors[idx] != 0 {
                norm_errors[idx] = 1;
                votes[idx] = 0;
            }
        }

        Ok(Self {
            n_items,
            k_judges,
            votes,
            gold: vec![1u32; n_items],
            errors: norm_errors,
            strata: compacted_strata,
            n_strata,
        })
    }

    /// Constructs a `PanelAudit` from a row-major `n_items x k_judges` binary correctness matrix
    /// (non-zero = correct, `0` = error).
    pub fn from_binary_correctness(
        correct: &[u8],
        n_items: usize,
        k_judges: usize,
        strata: Option<&[usize]>,
    ) -> Result<Self, &'static str> {
        if n_items == 0 {
            return Err("n_items must be >= 1");
        }
        if k_judges == 0 {
            return Err("k_judges must be >= 1");
        }
        let expected_len = n_items
            .checked_mul(k_judges)
            .ok_or("n_items * k_judges overflows usize")?;
        if correct.len() != expected_len {
            return Err("correct length must equal n_items * k_judges");
        }
        let mut errors = vec![0u8; expected_len];
        for idx in 0..expected_len {
            if correct[idx] == 0 {
                errors[idx] = 1;
            }
        }
        Self::from_binary_errors(&errors, n_items, k_judges, strata)
    }

    /// Runs the full diagnostic suite with default configuration (`PanelAuditConfig::default()`).
    pub fn run(&self) -> PanelReport {
        self.run_with_config(&PanelAuditConfig::default())
    }

    /// Runs the full diagnostic suite with custom configuration.
    pub fn run_with_config(&self, cfg: &PanelAuditConfig) -> PanelReport {
        let n = self.n_items;
        let k = self.k_judges;
        let n_f = n as f64;
        let k_f = k as f64;

        // 1. Sufficient error counts, individual accuracies, unanimity, and majority accuracy.
        let mut sum_e = vec![0u64; k];
        let mut sum_ee = vec![0u64; k * k];
        let mut err_offsets = Vec::with_capacity(n + 1);
        let mut err_judges = Vec::new();
        err_offsets.push(0usize);

        let mut panel_correct_sum = 0.0_f64;
        let mut unanimous_count = 0u64;
        let mut unanimous_correct = 0u64;
        let mut all_wrong_count = 0u64;
        let mut vote_scratch = vec![0u32; k];

        for i in 0..n {
            let row = i * k;
            let g = self.gold[i];
            let first_v = self.votes[row];
            let mut is_unanimous = true;
            let mut err_cnt_item = 0usize;

            for j in 0..k {
                let v = self.votes[row + j];
                vote_scratch[j] = v;
                if v != first_v {
                    is_unanimous = false;
                }
                if self.errors[row + j] != 0 {
                    err_cnt_item += 1;
                    sum_e[j] += 1;
                    err_judges.push(j);
                }
            }
            err_offsets.push(err_judges.len());

            // Update pairwise co-error counts sparsely over judges that erred on item i.
            let item_errs = &err_judges[err_offsets[i]..err_offsets[i + 1]];
            for idx_a in 0..item_errs.len() {
                let a = item_errs[idx_a];
                sum_ee[a * k + a] += 1;
                for idx_b in (idx_a + 1)..item_errs.len() {
                    let b = item_errs[idx_b];
                    sum_ee[a * k + b] += 1;
                    sum_ee[b * k + a] += 1;
                }
            }

            if is_unanimous {
                unanimous_count += 1;
                if first_v == g {
                    unanimous_correct += 1;
                }
            }
            if err_cnt_item == k {
                all_wrong_count += 1;
            }
            panel_correct_sum += plurality_credit(&mut vote_scratch, g);
        }

        let judge_accuracies: Vec<f64> = sum_e.iter().map(|&e| 1.0 - (e as f64) / n_f).collect();
        let mut best_single_index = 0usize;
        let mut best_single_accuracy = judge_accuracies[0];
        for (j, &acc) in judge_accuracies.iter().enumerate().skip(1) {
            if acc > best_single_accuracy {
                best_single_accuracy = acc;
                best_single_index = j;
            }
        }
        let panel_accuracy = panel_correct_sum / n_f;
        let panel_lift = panel_accuracy - best_single_accuracy;
        let unanimous_rate = (unanimous_count as f64) / n_f;
        let unanimous_accuracy = if unanimous_count > 0 {
            (unanimous_correct as f64) / (unanimous_count as f64)
        } else {
            0.0
        };
        let all_wrong_rate = (all_wrong_count as f64) / n_f;

        // 2. Pairwise phi matrix and Kish / spectral effective panel size metrics.
        let (phi_matrix, mean_phi, std_phi) = phi_from_counts(k, n as u64, &sum_e, &sum_ee);
        let (n_eff_kish, n_eff_kish_raw) = kish_from_mean_phi(k, mean_phi);
        let (n_eff_eigen, lambda_max, effective_rank) = spectral_metrics(&phi_matrix, k);
        let independence_ratio = n_eff_kish / k_f;
        let asymptotic_ceiling = if mean_phi > 1e-9 {
            1.0 / mean_phi
        } else {
            f64::INFINITY
        };

        // 3. Exact nominal Krippendorff's alpha.
        let krippendorff_alpha = compute_krippendorff_alpha(&self.votes, n, k);

        // 4. Item-aware stratified Condorcet null model (exact binary DP + L1-cached multiclass).
        let condorcet_predicted_accuracy = if cfg.n_condorcet_sims > 0 {
            self.run_condorcet_null(cfg.n_condorcet_sims, cfg.seed)
        } else {
            panel_accuracy
        };
        let condorcet_gap = condorcet_predicted_accuracy - panel_accuracy;

        // 5. Fast exact stratified permutation omnibus test.
        let permutation_p_value = self.run_stratified_permutation_test(
            &sum_e,
            cfg.n_permutations,
            cfg.seed,
        );

        // 6. Item-level bootstrap 95% CI on n_eff_kish.
        let (n_eff_ci_low, n_eff_ci_high) = self.run_bootstrap_neff_ci(
            &err_offsets,
            &err_judges,
            n_eff_kish,
            cfg.n_bootstrap,
            cfg.seed,
        );

        // 7. Leave-one-out judge analysis.
        let loo_analysis = self.run_loo_analysis(panel_accuracy, n_eff_kish, &phi_matrix);

        // 8. Greedy pruning subset capturing >= target ratio of full-panel n_eff_kish.
        let optimal_subset = select_optimal_subset(
            k,
            best_single_index,
            &judge_accuracies,
            &phi_matrix,
            n_eff_kish,
            cfg.subset_target_ratio,
        );

        PanelReport {
            k_judges: k,
            n_items: n,
            judge_accuracies,
            panel_accuracy,
            best_single_accuracy,
            best_single_index,
            panel_lift,
            phi_matrix,
            mean_phi,
            std_phi,
            n_eff_kish,
            n_eff_kish_raw,
            n_eff_eigen,
            lambda_max,
            effective_rank,
            independence_ratio,
            asymptotic_ceiling,
            krippendorff_alpha,
            unanimous_rate,
            unanimous_accuracy,
            all_wrong_rate,
            condorcet_predicted_accuracy,
            condorcet_gap,
            permutation_p_value,
            n_eff_ci_low,
            n_eff_ci_high,
            loo_analysis,
            optimal_subset,
        }
    }

    fn run_condorcet_null(&self, n_sims: usize, seed: u64) -> f64 {
        let n = self.n_items;
        let k = self.k_judges;

        // Group items by (stratum, gold_label) in deterministic first-appearance order.
        let mut bucket_map: HashMap<(usize, u32), usize> = HashMap::new();
        let mut bucket_keys: Vec<(usize, u32)> = Vec::new();
        let mut buckets: Vec<Vec<usize>> = Vec::new();
        let mut stratum_items: Vec<Vec<usize>> = vec![Vec::new(); self.n_strata];

        for i in 0..n {
            let s = self.strata[i];
            let g = self.gold[i];
            stratum_items[s].push(i);
            let b_idx = if let Some(&idx) = bucket_map.get(&(s, g)) {
                idx
            } else {
                let idx = buckets.len();
                bucket_map.insert((s, g), idx);
                bucket_keys.push((s, g));
                buckets.push(Vec::new());
                idx
            };
            buckets[b_idx].push(i);
        }

        // Classify each bucket into:
        // 1. ExactBinary: <= 2 distinct labels (including gold) OR gold never voted -> O(K^2) exact DP.
        // 2. Multiclass (C <= 16): L1-cached per-judge C-entry cumulative frequency table, O(K) zero-sort.
        // 3. General (C > 16 or denom > u32::MAX): fallback to exact general plurality_credit.
        let mut exact_weighted_acc = 0.0_f64;
        let mut sim_buckets: Vec<CondorcetSimBucket> = Vec::new();
        let mut counts_buf = vec![0u32; k * 16];
        let mut probs_buf = vec![0.0_f64; k];

        for (b_idx, bucket) in buckets.iter().enumerate() {
            let (s, g) = bucket_keys[b_idx];
            let weight = bucket.len();
            let singleton = weight < 2;
            let sample_slice: &[usize] = if singleton {
                &stratum_items[s]
            } else {
                bucket.as_slice()
            };
            let denom = sample_slice.len();

            if denom <= i32::MAX as usize {
                let mut distinct = [0u32; 16];
                distinct[0] = g;
                let mut n_labels: usize = 1;
                let mut overflow = false;
                for v in &mut counts_buf {
                    *v = 0;
                }

                'scan: for &u in sample_slice {
                    let row = u * k;
                    for j in 0..k {
                        let v = if singleton {
                            if self.errors[row + j] == 0 {
                                g
                            } else {
                                let raw = self.votes[row + j];
                                if raw != g { raw } else { g.wrapping_add(1) }
                            }
                        } else {
                            self.votes[row + j]
                        };
                        let mut found = None;
                        for (idx, &lbl) in distinct[..n_labels].iter().enumerate() {
                            if lbl == v {
                                found = Some(idx);
                                break;
                            }
                        }
                        if let Some(idx) = found {
                            counts_buf[j * 16 + idx] += 1;
                        } else if n_labels < 16 {
                            let idx = n_labels;
                            distinct[idx] = v;
                            n_labels += 1;
                            counts_buf[j * 16 + idx] += 1;
                        } else {
                            overflow = true;
                            break 'scan;
                        }
                    }
                }

                if !overflow {
                    let mut any_gold = false;
                    for j in 0..k {
                        if counts_buf[j * 16] > 0 {
                            any_gold = true;
                            break;
                        }
                    }
                    if !any_gold {
                        // Gold has zero empirical probability for every judge -> exact accuracy 0.0.
                        continue;
                    }
                    if n_labels <= 2 {
                        // Binary exact Poisson-Binomial DP (zero Monte Carlo variance).
                        let inv_denom = 1.0 / (denom as f64);
                        for j in 0..k {
                            probs_buf[j] = (counts_buf[j * 16] as f64) * inv_denom;
                        }
                        let acc = poisson_binomial_majority_accuracy(&probs_buf);
                        exact_weighted_acc += (weight as f64) * acc;
                        continue;
                    }

                    // Multiclass fast path (3 <= C <= 16): pre-scale exact cumulative frequency
                    // thresholds for multiply-free exact draws.
                    let denom_u32 = denom as u32;
                    if (n_labels == 3 || n_labels == 4) && denom <= 512 && k <= i16::MAX as usize {
                        // Exact 16-bit signed-biased thresholds in [-32768, -32768 + scale*denom]:
                        // 8 draws fit in one 128-bit SIMD register using only 2 u64 RNG steps.
                        let scale = 65_535u32 / denom_u32;
                        let limit = (-32_768_i32 + (scale * denom_u32) as i32) as i16;
                        if n_labels == 3 {
                            let mut cdfs = Vec::with_capacity(k);
                            for j in 0..k {
                                let base = j * 16;
                                let c0 = counts_buf[base];
                                let c1 = c0 + counts_buf[base + 1];
                                let t0 = (-32_768_i32 + (c0 * scale) as i32) as i16;
                                let t1 = (-32_768_i32 + (c1 * scale) as i32) as i16;
                                cdfs.push((t0, t1));
                            }
                            sim_buckets.push(CondorcetSimBucket::C3I16 {
                                weight,
                                limit,
                                cdfs,
                            });
                        } else {
                            let mut cdfs = Vec::with_capacity(k);
                            for j in 0..k {
                                let base = j * 16;
                                let c0 = counts_buf[base];
                                let c1 = c0 + counts_buf[base + 1];
                                let c2 = c1 + counts_buf[base + 2];
                                let t0 = (-32_768_i32 + (c0 * scale) as i32) as i16;
                                let t1 = (-32_768_i32 + (c1 * scale) as i32) as i16;
                                let t2 = (-32_768_i32 + (c2 * scale) as i32) as i16;
                                cdfs.push((t0, t1, t2));
                            }
                            sim_buckets.push(CondorcetSimBucket::C4I16 {
                                weight,
                                limit,
                                cdfs,
                            });
                        }
                        continue;
                    }

                    let scale = (i32::MAX as u32) / denom_u32;
                    let limit = (scale * denom_u32) as i32;
                    if n_labels == 3 {
                        let mut cdfs = Vec::with_capacity(k);
                        for j in 0..k {
                            let base = j * 16;
                            let c0 = counts_buf[base];
                            let c1 = c0 + counts_buf[base + 1];
                            cdfs.push(((c0 * scale) as i32, (c1 * scale) as i32));
                        }
                        sim_buckets.push(CondorcetSimBucket::C3 {
                            weight,
                            limit,
                            cdfs,
                        });
                    } else if n_labels == 4 {
                        let mut cdfs = Vec::with_capacity(k);
                        for j in 0..k {
                            let base = j * 16;
                            let c0 = counts_buf[base];
                            let c1 = c0 + counts_buf[base + 1];
                            let c2 = c1 + counts_buf[base + 2];
                            cdfs.push((
                                (c0 * scale) as i32,
                                (c1 * scale) as i32,
                                (c2 * scale) as i32,
                            ));
                        }
                        sim_buckets.push(CondorcetSimBucket::C4 {
                            weight,
                            limit,
                            cdfs,
                        });
                    } else {
                        let mut cdfs = Vec::with_capacity(k);
                        for j in 0..k {
                            let base = j * 16;
                            let mut row_cdf = [0i32; 16];
                            let mut run = 0u32;
                            for c in 0..n_labels {
                                run += counts_buf[base + c];
                                row_cdf[c] = (run * scale) as i32;
                            }
                            cdfs.push(row_cdf);
                        }
                        sim_buckets.push(CondorcetSimBucket::CSmall {
                            weight,
                            limit,
                            n_labels,
                            cdfs,
                        });
                    }
                    continue;
                }
            }

            // General fallback for buckets with > 16 distinct labels.
            sim_buckets.push(CondorcetSimBucket::General {
                weight,
                gold: g,
                singleton,
                sample_items: sample_slice.to_vec(),
            });
        }

        if sim_buckets.is_empty() {
            return exact_weighted_acc / (n as f64);
        }

        // Parallelize multiclass/general simulation across deterministic chunks when n_sims >= 256.
        let votes_ref = &self.votes;
        let errors_ref = &self.errors;
        let sim_buckets_ref = &sim_buckets;
        let chunk_sums = run_parallel_chunks(n_sims, |chunk_idx, chunk_sims| {
            if chunk_sims == 0 {
                return 0.0_f64;
            }
            let mut rng2 = Rng2::from_seed(seed, DOMAIN_CONDORCET, chunk_idx);
            let mut rng_fb = Rng::from_seed(derive_sub_seed(
                seed,
                DOMAIN_CONDORCET_FB,
                chunk_idx,
            ));
            let mut sim_votes: Vec<u32> = Vec::new();
            let mut chunk_credit = 0.0_f64;

            for b in sim_buckets_ref {
                match b {
                    CondorcetSimBucket::C3I16 {
                        weight,
                        limit,
                        cdfs,
                    } => {
                        let n_draws = chunk_sims * (*weight);
                        let mut cred_x6 = 0u64;
                        let blocks = n_draws / 8;
                        for _ in 0..blocks {
                            cred_x6 +=
                                draw_c3_i16_batch8_credit_x6(cdfs, *limit, &mut rng2, &mut rng_fb);
                        }
                        for _ in 0..(n_draws & 7) {
                            cred_x6 += draw_c3_i16_credit_x6(cdfs, *limit, &mut rng_fb);
                        }
                        chunk_credit += (cred_x6 as f64) * (1.0 / 6.0);
                    }
                    CondorcetSimBucket::C4I16 {
                        weight,
                        limit,
                        cdfs,
                    } => {
                        let n_draws = chunk_sims * (*weight);
                        let mut cred_x12 = 0u64;
                        let blocks = n_draws / 8;
                        for _ in 0..blocks {
                            cred_x12 += draw_c4_i16_batch8_credit_x12(
                                cdfs,
                                *limit,
                                &mut rng2,
                                &mut rng_fb,
                            );
                        }
                        for _ in 0..(n_draws & 7) {
                            cred_x12 += draw_c4_i16_credit_x12(cdfs, *limit, &mut rng_fb);
                        }
                        chunk_credit += (cred_x12 as f64) * (1.0 / 12.0);
                    }
                    CondorcetSimBucket::C3 {
                        weight,
                        limit,
                        cdfs,
                    } => {
                        let n_draws = chunk_sims * (*weight);
                        let mut cred_x6 = 0u64;
                        let blocks = n_draws / 4;
                        for _ in 0..blocks {
                            cred_x6 +=
                                draw_c3_batch4_credit_x6(cdfs, *limit, &mut rng2, &mut rng_fb);
                        }
                        for _ in 0..(n_draws & 3) {
                            cred_x6 += draw_c3_credit_x6(cdfs, *limit, &mut rng_fb);
                        }
                        chunk_credit += (cred_x6 as f64) * (1.0 / 6.0);
                    }
                    CondorcetSimBucket::C4 {
                        weight,
                        limit,
                        cdfs,
                    } => {
                        let n_draws = chunk_sims * (*weight);
                        let mut cred_x12 = 0u64;
                        let blocks = n_draws / 4;
                        for _ in 0..blocks {
                            cred_x12 +=
                                draw_c4_batch4_credit_x12(cdfs, *limit, &mut rng2, &mut rng_fb);
                        }
                        for _ in 0..(n_draws & 3) {
                            cred_x12 += draw_c4_credit_x12(cdfs, *limit, &mut rng_fb);
                        }
                        chunk_credit += (cred_x12 as f64) * (1.0 / 12.0);
                    }
                    CondorcetSimBucket::CSmall {
                        weight,
                        limit,
                        n_labels,
                        cdfs,
                    } => {
                        let n_draws = chunk_sims * (*weight);
                        let mut acc = 0.0_f64;
                        let blocks = n_draws / 4;
                        for _ in 0..blocks {
                            acc += draw_csmall_batch4_credit(
                                cdfs,
                                *n_labels,
                                *limit,
                                &mut rng2,
                                &mut rng_fb,
                            );
                        }
                        for _ in 0..(n_draws & 3) {
                            acc += draw_csmall_credit(cdfs, *n_labels, *limit, &mut rng_fb);
                        }
                        chunk_credit += acc;
                    }
                    CondorcetSimBucket::General {
                        weight,
                        gold,
                        singleton,
                        sample_items,
                    } => {
                        if sim_votes.len() != k {
                            sim_votes.resize(k, 0);
                        }
                        let slen = sample_items.len();
                        let n_draws = chunk_sims * (*weight);
                        let g = *gold;
                        let mut acc = 0.0_f64;
                        for _ in 0..n_draws {
                            if *singleton {
                                for j in 0..k {
                                    let u = sample_items[rng_fb.gen_range_usize(slen)];
                                    let idx = u * k + j;
                                    if errors_ref[idx] == 0 {
                                        sim_votes[j] = g;
                                    } else {
                                        let v = votes_ref[idx];
                                        sim_votes[j] = if v != g { v } else { g.wrapping_add(1) };
                                    }
                                }
                            } else {
                                for j in 0..k {
                                    let u = sample_items[rng_fb.gen_range_usize(slen)];
                                    sim_votes[j] = votes_ref[u * k + j];
                                }
                            }
                            acc += plurality_credit(&mut sim_votes, g);
                        }
                        chunk_credit += acc;
                    }
                }
            }
            chunk_credit
        });

        let mut sim_credit_total = 0.0_f64;
        for s in chunk_sums {
            sim_credit_total += s;
        }
        (exact_weighted_acc + sim_credit_total / (n_sims as f64)) / (n as f64)
    }

    /// Exact stratified omnibus permutation test for positive excess pairwise error correlation.
    ///
    /// Exploits the exact algebraic identity:
    /// `sum_{a < b} phi_{a,b} = (1 / (2N)) * sum_i (sum_j w_j e_{i,j})^2 - const`
    /// where `w_j = 1 / sqrt(p_j (1 - p_j))` is invariant under every within-stratum
    /// permutation of each judge's error vector. Uses direct in-place Fisher-Yates
    /// shuffling on the accumulator (exact multiset invariance of `sum_i acc[i]^2`),
    /// free first-judge initialization, and deterministic multi-core chunking.
    fn run_stratified_permutation_test(&self, sum_e: &[u64], n_perms: usize, seed: u64) -> f64 {
        let n = self.n_items;
        let k = self.k_judges;
        if k < 2 || n < 2 || n_perms == 0 {
            return 1.0;
        }

        let n_f = n as f64;
        let mut weights = vec![0.0_f64; k];
        let mut active_judges = 0usize;
        for j in 0..k {
            let p = (sum_e[j] as f64) / n_f;
            let var = p * (1.0 - p);
            if var > 1e-15 {
                weights[j] = 1.0 / var.sqrt();
                active_judges += 1;
            }
        }
        if active_judges < 2 {
            return 1.0;
        }

        // Observed statistic S_sq = sum_i (sum_j w_j * e_{i,j})^2.
        let mut s_sq_obs = 0.0_f64;
        let mut stratum_sizes = vec![0usize; self.n_strata];
        let mut stratum_err_counts = vec![0usize; self.n_strata * k];

        for i in 0..n {
            let s = self.strata[i];
            stratum_sizes[s] += 1;
            let row = i * k;
            let mut si = 0.0_f64;
            for j in 0..k {
                if self.errors[row + j] != 0 {
                    stratum_err_counts[s * k + j] += 1;
                    si += weights[j];
                }
            }
            s_sq_obs += si * si;
        }

        // Precompute constant strata contributions and dynamic partial Fisher-Yates tasks.
        let mut const_s_sq = 0.0_f64;
        let mut dyn_strata: Vec<(usize, f64, Vec<(usize, f64)>)> = Vec::new();
        let mut max_m_s = 0usize;

        for s in 0..self.n_strata {
            let m_s = stratum_sizes[s];
            if m_s == 0 {
                continue;
            }
            let mut base = 0.0_f64;
            let mut tasks: Vec<(usize, f64)> = Vec::new();
            for j in 0..k {
                let w = weights[j];
                if w == 0.0 {
                    continue;
                }
                let c = stratum_err_counts[s * k + j];
                if c == 0 {
                    continue;
                } else if c == m_s {
                    base += w;
                } else if c <= m_s / 2 {
                    tasks.push((c, w));
                } else {
                    base += w;
                    tasks.push((m_s - c, -w));
                }
            }
            if tasks.is_empty() {
                const_s_sq += (m_s as f64) * base * base;
            } else if tasks.len() == 1 {
                let (r0, dw0) = tasks[0];
                let v1 = base + dw0;
                const_s_sq += (r0 as f64) * v1 * v1 + ((m_s - r0) as f64) * base * base;
            } else {
                // Sort descending by r so the largest partial shuffle becomes the free judge-0 init.
                tasks.sort_by(|a, b| b.0.cmp(&a.0));
                if m_s > max_m_s {
                    max_m_s = m_s;
                }
                dyn_strata.push((m_s, base, tasks));
            }
        }

        let tol = 1e-11 * s_sq_obs.abs().max(1.0);
        let threshold = s_sq_obs - tol;

        if dyn_strata.is_empty() {
            let ge_count = if const_s_sq >= threshold { n_perms } else { 0 };
            return ((1 + ge_count) as f64) / ((1 + n_perms) as f64);
        }

        let dyn_strata_ref = &dyn_strata;
        let chunk_counts = run_parallel_chunks(n_perms, |chunk_idx, chunk_perms| {
            if chunk_perms == 0 {
                return 0usize;
            }
            let mut rng = Rng::from_seed(derive_sub_seed(seed, DOMAIN_PERM, chunk_idx));
            let mut acc = vec![0.0_f64; max_m_s];
            let mut ge_count = 0usize;

            for _ in 0..chunk_perms {
                let mut s_sq_perm = const_s_sq;
                for (m_s, base, tasks) in dyn_strata_ref {
                    let m_s = *m_s;
                    let base = *base;
                    // Task 0 initializes acc in O(m_s) with zero RNG/swaps (exact by symmetry).
                    let (r0, dw0) = tasks[0];
                    let v_hi = base + dw0;
                    acc[..r0].fill(v_hi);
                    acc[r0..m_s].fill(base);

                    // Remaining tasks apply in-place partial Fisher-Yates directly on acc.
                    if m_s <= u32::MAX as usize {
                        for &(r, dw) in &tasks[1..] {
                            let mut t = 0usize;
                            while t + 1 < r {
                                let w = rng.next_u64();
                                let b0 = (m_s - t) as u32;
                                let p0 = t + fast_lemire_u32(w as u32, b0, &mut rng) as usize;
                                let val0 = acc[p0];
                                acc[p0] = acc[t];
                                acc[t] = val0 + dw;

                                let t1 = t + 1;
                                let b1 = (m_s - t1) as u32;
                                let p1 =
                                    t1 + fast_lemire_u32((w >> 32) as u32, b1, &mut rng) as usize;
                                let val1 = acc[p1];
                                acc[p1] = acc[t1];
                                acc[t1] = val1 + dw;

                                t += 2;
                            }
                            if t < r {
                                let b0 = (m_s - t) as u32;
                                let p0 =
                                    t + fast_lemire_u32(rng.next_u64() as u32, b0, &mut rng) as usize;
                                let val0 = acc[p0];
                                acc[p0] = acc[t];
                                acc[t] = val0 + dw;
                            }
                        }
                    } else {
                        for &(r, dw) in &tasks[1..] {
                            for t in 0..r {
                                let pick = t + rng.gen_range_usize(m_s - t);
                                let val = acc[pick];
                                acc[pick] = acc[t];
                                acc[t] = val + dw;
                            }
                        }
                    }

                    // 4-way unrolled sum of squares for auto-vectorized FMA.
                    let mut s0 = 0.0_f64;
                    let mut s1 = 0.0_f64;
                    let mut s2 = 0.0_f64;
                    let mut s3 = 0.0_f64;
                    let mut idx = 0usize;
                    let end4 = m_s & !3;
                    while idx < end4 {
                        let a0 = acc[idx];
                        let a1 = acc[idx + 1];
                        let a2 = acc[idx + 2];
                        let a3 = acc[idx + 3];
                        s0 += a0 * a0;
                        s1 += a1 * a1;
                        s2 += a2 * a2;
                        s3 += a3 * a3;
                        idx += 4;
                    }
                    let mut tail = 0.0_f64;
                    while idx < m_s {
                        let a = acc[idx];
                        tail += a * a;
                        idx += 1;
                    }
                    s_sq_perm += (s0 + s1) + (s2 + s3) + tail;
                }
                if s_sq_perm >= threshold {
                    ge_count += 1;
                }
            }
            ge_count
        });

        let ge_total: usize = chunk_counts.into_iter().sum();
        ((1 + ge_total) as f64) / ((1 + n_perms) as f64)
    }

    fn run_bootstrap_neff_ci(
        &self,
        err_offsets: &[usize],
        err_judges: &[usize],
        point_neff: f64,
        n_boot: usize,
        seed: u64,
    ) -> (f64, f64) {
        let n = self.n_items;
        let k = self.k_judges;
        if n_boot == 0 || k < 2 || n < 2 {
            return (point_neff, point_neff);
        }

        let n_f = n as f64;
        let n_pairs_f = ((k * (k - 1)) / 2) as f64;

        let mut boot_vals: Vec<f64> = if k <= 64 {
            // Fast bitmask-deduplicated pattern path for K <= 64:
            // collapses identical error patterns and skips all-correct items (mask == 0).
            let mut pat_map: HashMap<u64, u32> = HashMap::new();
            let mut pat_masks: Vec<u64> = Vec::new();
            let mut item_pat: Vec<u32> = Vec::with_capacity(n);

            for i in 0..n {
                let row = i * k;
                let mut mask = 0u64;
                for j in 0..k {
                    if self.errors[row + j] != 0 {
                        mask |= 1u64 << j;
                    }
                }
                if mask == 0 {
                    item_pat.push(u32::MAX); // patched to sentinel n_nonzero below
                } else if let Some(&pid) = pat_map.get(&mask) {
                    item_pat.push(pid);
                } else {
                    let pid = pat_masks.len() as u32;
                    pat_map.insert(mask, pid);
                    pat_masks.push(mask);
                    item_pat.push(pid);
                }
            }

            let n_nonzero = pat_masks.len();
            if n_nonzero == 0 {
                // Zero errors across all items and judges -> constant n_eff == K.
                return (k as f64, k as f64);
            }
            let sentinel = n_nonzero as u32;
            for pid in &mut item_pat {
                if *pid == u32::MAX {
                    *pid = sentinel;
                }
            }

            let pat_masks_ref = &pat_masks;
            let item_pat_ref = &item_pat;
            let use_u32 = n <= u32::MAX as usize;
            let n_u32 = n as u32;
            let thresh = if use_u32 { n_u32.wrapping_neg() % n_u32 } else { 0 };

            let chunks = run_parallel_chunks(n_boot, |chunk_idx, chunk_boot| {
                let mut rng = Rng::from_seed(derive_sub_seed(seed, DOMAIN_BOOT, chunk_idx));
                let mut pat_mult = vec![0u32; n_nonzero + 1];
                let mut sum_e = vec![0u64; k];
                let mut sum_ee = vec![0u64; k * k];
                let mut p_buf = vec![0.0_f64; k];
                let mut sd_buf = vec![0.0_f64; k];
                let mut local = Vec::with_capacity(chunk_boot);

                for _ in 0..chunk_boot {
                    pat_mult.fill(0);
                    if use_u32 {
                        let pairs = n / 2;
                        for _ in 0..pairs {
                            let w = rng.next_u64();
                            let i0 = fast_range_u32_pre(w as u32, n_u32, thresh, &mut rng) as usize;
                            let i1 =
                                fast_range_u32_pre((w >> 32) as u32, n_u32, thresh, &mut rng)
                                    as usize;
                            pat_mult[item_pat_ref[i0] as usize] += 1;
                            pat_mult[item_pat_ref[i1] as usize] += 1;
                        }
                        if (n & 1) != 0 {
                            let i0 = fast_range_u32_pre(
                                rng.next_u64() as u32,
                                n_u32,
                                thresh,
                                &mut rng,
                            ) as usize;
                            pat_mult[item_pat_ref[i0] as usize] += 1;
                        }
                    } else {
                        for _ in 0..n {
                            let i0 = rng.gen_range_usize(n);
                            pat_mult[item_pat_ref[i0] as usize] += 1;
                        }
                    }

                    sum_e.fill(0);
                    sum_ee.fill(0);
                    for p in 0..n_nonzero {
                        let m = pat_mult[p] as u64;
                        if m == 0 {
                            continue;
                        }
                        let mut mask = pat_masks_ref[p];
                        while mask != 0 {
                            let a = mask.trailing_zeros() as usize;
                            mask &= mask - 1;
                            sum_e[a] += m;
                            let row = a * k;
                            let mut inner = mask;
                            while inner != 0 {
                                let b = inner.trailing_zeros() as usize;
                                inner &= inner - 1;
                                sum_ee[row + b] += m;
                            }
                        }
                    }

                    for j in 0..k {
                        let pj = (sum_e[j] as f64) / n_f;
                        p_buf[j] = pj;
                        sd_buf[j] = (pj * (1.0 - pj)).max(0.0).sqrt();
                    }
                    let mut phi_sum = 0.0_f64;
                    for a in 0..k {
                        let sa = sd_buf[a];
                        if sa <= 1e-15 {
                            continue;
                        }
                        let pa = p_buf[a];
                        let row = a * k;
                        for b in (a + 1)..k {
                            let sb = sd_buf[b];
                            if sb <= 1e-15 {
                                continue;
                            }
                            let pab = (sum_ee[row + b] as f64) / n_f;
                            let phi = ((pab - pa * p_buf[b]) / (sa * sb)).clamp(-1.0, 1.0);
                            phi_sum += phi;
                        }
                    }
                    let mean_phi_b = phi_sum / n_pairs_f;
                    let (neff_b, _) = kish_from_mean_phi(k, mean_phi_b);
                    local.push(neff_b);
                }
                local
            });

            let mut all = Vec::with_capacity(n_boot);
            for mut c in chunks {
                all.append(&mut c);
            }
            all
        } else {
            // General sparse path for K > 64 judges.
            let use_u32 = n <= u32::MAX as usize;
            let n_u32 = n as u32;
            let thresh = if use_u32 { n_u32.wrapping_neg() % n_u32 } else { 0 };

            let chunks = run_parallel_chunks(n_boot, |chunk_idx, chunk_boot| {
                let mut rng = Rng::from_seed(derive_sub_seed(seed, DOMAIN_BOOT, chunk_idx));
                let mut mult = vec![0u64; n];
                let mut sum_e = vec![0u64; k];
                let mut sum_ee = vec![0u64; k * k];
                let mut p_buf = vec![0.0_f64; k];
                let mut sd_buf = vec![0.0_f64; k];
                let mut local = Vec::with_capacity(chunk_boot);

                for _ in 0..chunk_boot {
                    mult.fill(0);
                    if use_u32 {
                        let pairs = n / 2;
                        for _ in 0..pairs {
                            let w = rng.next_u64();
                            let i0 = fast_range_u32_pre(w as u32, n_u32, thresh, &mut rng) as usize;
                            let i1 =
                                fast_range_u32_pre((w >> 32) as u32, n_u32, thresh, &mut rng)
                                    as usize;
                            mult[i0] += 1;
                            mult[i1] += 1;
                        }
                        if (n & 1) != 0 {
                            let i0 = fast_range_u32_pre(
                                rng.next_u64() as u32,
                                n_u32,
                                thresh,
                                &mut rng,
                            ) as usize;
                            mult[i0] += 1;
                        }
                    } else {
                        for _ in 0..n {
                            mult[rng.gen_range_usize(n)] += 1;
                        }
                    }
                    sum_e.fill(0);
                    sum_ee.fill(0);
                    for i in 0..n {
                        let m = mult[i];
                        if m == 0 {
                            continue;
                        }
                        let slice = &err_judges[err_offsets[i]..err_offsets[i + 1]];
                        for idx_a in 0..slice.len() {
                            let a = slice[idx_a];
                            sum_e[a] += m;
                            let row = a * k;
                            for idx_b in (idx_a + 1)..slice.len() {
                                let b = slice[idx_b];
                                sum_ee[row + b] += m;
                            }
                        }
                    }
                    for j in 0..k {
                        let pj = (sum_e[j] as f64) / n_f;
                        p_buf[j] = pj;
                        sd_buf[j] = (pj * (1.0 - pj)).max(0.0).sqrt();
                    }
                    let mut phi_sum = 0.0_f64;
                    for a in 0..k {
                        let sa = sd_buf[a];
                        if sa <= 1e-15 {
                            continue;
                        }
                        let pa = p_buf[a];
                        let row = a * k;
                        for b in (a + 1)..k {
                            let sb = sd_buf[b];
                            if sb <= 1e-15 {
                                continue;
                            }
                            let pab = (sum_ee[row + b] as f64) / n_f;
                            let phi = ((pab - pa * p_buf[b]) / (sa * sb)).clamp(-1.0, 1.0);
                            phi_sum += phi;
                        }
                    }
                    let mean_phi_b = phi_sum / n_pairs_f;
                    let (neff_b, _) = kish_from_mean_phi(k, mean_phi_b);
                    local.push(neff_b);
                }
                local
            });

            let mut all = Vec::with_capacity(n_boot);
            for mut c in chunks {
                all.append(&mut c);
            }
            all
        };

        boot_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let lo = quantile_sorted(&boot_vals, 0.025).clamp(1.0, k as f64);
        let hi = quantile_sorted(&boot_vals, 0.975).clamp(lo, k as f64);
        (lo, hi)
    }

    fn run_loo_analysis(
        &self,
        panel_accuracy: f64,
        n_eff_kish: f64,
        phi_matrix: &[f64],
    ) -> Vec<LooJudgeImpact> {
        let n = self.n_items;
        let k = self.k_judges;
        let n_f = n as f64;
        let mut results = Vec::with_capacity(k);

        if k == 1 {
            results.push(LooJudgeImpact {
                judge_idx: 0,
                accuracy_without: 0.0,
                delta_accuracy: -panel_accuracy,
                n_eff_without: 0.0,
                delta_n_eff: -n_eff_kish,
                redundant: false,
            });
            return results;
        }

        let mut scratch = vec![0u32; k - 1];
        // Precompute total off-diagonal phi sum and per-judge row off-diagonal sums.
        let mut total_off_phi = 0.0_f64;
        let mut row_off_phi = vec![0.0_f64; k];
        for a in 0..k {
            for b in (a + 1)..k {
                let v = phi_matrix[a * k + b];
                total_off_phi += v;
                row_off_phi[a] += v;
                row_off_phi[b] += v;
            }
        }
        let rem_pairs = ((k - 1) * (k - 2)) / 2;

        for drop_j in 0..k {
            let mut correct_without = 0.0_f64;
            for i in 0..n {
                let row = i * k;
                let mut pos = 0usize;
                for j in 0..k {
                    if j != drop_j {
                        scratch[pos] = self.votes[row + j];
                        pos += 1;
                    }
                }
                correct_without += plurality_credit(&mut scratch, self.gold[i]);
            }
            let accuracy_without = correct_without / n_f;
            let delta_accuracy = accuracy_without - panel_accuracy;

            let n_eff_without = if rem_pairs == 0 {
                1.0
            } else {
                let mean_phi_wo = (total_off_phi - row_off_phi[drop_j]) / (rem_pairs as f64);
                kish_from_mean_phi(k - 1, mean_phi_wo).0
            };
            let delta_n_eff = n_eff_without - n_eff_kish;

            let redundant = (delta_accuracy > 1e-12)
                || (delta_n_eff > 1e-12)
                || (delta_accuracy >= -1e-12 && delta_n_eff >= -1e-12);

            results.push(LooJudgeImpact {
                judge_idx: drop_j,
                accuracy_without,
                delta_accuracy,
                n_eff_without,
                delta_n_eff,
                redundant,
            });
        }

        results
    }
}

// ============================================================================
// OnlinePanelTracker — Zero-Copy Streaming Panel Health Monitor
// ============================================================================

/// Streaming panel correlation tracker maintaining exact `O(K^2)` integer
/// sufficient statistics (`n`, `sum_e`, `sum_ee`) with zero per-observation
/// heap allocations and exact shard merging.
#[derive(Clone, Debug)]
pub struct OnlinePanelTracker {
    k: usize,
    n: u64,
    sum_e: Vec<u64>,
    sum_ee: Vec<u64>,
}

impl OnlinePanelTracker {
    /// Fallible constructor returning `Err` if `k == 0` or `k > 16_384`.
    pub fn try_new(k: usize) -> Result<Self, &'static str> {
        if k == 0 || k > 16_384 {
            return Err("k_judges must be in 1..=16384");
        }
        let kk = k.checked_mul(k).ok_or("k * k overflows usize")?;
        Ok(Self {
            k,
            n: 0,
            sum_e: vec![0u64; k],
            sum_ee: vec![0u64; kk],
        })
    }

    /// Creates a new streaming tracker for a panel of `k` judges (`0 <= k <= 16_384`).
    pub fn new(k: usize) -> Self {
        if k == 0 {
            return Self {
                k: 0,
                n: 0,
                sum_e: Vec::new(),
                sum_ee: Vec::new(),
            };
        }
        Self::try_new(k).expect("OnlinePanelTracker::new: k must be <= 16384")
    }

    /// Number of judges `K`.
    #[inline]
    pub fn k_judges(&self) -> usize {
        self.k
    }

    /// Number of items observed so far.
    #[inline]
    pub fn n_items(&self) -> u64 {
        self.n
    }

    /// Observes one item's judge error vector (`0` = correct, non-zero = error).
    ///
    /// Silently ignores slices whose length does not match `k`. Performs zero heap allocations.
    #[inline]
    pub fn observe(&mut self, judge_errors: &[u8]) {
        if judge_errors.len() != self.k || self.k == 0 {
            return;
        }
        self.n += 1;
        let k = self.k;
        for a in 0..k {
            if judge_errors[a] != 0 {
                self.sum_e[a] += 1;
                let row = a * k;
                self.sum_ee[row + a] += 1;
                for b in (a + 1)..k {
                    if judge_errors[b] != 0 {
                        self.sum_ee[row + b] += 1;
                        self.sum_ee[b * k + a] += 1;
                    }
                }
            }
        }
    }

    /// Observes one item's discrete judge votes against `gold` directly without allocating.
    #[inline]
    pub fn observe_votes(&mut self, judge_votes: &[u32], gold: u32) {
        if judge_votes.len() != self.k || self.k == 0 {
            return;
        }
        self.n += 1;
        let k = self.k;
        for a in 0..k {
            if judge_votes[a] != gold {
                self.sum_e[a] += 1;
                let row = a * k;
                self.sum_ee[row + a] += 1;
                for b in (a + 1)..k {
                    if judge_votes[b] != gold {
                        self.sum_ee[row + b] += 1;
                        self.sum_ee[b * k + a] += 1;
                    }
                }
            }
        }
    }

    /// Computes the current `k x k` pairwise Pearson/phi error correlation matrix.
    pub fn phi_matrix(&self) -> Vec<f64> {
        phi_from_counts(self.k, self.n, &self.sum_e, &self.sum_ee).0
    }

    /// Computes the current mean off-diagonal error correlation.
    pub fn mean_phi(&self) -> f64 {
        phi_from_counts(self.k, self.n, &self.sum_e, &self.sum_ee).1
    }

    /// Computes the current Kish effective panel size `K / (1 + (K - 1) * max(mean_phi, 0))`.
    pub fn n_eff_kish(&self) -> f64 {
        if self.k == 0 {
            return 0.0;
        }
        let mean_phi = self.mean_phi();
        kish_from_mean_phi(self.k, mean_phi).0
    }

    /// Computes the current unclamped Kish effective panel size.
    pub fn n_eff_kish_raw(&self) -> f64 {
        if self.k == 0 {
            return 0.0;
        }
        let mean_phi = self.mean_phi();
        kish_from_mean_phi(self.k, mean_phi).1
    }

    /// Computes the current spectral effective panel size `K / lambda_max`.
    pub fn n_eff_eigen(&self) -> f64 {
        if self.k == 0 {
            return 0.0;
        }
        let phi = self.phi_matrix();
        spectral_metrics(&phi, self.k).0
    }

    /// Computes current per-judge empirical accuracies (returns `0.0`s if `n == 0`).
    pub fn judge_accuracies(&self) -> Vec<f64> {
        if self.n == 0 {
            return vec![0.0; self.k];
        }
        let n_f = self.n as f64;
        self.sum_e.iter().map(|&e| 1.0 - (e as f64) / n_f).collect()
    }

    /// Resets all sufficient statistics to zero without reallocating.
    pub fn reset(&mut self) {
        self.n = 0;
        for v in &mut self.sum_e {
            *v = 0;
        }
        for v in &mut self.sum_ee {
            *v = 0;
        }
    }

    /// Exact additive merge of another `OnlinePanelTracker` with identical `k`.
    ///
    /// # Panics
    /// Panics if `self.k != other.k`.
    pub fn merge(&mut self, other: &Self) {
        assert_eq!(
            self.k, other.k,
            "OnlinePanelTracker::merge requires identical k"
        );
        self.n = self.n.saturating_add(other.n);
        for (dst, &src) in self.sum_e.iter_mut().zip(other.sum_e.iter()) {
            *dst = dst.saturating_add(src);
        }
        for (dst, &src) in self.sum_ee.iter_mut().zip(other.sum_ee.iter()) {
            *dst = dst.saturating_add(src);
        }
    }
}

// ============================================================================
// Internal Math, Parallelism & Helper Routines
// ============================================================================

const DOMAIN_CONDORCET: u64 = 0xC04D_0CE7_0000_0001;
const DOMAIN_CONDORCET_FB: u64 = 0xC04D_0CE7_0000_00F1;
const DOMAIN_PERM: u64 = 0x50E8_0000_0000_0002;
const DOMAIN_BOOT: u64 = 0xB007_0000_0000_0003;

enum CondorcetSimBucket {
    C3I16 {
        weight: usize,
        limit: i16,
        cdfs: Vec<(i16, i16)>,
    },
    C4I16 {
        weight: usize,
        limit: i16,
        cdfs: Vec<(i16, i16, i16)>,
    },
    C3 {
        weight: usize,
        limit: i32,
        cdfs: Vec<(i32, i32)>,
    },
    C4 {
        weight: usize,
        limit: i32,
        cdfs: Vec<(i32, i32, i32)>,
    },
    CSmall {
        weight: usize,
        limit: i32,
        n_labels: usize,
        cdfs: Vec<[i32; 16]>,
    },
    General {
        weight: usize,
        gold: u32,
        singleton: bool,
        sample_items: Vec<usize>,
    },
}

/// SplitMix64 finalizer producing deterministic uncorrelated sub-seeds from `(seed, domain, chunk_idx)`.
#[inline]
fn derive_sub_seed(seed: u64, domain: u64, chunk_idx: usize) -> u64 {
    let mut x = seed
        .wrapping_add(domain.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_add((chunk_idx as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

#[cfg(not(target_arch = "wasm32"))]
#[inline]
fn panel_worker_count(total_work: usize) -> usize {
    if total_work < 256 {
        return 1;
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(8)
        .min(total_work)
}

#[cfg(target_arch = "wasm32")]
#[inline]
fn panel_worker_count(_total_work: usize) -> usize {
    1
}

/// Runs `worker(chunk_idx, chunk_count)` across deterministic chunks in `0..n_threads` order,
/// using zero-dependency `std::thread::scope` on native targets and single-threaded execution
/// on `wasm32-unknown-unknown`.
fn run_parallel_chunks<T, F>(total_work: usize, worker: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize, usize) -> T + Sync,
{
    let n_threads = panel_worker_count(total_work);
    if n_threads <= 1 {
        return vec![worker(0, total_work)];
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let base = total_work / n_threads;
        let rem = total_work % n_threads;
        let worker_ref = &worker;
        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(n_threads - 1);
            for tid in 1..n_threads {
                let count = base + usize::from(tid < rem);
                handles.push(scope.spawn(move || worker_ref(tid, count)));
            }
            let count0 = base + usize::from(0 < rem);
            let first = worker_ref(0, count0);
            let mut out = Vec::with_capacity(n_threads);
            out.push(first);
            for h in handles {
                out.push(h.join().expect("panel audit worker thread panicked"));
            }
            out
        })
    }

    #[cfg(target_arch = "wasm32")]
    {
        vec![worker(0, total_work)]
    }
}

/// Dual-stream xoshiro256++ PRNG: runs 2 independent 64-bit streams in scalar registers
/// (exploiting single-cycle 64-bit `rol` and superscalar ALU ILP) to emit 8x `i16` or 4x `i31`.
struct Rng2 {
    a: Rng,
    b: Rng,
}

impl Rng2 {
    fn from_seed(seed: u64, domain: u64, chunk_idx: usize) -> Self {
        let a = Rng::from_seed(derive_sub_seed(seed, domain, chunk_idx.wrapping_mul(2)));
        let b = Rng::from_seed(derive_sub_seed(
            seed,
            domain,
            chunk_idx.wrapping_mul(2).wrapping_add(1),
        ));
        Self { a, b }
    }

    #[inline(always)]
    fn next_u64x2(&mut self) -> (u64, u64) {
        (self.a.next_u64(), self.b.next_u64())
    }
}

#[cold]
#[inline(never)]
fn fast_range_u32_slow(bound: u32, thresh: u32, rng: &mut Rng) -> u32 {
    loop {
        let prod = ((rng.next_u64() as u32) as u64) * (bound as u64);
        if (prod as u32) >= thresh {
            return (prod >> 32) as u32;
        }
    }
}

/// Exact uniform integer in `0..bound` from a 32-bit uniform `val` using Lemire's method
/// with precomputed `thresh = bound.wrapping_neg() % bound`.
#[inline(always)]
fn fast_range_u32_pre(val: u32, bound: u32, thresh: u32, rng: &mut Rng) -> u32 {
    let prod = (val as u64) * (bound as u64);
    if (prod as u32) < thresh {
        return fast_range_u32_slow(bound, thresh, rng);
    }
    (prod >> 32) as u32
}

#[cold]
#[inline(never)]
fn sample_i16_lt_slow(limit: i16, rng: &mut Rng) -> i16 {
    loop {
        let w = rng.next_u64();
        let v0 = w as i16;
        if v0 < limit {
            return v0;
        }
        let v1 = (w >> 16) as i16;
        if v1 < limit {
            return v1;
        }
        let v2 = (w >> 32) as i16;
        if v2 < limit {
            return v2;
        }
        let v3 = (w >> 48) as i16;
        if v3 < limit {
            return v3;
        }
    }
}

#[cold]
#[inline(never)]
fn fix_rejected_lanes_i16x8(r: &mut [i16; 8], limit: i16, rng: &mut Rng) {
    for lane in 0..8 {
        if r[lane] >= limit {
            r[lane] = sample_i16_lt_slow(limit, rng);
        }
    }
}

/// Generates 8 exact uniform `i16` values in `[-32768, limit)` using only 2 `u64` RNG draws
/// and zero multiplications; fits in one 128-bit SSE2/NEON/WASM SIMD register.
#[inline(always)]
fn gen_i16x8_lt(rng2: &mut Rng2, limit: i16, fallback_rng: &mut Rng) -> [i16; 8] {
    let (w0, w1) = rng2.next_u64x2();
    let mut r = [
        w0 as i16,
        (w0 >> 16) as i16,
        (w0 >> 32) as i16,
        (w0 >> 48) as i16,
        w1 as i16,
        (w1 >> 16) as i16,
        (w1 >> 32) as i16,
        (w1 >> 48) as i16,
    ];
    let mut any_reject = 0i16;
    for lane in 0..8 {
        any_reject |= (r[lane] >= limit) as i16;
    }
    if any_reject != 0 {
        fix_rejected_lanes_i16x8(&mut r, limit, fallback_rng);
    }
    r
}

#[inline(always)]
fn sample_i16_lt(limit: i16, rng: &mut Rng) -> i16 {
    let v = rng.next_u64() as i16;
    if v >= limit {
        return sample_i16_lt_slow(limit, rng);
    }
    v
}

#[cold]
#[inline(never)]
fn sample_i31_lt_slow(limit: i32, rng: &mut Rng) -> i32 {
    loop {
        let v = ((rng.next_u64() as u32) & 0x7FFF_FFFF) as i32;
        if v < limit {
            return v;
        }
    }
}

#[cold]
#[inline(never)]
fn fix_rejected_lanes_i31x4(r: &mut [i32; 4], limit: i32, rng: &mut Rng) {
    for lane in 0..4 {
        if r[lane] >= limit {
            r[lane] = sample_i31_lt_slow(limit, rng);
        }
    }
}

/// Generates 4 independent exact uniform integers in `0..limit` (`1 <= limit <= i32::MAX`)
/// with ZERO multiplications using 2 `u64` xoshiro256++ draws.
#[inline(always)]
fn gen_i31x4_lt(rng2: &mut Rng2, limit: i32, fallback_rng: &mut Rng) -> [i32; 4] {
    let (w0, w1) = rng2.next_u64x2();
    let mut r = [
        ((w0 as u32) & 0x7FFF_FFFF) as i32,
        (((w0 >> 32) as u32) & 0x7FFF_FFFF) as i32,
        ((w1 as u32) & 0x7FFF_FFFF) as i32,
        (((w1 >> 32) as u32) & 0x7FFF_FFFF) as i32,
    ];
    let mut any_reject = 0i32;
    for lane in 0..4 {
        any_reject |= (r[lane] >= limit) as i32;
    }
    if any_reject != 0 {
        fix_rejected_lanes_i31x4(&mut r, limit, fallback_rng);
    }
    r
}

#[inline(always)]
fn sample_i31_lt(limit: i32, rng: &mut Rng) -> i32 {
    let v = ((rng.next_u64() as u32) & 0x7FFF_FFFF) as i32;
    if v >= limit {
        return sample_i31_lt_slow(limit, rng);
    }
    v
}

#[cold]
#[inline(never)]
fn fast_lemire_u32_slow(mut prod: u64, bound: u32, rng: &mut Rng) -> u32 {
    let thresh = bound.wrapping_neg() % bound;
    while (prod as u32) < thresh {
        prod = ((rng.next_u64() as u32) as u64) * (bound as u64);
    }
    (prod >> 32) as u32
}

/// Exact uniform integer in `0..bound` from a 32-bit uniform `val` using Lemire's
/// nearly-divisionless rejection method (`bound >= 1`).
#[inline(always)]
fn fast_lemire_u32(val: u32, bound: u32, rng: &mut Rng) -> u32 {
    let prod = (val as u64) * (bound as u64);
    if (prod as u32) < bound {
        return fast_lemire_u32_slow(prod, bound, rng);
    }
    (prod >> 32) as u32
}

/// Exact `O(K^2)` Poisson-Binomial DP for majority-vote accuracy of `K` independent
/// binary judges with individual accuracy probabilities `probs`.
fn poisson_binomial_majority_accuracy(probs: &[f64]) -> f64 {
    let k = probs.len();
    if k == 0 {
        return 0.0;
    }
    let mut stack_dp = [0.0_f64; 64];
    let mut heap_dp: Vec<f64>;
    let dp: &mut [f64] = if k < 64 {
        &mut stack_dp[..=k]
    } else {
        heap_dp = vec![0.0_f64; k + 1];
        &mut heap_dp
    };
    dp[0] = 1.0;
    for (j, &p_raw) in probs.iter().enumerate() {
        let p = p_raw.clamp(0.0, 1.0);
        let q = 1.0 - p;
        for c in (1..=j + 1).rev() {
            dp[c] = dp[c] * q + dp[c - 1] * p;
        }
        dp[0] *= q;
    }
    let half = k / 2;
    let mut acc = 0.0_f64;
    for &prob in &dp[(half + 1)..=k] {
        acc += prob;
    }
    if k % 2 == 0 {
        acc += 0.5 * dp[half];
    }
    acc.clamp(0.0, 1.0)
}

/// 8-wide 16-bit single-XMM multiply-free plurality credit (scaled by 6) for 3-class Condorcet.
#[inline(always)]
fn draw_c3_i16_batch8_credit_x6(
    cdfs: &[(i16, i16)],
    limit: i16,
    rng2: &mut Rng2,
    fallback_rng: &mut Rng,
) -> u64 {
    let k_i16 = cdfs.len() as i16;
    let mut c0 = [0i16; 8];
    let mut c01 = [0i16; 8];
    for &(t0, t1) in cdfs {
        let r = gen_i16x8_lt(rng2, limit, fallback_rng);
        for lane in 0..8 {
            let rv = r[lane];
            c0[lane] += (rv < t0) as i16;
            c01[lane] += (rv < t1) as i16;
        }
    }
    let mut total_x6 = 0u64;
    for lane in 0..8 {
        let v0 = c0[lane];
        let v01 = c01[lane];
        let v1 = v01 - v0;
        let v2 = k_i16 - v01;
        let max_other = v1.max(v2);
        let win = (v0 > max_other) as u64;
        let tie = (v0 == max_other) as u64;
        let tie3 = (v1 == v2) as u64;
        total_x6 += win * 6 + tie * (3 - tie3);
    }
    total_x6
}

/// Scalar tail for 3-class 16-bit multiply-free plurality credit (scaled by 6).
#[inline(always)]
fn draw_c3_i16_credit_x6(cdfs: &[(i16, i16)], limit: i16, rng: &mut Rng) -> u64 {
    let k_i16 = cdfs.len() as i16;
    let mut c0 = 0i16;
    let mut c01 = 0i16;
    for &(t0, t1) in cdfs {
        let r = sample_i16_lt(limit, rng);
        c0 += (r < t0) as i16;
        c01 += (r < t1) as i16;
    }
    let c1 = c01 - c0;
    let c2 = k_i16 - c01;
    let max_other = c1.max(c2);
    let win = (c0 > max_other) as u64;
    let tie = (c0 == max_other) as u64;
    let tie3 = (c1 == c2) as u64;
    win * 6 + tie * (3 - tie3)
}

/// 8-wide 16-bit single-XMM multiply-free plurality credit (scaled by 12) for 4-class Condorcet.
#[inline(always)]
fn draw_c4_i16_batch8_credit_x12(
    cdfs: &[(i16, i16, i16)],
    limit: i16,
    rng2: &mut Rng2,
    fallback_rng: &mut Rng,
) -> u64 {
    const CRED_X12: [u64; 5] = [0, 12, 6, 4, 3];
    let k_i16 = cdfs.len() as i16;
    let mut c0 = [0i16; 8];
    let mut c01 = [0i16; 8];
    let mut c012 = [0i16; 8];
    for &(t0, t1, t2) in cdfs {
        let r = gen_i16x8_lt(rng2, limit, fallback_rng);
        for lane in 0..8 {
            let rv = r[lane];
            c0[lane] += (rv < t0) as i16;
            c01[lane] += (rv < t1) as i16;
            c012[lane] += (rv < t2) as i16;
        }
    }
    let mut total_x12 = 0u64;
    for lane in 0..8 {
        let v0 = c0[lane];
        let v01 = c01[lane];
        let v012 = c012[lane];
        let v1 = v01 - v0;
        let v2 = v012 - v01;
        let v3 = k_i16 - v012;
        let max_other = v1.max(v2).max(v3);
        let win = (v0 > max_other) as u64;
        let tie = (v0 == max_other) as u64;
        let ties = 1 + (v1 == v0) as usize + (v2 == v0) as usize + (v3 == v0) as usize;
        total_x12 += win * 12 + tie * CRED_X12[ties];
    }
    total_x12
}

/// Scalar tail for 4-class 16-bit multiply-free plurality credit (scaled by 12).
#[inline(always)]
fn draw_c4_i16_credit_x12(cdfs: &[(i16, i16, i16)], limit: i16, rng: &mut Rng) -> u64 {
    const CRED_X12: [u64; 5] = [0, 12, 6, 4, 3];
    let k_i16 = cdfs.len() as i16;
    let mut c0 = 0i16;
    let mut c01 = 0i16;
    let mut c012 = 0i16;
    for &(t0, t1, t2) in cdfs {
        let r = sample_i16_lt(limit, rng);
        c0 += (r < t0) as i16;
        c01 += (r < t1) as i16;
        c012 += (r < t2) as i16;
    }
    let c1 = c01 - c0;
    let c2 = c012 - c01;
    let c3 = k_i16 - c012;
    let max_other = c1.max(c2).max(c3);
    let win = (c0 > max_other) as u64;
    let tie = (c0 == max_other) as u64;
    let ties = 1 + (c1 == c0) as usize + (c2 == c0) as usize + (c3 == c0) as usize;
    win * 12 + tie * CRED_X12[ties]
}

/// 4-wide zero-spill multiply-free SIMD plurality credit (scaled by 6) for 3-class Condorcet.
#[inline(always)]
fn draw_c3_batch4_credit_x6(
    cdfs: &[(i32, i32)],
    limit: i32,
    rng2: &mut Rng2,
    fallback_rng: &mut Rng,
) -> u64 {
    let k_i32 = cdfs.len() as i32;
    let mut c0 = [0i32; 4];
    let mut c01 = [0i32; 4];
    for &(t0, t1) in cdfs {
        let r = gen_i31x4_lt(rng2, limit, fallback_rng);
        for lane in 0..4 {
            let rv = r[lane];
            c0[lane] += (rv < t0) as i32;
            c01[lane] += (rv < t1) as i32;
        }
    }
    let mut total_x6 = 0u64;
    for lane in 0..4 {
        let v0 = c0[lane];
        let v01 = c01[lane];
        let v1 = v01 - v0;
        let v2 = k_i32 - v01;
        let max_other = v1.max(v2);
        let win = (v0 > max_other) as u64;
        let tie = (v0 == max_other) as u64;
        let tie3 = (v1 == v2) as u64;
        total_x6 += win * 6 + tie * (3 - tie3);
    }
    total_x6
}

/// Scalar tail for 3-class multiply-free plurality credit (scaled by 6).
#[inline(always)]
fn draw_c3_credit_x6(cdfs: &[(i32, i32)], limit: i32, rng: &mut Rng) -> u64 {
    let k_i32 = cdfs.len() as i32;
    let mut c0 = 0i32;
    let mut c01 = 0i32;
    for &(t0, t1) in cdfs {
        let r = sample_i31_lt(limit, rng);
        c0 += (r < t0) as i32;
        c01 += (r < t1) as i32;
    }
    let c1 = c01 - c0;
    let c2 = k_i32 - c01;
    let max_other = c1.max(c2);
    let win = (c0 > max_other) as u64;
    let tie = (c0 == max_other) as u64;
    let tie3 = (c1 == c2) as u64;
    win * 6 + tie * (3 - tie3)
}

/// 4-wide zero-spill multiply-free SIMD plurality credit (scaled by 12) for 4-class Condorcet.
#[inline(always)]
fn draw_c4_batch4_credit_x12(
    cdfs: &[(i32, i32, i32)],
    limit: i32,
    rng2: &mut Rng2,
    fallback_rng: &mut Rng,
) -> u64 {
    const CRED_X12: [u64; 5] = [0, 12, 6, 4, 3];
    let k_i32 = cdfs.len() as i32;
    let mut c0 = [0i32; 4];
    let mut c01 = [0i32; 4];
    let mut c012 = [0i32; 4];
    for &(t0, t1, t2) in cdfs {
        let r = gen_i31x4_lt(rng2, limit, fallback_rng);
        for lane in 0..4 {
            let rv = r[lane];
            c0[lane] += (rv < t0) as i32;
            c01[lane] += (rv < t1) as i32;
            c012[lane] += (rv < t2) as i32;
        }
    }
    let mut total_x12 = 0u64;
    for lane in 0..4 {
        let v0 = c0[lane];
        let v01 = c01[lane];
        let v012 = c012[lane];
        let v1 = v01 - v0;
        let v2 = v012 - v01;
        let v3 = k_i32 - v012;
        let max_other = v1.max(v2).max(v3);
        let win = (v0 > max_other) as u64;
        let tie = (v0 == max_other) as u64;
        let ties = 1 + (v1 == v0) as usize + (v2 == v0) as usize + (v3 == v0) as usize;
        total_x12 += win * 12 + tie * CRED_X12[ties];
    }
    total_x12
}

/// Scalar tail for 4-class multiply-free plurality credit (scaled by 12).
#[inline(always)]
fn draw_c4_credit_x12(cdfs: &[(i32, i32, i32)], limit: i32, rng: &mut Rng) -> u64 {
    const CRED_X12: [u64; 5] = [0, 12, 6, 4, 3];
    let k_i32 = cdfs.len() as i32;
    let mut c0 = 0i32;
    let mut c01 = 0i32;
    let mut c012 = 0i32;
    for &(t0, t1, t2) in cdfs {
        let r = sample_i31_lt(limit, rng);
        c0 += (r < t0) as i32;
        c01 += (r < t1) as i32;
        c012 += (r < t2) as i32;
    }
    let c1 = c01 - c0;
    let c2 = c012 - c01;
    let c3 = k_i32 - c012;
    let max_other = c1.max(c2).max(c3);
    let win = (c0 > max_other) as u64;
    let tie = (c0 == max_other) as u64;
    let ties = 1 + (c1 == c0) as usize + (c2 == c0) as usize + (c3 == c0) as usize;
    win * 12 + tie * CRED_X12[ties]
}

const INV_TIES_16: [f64; 17] = [
    0.0,
    1.0,
    1.0 / 2.0,
    1.0 / 3.0,
    1.0 / 4.0,
    1.0 / 5.0,
    1.0 / 6.0,
    1.0 / 7.0,
    1.0 / 8.0,
    1.0 / 9.0,
    1.0 / 10.0,
    1.0 / 11.0,
    1.0 / 12.0,
    1.0 / 13.0,
    1.0 / 14.0,
    1.0 / 15.0,
    1.0 / 16.0,
];

/// 4-wide multiply-free zero-sort `O(K)` histogram plurality credit for `5 <= C <= 16`.
#[inline(always)]
fn draw_csmall_batch4_credit(
    cdfs: &[[i32; 16]],
    n_labels: usize,
    limit: i32,
    rng2: &mut Rng2,
    fallback_rng: &mut Rng,
) -> f64 {
    let mut counts = [[0u32; 16]; 4];
    let cm1 = n_labels - 1;
    for cdf in cdfs {
        let r = gen_i31x4_lt(rng2, limit, fallback_rng);
        for lane in 0..4 {
            let rv = r[lane];
            let mut lbl = 0usize;
            for &t in &cdf[..cm1] {
                lbl += (rv >= t) as usize;
            }
            counts[lane][lbl] += 1;
        }
    }
    let mut total = 0.0_f64;
    for lane in 0..4 {
        let hist = &counts[lane];
        let gold_cnt = hist[0];
        if gold_cnt == 0 {
            continue;
        }
        let mut win = true;
        let mut ties = 1usize;
        for &c in &hist[1..n_labels] {
            if c > gold_cnt {
                win = false;
                break;
            }
            if c == gold_cnt {
                ties += 1;
            }
        }
        if win {
            total += INV_TIES_16[ties];
        }
    }
    total
}

/// Scalar tail for `5 <= C <= 16` multiply-free zero-sort `O(K)` histogram plurality credit.
#[inline(always)]
fn draw_csmall_credit(cdfs: &[[i32; 16]], n_labels: usize, limit: i32, rng: &mut Rng) -> f64 {
    let mut counts = [0u32; 16];
    let cm1 = n_labels - 1;
    for cdf in cdfs {
        let r = sample_i31_lt(limit, rng);
        let mut lbl = 0usize;
        for &t in &cdf[..cm1] {
            lbl += (r >= t) as usize;
        }
        counts[lbl] += 1;
    }

    let gold_cnt = counts[0];
    if gold_cnt == 0 {
        return 0.0;
    }
    let mut ties = 1usize;
    for &c in &counts[1..n_labels] {
        if c > gold_cnt {
            return 0.0;
        }
        if c == gold_cnt {
            ties += 1;
        }
    }
    INV_TIES_16[ties]
}

fn compact_strata(strata: Option<&[usize]>, n_items: usize) -> Result<(Vec<usize>, usize), &'static str> {
    match strata {
        None => Ok((vec![0usize; n_items], 1)),
        Some(raw) => {
            if raw.len() != n_items {
                return Err("strata length must equal n_items");
            }
            let mut map: HashMap<usize, usize> = HashMap::new();
            let mut out = Vec::with_capacity(n_items);
            for &s in raw {
                let next_id = map.len();
                let id = *map.entry(s).or_insert(next_id);
                out.push(id);
            }
            Ok((out, map.len().max(1)))
        }
    }
}

/// Computes uniform-tie-breaking plurality credit for `gold` given a mutable slice of votes.
#[inline]
fn plurality_credit(votes: &mut [u32], gold: u32) -> f64 {
    let m = votes.len();
    if m == 0 {
        return 0.0;
    }
    if m == 1 {
        return if votes[0] == gold { 1.0 } else { 0.0 };
    }
    // Fast zero-alloc insertion sort for typical panel sizes, unstable sort for large panels.
    if m <= 32 {
        for i in 1..m {
            let key = votes[i];
            let mut j = i;
            while j > 0 && votes[j - 1] > key {
                votes[j] = votes[j - 1];
                j -= 1;
            }
            votes[j] = key;
        }
    } else {
        votes.sort_unstable();
    }

    let mut max_cnt = 0usize;
    let mut tied_at_max = 0usize;
    let mut gold_at_max = false;

    let mut i = 0usize;
    while i < m {
        let val = votes[i];
        let mut j = i + 1;
        while j < m && votes[j] == val {
            j += 1;
        }
        let cnt = j - i;
        if cnt > max_cnt {
            max_cnt = cnt;
            tied_at_max = 1;
            gold_at_max = val == gold;
        } else if cnt == max_cnt {
            tied_at_max += 1;
            if val == gold {
                gold_at_max = true;
            }
        }
        i = j;
    }

    if gold_at_max && tied_at_max > 0 {
        1.0 / (tied_at_max as f64)
    } else {
        0.0
    }
}

fn phi_from_counts(k: usize, n: u64, sum_e: &[u64], sum_ee: &[u64]) -> (Vec<f64>, f64, f64) {
    if k == 0 {
        return (Vec::new(), 0.0, 0.0);
    }
    let mut phi = vec![0.0_f64; k * k];
    for i in 0..k {
        phi[i * k + i] = 1.0;
    }
    if k == 1 || n == 0 {
        return (phi, 0.0, 0.0);
    }

    let n_f = n as f64;
    let mut p = vec![0.0_f64; k];
    let mut sd = vec![0.0_f64; k];
    for j in 0..k {
        let pj = (sum_e[j] as f64) / n_f;
        p[j] = pj;
        sd[j] = (pj * (1.0 - pj)).max(0.0).sqrt();
    }

    let n_pairs = (k * (k - 1)) / 2;
    let mut sum_phi = 0.0_f64;
    for a in 0..k {
        for b in (a + 1)..k {
            let denom = sd[a] * sd[b];
            let val = if denom > 1e-15 {
                let pab = (sum_ee[a * k + b] as f64) / n_f;
                ((pab - p[a] * p[b]) / denom).clamp(-1.0, 1.0)
            } else {
                0.0
            };
            phi[a * k + b] = val;
            phi[b * k + a] = val;
            sum_phi += val;
        }
    }
    let mean_phi = sum_phi / (n_pairs as f64);

    let mut sq_diff = 0.0_f64;
    for a in 0..k {
        for b in (a + 1)..k {
            let d = phi[a * k + b] - mean_phi;
            sq_diff += d * d;
        }
    }
    let std_phi = (sq_diff / (n_pairs as f64)).max(0.0).sqrt();
    (phi, mean_phi, std_phi)
}

#[inline]
fn kish_from_mean_phi(k: usize, mean_phi: f64) -> (f64, f64) {
    if k == 0 {
        return (0.0, 0.0);
    }
    if k == 1 {
        return (1.0, 1.0);
    }
    let kf = k as f64;
    let n_eff_kish = kf / (1.0 + (kf - 1.0) * mean_phi.max(0.0));
    let denom_raw = 1.0 + (kf - 1.0) * mean_phi;
    let n_eff_kish_raw = if denom_raw > 1e-12 {
        kf / denom_raw
    } else {
        f64::INFINITY
    };
    (n_eff_kish, n_eff_kish_raw)
}

fn spectral_metrics(phi_matrix: &[f64], k: usize) -> (f64, f64, f64) {
    if k == 0 {
        return (0.0, 0.0, 0.0);
    }
    if k == 1 {
        return (1.0, 1.0, 1.0);
    }
    let eigs = symmetric_eigenvalues(phi_matrix, k);
    let lambda_max = eigs[0].max(1e-12);
    let n_eff_eigen = (k as f64) / lambda_max;

    let mut pos_sum = 0.0_f64;
    for &e in &eigs {
        if e > 1e-12 {
            pos_sum += e;
        }
    }
    let effective_rank = if pos_sum > 1e-12 {
        let mut entropy = 0.0_f64;
        for &e in &eigs {
            if e > 1e-12 {
                let q = e / pos_sum;
                entropy -= q * q.ln();
            }
        }
        entropy.exp().clamp(1.0, k as f64)
    } else {
        1.0
    };
    (n_eff_eigen, lambda_max, effective_rank)
}

fn compute_krippendorff_alpha(votes: &[u32], n_items: usize, k_judges: usize) -> f64 {
    if k_judges < 2 || n_items == 0 {
        return 1.0;
    }
    let mut marginal: HashMap<u32, u64> = HashMap::new();
    let mut within_agree: u128 = 0;
    let mut scratch = vec![0u32; k_judges];

    for i in 0..n_items {
        let row = i * k_judges;
        scratch.copy_from_slice(&votes[row..row + k_judges]);
        scratch.sort_unstable();
        let mut pos = 0usize;
        while pos < k_judges {
            let val = scratch[pos];
            let mut end = pos + 1;
            while end < k_judges && scratch[end] == val {
                end += 1;
            }
            let cnt = (end - pos) as u64;
            within_agree += (cnt as u128) * ((cnt - 1) as u128);
            *marginal.entry(val).or_insert(0) += cnt;
            pos = end;
        }
    }

    let total_within_pairs = (n_items as u128) * (k_judges as u128) * ((k_judges - 1) as u128);
    let d_o = 1.0 - (within_agree as f64) / (total_within_pairs as f64);

    let total_ratings = (n_items as u128) * (k_judges as u128);
    let total_all_pairs = total_ratings * (total_ratings - 1);
    let mut marginal_agree: u128 = 0;
    for &c in marginal.values() {
        marginal_agree += (c as u128) * ((c - 1) as u128);
    }
    let d_e = 1.0 - (marginal_agree as f64) / (total_all_pairs as f64);

    if d_e <= 1e-15 {
        return 1.0;
    }
    1.0 - d_o / d_e
}

fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    let m = sorted.len();
    if m == 0 {
        return 0.0;
    }
    if m == 1 {
        return sorted[0];
    }
    let pos = q.clamp(0.0, 1.0) * ((m - 1) as f64);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - (lo as f64);
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

fn select_optimal_subset(
    k: usize,
    best_single_index: usize,
    judge_accuracies: &[f64],
    phi_matrix: &[f64],
    full_neff: f64,
    target_ratio: f64,
) -> Vec<usize> {
    if k == 0 {
        return Vec::new();
    }
    let target = target_ratio.clamp(0.0, 1.0) * full_neff;
    let mut selected = Vec::with_capacity(k);
    let mut in_subset = vec![false; k];
    selected.push(best_single_index);
    in_subset[best_single_index] = true;

    let mut current_phi_sum = 0.0_f64;
    let mut current_neff = 1.0_f64;

    while current_neff < target - 1e-12 && selected.len() < k {
        let next_m = selected.len() + 1;
        let next_pairs = ((next_m * (next_m - 1)) / 2) as f64;
        let mut best_j = None;
        let mut best_neff = f64::NEG_INFINITY;
        let mut best_add_phi = 0.0_f64;
        let mut best_acc = f64::NEG_INFINITY;

        for j in 0..k {
            if in_subset[j] {
                continue;
            }
            let mut add_phi = 0.0_f64;
            for &s in &selected {
                add_phi += phi_matrix[s * k + j];
            }
            let cand_mean_phi = (current_phi_sum + add_phi) / next_pairs;
            let cand_neff = kish_from_mean_phi(next_m, cand_mean_phi).0;
            let acc = judge_accuracies[j];
            let better = (cand_neff > best_neff + 1e-12)
                || ((cand_neff - best_neff).abs() <= 1e-12 && acc > best_acc + 1e-12)
                || best_j.is_none();
            if better {
                best_j = Some(j);
                best_neff = cand_neff;
                best_add_phi = add_phi;
                best_acc = acc;
            }
        }

        if let Some(j) = best_j {
            selected.push(j);
            in_subset[j] = true;
            current_phi_sum += best_add_phi;
            current_neff = best_neff;
        } else {
            break;
        }
    }

    selected
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panel_clones_vs_independent_and_pruning() {
        let n = 400;
        let mut rng = Rng::from_seed(0x9A4E_0001);

        // Panel A: 6 identical clone judges with 75% accuracy
        let mut clone_errors = vec![0u8; n * 6];
        for i in 0..n {
            let err = if rng.next_f64() < 0.25 { 1 } else { 0 };
            for j in 0..6 {
                clone_errors[i * 6 + j] = err;
            }
        }
        let audit_clones = PanelAudit::from_binary_errors(&clone_errors, n, 6, None).unwrap();
        let rep_clones = audit_clones.run_with_config(&PanelAuditConfig {
            n_condorcet_sims: 500,
            n_permutations: 1_000,
            n_bootstrap: 400,
            ..Default::default()
        });
        assert!((rep_clones.mean_phi - 1.0).abs() < 1e-12);
        assert!((rep_clones.n_eff_kish - 1.0).abs() < 1e-12);
        assert!((rep_clones.n_eff_eigen - 1.0).abs() < 1e-9);
        assert!((rep_clones.lambda_max - 6.0).abs() < 1e-9);
        assert!((rep_clones.effective_rank - 1.0).abs() < 1e-9);
        assert!((rep_clones.krippendorff_alpha - 1.0).abs() < 1e-12);
        assert!((rep_clones.unanimous_rate - 1.0).abs() < 1e-12);
        assert!(rep_clones.condorcet_gap > 0.10, "expected large Condorcet gap, got {}", rep_clones.condorcet_gap);
        assert!(rep_clones.permutation_p_value <= 0.002);
        assert_eq!(rep_clones.optimal_subset.len(), 1, "5 of 6 clones should be pruned immediately");
        assert!(rep_clones.loo_analysis.iter().all(|l| l.redundant));

        // Panel B: 2 independent clusters of 3 clones each (K = 6, true N_eff = 2)
        let mut two_cluster = vec![0u8; n * 6];
        for i in 0..n {
            let e0 = if rng.next_f64() < 0.25 { 1 } else { 0 };
            let e1 = if rng.next_f64() < 0.25 { 1 } else { 0 };
            for j in 0..3 {
                two_cluster[i * 6 + j] = e0;
                two_cluster[i * 6 + 3 + j] = e1;
            }
        }
        let rep_2c = PanelAudit::from_binary_errors(&two_cluster, n, 6, None)
            .unwrap()
            .run_with_config(&PanelAuditConfig {
                n_condorcet_sims: 200,
                n_permutations: 500,
                n_bootstrap: 300,
                ..Default::default()
            });
        assert!((rep_2c.n_eff_eigen - 2.0).abs() < 0.25, "n_eff_eigen={}", rep_2c.n_eff_eigen);
        assert!((rep_2c.effective_rank - 2.0).abs() < 0.25, "eff_rank={}", rep_2c.effective_rank);
        assert_eq!(
            rep_2c.optimal_subset.len(),
            2,
            "greedy subset should pick 1 representative per cluster: {:?}",
            rep_2c.optimal_subset
        );
        // The two picked judges must belong to different clusters (<3 and >=3).
        let c0 = rep_2c.optimal_subset[0] / 3;
        let c1 = rep_2c.optimal_subset[1] / 3;
        assert_ne!(c0, c1);
    }

    #[test]
    fn panel_independent_stratified_null_and_condorcet_calibration() {
        // Difficulty confounding test: easy stratum (10% error) and hard stratum (60% error).
        // Within each stratum, 5 judges err INDEPENDENTLY.
        // Without stratification, judges look correlated (p < 0.001); WITH stratification,
        // the stratified permutation test must NOT reject!
        let n = 400;
        let k = 5;
        let mut rng = Rng::from_seed(0x9A4E_0002);
        let mut errors = vec![0u8; n * k];
        let mut strata = vec![0usize; n];
        for i in 0..n {
            let hard = i >= n / 2;
            strata[i] = if hard { 10 } else { 5 }; // arbitrary non-contiguous stratum IDs
            let err_p = if hard { 0.60 } else { 0.10 };
            for j in 0..k {
                if rng.next_f64() < err_p {
                    errors[i * k + j] = 1;
                }
            }
        }

        let unstrat = PanelAudit::from_binary_errors(&errors, n, k, None)
            .unwrap()
            .run_with_config(&PanelAuditConfig {
                n_condorcet_sims: 500,
                n_permutations: 2_000,
                n_bootstrap: 400,
                seed: 77,
                ..Default::default()
            });
        let strat = PanelAudit::from_binary_errors(&errors, n, k, Some(&strata))
            .unwrap()
            .run_with_config(&PanelAuditConfig {
                n_condorcet_sims: 1_000,
                n_permutations: 2_000,
                n_bootstrap: 400,
                seed: 77,
                ..Default::default()
            });

        // Unstratified wrongly detects difficulty confounding as correlation
        assert!(unstrat.permutation_p_value < 0.01);
        // Stratified correctly recognizes conditional independence!
        assert!(
            strat.permutation_p_value > 0.05,
            "stratified p-value {} falsely rejected conditional independence",
            strat.permutation_p_value
        );
        // Item-aware stratified Condorcet prediction closely matches actual panel accuracy!
        assert!(
            strat.condorcet_gap.abs() < 0.04,
            "stratified condorcet_gap {} too large under conditional independence",
            strat.condorcet_gap
        );
        // Bootstrap CI bounds contain point estimate cleanly
        assert!(strat.n_eff_ci_low <= strat.n_eff_kish + 0.5);
        assert!(strat.n_eff_ci_high >= strat.n_eff_kish - 0.5);
        assert!(strat.n_eff_ci_low <= strat.n_eff_ci_high);
    }

    #[test]
    fn panel_permutation_algebraic_identity_and_speed() {
        // Verify our fast S_sq identity matches direct mean_phi calculation to < 1e-11
        let n = 150;
        let k = 6;
        let mut rng = Rng::from_seed(0x9A4E_0003);
        let mut errors = vec![0u8; n * k];
        for i in 0..n {
            let shared = rng.next_f64() < 0.15;
            for j in 0..k {
                let err = shared || (rng.next_f64() < 0.10 + 0.03 * (j as f64));
                errors[i * k + j] = if err { 1 } else { 0 };
            }
        }
        let mut sum_e = vec![0u64; k];
        let mut sum_ee = vec![0u64; k * k];
        for i in 0..n {
            for a in 0..k {
                if errors[i * k + a] != 0 {
                    sum_e[a] += 1;
                    for b in 0..k {
                        if errors[i * k + b] != 0 {
                            sum_ee[a * k + b] += 1;
                        }
                    }
                }
            }
        }
        let (_, direct_mean_phi, _) = phi_from_counts(k, n as u64, &sum_e, &sum_ee);

        let n_f = n as f64;
        let w: Vec<f64> = sum_e
            .iter()
            .map(|&c| {
                let p = (c as f64) / n_f;
                let v = p * (1.0 - p);
                if v > 1e-15 { 1.0 / v.sqrt() } else { 0.0 }
            })
            .collect();
        let mut s_sq = 0.0;
        for i in 0..n {
            let mut si = 0.0;
            for j in 0..k {
                if errors[i * k + j] != 0 {
                    si += w[j];
                }
            }
            s_sq += si * si;
        }
        let mut diag_term = 0.0;
        for j in 0..k {
            diag_term += w[j] * w[j] * (sum_e[j] as f64);
        }
        let mut prod_term = 0.0;
        for a in 0..k {
            for b in (a + 1)..k {
                prod_term += w[a] * w[b] * ((sum_e[a] as f64) / n_f) * ((sum_e[b] as f64) / n_f);
            }
        }
        let n_pairs = ((k * (k - 1)) / 2) as f64;
        let reconstructed_mean_phi = ((s_sq - diag_term) / (2.0 * n_f) - prod_term) / n_pairs;
        assert!(
            (direct_mean_phi - reconstructed_mean_phi).abs() < 1e-11,
            "direct={direct_mean_phi}, reconstructed={reconstructed_mean_phi}"
        );

        // Verify 10,000 permutations runs cleanly and quickly
        let audit = PanelAudit::from_binary_errors(&errors, n, k, None).unwrap();
        let rep = audit.run_with_config(&PanelAuditConfig {
            n_condorcet_sims: 100,
            n_permutations: 10_000,
            n_bootstrap: 100,
            ..Default::default()
        });
        assert!(rep.permutation_p_value <= 0.001);
    }

    #[test]
    fn panel_edge_cases_constant_judges_anticorrelation_and_krippendorff() {
        // Constant 0-variance judges (always correct, always wrong) must produce zero NaNs
        let votes = vec![
            1, 0, 1,
            1, 0, 0,
            1, 0, 1,
            1, 0, 0,
        ];
        let gold = vec![1, 1, 1, 1];
        let rep = PanelAudit::from_votes(&votes, &gold, 4, 3, None)
            .unwrap()
            .run_with_config(&PanelAuditConfig {
                n_condorcet_sims: 100,
                n_permutations: 200,
                n_bootstrap: 100,
                ..Default::default()
            });
        assert!(!rep.mean_phi.is_nan());
        assert!(!rep.std_phi.is_nan());
        assert!(!rep.n_eff_kish.is_nan());
        assert!(!rep.n_eff_eigen.is_nan());
        assert!(!rep.effective_rank.is_nan());
        assert!(!rep.krippendorff_alpha.is_nan());
        assert!(!rep.unanimous_accuracy.is_nan());
        assert_eq!(rep.unanimous_rate, 0.0);
        assert_eq!(rep.unanimous_accuracy, 0.0);
        assert!(rep.phi_matrix.iter().all(|v| v.is_finite()));

        // Perfect anti-correlation between 2 judges
        let anti = vec![
            1, 0,
            0, 1,
            1, 0,
            0, 1,
        ];
        let rep_anti = PanelAudit::from_binary_errors(&anti, 4, 2, None)
            .unwrap()
            .run_with_config(&PanelAuditConfig {
                n_condorcet_sims: 50,
                n_permutations: 100,
                n_bootstrap: 50,
                ..Default::default()
            });
        assert!((rep_anti.mean_phi - (-1.0)).abs() < 1e-12);
        assert!((rep_anti.n_eff_kish - 2.0).abs() < 1e-12);
        assert!(rep_anti.n_eff_kish_raw.is_infinite());
        // Nominal Krippendorff's alpha for 2 always-disagreeing judges on N=4: -1 + 1/4 = -0.75
        assert!((rep_anti.krippendorff_alpha - (-0.75)).abs() < 1e-12);

        // Single-judge panel and validation checks
        assert!(PanelAudit::from_votes(&[], &[], 0, 2, None).is_err());
        assert!(PanelAudit::from_votes(&[1], &[1], 1, 0, None).is_err());
        assert!(PanelAudit::from_votes(&[1, 2], &[1], 1, 3, None).is_err());
        assert!(PanelAudit::from_votes(&[1, 2], &[1], 1, 2, Some(&[0, 1])).is_err());
        let rep1 = PanelAudit::from_binary_correctness(&[1, 0, 1], 3, 1, None)
            .unwrap()
            .run_with_config(&PanelAuditConfig {
                n_condorcet_sims: 50,
                n_permutations: 50,
                n_bootstrap: 50,
                ..Default::default()
            });
        assert!((rep1.panel_accuracy - 2.0 / 3.0).abs() < 1e-12);
        assert_eq!(rep1.optimal_subset, vec![0]);
    }

    #[test]
    fn online_panel_tracker_matches_batch_and_merges_exactly() {
        let n = 250;
        let k = 5;
        let mut rng = Rng::from_seed(0x0411_0001);
        let mut errors = vec![0u8; n * k];
        for i in 0..n {
            let common = rng.next_f64() < 0.2;
            for j in 0..k {
                let err = common || (rng.next_f64() < 0.15);
                errors[i * k + j] = if err { 1 } else { 0 };
            }
        }

        let batch = PanelAudit::from_binary_errors(&errors, n, k, None)
            .unwrap()
            .run_with_config(&PanelAuditConfig {
                n_condorcet_sims: 10,
                n_permutations: 10,
                n_bootstrap: 10,
                ..Default::default()
            });

        let mut full = OnlinePanelTracker::new(k);
        let mut s1 = OnlinePanelTracker::new(k);
        let mut s2 = OnlinePanelTracker::new(k);
        for i in 0..n {
            let row = &errors[i * k..(i + 1) * k];
            full.observe(row);
            if i < 100 {
                s1.observe(row);
            } else {
                s2.observe(row);
            }
        }
        // Invalid slice ignored safely
        full.observe(&[1, 0]);
        assert_eq!(full.n_items(), n as u64);

        s1.merge(&s2);
        assert_eq!(s1.n_items(), full.n_items());
        assert!((full.n_eff_kish() - batch.n_eff_kish).abs() < 1e-12);
        assert!((full.n_eff_eigen() - batch.n_eff_eigen).abs() < 1e-12);
        assert!((s1.n_eff_kish() - full.n_eff_kish()).abs() < 1e-12);
        assert!((s1.n_eff_eigen() - full.n_eff_eigen()).abs() < 1e-12);
        for (a, b) in s1.phi_matrix().iter().zip(batch.phi_matrix.iter()) {
            assert!((a - b).abs() < 1e-12);
        }

        // Edge cases: OnlinePanelTracker::new(0) and observe_votes
        let mut zero_t = OnlinePanelTracker::new(0);
        zero_t.observe(&[]);
        assert_eq!(zero_t.n_eff_kish(), 0.0);
        assert_eq!(zero_t.n_eff_eigen(), 0.0);

        let mut vote_t = OnlinePanelTracker::new(3);
        vote_t.observe_votes(&[10, 20, 10], 10);
        vote_t.observe_votes(&[10, 10, 10], 10);
        assert_eq!(vote_t.n_items(), 2);
        let accs = vote_t.judge_accuracies();
        assert!((accs[0] - 1.0).abs() < 1e-12);
        assert!((accs[1] - 0.5).abs() < 1e-12);
        assert!((accs[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn panel_multiclass_mcq_and_permutation_type1_calibration() {
        // 4-class MCQ with 5 independent judges and per-class confusion structure
        let n = 400;
        let k = 5;
        let mut rng = Rng::from_seed(0x4C1A_5501);
        let mut votes = vec![0u32; n * k];
        let mut gold = vec![0u32; n];
        for i in 0..n {
            let g = (i % 4) as u32;
            gold[i] = g;
            for j in 0..k {
                let v = if rng.next_f64() < 0.65 {
                    g
                } else {
                    (g + 1 + (rng.gen_range_usize(3) as u32)) % 4
                };
                votes[i * k + j] = v;
            }
        }
        let rep = PanelAudit::from_votes(&votes, &gold, n, k, None)
            .unwrap()
            .run_with_config(&PanelAuditConfig {
                n_condorcet_sims: 800,
                n_permutations: 1_000,
                n_bootstrap: 400,
                ..Default::default()
            });
        // Multi-class independent judges should exhibit strong positive lift and near-zero Condorcet gap
        assert!(rep.panel_lift > 0.15, "expected positive lift, got {}", rep.panel_lift);
        assert!(rep.condorcet_gap.abs() < 0.04, "condorcet_gap={}", rep.condorcet_gap);
        assert!(rep.n_eff_kish > 4.0, "expected n_eff_kish near 5, got {}", rep.n_eff_kish);
        // Exact analytical Krippendorff's alpha for 4 balanced classes with p=0.65:
        // P(agree) = 0.65^2 + 3*(0.35/3)^2 = 0.46333..., D_e = 0.75 -> alpha = 0.28444...
        let expected_alpha = 1.0 - (1.0 - (0.65_f64.powi(2) + 0.35_f64.powi(2) / 3.0)) / 0.75;
        assert!(
            (rep.krippendorff_alpha - expected_alpha).abs() < 0.05,
            "krippendorff_alpha={}, expected ~{expected_alpha:.4}",
            rep.krippendorff_alpha
        );

        // N=1 degenerate edge case: zero NaNs, zero panics
        let rep_n1 = PanelAudit::from_votes(&[7, 9], &[7], 1, 2, None)
            .unwrap()
            .run_with_config(&PanelAuditConfig {
                n_condorcet_sims: 20,
                n_permutations: 50,
                n_bootstrap: 20,
                ..Default::default()
            });
        assert!((rep_n1.panel_accuracy - 0.5).abs() < 1e-12);
        assert!(rep_n1.krippendorff_alpha.is_finite());
        assert_eq!(rep_n1.permutation_p_value, 1.0);

        // Stratified permutation omnibus test Type-I error calibration under conditional independence
        let n_trials = 100;
        let mut rejects = 0usize;
        let m_items = 80;
        let m_judges = 4;
        let strata: Vec<usize> = (0..m_items).map(|i| i / 40).collect();
        let mut errs = vec![0u8; m_items * m_judges];
        for trial in 0..n_trials {
            for i in 0..m_items {
                let ep = if strata[i] == 0 { 0.15 } else { 0.55 };
                for j in 0..m_judges {
                    errs[i * m_judges + j] = if rng.next_f64() < ep { 1 } else { 0 };
                }
            }
            let r = PanelAudit::from_binary_errors(&errs, m_items, m_judges, Some(&strata))
                .unwrap()
                .run_with_config(&PanelAuditConfig {
                    n_condorcet_sims: 0,
                    n_permutations: 400,
                    n_bootstrap: 0,
                    seed: 10_000 + trial as u64,
                    ..Default::default()
                });
            if r.permutation_p_value <= 0.05 {
                rejects += 1;
            }
        }
        // Binomial(100, 0.05) 99.9% upper bound is ~12; verify tight calibration
        assert!(
            rejects <= 11,
            "Stratified permutation test rejected {rejects}/{n_trials} times at alpha=0.05"
        );
    }

    #[test]
    fn condorcet_binary_exact_dp_zero_variance_and_analytical_parity() {
        // Verify Poisson-Binomial DP matches closed-form majority probabilities to < 1e-14.
        // K=1
        assert!((poisson_binomial_majority_accuracy(&[0.73]) - 0.73).abs() < 1e-14);
        // K=2 (with 0.5 tie-breaking -> exact mean of p0, p1)
        assert!((poisson_binomial_majority_accuracy(&[0.8, 0.6]) - 0.7).abs() < 1e-14);
        // K=3 homogeneous: 3p^2 - 2p^3
        let p = 0.72_f64;
        let expected_k3 = 3.0 * p * p - 2.0 * p * p * p;
        assert!((poisson_binomial_majority_accuracy(&[p, p, p]) - expected_k3).abs() < 1e-14);
        // K=4 heterogeneous: brute-force 2^4 enumeration
        let ps = [0.65_f64, 0.70, 0.80, 0.55];
        let mut exact_k4 = 0.0_f64;
        for mask in 0u32..16 {
            let mut prob = 1.0_f64;
            let mut cnt = 0u32;
            for j in 0..4 {
                if (mask & (1 << j)) != 0 {
                    prob *= ps[j];
                    cnt += 1;
                } else {
                    prob *= 1.0 - ps[j];
                }
            }
            if cnt > 2 {
                exact_k4 += prob;
            } else if cnt == 2 {
                exact_k4 += 0.5 * prob;
            }
        }
        assert!((poisson_binomial_majority_accuracy(&ps) - exact_k4).abs() < 1e-14);

        // End-to-end binary PanelAudit: exact DP produces bit-identical Condorcet accuracy
        // across completely different PRNG seeds (zero Monte Carlo variance).
        let n = 100;
        let k = 4;
        let mut votes = vec![0u32; n * k];
        let gold = vec![1u32; n];
        // Exact per-judge accuracies: 0.65, 0.70, 0.80, 0.55
        let counts = [65usize, 70, 80, 55];
        for j in 0..k {
            for i in 0..counts[j] {
                votes[i * k + j] = 1;
            }
        }
        let audit = PanelAudit::from_votes(&votes, &gold, n, k, None).unwrap();
        let rep_a = audit.run_with_config(&PanelAuditConfig {
            n_condorcet_sims: 500,
            n_permutations: 0,
            n_bootstrap: 0,
            seed: 1,
            ..Default::default()
        });
        let rep_b = audit.run_with_config(&PanelAuditConfig {
            n_condorcet_sims: 500,
            n_permutations: 0,
            n_bootstrap: 0,
            seed: 999_999,
            ..Default::default()
        });
        assert!((rep_a.condorcet_predicted_accuracy - exact_k4).abs() < 1e-14);
        assert_eq!(
            rep_a.condorcet_predicted_accuracy.to_bits(),
            rep_b.condorcet_predicted_accuracy.to_bits(),
            "binary Condorcet DP must have zero Monte Carlo variance across seeds"
        );
    }

    #[test]
    fn condorcet_multiclass_fast_paths_and_parallel_reproducibility() {
        // Test C=3, C=4, C=7 (CSmall), and C=20 (General fallback) against exact analytical expectations
        // and verify bit-for-bit parallel reproducibility.
        for &n_classes in &[3u32, 4u32, 7u32, 20u32] {
            let n = 120usize;
            let k = 5usize;
            let mut rng = Rng::from_seed(0xABCD_0000 + n_classes as u64);
            let mut votes = vec![0u32; n * k];
            let mut gold = vec![0u32; n];
            for i in 0..n {
                let g = (i as u32) % n_classes;
                gold[i] = g;
                for j in 0..k {
                    votes[i * k + j] = if rng.next_f64() < 0.60 {
                        g
                    } else {
                        (g + 1 + (rng.gen_range_usize((n_classes - 1) as usize) as u32)) % n_classes
                    };
                }
            }
            let audit = PanelAudit::from_votes(&votes, &gold, n, k, None).unwrap();
            let cfg = PanelAuditConfig {
                n_condorcet_sims: 600,
                n_permutations: 600,
                n_bootstrap: 400,
                seed: 2026,
                ..Default::default()
            };
            let r1 = audit.run_with_config(&cfg);
            let r2 = audit.run_with_config(&cfg);

            // Exact bit-for-bit reproducibility under multi-core execution.
            assert_eq!(
                r1.condorcet_predicted_accuracy.to_bits(),
                r2.condorcet_predicted_accuracy.to_bits(),
                "C={n_classes} condorcet must be bit-for-bit deterministic"
            );
            assert_eq!(
                r1.permutation_p_value.to_bits(),
                r2.permutation_p_value.to_bits(),
                "C={n_classes} permutation p-value must be bit-for-bit deterministic"
            );
            assert_eq!(r1.n_eff_ci_low.to_bits(), r2.n_eff_ci_low.to_bits());
            assert_eq!(r1.n_eff_ci_high.to_bits(), r2.n_eff_ci_high.to_bits());

            // Independent panel -> Condorcet predicted accuracy closely matches empirical accuracy.
            assert!(
                r1.condorcet_gap.abs() < 0.08,
                "C={n_classes}: condorcet_pred={} vs panel_acc={} (gap={})",
                r1.condorcet_predicted_accuracy,
                r1.panel_accuracy,
                r1.condorcet_gap
            );
        }

        // Large-K (> 64) fallback path for binary DP and bootstrap CI (K=70, N=60).
        let n = 60usize;
        let k = 70usize;
        let mut rng = Rng::from_seed(0xBEEF_7070);
        let mut errs = vec![0u8; n * k];
        for i in 0..n {
            let shared = rng.next_f64() < 0.20;
            for j in 0..k {
                errs[i * k + j] = u8::from(shared || rng.next_f64() < 0.15);
            }
        }
        let rep70 = PanelAudit::from_binary_errors(&errs, n, k, None)
            .unwrap()
            .run_with_config(&PanelAuditConfig {
                n_condorcet_sims: 300,
                n_permutations: 300,
                n_bootstrap: 300,
                seed: 7,
                ..Default::default()
            });
        assert!(rep70.condorcet_predicted_accuracy > 0.95);
        assert!(rep70.n_eff_ci_low >= 1.0 && rep70.n_eff_ci_low <= rep70.n_eff_ci_high);
        assert!(rep70.n_eff_ci_high <= k as f64);
        assert!(rep70.permutation_p_value <= 0.02);
    }

    #[test]
    fn condorcet_i16_and_i31_exact_unbiased_parity() {
        // Test C=3 and C=4 across denom=100 (C3I16/C4I16), denom=512 (boundary), and denom=600 (C3/C4 i31)
        // against exact brute-force K=3 enumeration of plurality credit.
        for &(n_classes, denom) in &[(3usize, 100usize), (3, 512), (3, 600), (4, 100), (4, 600)] {
            let k = 3usize;
            let gold = vec![0u32; denom];
            let mut votes = vec![0u32; denom * k];
            // Heterogeneous per-judge class counts summing to denom (including a 0-count edge case).
            let mut judge_probs = vec![vec![0.0_f64; n_classes]; k];
            for j in 0..k {
                let c0 = (denom * (45 + 10 * j)) / 100;
                let c1 = if j == 2 { 0 } else { (denom - c0) / 2 };
                let c2 = if n_classes == 3 {
                    denom - c0 - c1
                } else {
                    (denom - c0 - c1) / 2
                };
                let c3 = if n_classes == 4 { denom - c0 - c1 - c2 } else { 0 };
                let counts = if n_classes == 3 {
                    vec![c0, c1, c2]
                } else {
                    vec![c0, c1, c2, c3]
                };
                let mut pos = 0usize;
                for (cls, &cnt) in counts.iter().enumerate() {
                    judge_probs[j][cls] = (cnt as f64) / (denom as f64);
                    for _ in 0..cnt {
                        votes[pos * k + j] = cls as u32;
                        pos += 1;
                    }
                }
                assert_eq!(pos, denom);
            }

            // Exact expected plurality credit by enumerating all C^3 outcomes.
            let mut exact = 0.0_f64;
            for a in 0..n_classes {
                for b in 0..n_classes {
                    for c in 0..n_classes {
                        let prob = judge_probs[0][a] * judge_probs[1][b] * judge_probs[2][c];
                        let mut scratch = [a as u32, b as u32, c as u32];
                        exact += prob * plurality_credit(&mut scratch, 0);
                    }
                }
            }

            let rep = PanelAudit::from_votes(&votes, &gold, denom, k, None)
                .unwrap()
                .run_with_config(&PanelAuditConfig {
                    n_condorcet_sims: 800,
                    n_permutations: 0,
                    n_bootstrap: 0,
                    seed: 12345,
                    ..Default::default()
                });
            let err = (rep.condorcet_predicted_accuracy - exact).abs();
            // Standard error with >= 80,000 draws is <= 0.0018; 5-sigma guard is 0.01.
            assert!(
                err < 0.01,
                "C={n_classes}, denom={denom}: got {}, exact={}, err={err}",
                rep.condorcet_predicted_accuracy,
                exact
            );
        }
    }
}



