//! Compact (`<= 1024` bytes) Associative, Commutative Distributed Evidence Monoids (`EvidenceSketch`).
//!
//! Designed for distributed multi-agent fleets (1,000+ workers merging calibration nonconformity
//! distributions and anytime-valid E-process evidence in sub-microsecond time):
//! - Fixed-size **712-byte** little-endian binary wire format with magic header (`b"CLTR"`),
//!   version, flags, compensated mean/variance sufficient statistics (`CompSum`), a 64-bin
//!   signed log-spaced conformal nonconformity histogram, a 16-point geometric betting
//!   log-wealth vector, and an FNV-1a 64-bit integrity checksum.
//! - Zero-dependency JSON serialization (`to_json` / `from_json`).
//! - Exact associative and commutative `merge`, plus exact Leave-One-Out `subtract` for
//!   cold-start peer borrowing without self-contamination.

use crate::math::{fnv1a_64, log_mean_exp_grid, CompSum, DEFAULT_BET_GRID, GRID_LEN};

/// Wire format magic bytes (`b"CLTR"`).
pub const SKETCH_MAGIC: [u8; 4] = *b"CLTR";

/// Current binary wire format version.
pub const SKETCH_VERSION: u16 = 1;

/// Number of signed log-spaced histogram bins (32 negative + 32 positive).
pub const HIST_BINS: usize = 64;

/// Fixed binary serialized size in bytes:
/// - Header (`magic`: 4, `version`: 2, `flags`: 2, `n_samples`: 8) = 16 B
/// - Stats (`sum_x`: 16, `sum_x2`: 16, `min_val`: 8, `max_val`: 8) = 48 B
/// - Histogram (`bins: [u64; 64]`) = 512 B
/// - Betting log-wealth (`bet_log_wealth: [f64; 16]`) = 128 B
/// - Checksum (`fnv1a_64`: 8) = 8 B
/// Total = 712 bytes (strictly `<= 1024` bytes).
pub const WIRE_SIZE: usize = 16 + 48 + (HIST_BINS * 8) + (GRID_LEN * 8) + 8;

/// Base scale `s0` for the signed log-spaced histogram transform `sign(x) * ln(1 + |x| / s0)`.
const HIST_S0: f64 = 0.02;

/// Maximum finite inner bin edge (`200.0`); values with `|x| >= 200` map to tail bins 0 / 63,
/// with interpolation bounded by the exact observed `min_val` / `max_val`.
const HIST_X_MAX: f64 = 200.0;

/// Step size in log1p-space across the 31 inner bins per sign: `ln(1 + 200 / 0.02) / 31`.
const HIST_DY: f64 = 9.210440366976517 / 31.0;

/// Compact, associative, commutative distributed evidence monoid (`712` bytes serialized).
#[derive(Clone, Debug)]
pub struct EvidenceSketch {
    /// Wire magic header (`b"CLTR"`).
    pub magic: [u8; 4],
    /// Wire format version.
    pub version: u16,
    /// Optional metadata flags (preserved via bitwise OR on merge).
    pub flags: u16,
    /// Total number of scalar nonconformity/metric samples absorbed.
    pub n_samples: u64,
    /// Compensated sum of observed scalar values.
    pub sum_x: CompSum,
    /// Compensated sum of squared observed scalar values.
    pub sum_x2: CompSum,
    /// Minimum observed scalar value (`0.0` when `n_samples == 0`).
    pub min_val: f64,
    /// Maximum observed scalar value (`0.0` when `n_samples == 0`).
    pub max_val: f64,
    /// 64-bin signed log-spaced conformal nonconformity histogram.
    pub bins: [u64; HIST_BINS],
    /// 16-point geometric betting log-wealth vector (`sum log(1 + lambda_g * x_i)`).
    pub bet_log_wealth: [f64; GRID_LEN],
}

impl Default for EvidenceSketch {
    fn default() -> Self {
        Self::new()
    }
}

impl EvidenceSketch {
    /// Create an empty identity `EvidenceSketch`.
    #[inline]
    pub fn new() -> Self {
        Self {
            magic: SKETCH_MAGIC,
            version: SKETCH_VERSION,
            flags: 0,
            n_samples: 0,
            sum_x: CompSum::default(),
            sum_x2: CompSum::default(),
            min_val: 0.0,
            max_val: 0.0,
            bins: [0u64; HIST_BINS],
            bet_log_wealth: [0.0; GRID_LEN],
        }
    }

    /// Observe a single scalar nonconformity / metric value into the sufficient statistics
    /// and 64-bin signed log-spaced histogram.
    ///
    /// Non-finite values (`NaN`, `Inf`, or values whose magnitude would overflow `sum_x2`)
    /// are ignored so corrupt or extreme inputs can never poison the monoid.
    #[inline]
    pub fn observe_score(&mut self, score: f64) {
        if !score.is_finite() {
            return;
        }
        let sq = score * score;
        if !sq.is_finite() {
            return;
        }
        let mut next_x = self.sum_x;
        next_x.add(score);
        let mut next_x2 = self.sum_x2;
        next_x2.add(sq);
        if !next_x.value().is_finite() || !next_x2.value().is_finite() {
            return;
        }
        if self.n_samples == 0 {
            self.min_val = score;
            self.max_val = score;
        } else {
            if score < self.min_val {
                self.min_val = score;
            }
            if score > self.max_val {
                self.max_val = score;
            }
        }
        self.n_samples += 1;
        self.sum_x = next_x;
        self.sum_x2 = next_x2;
        let idx = bin_index(score);
        self.bins[idx] += 1;
    }

    /// Absorb a bounded E-process betting signal `x` (clamped to `[-1.0, 1.0]`) into
    /// the 16-point geometric betting log-wealth vector.
    ///
    /// Non-finite signals are ignored.
    #[inline]
    pub fn observe_bet(&mut self, bounded_signal: f64) {
        if !bounded_signal.is_finite() {
            return;
        }
        let x = bounded_signal.clamp(-1.0, 1.0);
        if x == 0.0 {
            return;
        }
        for g in 0..GRID_LEN {
            self.bet_log_wealth[g] += (DEFAULT_BET_GRID[g] * x).ln_1p();
        }
    }

    /// Convenience helper observing both a scalar score and an optional betting signal.
    #[inline]
    pub fn observe(&mut self, score: f64, bet_signal: Option<f64>) {
        self.observe_score(score);
        if let Some(sig) = bet_signal {
            self.observe_bet(sig);
        }
    }

    /// Associative, commutative monoid merge: absorbs `other` into `self` in-place.
    ///
    /// - Histogram counts and `n_samples` merge by exact integer addition.
    /// - `sum_x` and `sum_x2` merge via compensated Neumaier summation (`CompSum::merge`).
    /// - `min_val` / `max_val` merge via exact lattice `min` / `max` with empty identity.
    /// - `bet_log_wealth` merges additively because
    ///   `log(prod_{i in A U B} f_i(lambda)) = sum_{i in A} log f_i + sum_{i in B} log f_i`.
    #[inline]
    pub fn merge(&mut self, other: &Self) {
        let other_empty = other.n_samples == 0
            && other.flags == 0
            && other.bet_log_wealth == [0.0; GRID_LEN];
        if other_empty {
            return;
        }
        let self_empty = self.n_samples == 0
            && self.flags == 0
            && self.bet_log_wealth == [0.0; GRID_LEN];
        if self_empty {
            *self = other.clone();
            return;
        }

        self.flags |= other.flags;

        if other.n_samples > 0 {
            if self.n_samples == 0 {
                self.min_val = other.min_val;
                self.max_val = other.max_val;
            } else {
                if other.min_val < self.min_val {
                    self.min_val = other.min_val;
                }
                if other.max_val > self.max_val {
                    self.max_val = other.max_val;
                }
            }
            self.n_samples = self.n_samples.saturating_add(other.n_samples);
            self.sum_x.merge(other.sum_x);
            self.sum_x2.merge(other.sum_x2);
            for i in 0..HIST_BINS {
                self.bins[i] = self.bins[i].saturating_add(other.bins[i]);
            }
        }

        for g in 0..GRID_LEN {
            self.bet_log_wealth[g] += other.bet_log_wealth[g];
        }
    }

    /// Exact Leave-One-Out (LOO) peer subtraction: returns `self - local`.
    ///
    /// Enables cold-start agents to borrow fleet-wide calibration quantiles and E-process
    /// evidence from peers without contaminating the calibration distribution with their
    /// own local observations.
    pub fn subtract(&self, local: &Self) -> Result<Self, &'static str> {
        if local.n_samples > self.n_samples {
            return Err("local n_samples exceeds global n_samples");
        }
        let mut bins = [0u64; HIST_BINS];
        for i in 0..HIST_BINS {
            if local.bins[i] > self.bins[i] {
                return Err("local histogram bin count exceeds global bin count");
            }
            bins[i] = self.bins[i] - local.bins[i];
        }

        let n_samples = self.n_samples - local.n_samples;
        let mut sum_x = self.sum_x;
        sum_x.add(-local.sum_x.sum);
        sum_x.add(-local.sum_x.comp);

        let mut sum_x2 = self.sum_x2;
        sum_x2.add(-local.sum_x2.sum);
        sum_x2.add(-local.sum_x2.comp);

        let mut bet_log_wealth = [0.0_f64; GRID_LEN];
        for g in 0..GRID_LEN {
            let v = self.bet_log_wealth[g] - local.bet_log_wealth[g];
            if !v.is_finite() {
                return Err("non-finite betting log-wealth after subtraction");
            }
            bet_log_wealth[g] = v;
        }

        let (min_val, max_val) = if n_samples == 0 {
            sum_x = CompSum::default();
            sum_x2 = CompSum::default();
            (0.0, 0.0)
        } else {
            let new_min = if local.n_samples == 0 || local.min_val > self.min_val {
                self.min_val
            } else {
                // Local may have held the minimum: tighten to lowest populated bin bound.
                first_nonempty_bin(&bins)
                    .map(|b| {
                        let (lo, _) = bin_bounds(b);
                        if lo.is_finite() {
                            lo.max(self.min_val)
                        } else {
                            self.min_val
                        }
                    })
                    .unwrap_or(self.min_val)
            };

            let new_max = if local.n_samples == 0 || local.max_val < self.max_val {
                self.max_val
            } else {
                // Local may have held the maximum: tighten to highest populated bin bound.
                last_nonempty_bin(&bins)
                    .map(|b| {
                        let (_, hi) = bin_bounds(b);
                        if hi.is_finite() {
                            hi.min(self.max_val)
                        } else {
                            self.max_val
                        }
                    })
                    .unwrap_or(self.max_val)
            };

            let min_c = new_min.min(new_max);
            let max_c = new_max.max(min_c);
            (min_c, max_c)
        };

        Ok(Self {
            magic: SKETCH_MAGIC,
            version: SKETCH_VERSION,
            flags: self.flags,
            n_samples,
            sum_x,
            sum_x2,
            min_val,
            max_val,
            bins,
            bet_log_wealth,
        })
    }

    /// Sample mean of observed scores (`0.0` if empty).
    #[inline]
    pub fn mean(&self) -> f64 {
        if self.n_samples == 0 {
            0.0
        } else {
            self.sum_x.value() / (self.n_samples as f64)
        }
    }

    /// Unbiased sample variance of observed scores (`0.0` if `n_samples < 2`).
    #[inline]
    pub fn variance(&self) -> f64 {
        if self.n_samples < 2 {
            return 0.0;
        }
        let n = self.n_samples as f64;
        let sx = self.sum_x.value();
        let sx2 = self.sum_x2.value();
        let ss = (sx2 - (sx * sx) / n).max(0.0);
        ss / (n - 1.0)
    }

    /// Log E-value of the mixture betting supermartingale (`log_mean_exp_grid`).
    #[inline]
    pub fn log_e_value(&self) -> f64 {
        log_mean_exp_grid(&self.bet_log_wealth)
    }

    /// Anytime-valid E-value `mean_g exp(bet_log_wealth[g])`.
    #[inline]
    pub fn e_value(&self) -> f64 {
        self.log_e_value().exp()
    }

    /// Fast histogram quantile estimate for `q in [0, 1]` using exact `[min_val, max_val]`
    /// clamped linear interpolation inside the target signed log-spaced bin.
    ///
    /// Returns `0.0` if the sketch is empty or `q` is non-finite.
    pub fn quantile(&self, q: f64) -> f64 {
        if self.n_samples == 0 || !q.is_finite() {
            return 0.0;
        }
        let q = q.clamp(0.0, 1.0);
        if q == 0.0 {
            return self.min_val;
        }
        if q == 1.0 {
            return self.max_val;
        }

        let target = q * (self.n_samples as f64);
        let mut cum = 0u64;
        for (idx, &count) in self.bins.iter().enumerate() {
            if count == 0 {
                continue;
            }
            let next_cum = cum + count;
            if (next_cum as f64) >= target {
                let frac = ((target - (cum as f64)) / (count as f64)).clamp(0.0, 1.0);
                let (raw_lo, raw_hi) = bin_bounds(idx);
                let lo = raw_lo.max(self.min_val);
                let hi = raw_hi.min(self.max_val);
                if lo >= hi {
                    return lo;
                }
                return lo + frac * (hi - lo);
            }
            cum = next_cum;
        }
        self.max_val
    }

    /// Finite-sample distributed conformal threshold at miscoverage level `alpha in (0, 1)`.
    ///
    /// Computes the conservative conformal rank `ceil((n + 1) * (1 - alpha)) / n` and queries
    /// the histogram quantile (returning `f64::INFINITY` if `n == 0` or rank exceeds `n`).
    pub fn conformal_quantile(&self, alpha: f64) -> f64 {
        if self.n_samples == 0 || !alpha.is_finite() || alpha <= 0.0 {
            return f64::INFINITY;
        }
        if alpha >= 1.0 {
            return self.min_val;
        }
        let k = (((self.n_samples + 1) as f64) * (1.0 - alpha)).ceil();
        if k > (self.n_samples as f64) {
            return f64::INFINITY;
        }
        let q = (k / (self.n_samples as f64)).clamp(0.0, 1.0);
        self.quantile(q)
    }

    /// Zero-allocation serialization into a fixed `[u8; WIRE_SIZE]` (`712` bytes) buffer.
    pub fn to_fixed_bytes(&self) -> [u8; WIRE_SIZE] {
        let mut buf = [0u8; WIRE_SIZE];
        let mut off = 0usize;

        buf[off..off + 4].copy_from_slice(&SKETCH_MAGIC);
        off += 4;
        buf[off..off + 2].copy_from_slice(&SKETCH_VERSION.to_le_bytes());
        off += 2;
        buf[off..off + 2].copy_from_slice(&self.flags.to_le_bytes());
        off += 2;
        buf[off..off + 8].copy_from_slice(&self.n_samples.to_le_bytes());
        off += 8;

        write_f64_le(&mut buf, &mut off, self.sum_x.sum);
        write_f64_le(&mut buf, &mut off, self.sum_x.comp);
        write_f64_le(&mut buf, &mut off, self.sum_x2.sum);
        write_f64_le(&mut buf, &mut off, self.sum_x2.comp);
        write_f64_le(&mut buf, &mut off, self.min_val);
        write_f64_le(&mut buf, &mut off, self.max_val);

        for &c in &self.bins {
            buf[off..off + 8].copy_from_slice(&c.to_le_bytes());
            off += 8;
        }
        for &w in &self.bet_log_wealth {
            write_f64_le(&mut buf, &mut off, w);
        }

        debug_assert_eq!(off, WIRE_SIZE - 8);
        let checksum = fnv1a_64(&buf[..off]);
        buf[off..off + 8].copy_from_slice(&checksum.to_le_bytes());
        buf
    }

    /// Serialize into a `Vec<u8>` of exact length `WIRE_SIZE == 712` (strictly `<= 1024` bytes).
    pub fn to_bytes(&self) -> Vec<u8> {
        self.to_fixed_bytes().to_vec()
    }

    /// Deserialize from binary wire bytes, validating length, magic, version, checksum,
    /// finiteness of all floating-point fields, and exact histogram-sum integrity.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() != WIRE_SIZE {
            return Err("invalid EvidenceSketch wire length");
        }
        if bytes[0..4] != SKETCH_MAGIC {
            return Err("invalid EvidenceSketch magic bytes");
        }
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        if version != SKETCH_VERSION {
            return Err("unsupported EvidenceSketch version");
        }

        let payload_len = WIRE_SIZE - 8;
        let expected_chk = fnv1a_64(&bytes[..payload_len]);
        let stored_chk = read_u64_le(bytes, payload_len);
        if expected_chk != stored_chk {
            return Err("EvidenceSketch checksum mismatch");
        }

        let flags = u16::from_le_bytes([bytes[6], bytes[7]]);
        let n_samples = read_u64_le(bytes, 8);
        let mut off = 16usize;

        let sx_sum = read_finite_f64_le(bytes, &mut off)?;
        let sx_comp = read_finite_f64_le(bytes, &mut off)?;
        let sx2_sum = read_finite_f64_le(bytes, &mut off)?;
        let sx2_comp = read_finite_f64_le(bytes, &mut off)?;
        let min_val = read_finite_f64_le(bytes, &mut off)?;
        let max_val = read_finite_f64_le(bytes, &mut off)?;

        if n_samples > 0 && min_val > max_val {
            return Err("invalid EvidenceSketch bounds: min_val > max_val");
        }

        let mut bins = [0u64; HIST_BINS];
        let mut bin_sum = 0u64;
        for slot in &mut bins {
            let c = read_u64_le(bytes, off);
            off += 8;
            *slot = c;
            bin_sum = bin_sum
                .checked_add(c)
                .ok_or("EvidenceSketch histogram count overflow")?;
        }
        if bin_sum != n_samples {
            return Err("EvidenceSketch histogram sum does not match n_samples");
        }

        let mut bet_log_wealth = [0.0_f64; GRID_LEN];
        for slot in &mut bet_log_wealth {
            *slot = read_finite_f64_le(bytes, &mut off)?;
        }

        Ok(Self {
            magic: SKETCH_MAGIC,
            version,
            flags,
            n_samples,
            sum_x: CompSum {
                sum: sx_sum,
                comp: sx_comp,
            },
            sum_x2: CompSum {
                sum: sx2_sum,
                comp: sx2_comp,
            },
            min_val,
            max_val,
            bins,
            bet_log_wealth,
        })
    }

    /// Zero-dependency canonical JSON serialization.
    pub fn to_json(&self) -> String {
        let mut s = String::with_capacity(1400);
        s.push_str("{\"magic\":\"CLTR\",\"version\":");
        s.push_str(&self.version.to_string());
        s.push_str(",\"flags\":");
        s.push_str(&self.flags.to_string());
        s.push_str(",\"n_samples\":");
        s.push_str(&self.n_samples.to_string());
        s.push_str(",\"sum_x\":[");
        push_f64_json(&mut s, self.sum_x.sum);
        s.push(',');
        push_f64_json(&mut s, self.sum_x.comp);
        s.push_str("],\"sum_x2\":[");
        push_f64_json(&mut s, self.sum_x2.sum);
        s.push(',');
        push_f64_json(&mut s, self.sum_x2.comp);
        s.push_str("],\"min_val\":");
        push_f64_json(&mut s, self.min_val);
        s.push_str(",\"max_val\":");
        push_f64_json(&mut s, self.max_val);
        s.push_str(",\"bins\":[");
        for (i, &c) in self.bins.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&c.to_string());
        }
        s.push_str("],\"bet_log_wealth\":[");
        for (i, &w) in self.bet_log_wealth.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            push_f64_json(&mut s, w);
        }
        s.push_str("]}");
        s
    }

    /// Zero-dependency JSON deserialization with full structural and finiteness validation.
    pub fn from_json(json: &str) -> Result<Self, &'static str> {
        let magic_str = extract_json_str(json, "\"magic\"")?;
        if magic_str != "CLTR" {
            return Err("invalid EvidenceSketch JSON magic");
        }
        let version = u16::try_from(extract_json_u64(json, "\"version\"")?)
            .map_err(|_| "EvidenceSketch JSON version out of u16 range")?;
        if version != SKETCH_VERSION {
            return Err("unsupported EvidenceSketch JSON version");
        }
        let flags = u16::try_from(extract_json_u64(json, "\"flags\"")?)
            .map_err(|_| "EvidenceSketch JSON flags out of u16 range")?;
        let n_samples = extract_json_u64(json, "\"n_samples\"")?;
        let sx = extract_json_f64_array(json, "\"sum_x\"", 2)?;
        let sx2 = extract_json_f64_array(json, "\"sum_x2\"", 2)?;
        let min_val = extract_json_f64(json, "\"min_val\"")?;
        let max_val = extract_json_f64(json, "\"max_val\"")?;
        if n_samples > 0 && min_val > max_val {
            return Err("invalid EvidenceSketch JSON bounds");
        }

        let bins_vec = extract_json_u64_array(json, "\"bins\"", HIST_BINS)?;
        let mut bins = [0u64; HIST_BINS];
        let mut bin_sum = 0u64;
        for i in 0..HIST_BINS {
            bins[i] = bins_vec[i];
            bin_sum = bin_sum
                .checked_add(bins[i])
                .ok_or("EvidenceSketch JSON bin overflow")?;
        }
        if bin_sum != n_samples {
            return Err("EvidenceSketch JSON histogram sum mismatch");
        }

        let bet_vec = extract_json_f64_array(json, "\"bet_log_wealth\"", GRID_LEN)?;
        let mut bet_log_wealth = [0.0_f64; GRID_LEN];
        bet_log_wealth.copy_from_slice(&bet_vec);

        Ok(Self {
            magic: SKETCH_MAGIC,
            version,
            flags,
            n_samples,
            sum_x: CompSum {
                sum: sx[0],
                comp: sx[1],
            },
            sum_x2: CompSum {
                sum: sx2[0],
                comp: sx2[1],
            },
            min_val,
            max_val,
            bins,
            bet_log_wealth,
        })
    }
}

/// Map a finite scalar score `x` to its monotone signed log-spaced bin index in `0..64`.
#[inline]
pub fn bin_index(x: f64) -> usize {
    if x >= 0.0 {
        if x >= HIST_X_MAX {
            return HIST_BINS - 1;
        }
        let k = ((x / HIST_S0).ln_1p() / HIST_DY).floor() as usize;
        32 + k.min(31)
    } else {
        let ax = -x;
        if ax >= HIST_X_MAX {
            return 0;
        }
        let k = ((ax / HIST_S0).ln_1p() / HIST_DY).floor() as usize;
        31 - k.min(31)
    }
}

/// Exact `[lower, upper)` nominal edges of histogram bin `idx in 0..64`.
#[inline]
pub fn bin_bounds(idx: usize) -> (f64, f64) {
    if idx >= 32 {
        let k = (idx - 32).min(31);
        let lo = HIST_S0 * ((k as f64) * HIST_DY).exp_m1();
        let hi = if k == 31 {
            f64::INFINITY
        } else {
            HIST_S0 * (((k + 1) as f64) * HIST_DY).exp_m1()
        };
        (lo, hi)
    } else {
        let k = 31 - idx;
        let hi = -HIST_S0 * ((k as f64) * HIST_DY).exp_m1();
        let lo = if k == 31 {
            f64::NEG_INFINITY
        } else {
            -HIST_S0 * (((k + 1) as f64) * HIST_DY).exp_m1()
        };
        (lo, hi)
    }
}

#[inline]
fn first_nonempty_bin(bins: &[u64; HIST_BINS]) -> Option<usize> {
    bins.iter().position(|&c| c > 0)
}

#[inline]
fn last_nonempty_bin(bins: &[u64; HIST_BINS]) -> Option<usize> {
    bins.iter().rposition(|&c| c > 0)
}

#[inline]
fn write_f64_le(buf: &mut [u8], off: &mut usize, v: f64) {
    let clean = if v.is_finite() { v } else { 0.0 };
    buf[*off..*off + 8].copy_from_slice(&clean.to_le_bytes());
    *off += 8;
}

#[inline]
fn read_u64_le(bytes: &[u8], off: usize) -> u64 {
    u64::from_le_bytes([
        bytes[off],
        bytes[off + 1],
        bytes[off + 2],
        bytes[off + 3],
        bytes[off + 4],
        bytes[off + 5],
        bytes[off + 6],
        bytes[off + 7],
    ])
}

#[inline]
fn read_finite_f64_le(bytes: &[u8], off: &mut usize) -> Result<f64, &'static str> {
    let v = f64::from_le_bytes([
        bytes[*off],
        bytes[*off + 1],
        bytes[*off + 2],
        bytes[*off + 3],
        bytes[*off + 4],
        bytes[*off + 5],
        bytes[*off + 6],
        bytes[*off + 7],
    ]);
    *off += 8;
    if !v.is_finite() {
        return Err("non-finite float in EvidenceSketch wire payload");
    }
    Ok(v)
}

fn push_f64_json(out: &mut String, v: f64) {
    let clean = if v.is_finite() { v } else { 0.0 };
    // Format with full round-trip precision.
    use std::fmt::Write;
    let _ = write!(out, "{clean:.17e}");
}

fn find_json_value_start<'a>(json: &'a str, key: &str) -> Result<&'a str, &'static str> {
    let pos = json.find(key).ok_or("missing key in EvidenceSketch JSON")?;
    let rest = &json[pos + key.len()..];
    let colon = rest
        .find(':')
        .ok_or("missing colon in EvidenceSketch JSON")?;
    Ok(rest[colon + 1..].trim_start())
}

fn extract_json_str<'a>(json: &'a str, key: &str) -> Result<&'a str, &'static str> {
    let rest = find_json_value_start(json, key)?;
    let inner = rest
        .strip_prefix('"')
        .ok_or("expected JSON string opening quote")?;
    let end = inner
        .find('"')
        .ok_or("expected JSON string closing quote")?;
    Ok(&inner[..end])
}

fn extract_json_token<'a>(json: &'a str, key: &str) -> Result<&'a str, &'static str> {
    let rest = find_json_value_start(json, key)?;
    let end = rest
        .find(|c: char| c == ',' || c == '}' || c == ']' || c.is_whitespace())
        .unwrap_or(rest.len());
    Ok(&rest[..end])
}

fn extract_json_u64(json: &str, key: &str) -> Result<u64, &'static str> {
    extract_json_token(json, key)?
        .parse::<u64>()
        .map_err(|_| "invalid u64 in EvidenceSketch JSON")
}

fn extract_json_f64(json: &str, key: &str) -> Result<f64, &'static str> {
    let v = extract_json_token(json, key)?
        .parse::<f64>()
        .map_err(|_| "invalid f64 in EvidenceSketch JSON")?;
    if !v.is_finite() {
        return Err("non-finite f64 in EvidenceSketch JSON");
    }
    Ok(v)
}

fn extract_json_array_body<'a>(json: &'a str, key: &str) -> Result<&'a str, &'static str> {
    let rest = find_json_value_start(json, key)?;
    let inner = rest
        .strip_prefix('[')
        .ok_or("expected '[' in EvidenceSketch JSON")?;
    let end = inner
        .find(']')
        .ok_or("expected ']' in EvidenceSketch JSON")?;
    Ok(&inner[..end])
}

fn extract_json_u64_array(json: &str, key: &str, expected_len: usize) -> Result<Vec<u64>, &'static str> {
    let body = extract_json_array_body(json, key)?;
    let mut out = Vec::with_capacity(expected_len);
    for tok in body.split(',') {
        let v = tok
            .trim()
            .parse::<u64>()
            .map_err(|_| "invalid u64 array element in EvidenceSketch JSON")?;
        out.push(v);
    }
    if out.len() != expected_len {
        return Err("unexpected u64 array length in EvidenceSketch JSON");
    }
    Ok(out)
}

fn extract_json_f64_array(json: &str, key: &str, expected_len: usize) -> Result<Vec<f64>, &'static str> {
    let body = extract_json_array_body(json, key)?;
    let mut out = Vec::with_capacity(expected_len);
    for tok in body.split(',') {
        let v = tok
            .trim()
            .parse::<f64>()
            .map_err(|_| "invalid f64 array element in EvidenceSketch JSON")?;
        if !v.is_finite() {
            return Err("non-finite f64 array element in EvidenceSketch JSON");
        }
        out.push(v);
    }
    if out.len() != expected_len {
        return Err("unexpected f64 array length in EvidenceSketch JSON");
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Rng;

    fn assert_sketches_close(a: &EvidenceSketch, b: &EvidenceSketch, tol: f64) {
        assert_eq!(a.n_samples, b.n_samples);
        assert_eq!(a.bins, b.bins);
        assert!(
            (a.sum_x.value() - b.sum_x.value()).abs() <= tol,
            "sum_x mismatch: {} vs {}",
            a.sum_x.value(),
            b.sum_x.value()
        );
        assert!(
            (a.sum_x2.value() - b.sum_x2.value()).abs() <= tol,
            "sum_x2 mismatch: {} vs {}",
            a.sum_x2.value(),
            b.sum_x2.value()
        );
        assert!((a.min_val - b.min_val).abs() <= tol);
        assert!((a.max_val - b.max_val).abs() <= tol);
        for g in 0..GRID_LEN {
            assert!(
                (a.bet_log_wealth[g] - b.bet_log_wealth[g]).abs() <= tol,
                "bet_log_wealth[{g}] mismatch: {} vs {}",
                a.bet_log_wealth[g],
                b.bet_log_wealth[g]
            );
        }
    }

    fn random_sketch(seed: u64, n: usize) -> EvidenceSketch {
        let mut rng = Rng::from_seed(seed);
        let mut s = EvidenceSketch::new();
        for _ in 0..n {
            let x = (rng.next_f64() - 0.35) * 12.0;
            let bet = (rng.next_f64() - 0.45) * 1.6;
            s.observe(x, Some(bet));
        }
        s
    }

    #[test]
    fn wire_size_strictly_under_1024_bytes_and_roundtrips() {
        assert_eq!(WIRE_SIZE, 712);
        assert!(WIRE_SIZE <= 1024, "wire size {} must be <= 1024 B", WIRE_SIZE);

        // Empty sketch roundtrip (checks finite min/max on empty identity!).
        let empty = EvidenceSketch::new();
        let empty_bytes = empty.to_bytes();
        assert_eq!(empty_bytes.len(), WIRE_SIZE);
        assert!(empty_bytes.len() <= 1024);
        let empty_rt = EvidenceSketch::from_bytes(&empty_bytes).unwrap();
        assert_sketches_close(&empty, &empty_rt, 0.0);

        // Populated sketch binary + JSON roundtrip.
        let s = random_sketch(42, 500);
        let bytes = s.to_bytes();
        assert!(bytes.len() <= 1024);
        let rt = EvidenceSketch::from_bytes(&bytes).unwrap();
        assert_sketches_close(&s, &rt, 0.0);

        let json = s.to_json();
        let rt_json = EvidenceSketch::from_json(&json).unwrap();
        assert_sketches_close(&s, &rt_json, 1e-12);
    }

    #[test]
    fn wire_deserialization_rejects_corruption() {
        let s = random_sketch(99, 100);
        let valid = s.to_bytes();

        // Wrong length
        assert!(EvidenceSketch::from_bytes(&valid[..100]).is_err());

        // Bad magic
        let mut bad_magic = valid.clone();
        bad_magic[0] = b'X';
        assert!(EvidenceSketch::from_bytes(&bad_magic).is_err());

        // Bad version (recomputed checksum so we specifically test version check)
        let mut bad_ver = valid.clone();
        bad_ver[4..6].copy_from_slice(&99u16.to_le_bytes());
        let chk = fnv1a_64(&bad_ver[..WIRE_SIZE - 8]);
        bad_ver[WIRE_SIZE - 8..].copy_from_slice(&chk.to_le_bytes());
        assert!(EvidenceSketch::from_bytes(&bad_ver).is_err());

        // Bad checksum (bit flip)
        let mut flipped = valid.clone();
        flipped[100] ^= 0x01;
        assert!(EvidenceSketch::from_bytes(&flipped).is_err());

        // Injected NaN float (with valid checksum)
        let mut nan_buf = valid.clone();
        nan_buf[16..24].copy_from_slice(&f64::NAN.to_le_bytes());
        let chk = fnv1a_64(&nan_buf[..WIRE_SIZE - 8]);
        nan_buf[WIRE_SIZE - 8..].copy_from_slice(&chk.to_le_bytes());
        assert!(EvidenceSketch::from_bytes(&nan_buf).is_err());

        // Histogram sum mismatch (with valid checksum)
        let mut bad_hist = valid.clone();
        bad_hist[8..16].copy_from_slice(&(s.n_samples + 1).to_le_bytes());
        let chk = fnv1a_64(&bad_hist[..WIRE_SIZE - 8]);
        bad_hist[WIRE_SIZE - 8..].copy_from_slice(&chk.to_le_bytes());
        assert!(EvidenceSketch::from_bytes(&bad_hist).is_err());
    }

    #[test]
    fn exact_associativity_commutativity_and_identity() {
        let a = random_sketch(1, 250);
        let b = random_sketch(2, 310);
        let c = random_sketch(3, 190);
        let empty = EvidenceSketch::new();

        // Identity
        let mut a_id = a.clone();
        a_id.merge(&empty);
        assert_sketches_close(&a, &a_id, 1e-12);
        let mut id_a = empty.clone();
        id_a.merge(&a);
        assert_sketches_close(&a, &id_a, 1e-12);

        // Commutativity: merge(a, b) == merge(b, a)
        let mut ab = a.clone();
        ab.merge(&b);
        let mut ba = b.clone();
        ba.merge(&a);
        assert_sketches_close(&ab, &ba, 1e-12);

        // Associativity: merge(merge(a, b), c) == merge(a, merge(b, c))
        let mut ab_c = a.clone();
        ab_c.merge(&b);
        ab_c.merge(&c);

        let mut bc = b.clone();
        bc.merge(&c);
        let mut a_bc = a.clone();
        a_bc.merge(&bc);

        assert_sketches_close(&ab_c, &a_bc, 1e-12);
    }

    #[test]
    fn exact_leave_one_out_subtraction() {
        let peers = random_sketch(10, 400);
        let local = random_sketch(20, 80);

        let mut global = peers.clone();
        global.merge(&local);

        let recovered = global.subtract(&local).unwrap();
        assert_eq!(recovered.n_samples, peers.n_samples);
        assert_eq!(recovered.bins, peers.bins);
        assert!((recovered.mean() - peers.mean()).abs() < 1e-12);
        assert!((recovered.variance() - peers.variance()).abs() < 1e-10);
        for g in 0..GRID_LEN {
            assert!((recovered.bet_log_wealth[g] - peers.bet_log_wealth[g]).abs() < 1e-12);
        }
        assert!((recovered.log_e_value() - peers.log_e_value()).abs() < 1e-12);

        // Subtracting everything yields exact empty identity.
        let zeroed = local.subtract(&local).unwrap();
        assert_eq!(zeroed.n_samples, 0);
        assert_eq!(zeroed.bins, [0u64; HIST_BINS]);
        assert!((zeroed.log_e_value()).abs() < 1e-12);

        // Underflow must return Err.
        assert!(local.subtract(&global).is_err());
    }

    #[test]
    fn distributed_1000_worker_fleet_matches_single_node_and_quantiles_are_accurate() {
        let mut rng = Rng::from_seed(55555);
        let mut single = EvidenceSketch::new();
        let mut fleet: Vec<EvidenceSketch> = (0..1000).map(|_| EvidenceSketch::new()).collect();

        // 1000 workers x 50 samples = 50,000 samples across Uniform(0, 1).
        for worker in &mut fleet {
            for _ in 0..50 {
                let x = rng.next_f64();
                let bet = (rng.next_f64() - 0.5) * 0.8; // zero-mean null signal
                worker.observe(x, Some(bet));
                single.observe(x, Some(bet));
            }
        }

        let mut merged = EvidenceSketch::new();
        for worker in &fleet {
            merged.merge(worker);
        }

        assert_sketches_close(&single, &merged, 1e-9);
        assert!((merged.mean() - 0.5).abs() < 0.01);
        assert!((merged.variance() - (1.0 / 12.0)).abs() < 0.005);

        // Histogram quantile accuracy across multiple quantiles on Uniform(0, 1).
        for &q in &[0.10, 0.25, 0.50, 0.75, 0.90, 0.95] {
            let est = merged.quantile(q);
            assert!(
                (est - q).abs() < 0.03,
                "Uniform(0,1) quantile({q}) estimate {est:.4} too far from {q}"
            );
        }

        // Test signed negative/positive values and monotonicity of bin_index / bin_bounds.
        let mut signed = EvidenceSketch::new();
        let mut rng2 = Rng::from_seed(8888);
        for _ in 0..20_000 {
            let u = rng2.next_f64() * 20.0 - 10.0; // Uniform(-10, 10)
            signed.observe_score(u);
        }
        assert!((signed.quantile(0.5) - 0.0).abs() < 0.4);
        assert!((signed.quantile(0.25) - (-5.0)).abs() < 0.8);
        assert!((signed.quantile(0.75) - 5.0).abs() < 0.8);

        for b in 0..(HIST_BINS - 1) {
            let (_, hi) = bin_bounds(b);
            let (lo_next, _) = bin_bounds(b + 1);
            assert!(
                (hi - lo_next).abs() < 1e-12,
                "bin boundary discontinuity between {b} and {}",
                b + 1
            );
        }
    }

    #[test]
    fn e_process_type1_validity_and_alternative_power() {
        // Under H0: E[x] = 0, Ville's inequality guarantees P(e_value >= 1/alpha) <= alpha.
        let alpha = 0.05;
        let thresh = 1.0 / alpha;
        let trials = 400;
        let mut rng = Rng::from_seed(77777);
        let mut rejections = 0usize;

        for _ in 0..trials {
            let mut s = EvidenceSketch::new();
            let mut crossed = false;
            for _ in 0..100 {
                let x = if rng.next_f64() < 0.5 { 0.6 } else { -0.6 };
                s.observe_bet(x);
                if s.e_value() >= thresh {
                    crossed = true;
                    break;
                }
            }
            if crossed {
                rejections += 1;
            }
        }
        let type1 = rejections as f64 / trials as f64;
        assert!(
            type1 <= alpha + 0.02,
            "Type-I error {type1:.4} exceeded Ville bound {alpha}"
        );

        // Under H1: positive drift E[x] = +0.3 => e_value explodes cleanly in log-space.
        let mut alt = EvidenceSketch::new();
        for _ in 0..200 {
            let x = if rng.next_f64() < 0.75 { 0.6 } else { -0.6 };
            alt.observe_bet(x);
        }
        assert!(alt.e_value() > 100.0);
        assert!(alt.log_e_value().is_finite());

        // Non-finite inputs are ignored cleanly.
        let mut clean = EvidenceSketch::new();
        clean.observe_score(f64::NAN);
        clean.observe_score(f64::INFINITY);
        clean.observe_score(1e200); // sq overflows => ignored
        clean.observe_bet(f64::NAN);
        assert_eq!(clean.n_samples, 0);
        assert_eq!(clean.mean(), 0.0);
        assert_eq!(clean.variance(), 0.0);
        assert_eq!(clean.quantile(0.5), 0.0);
        assert_eq!(clean.conformal_quantile(0.1), f64::INFINITY);
    }
}
