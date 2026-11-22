//! Shared numerically stable math, PRNG (xoshiro256++), and betting grids.

/// Number of geometric betting grid points (16 f64s = 128 bytes = 2 cache lines).
/// Sized for clean SIMD/AVX2/NEON vectorization with tight regret across all gaps.
pub const GRID_LEN: usize = 16;

/// Symmetric two-sided geometric betting grid on (0, 1):
/// dense near 0 (small gaps Delta -> 0) and dense near 1 (strong consensus Delta -> 1).
/// Strictly inside (0, 1) so `1 + lambda * x > 0` for all `x >= -1`.
pub const DEFAULT_BET_GRID: [f64; GRID_LEN] = [
    0.005, 0.012, 0.025, 0.050, 0.090, 0.150, 0.240, 0.360,
    0.500, 0.640, 0.760, 0.850, 0.910, 0.950, 0.975, 0.990,
];

/// Precomputed `ln(1 + lambda)` for `DEFAULT_BET_GRID`.
pub const LOG1P_GRID: [f64; GRID_LEN] = [
    0.0049875415110389695,
    0.011928570615475948,
    0.024692612590371502,
    0.04879016416943205,
    0.08617769624105241,
    0.13976194237515863,
    0.21511137961694548,
    0.30748469974796055,
    0.4054651081081644,
    0.494696241836107,
    0.5653138090500592,
    0.6151856390902335,
    0.6471032420585385,
    0.6678293725756554,
    0.6805683992356137,
    0.6881346387364007,
];

/// Precomputed `ln(1 - lambda)` for `DEFAULT_BET_GRID`.
pub const LOG1M_GRID: [f64; GRID_LEN] = [
    -0.005012541823544286,
    -0.012072581800718716,
    -0.025317807984289882,
    -0.05129329438755058,
    -0.09431067947124129,
    -0.16251892949777494,
    -0.2744368457017603,
    -0.4462871026284195,
    -0.6931471805599453,
    -1.0216512475319814,
    -1.4271163556401458,
    -1.8971199848858813,
    -2.4079456086518722,
    -2.995732273553991,
    -3.6888794541139363,
    -4.605170185988091,
];

/// Natural log of `GRID_LEN` = ln(16).
pub const LN_GRID_LEN: f64 = 2.772588722239781;

/// Numerically stable `log((1/N) * sum_i exp(x_i))` over a fixed-size slice.
/// Returns `-inf` if all entries are `-inf` or slice is empty.
#[inline]
pub fn log_mean_exp(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NEG_INFINITY;
    }
    let mut m = f64::NEG_INFINITY;
    for &v in xs {
        if v > m {
            m = v;
        }
    }
    if !m.is_finite() {
        return m;
    }
    let mut s = 0.0_f64;
    for &v in xs {
        let d = v - m;
        if d > -745.0 {
            s += d.exp();
        }
    }
    m + s.ln() - (xs.len() as f64).ln()
}

/// Fast `log_mean_exp` specialized for `GRID_LEN = 16`.
#[inline]
pub fn log_mean_exp_grid(xs: &[f64; GRID_LEN]) -> f64 {
    let mut m = xs[0];
    for &v in &xs[1..] {
        if v > m {
            m = v;
        }
    }
    if !m.is_finite() {
        return m;
    }
    let mut s = 0.0_f64;
    for &v in xs {
        let d = v - m;
        if d > -745.0 {
            s += d.exp();
        }
    }
    m + s.ln() - LN_GRID_LEN
}

/// Deterministic, fast, high-quality xoshiro256++ PRNG (zero external deps).
#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn from_seed(seed: u64) -> Self {
        let mut z = seed;
        let mut s = [0u64; 4];
        for slot in &mut s {
            z = z.wrapping_add(0x9e3779b97f4a7c15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
            *slot = x ^ (x >> 31);
        }
        Self { s }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = (self.s[0].wrapping_add(self.s[3]))
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform float in [0, 1).
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    /// Uniform integer in [0, bound) using Lemire's nearly-divisionless rejection method.
    #[inline]
    pub fn gen_range_usize(&mut self, bound: usize) -> usize {
        if bound <= 1 {
            return 0;
        }
        let b = bound as u64;
        let mut x = self.next_u64();
        let mut m = (x as u128) * (b as u128);
        let mut l = m as u64;
        if l < b {
            let t = b.wrapping_neg() % b;
            while l < t {
                x = self.next_u64();
                m = (x as u128) * (b as u128);
                l = m as u64;
            }
        }
        (m >> 64) as usize
    }

    /// Fisher-Yates in-place shuffle.
    #[inline]
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        let n = slice.len();
        for i in (1..n).rev() {
            let j = self.gen_range_usize(i + 1);
            slice.swap(i, j);
        }
    }
}

/// FNV-1a 64-bit hash for fast deterministic label keying / canonical strings.
#[inline]
pub fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Compensated (Kahan/Neumaier) accumulator for exact floating-point sums.
#[derive(Clone, Copy, Debug, Default)]
pub struct CompSum {
    pub sum: f64,
    pub comp: f64,
}

impl CompSum {
    #[inline]
    pub fn add(&mut self, x: f64) {
        let t = self.sum + x;
        if self.sum.abs() >= x.abs() {
            self.comp += (self.sum - t) + x;
        } else {
            self.comp += (x - t) + self.sum;
        }
        self.sum = t;
    }

    #[inline]
    pub fn value(&self) -> f64 {
        self.sum + self.comp
    }

    #[inline]
    pub fn merge(&mut self, other: Self) {
        self.add(other.sum);
        self.add(other.comp);
    }
}

/// Exact Jacobi eigenvalue algorithm for small symmetric `k x k` correlation matrices.
/// Returns sorted descending eigenvalues. Zero external dependencies, deterministic.
pub fn symmetric_eigenvalues(matrix: &[f64], k: usize) -> Vec<f64> {
    if k == 0 {
        return Vec::new();
    }
    if k == 1 {
        return vec![matrix[0]];
    }
    let mut a = matrix.to_vec();
    let max_sweeps = 64 * k * k;
    for _ in 0..max_sweeps {
        // Find largest off-diagonal element
        let mut p = 0;
        let mut q = 1;
        let mut max_off = 0.0_f64;
        for i in 0..k {
            for j in (i + 1)..k {
                let v = a[i * k + j].abs();
                if v > max_off {
                    max_off = v;
                    p = i;
                    q = j;
                }
            }
        }
        if max_off < 1e-13 {
            break;
        }
        let app = a[p * k + p];
        let aqq = a[q * k + q];
        let apq = a[p * k + q];
        let tau = (aqq - app) / (2.0 * apq);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;

        a[p * k + p] = app - t * apq;
        a[q * k + q] = aqq + t * apq;
        a[p * k + q] = 0.0;
        a[q * k + p] = 0.0;

        for r in 0..k {
            if r != p && r != q {
                let arp = a[r * k + p];
                let arq = a[r * k + q];
                let new_rp = c * arp - s * arq;
                let new_rq = s * arp + c * arq;
                a[r * k + p] = new_rp;
                a[p * k + r] = new_rp;
                a[r * k + q] = new_rq;
                a[q * k + r] = new_rq;
            }
        }
    }
    let mut eigs: Vec<f64> = (0..k).map(|i| a[i * k + i]).collect();
    eigs.sort_by(|x, y| y.partial_cmp(x).unwrap_or(std::cmp::Ordering::Equal));
    eigs
}
