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
    0.004987541511039074,
    0.011928570865273802,
    0.0246926125903715,
    0.04879016416943201,
    0.08617769624105233,
    0.13976194237515868,
    0.2151113796169455,
    0.3074846997479606,
    0.4054651081081644,
    0.49469624183610705,
    0.5653138090500605,
    0.6151856390902335,
    0.6471032420585385,
    0.6678293725756554,
    0.6805683983530851,
    0.688134638736401,
];

/// Precomputed `ln(1 - lambda)` for `DEFAULT_BET_GRID`.
pub const LOG1M_GRID: [f64; GRID_LEN] = [
    -0.005012541823544282,
    -0.012072581234269237,
    -0.025317807984289876,
    -0.051293294387550536,
    -0.09431067947124132,
    -0.1625189294977749,
    -0.2744368457017603,
    -0.4462871026284195,
    -0.6931471805599453,
    -1.0216512475319814,
    -1.4271163556401458,
    -1.897119984885881,
    -2.4079456086518722,
    -2.99573227355399,
    -3.6888794541139354,
    -4.60517018598809,
];

/// Natural log of `GRID_LEN` = ln(16).
pub const LN_GRID_LEN: f64 = 2.772588722239781;

// IEEE-754 double-precision constants for fast_ln / fast_ln_1p / fast_exp_nonpos.
const MIN_NORMAL_F64_BITS: u64 = 0x0010_0000_0000_0000;
const INF_F64_BITS: u64 = 0x7FF0_0000_0000_0000;
/// `(1 << 52) - (SQRT_2.to_bits() & ((1 << 52) - 1))`, folds mantissa `[1, 2)` into `[sqrt(0.5), sqrt(2))`.
const SQRT2_FOLD_OFFSET: u64 = 0x0009_5F61_9980_C433;
const EXP_BIAS_BITS: u64 = 0x3FF0_0000_0000_0000;
/// Exact top 33 bits of `ln(2)` (`0x3FE6_2E42_FEE0_0000`), so `k * LN2_HI` is exact for all `|k| <= 1024`.
const LN2_HI: f64 = 0.69314718036912381649017333984375;
/// Residual `ln(2) - LN2_HI` (`0x3DEA_39EF_3579_3C76`).
const LN2_LO: f64 = 1.90821492927058770002e-10;
const INV_LN2: f64 = std::f64::consts::LOG2_E;
/// IEEE-754 integer-conversion bias `2^52`.
const F64_INT_MAGIC_BITS: u64 = 0x4330_0000_0000_0000;
/// `2^52 + 1023` for branchless SIMD-friendly biased-exponent to `f64` conversion.
const F64_INT_MAGIC_PLUS_BIAS: f64 = 4_503_599_627_371_519.0;
/// `3 * 2^51` for branchless round-to-nearest-even in `fast_exp_nonpos_clamped`.
const EXP_ROUND_MAGIC: f64 = 6_755_399_441_055_744.0;

#[cold]
#[inline(never)]
fn ln_fallback(x: f64) -> f64 {
    x.ln()
}

#[cold]
#[inline(never)]
fn ln_1p_fallback(x: f64) -> f64 {
    x.ln_1p()
}

/// Fast natural logarithm for normal positive finite `x in [f64::MIN_POSITIVE, f64::MAX]`.
///
/// Uses branchless IEEE-754 exponent extraction, range folding to `[sqrt(0.5), sqrt(2))`,
/// odd polynomial in `u = (m - 1) / (m + 1)` through `u^17 / 17`, and Cody-Waite `ln(2)`
/// reconstruction. Zero libm calls and fully SIMD-vectorizable.
#[inline(always)]
pub(crate) fn fast_ln_normal(x: f64) -> f64 {
    let bits = x.to_bits();
    let k_biased = bits.wrapping_add(SQRT2_FOLD_OFFSET) >> 52;
    let k = f64::from_bits(F64_INT_MAGIC_BITS | k_biased) - F64_INT_MAGIC_PLUS_BIAS;
    let m_bits = bits
        .wrapping_sub(k_biased << 52)
        .wrapping_add(EXP_BIAS_BITS);
    let m = f64::from_bits(m_bits);

    let u = (m - 1.0) / (m + 1.0);
    let u2 = u * u;
    let u4 = u2 * u2;
    let u8 = u4 * u4;
    let c1 = (1.0 / 3.0) + u2 * (1.0 / 5.0);
    let c2 = (1.0 / 7.0) + u2 * (1.0 / 9.0);
    let c3 = (1.0 / 11.0) + u2 * (1.0 / 13.0);
    let c4 = (1.0 / 15.0) + u2 * (1.0 / 17.0);
    let p = (c1 + u4 * c2) + u8 * (c3 + u4 * c4);
    let s = 2.0 * u;
    let ln_m = s + s * (u2 * p);
    k * LN2_HI + (k * LN2_LO + ln_m)
}

/// Fast natural logarithm `ln(x)` with `< 1e-15` relative error across `(0, +inf)`.
///
/// Uses IEEE-754 exponent extraction + range folding to `[sqrt(0.5), sqrt(2))` + odd
/// polynomial in `u = (m - 1) / (m + 1)` up to `u^17 / 17`, falling back to `f64::ln`
/// only for non-normal, non-positive, or non-finite inputs.
#[inline]
pub fn fast_ln(x: f64) -> f64 {
    let bits = x.to_bits();
    if bits.wrapping_sub(MIN_NORMAL_F64_BITS) >= (INF_F64_BITS - MIN_NORMAL_F64_BITS) {
        return ln_fallback(x);
    }
    fast_ln_normal(x)
}

/// Pure-arithmetic, branchless `ln(1 + x)` for finite inputs strictly above `-0.999999`.
///
/// Bit-for-bit identical to [`fast_ln_1p`] on `(-0.999999, +inf)`, with zero libm calls
/// so tight loops in `BettingCS` and `AscdStopper` auto-vectorize cleanly.
#[inline(always)]
pub(crate) fn fast_ln_1p_bounded(x: f64) -> f64 {
    debug_assert!(x.is_finite() && x > -0.999999);
    let taylor = x * (1.0 - x * (0.5 - x * ((1.0 / 3.0) - x * 0.25)));
    let general = fast_ln_normal(1.0 + x);
    if x.abs() < 1e-4 {
        taylor
    } else {
        general
    }
}

/// Fast natural logarithm `ln(1 + x)` accurate to `< 1e-15` across `(-0.999999, 1.0]`
/// (and `< 1e-15` relative error for all `x > -0.999999`).
///
/// Uses a 4-term Horner series for `|x| < 1e-4` to prevent cancellation, [`fast_ln`]
/// on `1.0 + x` for `x > -0.999999`, and falls back to `f64::ln_1p` only for extreme
/// edges `x <= -0.999999` or non-finite inputs.
#[inline]
pub fn fast_ln_1p(x: f64) -> f64 {
    if x.abs() < 1e-4 {
        return x * (1.0 - x * (0.5 - x * ((1.0 / 3.0) - x * 0.25)));
    }
    if x <= -0.999999 || !x.is_finite() {
        return ln_1p_fallback(x);
    }
    fast_ln_normal(1.0 + x)
}

/// Pure-arithmetic, SIMD-friendly `exp(d)` for finite `d in [-708.0, 0.0]`.
/// Uses degree-10 Chebyshev polynomial on `[-ln(2)/2, ln(2)/2]` with `< 1.3e-15`
/// max relative error and exact `exp(0.0) == 1.0`.
#[inline(always)]
pub(crate) fn fast_exp_nonpos_clamped(d: f64) -> f64 {
    let t = d * INV_LN2 + EXP_ROUND_MAGIC;
    let k = t - EXP_ROUND_MAGIC;
    let r = (d - k * LN2_HI) - k * LN2_LO;
    let r2 = r * r;
    let r4 = r2 * r2;
    let c0 = 0.5 + r * 0.16666666666648303;
    let c1 = 0.041666666666651364 + r * 0.008333333353717156;
    let c2 = 0.0013888888905871347 + r * 0.0001984120875699232;
    let c3 = 2.4801536409064087e-05 + r * (2.7625102005388108e-06 + r * 2.761379555451986e-07);
    let tail = (c0 + r2 * c1) + r4 * (c2 + r2 * c3);
    let e_r = (1.0 + r) + r2 * tail;
    let pow2k_bits = (t.to_bits().wrapping_add(1023) & 0x7FF) << 52;
    e_r * f64::from_bits(pow2k_bits)
}

/// Ultra-fast Cody-Waite-free `exp(d) = 2^(d * log2(e))` for `d in [-45.0, 0.0]`.
///
/// Computes `2^f` directly on `f = d * log2(e) - round(d * log2(e)) in [-0.5, 0.5]`
/// via degree-9 Chebyshev polynomial, achieving `< 3.8e-14` max relative error across `[-45, 0]`
/// and exact `exp(0.0) == 1.0`.
#[inline(always)]
fn fast_exp_nonpos_45(d: f64) -> f64 {
    let x = d * INV_LN2;
    let t = x + EXP_ROUND_MAGIC;
    let k = t - EXP_ROUND_MAGIC;
    let f = x - k;
    let f2 = f * f;
    let f4 = f2 * f2;
    let c0 = 0.6931471805599453 + f * 0.2402265069581299;
    let c1 = 0.055504108664760424 + f * 0.009618129159402558;
    let c2 = 0.0013333558179043112 + f * 0.00015403455852453838;
    let c3 = 1.525268684626772e-05 + f * (1.3255224878668817e-06 + f * 1.0203121063391729e-07);
    let p = (c0 + f2 * c1) + f4 * (c2 + f2 * c3);
    let e_f = 1.0 + f * p;
    let pow2k_bits = (t.to_bits().wrapping_add(1023) & 0x7FF) << 52;
    e_f * f64::from_bits(pow2k_bits)
}

/// Zero-division, SIMD-vectorizable `(ln(1 + y), ln(1 - y))` for `|y| <= 0.06`.
/// Degree-3 Chebyshev odd/even polynomials on `[0, 0.06^2]`. Max absolute error `< 4e-17`.
#[inline(always)]
fn eval_ln_pair_006(y: f64) -> (f64, f64) {
    let z = y * y;
    let z2 = z * z;
    let po = (0.3333333333332133 + z * 0.20000000106674995)
        + z2 * (0.14285566202169223 + z * 0.1117684082481575);
    let pe = (0.49999999999977995 + z * 0.3333333352891941)
        + z2 * (0.24999728494136256 + z * 0.20120511044913603);
    let odd = y + y * (z * po);
    let even = -0.5 * z * (1.0 + z * pe);
    (odd + even, even - odd)
}

/// Zero-division, SIMD-vectorizable `(ln(1 + y), ln(1 - y))` for `|y| <= 0.23`.
/// Degree-6 Chebyshev odd/even polynomials on `[0, 0.23^2]`. Max absolute error `< 1.7e-16`.
#[inline(always)]
fn eval_ln_pair_023(y: f64) -> (f64, f64) {
    let z = y * y;
    let z2 = z * z;
    let z4 = z2 * z2;

    let o0 = 0.3333333333333432 + z * 0.19999999998175036;
    let o1 = 0.14285714836319627 + z * 0.11111048968612353;
    let o2 = 0.09094236379360086 + z * (0.07601495532046833 + z * 0.0787477881924781);
    let po = (o0 + z2 * o1) + z4 * o2;

    let e0 = 0.5000000000000187 + z * 0.33333333329882764;
    let e1 = 0.2500000104104825 + z * 0.19999882508384498;
    let e2 = 0.16672957193711196 + z * (0.1411404146684336 + z * 0.1478340300833966);
    let pe = (e0 + z2 * e1) + z4 * e2;

    let odd = y + y * (z * po);
    let even = -0.5 * z * (1.0 + z * pe);
    (odd + even, even - odd)
}

/// Zero-division, SIMD-vectorizable `(ln(1 + y), ln(1 - y))` for `|y| <= 0.36`.
///
/// Splits into odd part `O(y) = 0.5 * ln((1+y)/(1-y)) = y * (1 + y^2 * P_odd(y^2))`
/// and even part `E(y^2) = 0.5 * ln(1 - y^2) = -0.5 * y^2 * (1 + y^2 * P_even(y^2))`
/// using degree-9 Chebyshev polynomials on `[0, 0.36^2]`. Max absolute error `< 1.2e-16`.
#[inline(always)]
fn eval_ln_pair_036(y: f64) -> (f64, f64) {
    let z = y * y;
    let z2 = z * z;
    let z4 = z2 * z2;
    let z8 = z4 * z4;

    let o0 = 0.33333333333333315 + z * 0.2000000000003198;
    let o1 = 0.14285714277595413 + z * 0.11111111908393528;
    let o2 = 0.09090869464043837 + z * 0.07693433820872884;
    let o3 = 0.06647316529073423 + z * 0.0608578594034101;
    let o4 = 0.03999541307218405 + z * 0.08865924383231263;
    let po = ((o0 + z2 * o1) + z4 * (o2 + z2 * o3)) + z8 * o4;

    let e0 = 0.4999999999999996 + z * 0.3333333333339476;
    let e1 = 0.24999999984406487 + z * 0.2000000153127349;
    let e2 = 0.16666590560776134 + z * 0.142878769969709;
    let e3 = 0.12462840684866185 + z * 0.1150173885521087;
    let e4 = 0.07574030644574008 + z * 0.16967351229033514;
    let pe = ((e0 + z2 * e1) + z4 * (e2 + z2 * e3)) + z8 * e4;

    let odd = y + y * (z * po);
    let even = -0.5 * z * (1.0 + z * pe);
    (odd + even, even - odd)
}

/// Zero-division, SIMD-vectorizable `(ln(1 + y), ln(1 - y))` for `|y| <= 0.56`.
/// Degree-13 Chebyshev odd/even polynomials on `[0, 0.56^2]`. Max absolute error `< 2.3e-16`.
#[inline(always)]
fn eval_ln_pair_056(y: f64) -> (f64, f64) {
    let z = y * y;
    let z2 = z * z;
    let z4 = z2 * z2;
    let z8 = z4 * z4;

    let o0 = 0.3333333333333331 + z * 0.20000000000027998;
    let o1 = 0.1428571427993337 + z * 0.11111111579711032;
    let o2 = 0.090908893425408 + z * 0.07692804093354065;
    let o3 = 0.06658625424494236 + z * 0.05970177388982569;
    let o4 = 0.046007078383930644 + z * 0.08237134080459094;
    let o5 = -0.08197743093847776 + z * 0.34094915046375446;
    let o6 = -0.40549901815153244 + z * 0.35800933935956725;
    let po = ((o0 + z2 * o1) + z4 * (o2 + z2 * o3)) + z8 * ((o4 + z2 * o5) + z4 * o6);

    let e0 = 0.49999999999999956 + z * 0.33333333333387816;
    let e1 = 0.24999999988750332 + z * 0.20000000911886556;
    let e2 = 0.16666628237305464 + z * 0.14286680237428892;
    let e3 = 0.12484352910346438 + z * 0.11281998368020701;
    let e4 = 0.08711079628814994 + z * 0.158521738540863;
    let e5 = -0.1607261106985144 + z * 0.662307709902181;
    let e6 = -0.7891789251122413 + z * 0.6955651209929055;
    let pe = ((e0 + z2 * e1) + z4 * (e2 + z2 * e3)) + z8 * ((e4 + z2 * e5) + z4 * e6);

    let odd = y + y * (z * po);
    let even = -0.5 * z * (1.0 + z * pe);
    (odd + even, even - odd)
}

#[cold]
#[inline(never)]
fn add_bet_grid_unit(pos: &mut [f64; GRID_LEN], neg: &mut [f64; GRID_LEN], positive: bool) {
    if positive {
        for g in 0..GRID_LEN {
            pos[g] += LOG1P_GRID[g];
            neg[g] += LOG1M_GRID[g];
        }
    } else {
        for g in 0..GRID_LEN {
            pos[g] += LOG1M_GRID[g];
            neg[g] += LOG1P_GRID[g];
        }
    }
}

/// Adds `ln(1 + DEFAULT_BET_GRID[g] * diff)` to `pos[g]` and
/// `ln(1 - DEFAULT_BET_GRID[g] * diff)` to `neg[g]` for all `g in 0..GRID_LEN`,
/// assuming `diff in [-1.0, 1.0]`.
///
/// Uses tiered zero-division odd/even shared-power polynomials (`<= 0.06` deg-3,
/// `<= 0.23` deg-6, `<= 0.36` deg-9, `<= 0.56` deg-13) achieving `< 1e-15` exact accuracy
/// with full SSE2/AVX2 auto-vectorization and zero libm calls.
#[inline]
pub(crate) fn add_bet_grid_pair(
    pos: &mut [f64; GRID_LEN],
    neg: &mut [f64; GRID_LEN],
    diff: f64,
) {
    let ad = diff.abs();
    if ad <= 0.06 {
        for g in 0..GRID_LEN {
            let (lp, lm) = eval_ln_pair_006(DEFAULT_BET_GRID[g] * diff);
            pos[g] += lp;
            neg[g] += lm;
        }
    } else if ad <= 0.23 {
        // g=0..6 has DEFAULT_BET_GRID[g] <= 0.15 -> |y| <= 0.0345 <= 0.06.
        for g in 0..6 {
            let (lp, lm) = eval_ln_pair_006(DEFAULT_BET_GRID[g] * diff);
            pos[g] += lp;
            neg[g] += lm;
        }
        for g in 6..GRID_LEN {
            let (lp, lm) = eval_ln_pair_023(DEFAULT_BET_GRID[g] * diff);
            pos[g] += lp;
            neg[g] += lm;
        }
    } else if ad <= 0.359 {
        // g=0..4 <= 0.05 -> |y| <= 0.018; g=4..10 <= 0.64 -> |y| <= 0.2298 <= 0.23.
        for g in 0..4 {
            let (lp, lm) = eval_ln_pair_006(DEFAULT_BET_GRID[g] * diff);
            pos[g] += lp;
            neg[g] += lm;
        }
        for g in 4..10 {
            let (lp, lm) = eval_ln_pair_023(DEFAULT_BET_GRID[g] * diff);
            pos[g] += lp;
            neg[g] += lm;
        }
        for g in 10..GRID_LEN {
            let (lp, lm) = eval_ln_pair_036(DEFAULT_BET_GRID[g] * diff);
            pos[g] += lp;
            neg[g] += lm;
        }
    } else if ad <= 0.56 {
        // g=0..4 <= 0.05 -> <= 0.028; g=4..8 <= 0.36 -> <= 0.2016 <= 0.23;
        // g=8..10 <= 0.64 -> <= 0.3584 <= 0.36; g=10..16 <= 0.99 -> <= 0.5544 <= 0.56.
        for g in 0..4 {
            let (lp, lm) = eval_ln_pair_006(DEFAULT_BET_GRID[g] * diff);
            pos[g] += lp;
            neg[g] += lm;
        }
        for g in 4..8 {
            let (lp, lm) = eval_ln_pair_023(DEFAULT_BET_GRID[g] * diff);
            pos[g] += lp;
            neg[g] += lm;
        }
        for g in 8..10 {
            let (lp, lm) = eval_ln_pair_036(DEFAULT_BET_GRID[g] * diff);
            pos[g] += lp;
            neg[g] += lm;
        }
        for g in 10..GRID_LEN {
            let (lp, lm) = eval_ln_pair_056(DEFAULT_BET_GRID[g] * diff);
            pos[g] += lp;
            neg[g] += lm;
        }
    } else if ad == 1.0 {
        add_bet_grid_unit(pos, neg, diff > 0.0);
    } else {
        // g=0..4 <= 0.05; g=4..8 <= 0.36 for all |diff| <= 1.0.
        for g in 0..4 {
            let (lp, lm) = eval_ln_pair_006(DEFAULT_BET_GRID[g] * diff);
            pos[g] += lp;
            neg[g] += lm;
        }
        for g in 4..8 {
            let (lp, lm) = eval_ln_pair_036(DEFAULT_BET_GRID[g] * diff);
            pos[g] += lp;
            neg[g] += lm;
        }
        for g in 8..GRID_LEN {
            let ld = DEFAULT_BET_GRID[g] * diff;
            pos[g] += fast_ln_normal(1.0 + ld);
            neg[g] += fast_ln_normal(1.0 - ld);
        }
    }
}

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
        if d > -45.0 {
            s += fast_exp_nonpos_clamped(d.max(-708.0));
        }
    }
    m + fast_ln_normal(s) - fast_ln_normal(xs.len() as f64)
}

/// Fast tree-reduced maximum over `GRID_LEN = 16` elements.
#[inline(always)]
pub(crate) fn max_grid(xs: &[f64; GRID_LEN]) -> f64 {
    let mut a = [0.0_f64; 8];
    for i in 0..8 {
        let x = xs[i];
        let y = xs[i + 8];
        a[i] = if x > y { x } else { y };
    }
    let b0 = if a[0] > a[4] { a[0] } else { a[4] };
    let b1 = if a[1] > a[5] { a[1] } else { a[5] };
    let b2 = if a[2] > a[6] { a[2] } else { a[6] };
    let b3 = if a[3] > a[7] { a[3] } else { a[7] };
    let c0 = if b0 > b2 { b0 } else { b2 };
    let c1 = if b1 > b3 { b1 } else { b3 };
    if c0 > c1 {
        c0
    } else {
        c1
    }
}

/// Fast `log_mean_exp_grid` when `m = max_grid(xs)` has already been computed.
#[inline]
pub(crate) fn log_mean_exp_grid_with_max(xs: &[f64; GRID_LEN], m: f64) -> f64 {
    if !m.is_finite() {
        return m;
    }
    let terms: [f64; GRID_LEN] = std::array::from_fn(|g| {
        let d = xs[g] - m;
        let d_c = if d > -45.0 { d } else { -45.0 };
        let term = fast_exp_nonpos_45(d_c);
        if d > -45.0 {
            term
        } else {
            0.0
        }
    });
    let mut s0 = 0.0_f64;
    let mut s1 = 0.0_f64;
    for i in 0..8 {
        s0 += terms[i * 2];
        s1 += terms[i * 2 + 1];
    }
    m + fast_ln_normal(s0 + s1) - LN_GRID_LEN
}

/// Fast `log_mean_exp` specialized for `GRID_LEN = 16`.
#[inline]
pub fn log_mean_exp_grid(xs: &[f64; GRID_LEN]) -> f64 {
    let m = max_grid(xs);
    log_mean_exp_grid_with_max(xs, m)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_ln_and_fast_ln_1p_accuracy_10k_points_and_edges() {
        assert_eq!(LN2_HI.to_bits(), 0x3FE6_2E42_FEE0_0000);
        assert_eq!(LN2_HI + LN2_LO, std::f64::consts::LN_2);
        for g in 0..GRID_LEN {
            assert!((LOG1P_GRID[g] - DEFAULT_BET_GRID[g].ln_1p()).abs() < 1e-15);
            assert!((LOG1M_GRID[g] - (-DEFAULT_BET_GRID[g]).ln_1p()).abs() < 1e-15);
        }

        // 10,000 points for fast_ln across (1e-6, 1e6) plus wide exponents (1e-300..1e300).
        let n_pts = 10_000usize;
        let mut max_abs_ln = 0.0_f64;
        let mut max_rel_ln = 0.0_f64;
        for i in 0..n_pts {
            let t = (i as f64) / ((n_pts - 1) as f64);
            let x = (10.0_f64).powf(-6.0 + 12.0 * t);
            let got = fast_ln(x);
            let ref_v = x.ln();
            let err = (got - ref_v).abs();
            max_abs_ln = max_abs_ln.max(err);
            max_rel_ln = max_rel_ln.max(err / ref_v.abs().max(1.0));
        }
        assert!(
            max_abs_ln < 1e-14,
            "fast_ln max_abs_err={max_abs_ln:.3e} on [1e-6, 1e6]"
        );
        assert!(
            max_rel_ln < 1e-14,
            "fast_ln max_rel_err={max_rel_ln:.3e} on [1e-6, 1e6]"
        );

        // Full normal exponent range [1e-300, 1e300] + exact powers and sqrt(2) boundary.
        let mut max_rel_wide = 0.0_f64;
        for exp in -300..=300 {
            for &mant in &[0.7071067811865475, 1.0, 1.414213562373095, 1.4142135623730951, 1.9999]
            {
                let x = mant * (10.0_f64).powi(exp);
                if x.is_finite() && x > 0.0 {
                    let got = fast_ln(x);
                    let ref_v = x.ln();
                    let rel = (got - ref_v).abs() / ref_v.abs().max(1.0);
                    max_rel_wide = max_rel_wide.max(rel);
                }
            }
        }
        assert!(
            max_rel_wide < 1e-14,
            "fast_ln max_rel_err={max_rel_wide:.3e} across [1e-300, 1e300]"
        );

        // 10,000 points for fast_ln_1p across (-0.999999, 1.0] (linear + log-near-(-1) + near-0).
        let mut max_abs_1p = 0.0_f64;
        for i in 0..n_pts {
            let t = (i as f64) / ((n_pts - 1) as f64);
            let x = -0.9999989 + 1.9999989 * t;
            let got = fast_ln_1p(x);
            let got_b = fast_ln_1p_bounded(x);
            let ref_v = x.ln_1p();
            assert_eq!(got, got_b, "fast_ln_1p_bounded mismatch at x={x}");
            max_abs_1p = max_abs_1p.max((got - ref_v).abs());
        }
        // Dense near -0.999999 and near 0 (both sides of 1e-4 threshold).
        for i in 1..2000 {
            let t = (i as f64) / 2000.0;
            let y = ((-6.0_f64) * std::f64::consts::LN_10 * (1.0 - t)
                + std::f64::consts::LN_2 * t)
                .exp();
            let x = y - 1.0;
            if x > -0.999999 {
                let err = (fast_ln_1p(x) - x.ln_1p()).abs();
                max_abs_1p = max_abs_1p.max(err);
            }
            let x_small = -2e-4 + 4e-4 * t;
            let err_s = (fast_ln_1p(x_small) - x_small.ln_1p()).abs();
            max_abs_1p = max_abs_1p.max(err_s);
        }
        assert!(
            max_abs_1p < 1e-14,
            "fast_ln_1p max_abs_err={max_abs_1p:.3e} on (-0.999999, 1.0]"
        );

        // Special & edge values.
        assert_eq!(fast_ln(1.0), 0.0);
        assert_eq!(fast_ln(0.0), f64::NEG_INFINITY);
        assert!(fast_ln(-1.0).is_nan());
        assert!(fast_ln(f64::NAN).is_nan());
        assert_eq!(fast_ln(f64::INFINITY), f64::INFINITY);
        assert!((fast_ln(1e-320) - (1e-320_f64).ln()).abs() < 1e-14);
        assert!((fast_ln(f64::MIN_POSITIVE) - f64::MIN_POSITIVE.ln()).abs() < 1e-12);
        assert!((fast_ln(f64::MAX) - f64::MAX.ln()).abs() < 1e-12);

        assert_eq!(fast_ln_1p(0.0), 0.0);
        assert_eq!(fast_ln_1p(-1.0), f64::NEG_INFINITY);
        assert!(fast_ln_1p(-1.5).is_nan());
        assert!(fast_ln_1p(f64::NAN).is_nan());
        assert_eq!(fast_ln_1p(f64::INFINITY), f64::INFINITY);
        assert!((fast_ln_1p(-0.9999995) - (-0.9999995_f64).ln_1p()).abs() < 1e-15);
        assert!((fast_ln_1p(1e-12) - (1e-12_f64).ln_1p()).abs() < 1e-26);
    }

    #[test]
    fn fast_exp_and_log_mean_exp_accuracy() {
        let mut max_rel = 0.0_f64;
        for i in 0..=10_000 {
            let d = -708.0 * (i as f64) / 10_000.0;
            let got = fast_exp_nonpos_clamped(d);
            let ref_v = d.exp();
            if ref_v > 0.0 {
                max_rel = max_rel.max((got - ref_v).abs() / ref_v);
            }
        }
        assert!(max_rel < 1e-14, "fast_exp_nonpos_clamped max_rel={max_rel:.3e}");
        assert_eq!(fast_exp_nonpos_clamped(0.0), 1.0);

        let mut rng = Rng::from_seed(0xABCD_1234);
        for _ in 0..500 {
            let mut arr = [0.0_f64; GRID_LEN];
            for slot in &mut arr {
                *slot = -60.0 + 120.0 * rng.next_f64();
            }
            let got_g = log_mean_exp_grid(&arr);
            let got_s = log_mean_exp(&arr);
            let m = arr.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let ref_v = m
                + arr
                    .iter()
                    .map(|&v| if v - m > -745.0 { (v - m).exp() } else { 0.0 })
                    .sum::<f64>()
                    .ln()
                - (GRID_LEN as f64).ln();
            assert!((got_g - ref_v).abs() < 1e-13, "grid err: {} vs {}", got_g, ref_v);
            assert!((got_s - ref_v).abs() < 1e-13, "slice err: {} vs {}", got_s, ref_v);
        }
        assert_eq!(log_mean_exp(&[]), f64::NEG_INFINITY);
        assert_eq!(log_mean_exp_grid(&[f64::NEG_INFINITY; GRID_LEN]), f64::NEG_INFINITY);
    }

    #[test]
    fn add_bet_grid_pair_exact_accuracy_across_full_range() {
        let mut max_err = 0.0_f64;
        for i in 0..=10_000 {
            let diff = -1.0 + 2.0 * (i as f64) / 10_000.0;
            let mut pos = [0.0_f64; GRID_LEN];
            let mut neg = [0.0_f64; GRID_LEN];
            add_bet_grid_pair(&mut pos, &mut neg, diff);
            for g in 0..GRID_LEN {
                let ld = DEFAULT_BET_GRID[g] * diff;
                let ref_p = ld.ln_1p();
                let ref_m = (-ld).ln_1p();
                max_err = max_err.max((pos[g] - ref_p).abs());
                max_err = max_err.max((neg[g] - ref_m).abs());
            }
        }
        assert!(
            max_err < 1e-14,
            "add_bet_grid_pair max_err={max_err:.3e} across [-1, 1]"
        );
    }
}
