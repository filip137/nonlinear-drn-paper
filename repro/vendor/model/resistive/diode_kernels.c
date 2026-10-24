/* Compiled Shockley coordinate updates for the CD minimizer (opt-in; built by _kernels.py).
 *
 * Each function repeats, node by node, the arithmetic of the torch path in minimizer.py,
 * ExponentialDoubleDiodeUpdater.lambert_hidden and
 * ExponentialSingleDiodeUpdater._lambert_single_oriented, followed by the relaxation
 * v_old + omega (v - v_old). Both families share one Lambert-W rule (shockley_w,
 * refine_w), one reconstruction (shockley_root) and one overflow guard (EXP_GUARD), as the
 * torch path does; the Newton polish loops keep their per-family global stopping rules.
 * One difference: below the asymptotic threshold W0 is evaluated in FP64 (Halley), where
 * torchlambertw computes in FP32; the polish removes it.
 *
 * Arrays are row-major rows x cols; a has row stride a_stride (0 broadcasts one row).
 * Returns 0 on success, 1 for a non-finite input, 2 for a non-finite result.
 */
#include <float.h>
#include <math.h>

static double clamp_b(double b, int clip, double limit) {
    /* Like torch.clamp: NaN stays NaN. */
    if (clip) {
        if (b < -limit) return -limit;
        if (b > limit) return limit;
    }
    return b;
}

static double sign_of(double x) {
    if (isnan(x)) return x;
    return (double)((x > 0.0) - (x < 0.0));
}

/* Asymptotic W0(z) from l1 = log(z). */
static double series(double l1, int terms) {
    double l2 = log(l1), w = l1 - l2;
    if (terms >= 2) w += l2 / l1;
    if (terms >= 3) w += l2 * (-2.0 + l2) / (2.0 * l1 * l1);
    if (terms >= 4) w += l2 * (6.0 - 9.0 * l2 + 2.0 * l2 * l2) / (6.0 * l1 * l1 * l1);
    return w;
}

/* Principal-branch W0(z) for z >= 0, as custom_lambert_halley.lambertw: Winitzki seed
 * (log(z) for log1p(z) above 1e6), then Halley steps divided through by e^w so they cannot
 * overflow, until a step is within 4 ulp, at most 5 steps (4 suffice for every z). */
static double lambertw0(double z) {
    if (z == 0.0) return 0.0;
    const double lp = z > 1e6 ? log(z) : log1p(z);
    double w = lp * (1.0 - log1p(lp) / (2.0 + lp));
    for (int k = 0; k < 5; k++) {
        const double f = w - z * exp(-w), wp1 = w + 1.0;
        const double step = f / (wp1 - (w + 2.0) * f / (2.0 * wp1));
        w -= step;
        if (!(fabs(step) > 4.0 * DBL_EPSILON * fabs(w))) break;
    }
    return w;
}

/* W0(exp(lz)) by the rule of _shockley_lambertw: the series above the threshold
 * (LOG_THRESHOLD), FP64 Halley at or below it. */
static double shockley_w(double lz, double log_thresh, int terms) {
    return lz > log_thresh ? series(lz, terms) : lambertw0(exp(lz));
}

/* Three Newton steps on w + log(w) = lz; W = 0 (exp underflow) stays exact. */
static double refine_w(double w, double lz) {
    if (!(w > 0.0)) return w;
    for (int k = 0; k < 3; k++) w = w - (w + log(w) - lz) * (w / (w + 1.0));
    return w;
}

/* linear - V_t W, rebuilt without cancellation whenever lz > 0 (_shockley_root). */
static double shockley_root(double lz, double w, double linear, double off, double vt, double lp) {
    return lz > 0.0 ? off + vt * (log(w) - lp) : linear - vt * w;
}

/* Overflow guard for the polish exponent (_exp_guard); it never binds near a root. */
#define EXP_GUARD (log(DBL_MAX) - 2.0)
#define LOG_THRESHOLD(z_thresh) fmin(fmax(log(z_thresh), 8.0), log((double)FLT_MAX) - 2.0)

/* 2 a t - |b| + I_s exp((t - V_off) / V_t) = 0 for t = |v| >= 0, v = -sign(b) t. */
int double_diode_update(long rows, long cols, const double *a, long a_stride, const double *b,
                        const double *v_old, double *v, double Is, double vt, double voff,
                        int clip_b, double b_clip, double z_thresh, int terms, int polish,
                        int max_iter, double residual_tol, double omega) {
    const double floor = Is * exp(-voff / vt), log_thresh = LOG_THRESHOLD(z_thresh);
    int any_active = 0;
#define DRIVE(i) fabs(clamp_b(b[i], clip_b, b_clip))
#define LOG_PREFACTOR(ai) log(Is / (2.0 * (ai) * vt))
#define LOG_Z(ai, drive) (LOG_PREFACTOR(ai) + ((drive) / (2.0 * (ai)) - voff) / vt)
    /* Pass 1: W0 into v (zero branch: 0). */
    for (long r = 0; r < rows; r++) {
        for (long c = 0; c < cols; c++) {
            const long i = r * cols + c;
            const double ai = a[r * a_stride + c], drive = DRIVE(i);
            v[i] = 0.0;
            if (drive > floor) {
                const double lz = LOG_Z(ai, drive);
                if (isnan(lz) || lz == INFINITY) return 1;
                v[i] = shockley_w(lz, log_thresh, terms);
                any_active = 1;
            }
        }
    }
    /* Pass 2: refine W, then the magnitude t from W. */
    for (long r = 0; r < rows; r++) {
        for (long c = 0; c < cols; c++) {
            const long i = r * cols + c;
            const double ai = a[r * a_stride + c], drive = DRIVE(i);
            if (!(drive > floor)) continue;
            const double lz = LOG_Z(ai, drive);
            const double w = polish ? v[i] : refine_w(v[i], lz);
            double t = shockley_root(lz, w, drive / (2.0 * ai), voff, vt, LOG_PREFACTOR(ai));
            if (t < 0.0) t = 0.0;
            v[i] = t;
        }
    }
#undef LOG_Z
#undef LOG_PREFACTOR
    if (polish && any_active) {
        const double tolerance = residual_tol * (double)rows;
        for (int k = 0; k < max_iter; k++) {
            double sumsq = 0.0;
            for (long r = 0; r < rows; r++) {
                for (long c = 0; c < cols; c++) {
                    const long i = r * cols + c;
                    const double drive = DRIVE(i);
                    if (!(drive > floor)) continue;
                    const double ai = a[r * a_stride + c], t = v[i];
                    double arg = (t - voff) / vt;
                    if (arg > EXP_GUARD) arg = EXP_GUARD;
                    const double current = Is * exp(arg);
                    const double residual = 2.0 * ai * t - drive + current;
                    const double derivative = 2.0 * ai + current / vt;
                    double next = t - residual / derivative;
                    if (next < 0.0) next = 0.0;
                    v[i] = next;
                    sumsq += residual * residual;
                }
            }
            if (sqrt(sumsq) < tolerance) break;
        }
    }
#undef DRIVE
    for (long i = 0; i < rows * cols; i++) {
        const double target = -sign_of(clamp_b(b[i], clip_b, b_clip)) * v[i];
        if (!isfinite(target)) return 2;
        v[i] = omega == 1.0 ? target : v_old[i] + omega * (target - v_old[i]);
    }
    return 0;
}

/* d u + s b + I_s (exp((u - off) / V_t) - 1) = 0 in u = s v, s = +1 forward, -1 reverse. */
int single_diode_update(long rows, long cols, const double *a, long a_stride, const double *b,
                        const double *sign, const double *v_old, double *v, double Is, double vt,
                        double off, int clip_b, double b_clip, double a_min, double z_thresh,
                        int terms, int polish, double abs_tol, double rel_tol, int max_iter,
                        double omega) {
    const long n = rows * cols;
    for (long i = 0; i < n; i++)
        if (!isfinite(clamp_b(b[i], clip_b, b_clip))) return 1;
#define DIAGONAL(r, c) (2.0 * (a[(r) * a_stride + (c)] < a_min ? a_min : a[(r) * a_stride + (c)]))
    if (Is == 0.0) {
        for (long r = 0; r < rows; r++) {
            for (long c = 0; c < cols; c++) {
                const long i = r * cols + c;
                const double target = -clamp_b(b[i], clip_b, b_clip) / DIAGONAL(r, c);
                if (!isfinite(target)) return 2;
                v[i] = omega == 1.0 ? target : v_old[i] + omega * (target - v_old[i]);
            }
        }
        return 0;
    }
    const double log_thresh = LOG_THRESHOLD(z_thresh);
    const double log_is = log(Is), log_vt = log(vt);
    /* Pass 1: W0 into v. */
    for (long r = 0; r < rows; r++) {
        for (long c = 0; c < cols; c++) {
            const long i = r * cols + c;
            const double d = DIAGONAL(r, c), ob = sign[c] * clamp_b(b[i], clip_b, b_clip);
            const double linear = (Is - ob) / d;
            const double lz = (log_is - log(d) - log_vt) + (linear - off) / vt;
            v[i] = shockley_w(lz, log_thresh, terms);
        }
    }
    /* Newton on w + log(w) = log(z) when there is no polish. */
    if (!polish) {
        for (long r = 0; r < rows; r++) {
            for (long c = 0; c < cols; c++) {
                const long i = r * cols + c;
                const double d = DIAGONAL(r, c), ob = sign[c] * clamp_b(b[i], clip_b, b_clip);
                const double lz = (log_is - log(d) - log_vt) + ((Is - ob) / d - off) / vt;
                v[i] = refine_w(v[i], lz);
            }
        }
    }
    /* Pass 2: voltage from W0, without cancellation when the diode conducts. */
    for (long r = 0; r < rows; r++) {
        for (long c = 0; c < cols; c++) {
            const long i = r * cols + c;
            const double d = DIAGONAL(r, c), ob = sign[c] * clamp_b(b[i], clip_b, b_clip);
            const double linear = (Is - ob) / d, lp = log_is - log(d) - log_vt;
            v[i] = shockley_root(lp + (linear - off) / vt, v[i], linear, off, vt, lp);
        }
    }
    if (polish) {
        for (int k = 0; k < max_iter; k++) {
            double max_f = 0.0, magnitude = 0.0;
            for (long r = 0; r < rows; r++) {
                for (long c = 0; c < cols; c++) {
                    const long i = r * cols + c;
                    const double d = DIAGONAL(r, c), bw = clamp_b(b[i], clip_b, b_clip);
                    const double ob = sign[c] * bw, u = v[i];
                    double arg = (u - off) / vt;
                    if (arg > EXP_GUARD) arg = EXP_GUARD;
                    const double e = exp(arg);
                    const double residual = d * u + ob + Is * (e - 1.0);
                    const double derivative = d + (Is / vt) * e;
                    const double next = u - residual / derivative;
                    v[i] = next;
                    if (!(fabs(residual) <= max_f)) max_f = fabs(residual);
                    const double size = d * fabs(next) + fabs(bw) + Is + 1.0;
                    if (!(size <= magnitude)) magnitude = size;
                }
            }
            if (max_f < abs_tol && max_f / (magnitude + a_min) < rel_tol) break;
        }
    }
    for (long r = 0; r < rows; r++) {
        for (long c = 0; c < cols; c++) {
            const long i = r * cols + c;
            const double target = sign[c] * v[i];
            if (!isfinite(target)) return 2;
            v[i] = omega == 1.0 ? target : v_old[i] + omega * (target - v_old[i]);
        }
    }
#undef DIAGONAL
    return 0;
}
