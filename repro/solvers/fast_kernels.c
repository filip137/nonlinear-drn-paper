/* Node kernels of fast_cd.py: the Shockley coordinate solve of one layer, then the relaxation
 * v_old + omega (v - v_old). Compared with the framework kernel (diode_kernels.c), three
 * options cut the work per node:
 *
 *   small   z = exp(lz) < 1e-5 (log z < SMALL_LOG_Z): W0(z) = z - z^2 + 3/2 z^3, relative error
 *           below 8/3 z^3 < 3e-15. One exp, no Halley, no polish.
 *   polish  0: no Newton polish on the voltage; W is solved to roundoff instead (Halley, or the
 *           series refined by Newton on w + log w = log z). 1: per-node polish, stopping at the
 *           node's own tolerance (at most max_iter steps).
 *   warm    Newton on w + log w = log z from the node's W of the previous call (w_store), at
 *           most WARM_STEPS steps; if it does not reach roundoff, the cold path runs.
 *
 * Arrays have rows x cols entries (row-major); two_a and lp hold one value per column (node):
 * 2a and log_prefactor. counts[] accumulates per-node paths: 0 off (double diode, v = 0),
 * 1 small, 2 warm, 3 Halley, 4 series, 5 polish steps.
 * Returns 0 on success, 1 for a non-finite log z, 2 for a non-finite result.
 */
#include <float.h>
#include <math.h>

#define SMALL_LOG_Z (-11.512925464970229)    /* log(1e-5) */
#define LOG_THRESHOLD 23.025850929940457     /* log(1e10), as the framework */
#define WARM_STEPS 4
#define TOL (4.0 * DBL_EPSILON)

static double clamp_b(double b, double limit) {
    return b < -limit ? -limit : (b > limit ? limit : b);
}

/* Asymptotic W0 from l1 = log z, four terms of the de Bruijn series. */
static double series(double l1) {
    const double l2 = log(l1);
    return l1 - l2 + l2 / l1 + l2 * (l2 - 2.0) / (2.0 * l1 * l1)
           + l2 * (6.0 - 9.0 * l2 + 2.0 * l2 * l2) / (6.0 * l1 * l1 * l1);
}

/* W0(z), z >= 0: Winitzki seed, Halley steps until a step is within 4 ulp (at most 5). */
static double halley(double z) {
    if (z == 0.0) return 0.0;
    const double lp = z > 1e6 ? log(z) : log1p(z);
    double w = lp * (1.0 - log1p(lp) / (2.0 + lp));
    for (int k = 0; k < 5; k++) {
        const double f = w - z * exp(-w), wp1 = w + 1.0;
        const double step = f / (wp1 - (w + 2.0) * f / (2.0 * wp1));
        w -= step;
        if (!(fabs(step) > TOL * fabs(w))) break;
    }
    return w;
}

/* Newton on g(w) = w + log w - lz (g' = 1 + 1/w). Returns 1 once a step is within 4 ulp. */
static int newton_log(double *w, double lz, int steps) {
    double x = *w;
    for (int k = 0; k < steps; k++) {
        if (!(x > 0.0)) return 0;
        const double step = (x + log(x) - lz) * (x / (x + 1.0));
        x -= step;
        if (!(fabs(step) > TOL * fabs(x))) {
            if (!(x > 0.0)) return 0;
            *w = x;
            return 1;
        }
    }
    return 0;
}

/* W0(exp(lz)); *path is set to 1 small, 2 warm, 3 Halley, 4 series. */
static double solve_w(double lz, double *w_store, int small, int warm, int polish, int *path) {
    double w;
    if (small && lz < SMALL_LOG_Z) {
        const double z = exp(lz);
        w = z * (1.0 - z * (1.0 - 1.5 * z));
        *path = 1;
    } else {
        w = *w_store;
        if (warm && w > 0.0 && newton_log(&w, lz, WARM_STEPS)) {
            *path = 2;
        } else if (lz <= LOG_THRESHOLD) {
            w = halley(exp(lz));
            *path = 3;
        } else {
            w = series(lz);
            if (!polish) newton_log(&w, lz, 4);  /* series error <= 2e-5: two steps suffice */
            *path = 4;
        }
    }
    *w_store = w;
    return w;
}

/* Voltage from W: l - Vt W, or Voff + Vt (log W - lp) where lz > 0 (no cancellation). */
static double root(double lz, double w, double l, double voff, double vt, double lp) {
    return lz > 0.0 ? voff + vt * (log(w) - lp) : l - vt * w;
}

/* 2a v + b + sign(v) Is exp((|v| - Voff)/Vt) = 0: v = -sign(b) t, 2a (A - t) = Is exp((t - Voff)/Vt). */
int double_update(long rows, long cols, const double *two_a, const double *lp, const double *b,
                  const double *v_old, double *v, double *w_store, double Is, double vt, double voff,
                  double b_clip, int small, int warm, int polish, int max_iter, double tol,
                  double omega, long *counts) {
    const double floor = Is * exp(-voff / vt);
    for (long r = 0; r < rows; r++) {
        for (long c = 0; c < cols; c++) {
            const long i = r * cols + c;
            const double bc = clamp_b(b[i], b_clip), drive = fabs(bc);
            double target = 0.0;
            if (drive > floor) {
                const double d = two_a[c], A = drive / d, lz = lp[c] + (A - voff) / vt;
                if (isnan(lz) || lz == INFINITY) return 1;
                int path;
                const double w = solve_w(lz, &w_store[i], small, warm, polish, &path);
                counts[path]++;
                double t = root(lz, w, A, voff, vt, lp[c]);
                if (t < 0.0) t = 0.0;
                if (polish && path != 1) {
                    for (int k = 0; k < max_iter; k++) {
                        const double current = Is * exp((t - voff) / vt);
                        const double residual = d * t - drive + current;
                        double next = t - residual / (d + current / vt);
                        t = next < 0.0 ? 0.0 : next;
                        counts[5]++;
                        if (fabs(residual) < tol) break;
                    }
                }
                target = bc > 0.0 ? -t : (bc < 0.0 ? t : 0.0);
            } else {
                w_store[i] = 0.0;
                counts[0]++;
            }
            const double value = omega == 1.0 ? target : v_old[i] + omega * (target - v_old[i]);
            if (!isfinite(value)) return 2;
            v[i] = value;
        }
    }
    return 0;
}

/* 2a v + b + s Is (exp((s v - Voff)/Vt) - 1) = 0: in u = s v, 2a (l - u) = Is exp((u - Voff)/Vt),
 * l = (Is - s b)/2a. */
int single_update(long rows, long cols, const double *two_a, const double *lp, const double *b,
                  const double *sign, const double *v_old, double *v, double *w_store, double Is,
                  double vt, double voff, double b_clip, int small, int warm, int polish,
                  int max_iter, double abs_tol, double rel_tol, double omega, long *counts) {
    for (long r = 0; r < rows; r++) {
        for (long c = 0; c < cols; c++) {
            const long i = r * cols + c;
            const double d = two_a[c], bc = clamp_b(b[i], b_clip), ob = sign[c] * bc;
            const double l = (Is - ob) / d, lz = lp[c] + (l - voff) / vt;
            if (isnan(lz) || lz == INFINITY) return 1;
            int path;
            const double w = solve_w(lz, &w_store[i], small, warm, polish, &path);
            counts[path]++;
            double u = root(lz, w, l, voff, vt, lp[c]);
            if (polish && path != 1) {
                for (int k = 0; k < max_iter; k++) {
                    const double e = exp((u - voff) / vt);
                    const double residual = d * u + ob + Is * (e - 1.0);
                    u -= residual / (d + (Is / vt) * e);
                    counts[5]++;
                    const double size = d * fabs(u) + fabs(bc) + Is + 1.0;
                    if (fabs(residual) < abs_tol && fabs(residual) / size < rel_tol) break;
                }
            }
            const double target = sign[c] * u;
            const double value = omega == 1.0 ? target : v_old[i] + omega * (target - v_old[i]);
            if (!isfinite(value)) return 2;
            v[i] = value;
        }
    }
    return 0;
}
