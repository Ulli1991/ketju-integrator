// Ketju-integrator (MSTAR)
// Copyright (C) 2020-2022 Antti Rantala, Matias Mannerkoski, and contributors.
// Licensed under the GPLv3 license.
// See LICENSE.md or <https://www.gnu.org/licenses/> for details.
// Modified by Ulrich Steinwandel, 2026: rewritten in C++ with Kokkos support (GPU).
// Copyright (C) 2026 Ulrich Steinwandel. Licensed under the GPLv3 license.
//
// Host- and device-callable pieces shared by the engine (engine.cpp) and the host merger code (mergers.cpp):
// GADGET softening kernels (internal.h / integration.c), the osculating time since pericentre and the contact rule
// of the collisions inside MSTAR (bh_mergers.c, ketju_do_star_collisions), and the stop-collision distance rule.

#pragma once

#include <Kokkos_Core.hpp>
#include <cmath>

namespace mstar {
namespace detail {

// internal.h softened_inverse_r
KOKKOS_INLINE_FUNCTION double softened_inverse_r(double r, double h) {
    if (h <= 0) return 1. / r;
    const double hinv = 1. / h;
    const double u = r * hinv;
    if (u >= 1.) {
        return 1. / r;
    } else if (u < 0.5) {
        return -hinv * (-2.8 + u * u * (16. / 3. + u * u * (6.4 * u - 9.6)));
    } else {
        return -hinv * (-3.2 + 2. / 30. / u + u * u * (32. / 3. + u * (-16.0 + u * (9.6 - 32. / 15. * u))));
    }
}

// integration.c softened_inverse_r3
KOKKOS_INLINE_FUNCTION double softened_inverse_r3(double r, double h) {
    if (h <= 0) return 1. / (r * r * r);
    const double hinv = 1. / h;
    const double u = r * hinv;
    if (u >= 1) {
        return 1.0 / (r * r * r);
    } else if (u < 0.5) {
        return hinv * hinv * hinv * (32. / 3. + u * u * (32.0 * u - 38.4));
    } else {
        return hinv * hinv * hinv * (64. / 3. - 48.0 * u + 38.4 * u * u - 32. / 3. * u * u * u - 2. / 30. / (u * u * u));
    }
}

// Event location at the softening kernel's break points (integration.c softening_kink_theta). The spline's higher
// derivatives jump at u = r / h = 0.5 and 1; a GBS step across such a point loses the smoothness the extrapolation and
// its error estimate assume (Hairer, Norsett & Wanner I, II.6; Gear & Osterby 1984), so the step is landed on the
// crossing. r^2 of a pair over the step is the cubic Hermite interpolant of its end values and slopes
// (d r^2 / d theta = 2 r.v dt, theta = (t - t0) / dt in [0, 1]).
struct HermiteR2 {
    double g0, g1, d0, d1;
    KOKKOS_INLINE_FUNCTION double operator()(double t) const {
        const double t2 = t * t, t3 = t2 * t;
        return g0 * (2 * t3 - 3 * t2 + 1) + d0 * (t3 - 2 * t2 + t) + g1 * (-2 * t3 + 3 * t2) + d1 * (t3 - t2);
    }
    KOKKOS_INLINE_FUNCTION double slope(double t) const {   // d r^2 / d theta
        const double t2 = t * t;
        return g0 * (6 * t2 - 6 * t) + d0 * (3 * t2 - 4 * t + 1) + g1 * (-6 * t2 + 6 * t) + d1 * (3 * t2 - 2 * t);
    }
    // first theta in [0, 1] with r^2 = c, 2 if none: monotone pieces between the roots of the slope, bisection
    KOKKOS_INLINE_FUNCTION double first_crossing(double c) const {
        const double A = 6 * g0 + 3 * d0 - 6 * g1 + 3 * d1, B = -6 * g0 - 4 * d0 + 6 * g1 - 2 * d1, C = d0;
        double edge[4] = {0, 1, 1, 1};
        int ne = 1;
        if (fabs(A) > 1e-14 * (fabs(B) + fabs(C))) {
            const double disc = B * B - 4 * A * C;
            if (disc > 0) {
                const double q = -0.5 * (B + (B >= 0 ? sqrt(disc) : -sqrt(disc)));
                double r1 = q / A, r2 = q != 0 ? C / q : r1;
                if (r1 > r2) { const double tmp = r1; r1 = r2; r2 = tmp; }
                if (r1 > 0 && r1 < 1) edge[ne++] = r1;
                if (r2 > 0 && r2 < 1 && r2 != r1) edge[ne++] = r2;
            }
        } else if (B != 0) {
            const double r = -C / B;
            if (r > 0 && r < 1) edge[ne++] = r;
        }
        edge[ne++] = 1;
        for (int q = 0; q + 1 < ne; q++) {
            double a = edge[q], b = edge[q + 1];
            const double fa = (*this)(a) - c, fb = (*this)(b) - c;
            if (fa == 0) return a;
            if ((fa < 0) == (fb < 0) && fb != 0) continue;
            for (int it = 0; it < 50; it++) {
                const double m = 0.5 * (a + b), fm = (*this)(m) - c;
                if ((fm < 0) == (fa < 0) && fm != 0) a = m; else b = m;
            }
            return 0.5 * (a + b);
        }
        return 2;
    }
};

// Local error of a GBS step across a kink: the 4th time derivative of the pair's relative acceleration jumps by
// J = G M / h^2 [f] (v_r / h)^4, [f] = 1536 at u = 0.5 and 384 at u = 1 (f(u) the spline force in units of
// G M / h^2), and the extrapolation leaves a position error ~ J dt^6 / 6!. Crossings whose error is below
// softening_kink_error_ratio x the tolerance on the pair's separation (tol u h) need no landing (grazing crossings,
// short steps).
KOKKOS_INLINE_FUNCTION double softening_kink_error(double GM, double h, double jump, double vr, double dt) {
    const double w = vr / h, dt2 = dt * dt;
    return GM / (h * h) * jump * (w * w) * (w * w) * (dt2 * dt2 * dt2) / 720;
}

// First crossing of u = r / h = 0.5 or 1 that needs a landing, by a pair of total mass M over a step: dr0, dv0
// (r = x_j - x_i, v_j - v_i) at the start, dr1, dv1 at the end, dt the step's length, tol the GBS tolerance times
// softening_kink_error_ratio; 2 if none
KOKKOS_INLINE_FUNCTION double softening_kink_theta(const double* dr0, const double* dv0, const double* dr1,
                                                   const double* dv1, double dt, double h, double GM, double tol) {
    HermiteR2 p{0, 0, 0, 0};
    for (int k = 0; k < 3; k++) {
        p.g0 += dr0[k] * dr0[k];
        p.g1 += dr1[k] * dr1[k];
        p.d0 += dr0[k] * dv0[k];
        p.d1 += dr1[k] * dv1[k];
    }
    p.d0 *= 2 * dt;
    p.d1 *= 2 * dt;
    double theta = 2;
    for (int b = 0; b < 2; b++) {
        const double uc = b == 0 ? 0.5 : 1.0, jump = b == 0 ? 1536.0 : 384.0;
        const double t = p.first_crossing(uc * uc * h * h);
        if (!(t < theta)) continue;
        const double vr = fabs(p.slope(t)) / (2 * uc * h * fabs(dt));   // |dr/dt| at the crossing
        if (softening_kink_error(GM, h, jump, vr, dt) > tol * uc * h) theta = t;
    }
    // A passage hidden in one step: a turning point of r inside the step (pericentre or apocentre) can carry the
    // pair across a break point and back between the step's ends, which the interpolant then misses (algorithmic
    // regularisation takes close passages in few steps). Near the softened region (an end inside 2 h, or a
    // Keplerian pericentre below 2 h) such a step is shortened until the pair moves at most h / 4 (at most 20x per
    // retry), where the crossings are located as above.
    if (theta > 1 && (p.d0 < 0) != (p.d1 < 0)) {
        double v2 = 0, disp2 = 0;
        for (int k = 0; k < 3; k++) {
            v2 += dv0[k] * dv0[k];
            disp2 += (dr1[k] - dr0[k]) * (dr1[k] - dr0[k]);
        }
        const double disp = sqrt(disp2);
        bool near = p.g0 < 4 * h * h || p.g1 < 4 * h * h;
        if (!near && p.d0 < 0) {
            const double r0 = sqrt(p.g0), rv = p.d0 / (2 * dt);
            const double L2 = p.g0 * v2 - rv * rv, eps = 0.5 * v2 - GM / r0;
            const double e2 = 1 + 2 * eps * L2 / (GM * GM), e = sqrt(e2 > 0 ? e2 : 0);
            near = L2 / (GM * (1 + e)) < 2 * h;
        }
        if (near && disp > 0.25 * h && softening_kink_error(GM, h, 1536.0, disp / fabs(dt), dt) > tol * 0.5 * h)
            theta = fmax(0.25 * h / disp, 0.05);
    }
    return theta;
}

// bh_mergers.c time_since_pericentre: time since the pericentre of the osculating two-body orbit (r.v >= 0 side)
KOKKOS_INLINE_FUNCTION double time_since_pericentre(double mu, double r, double v2, double rv, double e) {
    const double eps = 0.5 * v2 - mu / r, h2 = r * r * v2 - rv * rv, p = h2 / mu;
    if (fabs(eps) * r < 1e-8 * mu) {   // parabolic (Barker)
        const double D = rv / sqrt(mu * p);
        return 0.5 * sqrt(p * p * p / mu) * (D + D * D * D / 3);
    }
    const double a = fabs(mu / (2 * eps));
    if (eps < 0) {
        const double E = atan2(rv / sqrt(mu * a), 1 - r / a);
        return (E - e * sin(E)) * sqrt(a * a * a / mu);
    }
    const double F = asinh(rv / (e * sqrt(mu * a)));
    return (e * sinh(F) - F) * sqrt(a * a * a / mu);
}

// ketju_do_star_collisions pair test: dr = x_j - x_i, dv = v_j - v_i; rv_prev: r.v at the last check (pericentre
// test only if use_prev), dt_check the time since then; rc = R_i + R_j
KOKKOS_INLINE_FUNCTION bool star_contact(const double* dr, const double* dv, double mi, double mj, double G, double rc,
                                         bool use_prev, double rv_prev, double dt_check) {
    double r2 = 0, v2 = 0, rv = 0;
    for (int k = 0; k < 3; ++k) {
        r2 += dr[k] * dr[k];
        v2 += dv[k] * dv[k];
        rv += dr[k] * dv[k];
    }
    bool hit = r2 < rc * rc;
    if (!hit && use_prev && rv_prev < 0 && rv >= 0) {
        const double mu = G * (mi + mj);
        const double h2 = r2 * v2 - rv * rv;
        const double e2 = 1 + (v2 - 2 * mu / sqrt(r2)) * h2 / (mu * mu);
        const double e = sqrt(e2 > 0 ? e2 : 0), rp = (h2 / mu) / (1 + e);
        // r.v can also turn positive at an osculating apocentre (perturbed pair)
        hit = rp < rc && time_since_pericentre(mu, sqrt(r2), v2, rv, e) <= 1.000001 * dt_check;
    }
    return hit;
}

// Collision distance of the stop rule (StopCollisions); type 1 star-star, 2 tidal disruption, 3 ISCO / BH-BH
KOKKOS_INLINE_FUNCTION double collision_distance(bool bh1, double m1, double R1, bool bh2, double m2, double R2,
                                                 double G, double c, bool bh_max_single, int& type) {
    const double r_isco = c > 0.0 ? 6.0 * G * (m1 + m2) / (c * c) : 0.0;
    if (bh1 && bh2) {
        type = 3;
        if (!(c > 0.0)) return 0.0;
        return bh_max_single ? 6.0 * G * (m1 > m2 ? m1 : m2) / (c * c) : 20.0 * G * (m1 + m2) / (c * c);
    }
    if (bh1 || bh2) {
        const double m_s = bh1 ? m2 : m1, R_s = bh1 ? R2 : R1, M_bh = bh1 ? m1 : m2;
        double r_t = 0.0;
        if (R_s > 0.0 && m_s > 0.0) {
            r_t = 1.3 * cbrt(M_bh / m_s) * R_s;
            if (r_t < R_s) r_t = R_s;
        }
        if (r_isco >= r_t) { type = 3; return r_isco; }
        type = 2;
        return r_t;
    }
    type = 1;
    return R1 + R2;
}

}  // namespace detail
}  // namespace mstar
