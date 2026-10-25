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
