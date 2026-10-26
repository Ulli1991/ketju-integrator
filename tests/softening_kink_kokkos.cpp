// Ketju-integrator tests: GBS steps landed on the softening kernel's break points (softening_kink_landing), on the
// C++/Kokkos library: the cases and bounds of unit-tests/softening_kink.cpp of the C library (branch ketju-c).
// Copyright (C) 2026 Ulrich Steinwandel.
// Licensed under the GPLv3 license.
// See LICENSE.md or <https://www.gnu.org/licenses/> for details.
//
// The GADGET spline's higher derivatives jump at u = r / h = 0.5 and 1. Without landing, a softened pair crossing
// them inside a GBS step defeats the error estimate: |dE/E| 1e2 - 1e6 x the tolerance, not converging with it.
// Bounds 1.3x the C library's measured values (gcc 15, x86-64; this library gives the same on the CPU).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>
#include <vector>

#include "mstar/mstar.hpp"

namespace {

struct EnsembleResult {
    double max_err, median_err;
    long kink_landings;
};

// 16 initial states (x_0 shifted by q 1e-9): the largest |dE/E| and the summed landing count.
// case 0: two stars (1, 0.7) on a softened orbit through h = 0.3 (apocentre separation ~2, impact 0.02), T = 20;
// case 1: the stella collision set-up (three stars, R = 0.05, h = 0.3, external tide), T = 20
EnsembleResult ensemble(int which, double tol, bool landing, double soft = 0.3) {
    EnsembleResult res = {0, 0, 0};
    std::vector<double> errs;
    for (int q = 0; q < 16; q++) {
        const int n = which == 0 ? 2 : 3;
        mstar::System S(0, n);
        S.constants.G = 1;
        S.constants.c = 1;
        S.options.PN_flags = mstar::PN_NONE;
        S.options.gbs_relative_tolerance = tol;
        S.options.output_time_relative_tolerance = 1e-12;
        S.options.star_star_softening = soft;
        S.options.softening_kink_landing = landing;
        if (which == 0) {
            const double x0[3] = {1, 0, 0}, x1[3] = {-1, 0.02, 0}, v0[3] = {0, 0, 0};
            S.set_particle(0, 1, x0, v0);
            S.set_particle(1, 0.7, x1, v0);
        } else {
            const double T[6] = {-0.002, 0.001, 0.0015, 0.0004, -0.0003, 0.0006};
            const double x[3][3] = {{1, 0, 0}, {-1, 0.02, 0}, {0, 3, 0.5}};
            const double v[3][3] = {{-0.3, 0, 0}, {0.3, 0, 0}, {0.2, 0, 0}};
            const double m[3] = {1, 0.7, 0.5};
            S.collision_radius = {0.05, 0.05, 0.05};
            S.options.enable_external_tide = true;
            for (int k = 0; k < 6; ++k) S.options.external_tidal_tensor[k] = T[k];
            for (int i = 0; i < 3; ++i) S.set_particle(i, m[i], x[i], v[i]);
        }
        S.pos[0] += q * 1e-9;
        S.run(20);
        res.max_err = std::max(res.max_err, std::fabs(S.perf.relative_energy_error));
        errs.push_back(std::fabs(S.perf.relative_energy_error));
        res.kink_landings += S.perf.kink_landings;
    }
    std::sort(errs.begin(), errs.end());
    res.median_err = errs[8];
    return res;
}

} // namespace

// The softened two-body orbit: with landing |dE/E| converges with the tolerance; without it does not
TEST(Softening_kink, two_body_orbit_converges) {
    const EnsembleResult a = ensemble(0, 1e-11, true), b = ensemble(0, 1e-13, true), off = ensemble(0, 1e-13, false);
    std::printf("  two-body: median / max |dE/E| tol 1e-11 %.3e / %.3e, 1e-13 %.3e / %.3e (landings %ld), landing off 1e-13 "
                "%.3e / %.3e\n", a.median_err, a.max_err, b.median_err, b.max_err, b.kink_landings, off.median_err, off.max_err);
    // Bounds 1.3x the largest of 8 sets of 16 states (C, CPU): medians 1.2e-10..2.7e-10 (1e-11), 1.2e-12..1.9e-11
    // (1e-13); maxima 6.0e-10..7.8e-9, 2.3e-10..2.4e-9, from single steps in the smooth core where the GBS error
    // estimate misses (no kink involved). A100 (FMA): 1.8e-10 / 7.7e-12, maxima 2.4e-9 / 4.6e-10.
    EXPECT_LT(a.median_err, 3.5e-10);
    EXPECT_LT(b.median_err, 2.5e-11);
    EXPECT_LT(b.median_err, a.median_err / 5);   // converges with the tolerance (sets: 6x - 180x for 100x)
    EXPECT_LT(a.max_err, 1e-8);
    EXPECT_LT(b.max_err, 3.1e-9);
    EXPECT_GT(b.kink_landings, 0);
    EXPECT_GT(off.max_err, 100 * b.max_err);   // negative check: without landing (CPU 1.18e-5, A100 3.7e-7)
}

// The stella collision with tide and softening (the pair crosses u = 1 and 0.5 before it merges)
TEST(Softening_kink, collision_tide_softening_converges) {
    const EnsembleResult a = ensemble(1, 1e-11, true), b = ensemble(1, 1e-13, true), off = ensemble(1, 1e-11, false);
    std::printf("  collision: max |dE/E| tol 1e-11 %.3e, 1e-13 %.3e (landings %ld), landing off 1e-11 %.3e\n", a.max_err,
                b.max_err, b.kink_landings, off.max_err);
    EXPECT_LT(a.max_err, 1.35e-12);   // measured 1.04e-12
    EXPECT_LT(b.max_err, 5.0e-13);    // measured 3.81e-13
    EXPECT_GT(b.kink_landings, 0);
    EXPECT_GT(off.max_err, 100 * a.max_err);   // negative check (measured 4.32e-8)
}

// No landing without softening, or with no_star_star_gravity (no star-star forces)
TEST(Softening_kink, off_without_softening) {
    EXPECT_EQ(ensemble(0, 1e-11, true, 0.0).kink_landings, 0);
}
