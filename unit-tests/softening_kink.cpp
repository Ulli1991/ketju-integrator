// Ketju-integrator tests: GBS steps landed on the softening kernel's break points (softening_kink_landing).
// Copyright (C) 2026 Ulrich Steinwandel.
// Licensed under the GPLv3 license.
// See LICENSE.md or <https://www.gnu.org/licenses/> for details.
//
// The GADGET spline's higher derivatives jump at u = r / h = 0.5 and 1. Without landing, a softened pair crossing
// them inside a GBS step defeats the error estimate: |dE/E| 1e2 - 1e6 x the tolerance, not converging with it.
// The same cases run on the C++/Kokkos library in kokkos/tests/softening_kink_kokkos.cpp. Bounds 1.3x measured
// (gcc 15, x86-64).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>
#include <mpi.h>
#include <vector>

#include "ketju_integrator/ketju_integrator.h"

namespace {

struct EnsembleResult {
    double max_err;
    long kink_landings;
};

// 16 initial states (x_0 shifted by q 1e-9): the largest |dE/E| and the summed landing count.
// case 0: two stars (1, 0.7) on a softened orbit through h = 0.3 (apocentre separation ~2, impact 0.02), T = 20;
// case 1: the stella collision set-up (three stars, R = 0.05, h = 0.3, external tide), T = 20
EnsembleResult ensemble(int which, double tol, bool landing, double soft = 0.3) {
    EnsembleResult res = {0, 0};
    for (int q = 0; q < 16; q++) {
        struct ketju_system R;
        const int n = which == 0 ? 2 : 3;
        ketju_create_system(&R, 0, n, MPI_COMM_WORLD);
        R.constants->G = 1;
        R.constants->c = 1;
        R.options->PN_flags = KETJU_PN_NONE;
        R.options->gbs_relative_tolerance = tol;
        R.options->output_time_relative_tolerance = 1e-12;
        R.options->star_star_softening = soft;
        R.options->softening_kink_landing = landing;
        double radius[3] = {0.05, 0.05, 0.05};
        if (which == 0) {
            R.physical_state->mass[0] = 1;
            R.physical_state->mass[1] = 0.7;
            R.physical_state->pos[0][0] = 1;
            R.physical_state->pos[1][0] = -1;
            R.physical_state->pos[1][1] = 0.02;
        } else {
            const double T[6] = {-0.002, 0.001, 0.0015, 0.0004, -0.0003, 0.0006};
            const double x[3][3] = {{1, 0, 0}, {-1, 0.02, 0}, {0, 3, 0.5}};
            const double v[3][3] = {{-0.3, 0, 0}, {0.3, 0, 0}, {0.2, 0, 0}};
            const double m[3] = {1, 0.7, 0.5};
            R.collision_radius = radius;
            R.options->enable_external_tide = true;
            for (int k = 0; k < 6; ++k) R.options->external_tidal_tensor[k] = T[k];
            for (int i = 0; i < 3; ++i) {
                R.physical_state->mass[i] = m[i];
                for (int k = 0; k < 3; k++) {
                    R.physical_state->pos[i][k] = x[i][k];
                    R.physical_state->vel[i][k] = v[i][k];
                }
            }
        }
        R.physical_state->pos[0][0] += q * 1e-9;
        ketju_run_integrator(&R, 20);
        res.max_err = std::max(res.max_err, std::fabs(R.perf->relative_energy_error));
        res.kink_landings += R.perf->kink_landings;
        ketju_free_system(&R);
    }
    return res;
}

} // namespace

// The softened two-body orbit: with landing |dE/E| converges with the tolerance; without it does not
TEST(Softening_kink, two_body_orbit_converges) {
    const EnsembleResult a = ensemble(0, 1e-11, true), b = ensemble(0, 1e-13, true), off = ensemble(0, 1e-13, false);
    std::printf("  two-body: max |dE/E| tol 1e-11 %.3e, 1e-13 %.3e (landings %ld), landing off 1e-13 %.3e\n", a.max_err,
                b.max_err, b.kink_landings, off.max_err);
    EXPECT_LT(a.max_err, 2.2e-8);   // measured 1.65e-8
    EXPECT_LT(b.max_err, 1.1e-9);   // measured 8.54e-10 (the maximum over 16 states scatters by 4x)
    EXPECT_LT(b.max_err, a.max_err / 10);   // converges with the tolerance
    EXPECT_GT(b.kink_landings, 0);
    EXPECT_GT(off.max_err, 100 * b.max_err);   // negative check: without landing (measured 1.18e-5)
}

// The stella collision with tide and softening (the pair crosses u = 1 and 0.5 before it merges)
TEST(Softening_kink, collision_tide_softening_converges) {
    const EnsembleResult a = ensemble(1, 1e-11, true), b = ensemble(1, 1e-13, true), off = ensemble(1, 1e-11, false);
    std::printf("  collision: max |dE/E| tol 1e-11 %.3e, 1e-13 %.3e (landings %ld), landing off 1e-11 %.3e\n", a.max_err,
                b.max_err, b.kink_landings, off.max_err);
    EXPECT_LT(a.max_err, 2.3e-12);    // measured 1.71e-12
    EXPECT_LT(b.max_err, 6.7e-13);    // measured 5.11e-13
    EXPECT_GT(b.kink_landings, 0);
    EXPECT_GT(off.max_err, 100 * a.max_err);   // negative check (measured 4.32e-8)
}

// No landing without softening, or with no_star_star_gravity (no star-star forces)
TEST(Softening_kink, off_without_softening) {
    EXPECT_EQ(ensemble(0, 1e-11, true, 0.0).kink_landings, 0);
}
