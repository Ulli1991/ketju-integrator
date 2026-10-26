// Ketju-integrator tests: GIZMO's additions (contact collisions of non-PN particles, the collision energy with tide and
// softening, the pericentre guard, the external tide, the collision_radius initialisation) on the C++/Kokkos library.
// The same cases, set-ups and bounds as the C library's unit-tests/gizmo_additions.cpp (branch ketju-c), on the mstar::System API.
// Copyright (C) 2026 Ulrich Steinwandel.
// Licensed under the GPLv3 license.
// See LICENSE.md or <https://www.gnu.org/licenses/> for details.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <gtest/gtest.h>
#include <vector>

#include "mstar/mstar.hpp"

namespace {

// internal.h softened_inverse_r (an independent copy for the energy as integrated)
double softened_inverse_r(double r, double h) {
    if (h <= 0) return 1. / r;
    const double hinv = 1. / h, u = r * hinv;
    if (u >= 1.) return 1. / r;
    if (u < 0.5) return -hinv * (-2.8 + u * u * (16. / 3. + u * u * (6.4 * u - 9.6)));
    return -hinv * (-3.2 + 2. / 30. / u + u * u * (32. / 3. + u * (-16.0 + u * (9.6 - 32. / 15. * u))));
}

double *P(mstar::System &S, int i) { return &S.pos[3 * i]; }
double *V(mstar::System &S, int i) { return &S.vel[3 * i]; }

// Newtonian energy, plus -1/2 sum m x.T.x of the external tide a = T x if T is given (xx, yy, zz, xy, yz, xz)
double energy(const mstar::System &S, const double *t = nullptr) {
    const double G = S.constants.G;
    double E = 0;
    for (int i = 0; i < S.num_particles; ++i) {
        const double *v = &S.vel[3 * i], *x = &S.pos[3 * i];
        E += 0.5 * S.mass[i] * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        for (int j = i + 1; j < S.num_particles; ++j) {
            double r2 = 0;
            for (int k = 0; k < 3; ++k) r2 += (x[k] - S.pos[3 * j + k]) * (x[k] - S.pos[3 * j + k]);
            E -= G * S.mass[i] * S.mass[j] / std::sqrt(r2);
        }
        if (t) {
            const double T[3][3] = {{t[0], t[3], t[5]}, {t[3], t[1], t[4]}, {t[5], t[4], t[2]}};
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b) E -= 0.5 * S.mass[i] * x[a] * T[a][b] * x[b];
        }
    }
    return E;
}

void setup_stars(mstar::System &S) {
    S.constants.G = 1;
    S.constants.c = 1;
    S.options.PN_flags = mstar::PN_NONE;
    S.options.gbs_relative_tolerance = 1e-12;
    S.options.output_time_relative_tolerance = 1e-12;
}

// Relative two-body state with pericentre rp, at separation r0 approaching, total mass M (G = 1), speed v0 at r0
void flyby_state(double M, double rp, double r0, double v0, double x[3], double v[3]) {
    const double vp = std::sqrt(v0 * v0 + 2 * M * (1 / rp - 1 / r0)), h = rp * vp;
    const double vt = h / r0, vr = -std::sqrt(v0 * v0 - vt * vt);
    x[0] = r0, x[1] = 0, x[2] = 0;
    v[0] = vr, v[1] = vt, v[2] = 0;
}

} // namespace

// collision_radius is empty (collisions off) after construction and after reset
TEST(Gizmo_collision_radius, null_after_create) {
    mstar::System S(2, 1);
    ASSERT_TRUE(S.collision_radius.empty());
    S.collision_radius.assign(3, 1.0);
    S.reset(2, 1);
    ASSERT_TRUE(S.collision_radius.empty());
}

// two stars falling together from rest merge on contact (r < R1 + R2) into one at their CoM; mass and momentum are
// conserved and the energy change is booked in dE_collision
TEST(Gizmo_star_collisions, head_on_contact) {
    for (int with_radius = 1; with_radius >= 0; --with_radius) {
        mstar::System S(0, 2);
        setup_stars(S);
        if (with_radius) S.collision_radius = {0.5, 0.5};
        const double m[2] = {0.5, 1.0}, Vy = 0.1, r0 = 10;
        for (int i = 0; i < 2; ++i) {
            S.mass[i] = m[i];
            P(S, i)[0] = (i ? -m[0] : m[1]) / 1.5 * r0;
            V(S, i)[1] = Vy;
        }
        const double E0 = energy(S), t_end = 50;
        S.run(t_end);
        if (!with_radius) { // feature off: no merger, the stars pass through each other
            EXPECT_EQ(S.num_particles, 2);
            EXPECT_EQ((int)S.mergers.size(), 0);
            continue;
        }
        ASSERT_EQ(S.num_particles, 1);
        ASSERT_EQ((int)S.mergers.size(), 1);
        const auto &mi = S.mergers[0];
        EXPECT_EQ(mi.is_collision, 1);
        EXPECT_EQ(mi.m1, 1.0); // the heavier star survives
        EXPECT_EQ(mi.m2, 0.5);
        EXPECT_EQ(mi.extra_index1, 1); // ... with its caller index
        EXPECT_EQ(mi.extra_index2, 0);
        EXPECT_EQ(S.extra_index[0], 1);
        EXPECT_EQ(S.mass[0], 1.5);
        const double x = 1 / r0, t_contact = std::sqrt(r0 * r0 * r0 / (2 * 1.5)) * (std::sqrt(x * (1 - x)) + std::acos(std::sqrt(x)));
        EXPECT_GE(mi.t_merger, t_contact * (1 - 1e-9));
        EXPECT_LE(mi.t_merger, 0.5 * M_PI * std::sqrt(r0 * r0 * r0 / (2 * 1.5)) * (1 + 1e-9)); // before r = 0
        EXPECT_NEAR(P(S, 0)[0], 0, 1e-10);
        EXPECT_NEAR(P(S, 0)[1], Vy * t_end, 1e-10);
        EXPECT_NEAR(V(S, 0)[0], 0, 1e-12);
        EXPECT_NEAR(V(S, 0)[1], Vy, 1e-12);
        EXPECT_EQ(S.collision_radius[0], 0.5);
        EXPECT_NEAR(energy(S), E0 + mi.dE_collision, 1e-10 * std::abs(E0));
        EXPECT_NEAR(energy(S), 0.5 * 1.5 * Vy * Vy, 1e-12);
        EXPECT_LT(std::abs(S.perf.relative_energy_error), 1e-10);
    }
}

// a flyby whose two-body pericentre is 0.8 (R1 + R2) collides even if no accepted step lands inside contact
// (pericentre passage test); 1.25 (R1 + R2) misses. A bound third star stays: E_end = E0 + dE_collision.
TEST(Gizmo_star_collisions, graze_and_miss) {
    for (double f : {0.8, 1.25}) {
        mstar::System S(0, 3);
        setup_stars(S);
        S.collision_radius = {0.05, 0.05, 0.05};
        double x[3], v[3];
        flyby_state(2, f * 0.1, 20, 0.6, x, v);
        for (int k = 0; k < 3; ++k) {
            P(S, 0)[k] = 0.5 * x[k];
            P(S, 1)[k] = -0.5 * x[k];
            V(S, 0)[k] = 0.5 * v[k];
            V(S, 1)[k] = -0.5 * v[k];
        }
        S.mass[0] = S.mass[1] = 1;
        S.mass[2] = 0.1;
        P(S, 2)[1] = 200;
        V(S, 2)[0] = 0.05;
        const double E0 = energy(S);
        S.run(80);
        if (f > 1) {
            EXPECT_EQ((int)S.mergers.size(), 0);
            EXPECT_EQ(S.num_particles, 3);
            EXPECT_NEAR(energy(S), E0, 1e-9 * std::abs(E0));
        } else {
            ASSERT_EQ((int)S.mergers.size(), 1);
            EXPECT_EQ(S.num_particles, 2);
            EXPECT_EQ(S.mergers[0].is_collision, 1);
            EXPECT_EQ(S.mass[0], 2.0);
            EXPECT_NEAR(energy(S), E0 + S.mergers[0].dE_collision, 1e-9 * std::abs(E0));
        }
        EXPECT_LT(std::abs(S.perf.relative_energy_error), 1e-9);
    }
}

// the booked energy change of a collision is minus the pair's internal energy plus the change of its potential energy
// with the other particles as it moves to its CoM; it equals the change of the total energy exactly
TEST(Gizmo_collision_energy, includes_other_members_potential) {
    mstar::System S(0, 3);
    setup_stars(S);
    S.collision_radius = {0.5, 0.5, 0.5};
    const double m[3] = {1.0, 3.0, 2.0};
    const double x[3][3] = {{0.3, 0.1, 0}, {-0.4, 0.0, 0.1}, {2.0, 1.5, -0.5}};
    const double v[3][3] = {{0.2, -0.5, 0.1}, {-0.3, 0.4, 0}, {0.05, 0, 0.1}};
    for (int i = 0; i < 3; ++i) S.set_particle(i, m[i], x[i], v[i]);
    double r01 = 0, v01 = 0, d0 = 0, d1 = 0, dc = 0;
    for (int k = 0; k < 3; ++k) {
        const double xc = (m[0] * x[0][k] + m[1] * x[1][k]) / (m[0] + m[1]);
        r01 += (x[0][k] - x[1][k]) * (x[0][k] - x[1][k]);
        v01 += (v[0][k] - v[1][k]) * (v[0][k] - v[1][k]);
        d0 += (x[2][k] - x[0][k]) * (x[2][k] - x[0][k]);
        d1 += (x[2][k] - x[1][k]) * (x[2][k] - x[1][k]);
        dc += (x[2][k] - xc) * (x[2][k] - xc);
    }
    const double E_pair = 0.5 * m[0] * m[1] / (m[0] + m[1]) * v01 - m[0] * m[1] / std::sqrt(r01);
    const double dE_others = m[2] * (m[0] / std::sqrt(d0) + m[1] / std::sqrt(d1) - (m[0] + m[1]) / std::sqrt(dc));
    const double E0 = energy(S);
    ASSERT_EQ(mstar::do_star_collisions(S, nullptr), 1); // r = 0.71 < R1 + R2 = 1
    ASSERT_EQ(S.num_particles, 2);
    const double dE = S.mergers[0].dE_collision;
    EXPECT_NEAR(energy(S) - E0, dE, 1e-13 * std::abs(E0));
    EXPECT_NEAR(dE, -E_pair + dE_others, 1e-13 * std::abs(E0));
    EXPECT_GT(std::abs(dE_others), 1e-3 * std::abs(E_pair));
    EXPECT_GT(std::abs((energy(S) - E0) - (-E_pair)), 1e8 * 1e-13 * std::abs(E0));
}

// a sign change of r.v counts as a pericentre passage only if the osculating orbit's pericentre time lies since the
// last check. Bound pair e = 0.99, a = 1, G M = 2: rp = 0.01 < R1 + R2 = 0.02.
TEST(Gizmo_pericentre_guard, osculating_pericentre_time) {
    const double M = 2, a = 1, e = 0.99, p = a * (1 - e * e), rc = 0.02;
    auto run = [&](double r, double dt_check) {
        mstar::System S(0, 2);
        setup_stars(S);
        S.collision_radius = {0.5 * rc, 0.5 * rc};
        const double cf = (p / r - 1) / e, f = std::acos(cf);
        const double vr = std::sqrt(M / p) * e * std::sin(f), vt = std::sqrt(M / p) * (1 + e * cf);
        const double xr[3] = {r * cf, r * std::sin(f), 0};
        const double vrel[3] = {vr * cf - vt * std::sin(f), vr * std::sin(f) + vt * cf, 0};
        for (int k = 0; k < 3; ++k) {
            P(S, 0)[k] = 0.5 * xr[k], P(S, 1)[k] = -0.5 * xr[k];
            V(S, 0)[k] = 0.5 * vrel[k], V(S, 1)[k] = -0.5 * vrel[k];
        }
        S.mass[0] = S.mass[1] = 1;
        S.time = 10;
        double rv_prev[5] = {0, -1e-3, 0, 0, 10 - dt_check};
        return mstar::do_star_collisions(S, rv_prev);
    };
    auto t_peri = [&](double r) {
        const double E = std::acos((1 - r / a) / e);
        return (E - e * std::sin(E)) * std::sqrt(a * a * a / M);
    };
    const double r1 = 0.05;
    EXPECT_EQ(run(r1, 2.0 * t_peri(r1)), 1);
    EXPECT_EQ(run(r1, 0.5 * t_peri(r1)), 0);
    const double r2 = 1.9;
    EXPECT_EQ(run(r2, 0.01), 0);
}

// hierarchical triple (K3 IC: three 1 Msun stars, a 50 AU e = 0.2 binary and a third star at ~1200 AU, radii 1 Rsun)
// integrated in host-sized calls: no pair comes near contact, so no collision; the energy is conserved
TEST(Gizmo_pericentre_guard, k3_triple_no_false_collision) {
    const double AU_per_kpc = 2.0626480624709636e8, G = 887.1287;
    const double X[3] = {2.0846989099999994e-06, 1.7938106899999997e-06, -3.8785095999999995e-06};
    const double Vv[3][3] = {{0, 2.8741549917192337, 0.038696562636978307}, {0, -1.9895475219637486, 0.038696562636978307},
                             {0, -0.88460746975548465, -0.077393125273956601}};
    mstar::System S(0, 3);
    setup_stars(S);
    S.constants.G = G;
    S.collision_radius.assign(3, 0.0046504673);
    for (int i = 0; i < 3; ++i) {
        S.mass[i] = 1;
        P(S, i)[0] = X[i] * AU_per_kpc;
        for (int k = 0; k < 3; ++k) V(S, i)[k] = Vv[i][k];
    }
    const double E0 = energy(S);
    const double dt_host = 1.25e-7 * 978.46e6 / 4.7405;
    for (int s = 0; s < 400; ++s) {
        S.run(dt_host);
        ASSERT_EQ((int)S.mergers.size(), 0) << "false collision in call " << s;
    }
    EXPECT_EQ(S.num_particles, 3);
    EXPECT_NEAR(energy(S), E0, 1e-8 * std::abs(E0));
}

// linear external tide a = T x. With T = -w^2 I the CoM of a binary moves exactly as X = X0 cos(w t); the energy
// including -1/2 sum m x.T.x is conserved (the tide's work enters B, not U)
TEST(Gizmo_external_tide, harmonic_com_and_energy) {
    const double w = 0.01, X0 = 1, t_end = 0.5 * M_PI / w * 0.6;
    double steps_off = 0;
    for (int tide = 0; tide <= 1; ++tide) {
        mstar::System S(0, 2);
        setup_stars(S);
        S.options.enable_external_tide = tide;
        const double T[6] = {-w * w, -w * w, -w * w, 0, 0, 0};
        for (int k = 0; k < 6; ++k) S.options.external_tidal_tensor[k] = T[k];
        for (int i = 0; i < 2; ++i) {
            S.mass[i] = 1;
            P(S, i)[0] = X0 + (i ? -0.5 : 0.5);
            V(S, i)[1] = i ? -std::sqrt(0.5) : std::sqrt(0.5);
        }
        const double E0 = energy(S, T), EN0 = energy(S);
        S.run(t_end);
        double Xc[3], Vc[3];
        for (int k = 0; k < 3; ++k) {
            Xc[k] = 0.5 * (P(S, 0)[k] + P(S, 1)[k]);
            Vc[k] = 0.5 * (V(S, 0)[k] + V(S, 1)[k]);
        }
        if (!tide) {
            EXPECT_NEAR(Xc[0], X0, 1e-12);
            EXPECT_NEAR(energy(S), EN0, 1e-10 * std::abs(EN0));
            steps_off = S.perf.successful_steps;
            continue;
        }
        EXPECT_NEAR(Xc[0], X0 * std::cos(w * t_end), 1e-10);
        EXPECT_NEAR(Vc[0], -X0 * w * std::sin(w * t_end), 1e-12);
        EXPECT_NEAR(Xc[1], 0, 1e-10);
        EXPECT_NEAR(energy(S, T), E0, 1e-10 * std::abs(E0));
        EXPECT_NEAR(S.perf.relative_energy_error, (energy(S, T) - E0) / E0, 1e-12);
        EXPECT_LT(std::abs(S.perf.relative_energy_error), 1e-10);
        EXPECT_GT(std::abs(energy(S) - EN0), 1e4 * 1e-10 * std::abs(E0));
        EXPECT_LT(S.perf.successful_steps, 2 * steps_off);
        // the library's energy() is the conserved one
        EXPECT_NEAR(S.energy(), energy(S, T), 1e-14 * std::abs(E0));
    }
}

// the booked collision energy is the change of the energy MSTAR conserves: star-star pairs softened by
// star_star_softening, and -1/2 sum m x.T.x of the external tide
TEST(Gizmo_collision_energy, tide_and_softening) {
    const double m[3] = {1.0, 3.0, 2.0};
    const double x[3][3] = {{0.3, 0.1, 0}, {-0.1, 0.0, 0.1}, {0.9, 0.6, -0.3}};
    const double v[3][3] = {{0.2, -0.5, 0.1}, {-0.3, 0.4, 0}, {0.05, 0, 0.1}};
    const double T[6] = {-0.02, 0.01, 0.015, 0.004, -0.003, 0.006};
    const double h = 0.8;
    double dE_variant[2];
    for (int full = 0; full <= 1; ++full) {
        mstar::System S(0, 3);
        setup_stars(S);
        S.collision_radius = {0.5, 0.5, 0.5};
        S.options.star_star_softening = full ? h : 0;
        S.options.enable_external_tide = full;
        for (int k = 0; k < 6; ++k) S.options.external_tidal_tensor[k] = T[k];
        for (int i = 0; i < 3; ++i) S.set_particle(i, m[i], x[i], v[i]);
        auto E = [&](const mstar::System &Q) {
            double e = energy(Q, full ? T : nullptr);
            for (int i = 0; i < Q.num_particles; ++i)
                for (int j = i + 1; j < Q.num_particles; ++j) {
                    double r2 = 0;
                    for (int k = 0; k < 3; ++k) r2 += (Q.pos[3 * i + k] - Q.pos[3 * j + k]) * (Q.pos[3 * i + k] - Q.pos[3 * j + k]);
                    e += Q.mass[i] * Q.mass[j] * (1 / std::sqrt(r2) - softened_inverse_r(std::sqrt(r2), Q.options.star_star_softening));
                }
            return e;
        };
        const double E0 = E(S);
        EXPECT_NEAR(S.energy(), E0, 1e-14 * std::abs(E0));
        ASSERT_EQ(mstar::do_star_collisions(S, nullptr), 1);
        dE_variant[full] = S.mergers[0].dE_collision;
        EXPECT_NEAR(E(S) - E0, dE_variant[full], 1e-13 * std::abs(E0));
    }
    EXPECT_GT(std::abs(dE_variant[1] - dE_variant[0]), 1e-2 * std::abs(dE_variant[0]));
}

// a collision during run() with softening and the tide: E_end - E0 = dE_collision + the integration error. h = 0.05
// below R1 + R2 = 0.1: no pair crosses the softening kernel's break points before merging (pairs that do defeat the GBS
// error estimate, in the C library too: mstar-compare-c softening_ensemble). Bitwise as the C library on the CPU
// (mstar-compare-c coll_tide_soft_1e-13: |dE/E| 2.0e-13)
TEST(Gizmo_collision_energy, run_with_tide_and_softening) {
    mstar::System S(0, 3);
    setup_stars(S);
    S.options.gbs_relative_tolerance = 1e-13;
    S.collision_radius = {0.05, 0.05, 0.05};
    S.options.star_star_softening = 0.05;
    S.options.enable_external_tide = true;
    const double T[6] = {-0.002, 0.001, 0.0015, 0.0004, -0.0003, 0.0006};
    for (int k = 0; k < 6; ++k) S.options.external_tidal_tensor[k] = T[k];
    const double x[3][3] = {{1, 0, 0}, {-1, 0.02, 0}, {0, 3, 0.5}};
    const double v[3][3] = {{-0.3, 0, 0}, {0.3, 0, 0}, {0.2, 0, 0}};
    const double m[3] = {1, 0.7, 0.5};
    for (int i = 0; i < 3; ++i) S.set_particle(i, m[i], x[i], v[i]);
    const double E0 = S.energy();
    S.run(20);
    ASSERT_EQ((int)S.mergers.size(), 1);
    EXPECT_NEAR(S.energy(), E0 + S.mergers[0].dE_collision, 5e-13 * std::abs(E0));
    EXPECT_LT(std::abs(S.perf.relative_energy_error), 5e-13);
    // negative check: without the booking the energy is off by the collision energy
    EXPECT_GT(std::abs(S.energy() - E0), 1e9 * 5e-13 * std::abs(E0));
}

int main(int argc, char **argv) {
    mstar::ScopeGuard guard(argc, argv);
    int run = 0, failed = 0;
    for (auto &t : testing::registry()) {
        if (argc > 1 && !std::strstr(t.name.c_str(), argv[1])) continue;
        const int before = testing::failures();
        t.run();
        ++run;
        const bool ok = testing::failures() == before;
        if (!ok) ++failed;
        std::printf("[%s] %s\n", ok ? "  OK  " : "FAILED", t.name.c_str());
        std::fflush(stdout);
    }
    std::printf("%d tests, %d failed (%s)\n", run, failed, mstar::execution_space_name());
    return failed ? 1 : 0;
}
