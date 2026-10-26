// Ketju-integrator tests: KETJU's unit tests (integration, mergers, MST construction) on the C++/Kokkos library.
// The same cases, set-ups and reference values as KETJU's unit-tests/{integration,mergers,mst_construction}.cpp (branch
// ketju-c), on the mstar::System API. Copyright (C) 2020-2022 Matias Mannerkoski and contributors (the cases);
// 2026 Ulrich Steinwandel (this port). Licensed under the GPLv3 license.
// See LICENSE.md or <https://www.gnu.org/licenses/> for details.

#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <vector>

#include "mstar/mstar.hpp"

namespace {
double *P(mstar::System &S, int i) { return &S.pos[3 * i]; }
double *V(mstar::System &S, int i) { return &S.vel[3 * i]; }
double *SP(mstar::System &S, int i) { return &S.spin[3 * i]; }
void unit_system(mstar::System &S, unsigned pn) {
    S.constants.c = 1;
    S.constants.G = 1;
    S.options.PN_flags = pn;
    S.options.gbs_relative_tolerance = 1e-12;
    S.options.output_time_relative_tolerance = 1e-12;
}
struct Expected { double pos[3], vel[3], spin[3]; };
void check_binary(mstar::System &S, const Expected (&e)[2], double reltol) {
    for (int p = 0; p < S.num_particles; ++p)
        for (int i = 0; i < 3; ++i) {
            EXPECT_NEAR(P(S, p)[i], e[p].pos[i], std::abs(e[p].pos[i] * reltol));
            EXPECT_NEAR(V(S, p)[i], e[p].vel[i], std::abs(e[p].vel[i] * reltol));
            EXPECT_NEAR(SP(S, p)[i], e[p].spin[i], std::abs(e[p].spin[i] * reltol));
        }
}
void equal_mass_binary(mstar::System &S, double m0, double m1) {
    S.mass[0] = m0; P(S, 0)[0] = 40; V(S, 0)[1] = .085; SP(S, 0)[0] = 1;
    S.mass[1] = m1; P(S, 1)[0] = -40; V(S, 1)[1] = -.085; SP(S, 1)[1] = 1;
}
} // namespace

// Comparisons to the old AR-CHAIN based integrator, with the reference values computed using gbs tolerance = 1e-14 and
// output time tolerance = 1e-12.
TEST(Integration_versus_old_integrator, binary_system_full_pn_no_spin) {
    mstar::System S(2, 0);
    unit_system(S, mstar::PN_DYNAMIC_ALL);
    equal_mass_binary(S, 1, 1);
    const double tend = 1e6;
    S.run(tend);
    const double reltol = 10 * S.options.gbs_relative_tolerance * S.perf.successful_steps;
    ASSERT_EQ(S.num_pn_particles, 2);
    ASSERT_NEAR(S.time, tend, tend * S.options.output_time_relative_tolerance);
    const Expected e[2] = {{{-15.675129114282553, -24.12600028245775, 0.0}, {0.07932781836871444, -0.051044405469834665, 0.0}, {1.0, 0.0, 0.0}},
                           {{15.675129114282553, 24.12600028245775, 0.0}, {-0.07932781836871444, 0.051044405469834665, 0.0}, {0.0, 1.0, 0.0}}};
    check_binary(S, e, reltol);
}

TEST(Integration_versus_old_integrator, binary_system_no_pn) {
    mstar::System S(2, 0);
    unit_system(S, mstar::PN_NONE);
    equal_mass_binary(S, 1, 1);
    const double tend = 1e6;
    S.run(tend);
    const double reltol = 20 * S.options.gbs_relative_tolerance * S.perf.successful_steps;
    ASSERT_EQ(S.num_pn_particles, 2);
    ASSERT_NEAR(S.time, tend, tend * S.options.output_time_relative_tolerance);
    const Expected e[2] = {{{28.093181660139404, -31.029347533391817, 0.0}, {0.05450807268322124, 0.060820810188436096, 0.0}, {1.0, 0.0, 0.0}},
                           {{-28.093181660139404, 31.029347533391817, 0.0}, {-0.05450807268322124, -0.060820810188436096, 0.0}, {0.0, 1.0, 0.0}}};
    check_binary(S, e, reltol);
}

TEST(Integration_threebody_terms, binary_system_with_and_without_threebody_terms) {
    mstar::System S(2, 0), S2(2, 0);
    unit_system(S, mstar::PN_1_0 | mstar::PN_THREEBODY);
    unit_system(S2, mstar::PN_1_0);
    for (auto *R : {&S, &S2}) {
        R->mass[0] = .5; P(*R, 0)[0] = 40; V(*R, 0)[1] = .085;
        R->mass[1] = 1.5; P(*R, 1)[0] = -40; V(*R, 1)[1] = -.085;
    }
    const double tend = 1e5;
    S.run(tend);
    S2.run(tend);
    const double reltol = S.perf.successful_steps * 2.5e-12; // rounding errors can accumulate over the integration
    ASSERT_NEAR(S.time, S2.time, 1e-12 * 1e6);
    ASSERT_EQ(S.num_particles, 2);
    for (int p = 0; p < S.num_particles; ++p)
        for (int i = 0; i < 3; ++i) {
            EXPECT_NEAR(S2.pos[3 * p + i], S.pos[3 * p + i], std::abs(40 * reltol));
            EXPECT_NEAR(S2.vel[3 * p + i], S.vel[3 * p + i], std::abs(0.085 * reltol));
        }
}

// The full PN case with different unit systems to check that the units don't matter
TEST(Integration_units, binary_system_full_pn_spin_generic_units) {
    mstar::System S(2, 0), S2(2, 0);
    unit_system(S, mstar::PN_ALL);
    equal_mass_binary(S, .5, 1.5);
    double tend = 1e6;
    S.run(tend);
    const double c = 12354, G = 0.0634;
    const double unit_len = G / (c * c), unit_time = G / (c * c * c), unit_vel = c, unit_spin = G / c;
    unit_system(S2, mstar::PN_ALL);
    S2.constants.c = c;
    S2.constants.G = G;
    S2.mass[0] = .5; P(S2, 0)[0] = 40 * unit_len; V(S2, 0)[1] = .085 * unit_vel; SP(S2, 0)[0] = 1 * unit_spin;
    S2.mass[1] = 1.5; P(S2, 1)[0] = -40 * unit_len; V(S2, 1)[1] = -.085 * unit_vel; SP(S2, 1)[1] = 1 * unit_spin;
    tend = 1e6 * unit_time;
    S2.run(tend);
    const double reltol = 200 * S.perf.successful_steps * 1e-12;
    ASSERT_NEAR(S.time, S2.time / unit_time, 1e-12 * 1e6);
    ASSERT_EQ(S.num_particles, 2);
    for (int p = 0; p < S.num_particles; ++p)
        for (int i = 0; i < 3; ++i) {
            EXPECT_NEAR(S2.pos[3 * p + i] / unit_len, S.pos[3 * p + i], std::abs(S.pos[3 * p + i] * reltol));
            EXPECT_NEAR(S2.vel[3 * p + i] / unit_vel, S.vel[3 * p + i], std::abs(S.vel[3 * p + i] * reltol));
            EXPECT_NEAR(S2.spin[3 * p + i] / unit_spin, S.spin[3 * p + i], std::abs(S.spin[3 * p + i] * reltol));
        }
}

// Some simple tests where the BHs merge after integration
TEST(Integration_with_mergers, head_on_merger) {
    mstar::System S(2, 0);
    unit_system(S, mstar::PN_ALL);
    S.mass[0] = 1; P(S, 0)[0] = 40;
    S.mass[1] = 1; P(S, 1)[0] = -40;
    const double tend = 1e6, tol = 1e-8;
    S.run(tend);
    ASSERT_EQ(S.num_pn_particles, 1);
    ASSERT_NEAR(S.time, tend, tend * S.options.output_time_relative_tolerance);
    ASSERT_LE(S.mass[0], 2.);
    ASSERT_GE(S.mass[0], 1.99);
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(P(S, 0)[i], 0., tol);
        EXPECT_NEAR(V(S, 0)[i], 0., tol);
        EXPECT_NEAR(SP(S, 0)[i], 0., tol);
    }
}

TEST(Integration_with_mergers, merger_from_orbit) {
    mstar::System S(2, 0);
    unit_system(S, mstar::PN_ALL);
    S.mass[0] = 1; P(S, 0)[0] = 40; V(S, 0)[1] = 0.05;
    S.mass[1] = 1; P(S, 1)[0] = -40;
    const double tend = 1e6;
    S.run(tend);
    ASSERT_EQ(S.num_pn_particles, 1);
    ASSERT_NEAR(S.time, tend, tend * S.options.output_time_relative_tolerance);
    // no point in checking the exact value, which is determined by the GR fits: just that the result is sensible
    ASSERT_LT(S.mass[0], 2.01);
    ASSERT_GT(S.mass[0], 1.8);
}

namespace {
// ids[extra_index[k]] is the caller datum now in slot k (KETJU permutes particle_extra_data the same way)
std::vector<int> permuted(const mstar::System &S, const std::vector<int> &ids) {
    std::vector<int> out(S.num_particles);
    for (int k = 0; k < S.num_particles; ++k) out[k] = ids[S.extra_index[k]];
    return out;
}
// KETJU's merger_infos[k].extra_index1/2 are positions in the permuted extra data; here MergerInfo::extra_index1/2 are
// the callers' own indices, whose position in the permuted array is the slot with System::extra_index[slot] == index
int slot_of(const mstar::System &S, int caller_index) {
    for (int k = 0; k < (int)S.extra_index.size(); ++k)
        if (S.extra_index[k] == caller_index) return k;
    return -1;
}
void merger_setup(mstar::System &S) {
    unit_system(S, mstar::PN_ALL);
    S.options.bh_merger_distance_in_Rs = 6;
}
} // namespace

TEST(Instant_mergers, binary_merger_with_extra_data) {
    mstar::System S(2, 1);
    const std::vector<int> ids = {1, 2, 3};
    merger_setup(S);
    S.mass[0] = 1; P(S, 0)[0] = 4; V(S, 0)[1] = 0.05;
    S.mass[1] = 1; P(S, 1)[0] = -4;
    S.mass[2] = 1e-3; P(S, 2)[0] = 10000;
    const double tend = 1;
    S.run(tend);
    ASSERT_EQ(S.num_pn_particles, 1);
    ASSERT_EQ(S.num_particles, 2);
    ASSERT_NEAR(S.time, tend, tend * S.options.output_time_relative_tolerance);
    ASSERT_EQ((int)S.mergers.size(), 1);
    // KETJU: extra_data = {1, 3, 2}, the merged-away particle at the end
    ASSERT_EQ(S.extra_index[0], 0);
    ASSERT_EQ(S.extra_index[1], 2);
    ASSERT_EQ(S.extra_index[2], 1);
    const auto d = permuted(S, ids);
    ASSERT_EQ(d[0], 1);
    ASSERT_EQ(d[1], 3);
    ASSERT_EQ(slot_of(S, S.mergers[0].extra_index1), 0);
    ASSERT_EQ(slot_of(S, S.mergers[0].extra_index2), 2);
    ASSERT_EQ(S.mergers[0].m1, 1);
    ASSERT_EQ(S.mergers[0].m2, 1);
    ASSERT_EQ(S.mergers[0].m_remnant, S.mass[0]);
}

// A bunch of different merger orders to test the indexing of the extra_data after mergers.
TEST(Instant_mergers, multiple_merger_with_extra_data1) {
    mstar::System S(3, 1);
    merger_setup(S);
    S.mass[0] = 1; P(S, 0)[0] = 4;
    // positions so that on the first step IDs 1 and 3 merge, then the remaining pair on the next step
    S.mass[1] = 1; P(S, 1)[1] = -24;
    S.mass[2] = 1; P(S, 2)[0] = -4;
    S.mass[3] = 1e-3; P(S, 3)[0] = 10000;
    const double tend = 1;
    S.run(tend);
    ASSERT_EQ(S.num_pn_particles, 1);
    ASSERT_EQ(S.num_particles, 2);
    ASSERT_NEAR(S.time, tend, tend * S.options.output_time_relative_tolerance);
    ASSERT_EQ((int)S.mergers.size(), 2);
    // KETJU: extra_data = {1, 4, 2, 3} (IDs 1..4)
    ASSERT_EQ(S.extra_index[0], 0);
    ASSERT_EQ(S.extra_index[1], 3);
    ASSERT_EQ(S.extra_index[2], 1);
    ASSERT_EQ(S.extra_index[3], 2);
    ASSERT_EQ(slot_of(S, S.mergers[0].extra_index1), 0);
    ASSERT_EQ(slot_of(S, S.mergers[0].extra_index2), 3);
    ASSERT_EQ(S.mergers[0].m1, 1);
    ASSERT_EQ(S.mergers[0].m2, 1);
    ASSERT_EQ(slot_of(S, S.mergers[1].extra_index1), 0);
    ASSERT_EQ(slot_of(S, S.mergers[1].extra_index2), 2);
    ASSERT_EQ(S.mergers[1].m_remnant, S.mass[0]);
}

TEST(Instant_mergers, multiple_merger_with_extra_data2) {
    mstar::System S(3, 1);
    merger_setup(S);
    S.mass[0] = 1; P(S, 0)[0] = 4;
    // positions so that on the first step IDs 2 and 3 merge, then the remaining pair on the next step
    S.mass[1] = 0.9; P(S, 1)[0] = -22;
    S.mass[2] = 1; P(S, 2)[0] = -23;
    S.mass[3] = 1e-3; P(S, 3)[0] = 10000;
    const double tend = 1;
    S.run(tend);
    ASSERT_EQ(S.num_pn_particles, 1);
    ASSERT_EQ(S.num_particles, 2);
    ASSERT_NEAR(S.time, tend, tend * S.options.output_time_relative_tolerance);
    ASSERT_EQ((int)S.mergers.size(), 2);
    ASSERT_EQ(S.extra_index[0], 0);
    ASSERT_EQ(S.extra_index[1], 3);
    ASSERT_EQ(S.extra_index[2], 1);
    ASSERT_EQ(S.extra_index[3], 2);
    ASSERT_EQ(slot_of(S, S.mergers[0].extra_index1), 2);
    ASSERT_EQ(slot_of(S, S.mergers[0].extra_index2), 3);
    ASSERT_EQ(S.mergers[0].m1, 0.9);
    ASSERT_EQ(S.mergers[0].m2, 1);
    ASSERT_EQ(slot_of(S, S.mergers[1].extra_index1), 0);
    ASSERT_EQ(slot_of(S, S.mergers[1].extra_index2), 2);
    ASSERT_EQ(S.mergers[1].m_remnant, S.mass[0]);
}

TEST(Instant_mergers, multiple_merger_with_extra_data3) {
    mstar::System S(4, 1);
    merger_setup(S);
    S.mass[0] = 1; P(S, 0)[0] = 10;
    // positions so that on the first step IDs 3 and 4 merge, then 1 and 2 on the next step
    S.mass[1] = 0.9; P(S, 1)[0] = 10 + (12 + 1e-12) * 1.9;
    S.mass[2] = 1; P(S, 2)[0] = -100;
    S.mass[3] = 2; P(S, 3)[0] = -105;
    S.mass[4] = 1e-3; P(S, 4)[0] = 10000;
    const double tend = 1;
    S.run(tend);
    ASSERT_EQ(S.num_pn_particles, 2);
    ASSERT_EQ(S.num_particles, 3);
    ASSERT_NEAR(S.time, tend, tend * S.options.output_time_relative_tolerance);
    ASSERT_EQ((int)S.mergers.size(), 2);
    // KETJU: extra_data = {1, 3, 5, 2, 4}
    ASSERT_EQ(S.extra_index[0], 0);
    ASSERT_EQ(S.extra_index[1], 2);
    ASSERT_EQ(S.extra_index[2], 4);
    ASSERT_EQ(S.extra_index[3], 1);
    ASSERT_EQ(S.extra_index[4], 3);
    ASSERT_EQ(slot_of(S, S.mergers[0].extra_index1), 1);
    ASSERT_EQ(slot_of(S, S.mergers[0].extra_index2), 4);
    ASSERT_EQ(S.mergers[0].m1, 1);
    ASSERT_EQ(S.mergers[0].m2, 2);
    ASSERT_EQ(S.mergers[0].m_remnant, S.mass[1]);
    ASSERT_EQ(slot_of(S, S.mergers[1].extra_index1), 0);
    ASSERT_EQ(slot_of(S, S.mergers[1].extra_index2), 3);
    ASSERT_EQ(S.mergers[1].m_remnant, S.mass[0]);
    ASSERT_EQ(S.mergers[1].m1, 1);
    ASSERT_EQ(S.mergers[1].m2, 0.9);
}

// KETJU's MST construction test checks the tree structure of the C library (not exposed here). Here: random stars
// in [-1, 1]^3, a short run with the host tree and with the team (device) tree built on every step; both must
// conserve the energy and agree on the final state (the tree changes the chain, so only to the tolerance).
namespace {
void mst_run(int N, bool device_tree, std::vector<double> &pos, mstar::PerformanceCounters &perf, std::mt19937 &gen) {
    mstar::System S(0, N);
    S.constants.G = 1;
    S.options.PN_flags = mstar::PN_NONE;
    S.options.gbs_relative_tolerance = 1e-10;
    S.options.device_tree_everywhere = device_tree;
    if (device_tree) S.options.device_tree_min_n = 0;
    std::uniform_real_distribution<> dist(-1, 1);
    for (int i = 0; i < N; ++i) {
        S.mass[i] = 1. / N;
        for (int k = 0; k < 3; ++k) S.pos[3 * i + k] = dist(gen);
    }
    S.run(1e-3);
    ASSERT_EQ(S.num_particles, N);
    pos = S.pos;
    perf = S.perf;
}
} // namespace

TEST(MST, random_stars_host_and_device_tree) {
    std::mt19937 gen(0x5eed);
    for (int N : {2, 4, 10, 20, 50, 100, 500, 1000}) {
        std::vector<double> p_host, p_dev;
        mstar::PerformanceCounters c_host, c_dev;
        const auto state = gen;   // the same stars for both trees
        mst_run(N, false, p_host, c_host, gen);
        gen = state;
        mst_run(N, true, p_dev, c_dev, gen);
        double mx = 0;
        for (size_t q = 0; q < p_host.size(); ++q) mx = std::fmax(mx, std::fabs(p_host[q] - p_dev[q]));
        std::printf("  MST N=%d: dE/E host %.2e device %.2e, max |dx| %.2e, steps %ld / %ld\n", N, c_host.relative_energy_error,
                    c_dev.relative_energy_error, mx, c_host.successful_steps, c_dev.successful_steps);
        EXPECT_GT(c_host.successful_steps, 0);
        EXPECT_LT(std::abs(c_host.relative_energy_error), 1e-8);
        EXPECT_LT(std::abs(c_dev.relative_energy_error), 1e-8);
        EXPECT_LT(mx, 1e-6);
    }
}
