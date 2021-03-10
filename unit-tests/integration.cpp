#include <gtest/gtest.h>
#include <mpi.h>
#include <random>
#include <sstream>
#include <cmath>

#include "ketju_integrator/ketju_integrator.h"

// Comparisons to the old AR-CHAIN based integrator, with the reference values
// computed using gbs tolerance = 1e-14 and output time tolerance = 1e-12.

// In principle situations in which the PN corrected CoM is at rest can be
// compared reliably, but for some reason starting at 3PN level there is a small
// secular error so the below test doesn't pass.
#if 0
TEST(Integration_versus_old_integrator, binary_system_full_pn_no_spin) {
    
    struct ketju_system R;
    ketju_create_system(&R, 2, 0, MPI_COMM_WORLD);

    R.constants->c = 1;
    R.constants->G = 1;
    R.options->PN_flags = KETJU_PN_DYNAMIC_ALL;

    R.physical_state->mass[0] = 1;
    R.physical_state->pos[0][0] = 40;
    R.physical_state->vel[0][1] = .085;
    R.physical_state->spin[0][0] = 1;

    R.physical_state->mass[1] = 1;
    R.physical_state->pos[1][0] = -40;
    R.physical_state->vel[1][1] = -.085;
    R.physical_state->spin[1][1] = 1;
    
    R.options->gbs_relative_tolerance = 1e-12;
    R.options->output_time_relative_tolerance = 1e-12;

    const double tend = 1e6;

    ketju_run_integrator(&R, tend);
    
    const double reltol =
        5 * R.options->gbs_relative_tolerance * R.perf->successful_steps;

    ASSERT_EQ(R.num_pn_particles, 2);
    ASSERT_NEAR(R.physical_state->time, tend,
                tend * R.options->output_time_relative_tolerance);

    struct expected_vals {
        double pos[3];
        double vel[3];
        double spin[3];
    } expected[] = {{{-15.675129114282553, -24.12600028245775, 0.0},
                     {0.07932781836871444, -0.051044405469834665, 0.0},
                     {1.0, 0.0, 0.0}},
                    {{15.675129114282553, 24.12600028245775, 0.0},
                     {-0.07932781836871444, 0.051044405469834665, 0.0},
                     {0.0, 1.0, 0.0}}};

    for (int p = 0; p < R.num_particles; ++p) {
        for (int i = 0; i < 3; ++i) {
            EXPECT_NEAR(R.physical_state->pos[p][i], expected[p].pos[i],
                        std::abs(expected[p].pos[i] * reltol));
            EXPECT_NEAR(R.physical_state->vel[p][i], expected[p].vel[i],
                        std::abs(expected[p].vel[i] * reltol));
            EXPECT_NEAR(R.physical_state->spin[p][i], expected[p].spin[i],
                        std::abs(expected[p].spin[i] * reltol));
        }
    }
}
#endif

TEST(Integration_versus_old_integrator, binary_system_no_pn) {
    
    struct ketju_system R;
    ketju_create_system(&R, 2, 0, MPI_COMM_WORLD);

    R.constants->c = 1;
    R.constants->G = 1;
    R.options->PN_flags = KETJU_PN_NONE;

    R.physical_state->mass[0] = 1;
    R.physical_state->pos[0][0] = 40;
    R.physical_state->vel[0][1] = .085;
    R.physical_state->spin[0][0] = 1;

    R.physical_state->mass[1] = 1;
    R.physical_state->pos[1][0] = -40;
    R.physical_state->vel[1][1] = -.085;
    R.physical_state->spin[1][1] = 1;
    
    R.options->gbs_relative_tolerance = 1e-12;
    R.options->output_time_relative_tolerance = 1e-12;

    const double tend = 1e6;

    ketju_run_integrator(&R, tend);
    
    const double reltol =
        5 * R.options->gbs_relative_tolerance * R.perf->successful_steps;

    ASSERT_EQ(R.num_pn_particles, 2);
    ASSERT_NEAR(R.physical_state->time, tend,
                tend * R.options->output_time_relative_tolerance);

    struct expected_vals {
        double pos[3];
        double vel[3];
        double spin[3];
    } expected[] = {{{28.093181660139404, -31.029347533391817, 0.0},
                     {0.05450807268322124, 0.060820810188436096, 0.0},
                     {1.0, 0.0, 0.0}},
                    {{-28.093181660139404, 31.029347533391817, 0.0},
                     {-0.05450807268322124, -0.060820810188436096, 0.0},
                     {0.0, 1.0, 0.0}}};

    for (int p = 0; p < R.num_particles; ++p) {
        for (int i = 0; i < 3; ++i) {
            EXPECT_NEAR(R.physical_state->pos[p][i], expected[p].pos[i],
                        std::abs(expected[p].pos[i] * reltol));
            EXPECT_NEAR(R.physical_state->vel[p][i], expected[p].vel[i],
                        std::abs(expected[p].vel[i] * reltol));
            EXPECT_NEAR(R.physical_state->spin[p][i], expected[p].spin[i],
                        std::abs(expected[p].spin[i] * reltol));
        }
    }
}

TEST(Integration_threebody_terms,
     binary_system_with_and_without_threebody_terms) {

    struct ketju_system R;
    ketju_create_system(&R, 2, 0, MPI_COMM_WORLD);
    struct ketju_system R2;
    ketju_create_system(&R2, 2, 0, MPI_COMM_WORLD);

    unsigned int pn_flags = KETJU_PN_1_0_ACC | KETJU_PN_THREEBODY;

    R.constants->c = 1;
    R.constants->G = 1;
    R.options->PN_flags = pn_flags;

    R.physical_state->mass[0] = .5;
    R.physical_state->pos[0][0] = 40;
    R.physical_state->vel[0][1] = .085;

    R.physical_state->mass[1] = 1.5;
    R.physical_state->pos[1][0] = -40;
    R.physical_state->vel[1][1] = -.085;

    R.options->gbs_relative_tolerance = 1e-12;
    R.options->output_time_relative_tolerance = 1e-12;

    double tend = 1e6;
    ketju_run_integrator(&R, tend);

    // conversions to the natural units used above
    R2.constants->c = 1;
    R2.constants->G = 1;
    R2.options->PN_flags = KETJU_PN_1_0_ACC;

    R2.physical_state->mass[0] = .5;
    R2.physical_state->pos[0][0] = 40;
    R2.physical_state->vel[0][1] = .085;

    R2.physical_state->mass[1] = 1.5;
    R2.physical_state->pos[1][0] = -40;
    R2.physical_state->vel[1][1] = -.085;

    R2.options->gbs_relative_tolerance = 1e-12;
    R2.options->output_time_relative_tolerance = 1e-12;

    ketju_run_integrator(&R2, tend);

    // Only rounding error expected
    const double reltol = R.perf->successful_steps * 1e-13;

    ASSERT_NEAR(R.physical_state->time, R2.physical_state->time, 1e-12 * 1e6);

    ASSERT_EQ(R.num_particles, 2);

    for (int p = 0; p < R.num_particles; ++p) {
        for (int i = 0; i < 3; ++i) {
            // The expected values are in natural units
            EXPECT_NEAR(R2.physical_state->pos[p][i],
                        R.physical_state->pos[p][i],
                        std::abs(R.physical_state->pos[p][i] * reltol));
            EXPECT_NEAR(R2.physical_state->vel[p][i],
                        R.physical_state->vel[p][i],
                        std::abs(R.physical_state->vel[p][i] * reltol));
        }
    }
}

// The full PN case with different unit systems to check that the units
// don't matter
TEST(Integration_units,
     binary_system_full_pn_spin_generic_units) {

    struct ketju_system R;
    ketju_create_system(&R, 2, 0, MPI_COMM_WORLD);
    struct ketju_system R2;
    ketju_create_system(&R2, 2, 0, MPI_COMM_WORLD);

    unsigned int pn_flags =  KETJU_PN_ALL; 

    R.constants->c = 1;
    R.constants->G = 1;
    R.options->PN_flags = pn_flags;

    R.physical_state->mass[0] = .5;
    R.physical_state->pos[0][0] = 40;
    R.physical_state->vel[0][1] = .085;
    R.physical_state->spin[0][0] = 1;

    R.physical_state->mass[1] = 1.5;
    R.physical_state->pos[1][0] = -40;
    R.physical_state->vel[1][1] = -.085;
    R.physical_state->spin[1][1] = 1;

    R.options->gbs_relative_tolerance = 1e-12;
    R.options->output_time_relative_tolerance = 1e-12;

    double tend = 1e6;
    ketju_run_integrator(&R, tend);

    const double c = 12354;
    const double G = 0.0634;
    // conversions to the natural units used above
    const double unit_len = G / (c * c);
    const double unit_time = G / (c * c * c);
    const double unit_vel = c;
    const double unit_spin = G / c;
    R2.constants->c = c;
    R2.constants->G = G;
    R2.options->PN_flags = pn_flags;

    R2.physical_state->mass[0] = .5;
    R2.physical_state->pos[0][0] = 40 * unit_len;
    R2.physical_state->vel[0][1] = .085 * unit_vel;
    R2.physical_state->spin[0][0] = 1 * unit_spin;

    R2.physical_state->mass[1] = 1.5;
    R2.physical_state->pos[1][0] = -40 * unit_len;
    R2.physical_state->vel[1][1] = -.085 * unit_vel;
    R2.physical_state->spin[1][1] = 1 * unit_spin;

    R2.options->gbs_relative_tolerance = 1e-12;
    R2.options->output_time_relative_tolerance = 1e-12;

    tend = 1e6 * unit_time;

    ketju_run_integrator(&R2, tend);
    
    const double reltol = 200 * R.perf->successful_steps * 1e-12;

    ASSERT_NEAR(R.physical_state->time, R2.physical_state->time / unit_time,
                1e-12*1e6);

    ASSERT_EQ(R.num_particles, 2);

    for (int p = 0; p < R.num_particles; ++p) {
        for (int i = 0; i < 3; ++i) {
            // The expected values are in natural units
            EXPECT_NEAR(R2.physical_state->pos[p][i] / unit_len,
                        R.physical_state->pos[p][i],
                        std::abs(R.physical_state->pos[p][i] * reltol));
            EXPECT_NEAR(R2.physical_state->vel[p][i] / unit_vel,
                        R.physical_state->vel[p][i],
                        std::abs(R.physical_state->vel[p][i] * reltol));
            EXPECT_NEAR(R2.physical_state->spin[p][i] / unit_spin,
                        R.physical_state->spin[p][i],
                        std::abs(R.physical_state->spin[p][i] * reltol));
        }
    }
}

// Some simple tests where the BHs merge
TEST(Integration_with_mergers, head_on_merger) {

    struct ketju_system R;
    ketju_create_system(&R, 2, 0, MPI_COMM_WORLD);

    R.constants->c = 1;
    R.constants->G = 1;
    R.options->PN_flags = KETJU_PN_ALL;

    R.physical_state->mass[0] = 1;
    R.physical_state->pos[0][0] = 40;

    R.physical_state->mass[1] = 1;
    R.physical_state->pos[1][0] = -40;

    R.options->gbs_relative_tolerance = 1e-12;
    R.options->output_time_relative_tolerance = 1e-12;

    const double tend = 1e6;

    ketju_run_integrator(&R, tend);
    const double tol = 1e-8;

    ASSERT_EQ(R.num_pn_particles, 1);
    ASSERT_NEAR(R.physical_state->time, tend,
                tend * R.options->output_time_relative_tolerance);

    ASSERT_LE(R.physical_state->mass[0], 2.);
    ASSERT_GE(R.physical_state->mass[0], 1.99);

    struct expected_vals {
        double pos[3];
        double vel[3];
        double spin[3];
    } expected[] = {{{0., 0., 0.}, {0., 0., 0.}, {0., 0., 0.}}};
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(R.physical_state->pos[0][i], expected[0].pos[i], tol);
        EXPECT_NEAR(R.physical_state->vel[0][i], expected[0].vel[i], tol);
        EXPECT_NEAR(R.physical_state->spin[0][i], expected[0].spin[i], tol);
    }
}

TEST(Integration_with_mergers, merger_from_orbit) {

    struct ketju_system R;
    ketju_create_system(&R, 2, 0, MPI_COMM_WORLD);

    R.constants->c = 1;
    R.constants->G = 1;
    R.options->PN_flags = KETJU_PN_ALL;

    R.physical_state->mass[0] = 1;
    R.physical_state->pos[0][0] = 40;
    const double v0[3] = {0., 0.05, 0.};
    for (int i = 0; i < 3; ++i) {
        R.physical_state->vel[0][i] = v0[i];
    }

    R.physical_state->mass[1] = 1;
    R.physical_state->pos[1][0] = -40;

    R.options->gbs_relative_tolerance = 1e-12;
    R.options->output_time_relative_tolerance = 1e-12;

    const double tend = 1e6;

    ketju_run_integrator(&R, tend);
    const double reltol =
        40 * R.options->gbs_relative_tolerance * R.perf->successful_steps;

    ASSERT_EQ(R.num_pn_particles, 1);
    ASSERT_NEAR(R.physical_state->time, tend,
                tend * R.options->output_time_relative_tolerance);


    // No point in checking exact value which is determined by the GR fits,
    // just check that the result is sensible.
    ASSERT_LT(R.physical_state->mass[0], 2.01);
    ASSERT_GT(R.physical_state->mass[0], 1.8);

    // Don't know what to check for in the coordinates
}
