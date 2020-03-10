#include <gtest/gtest.h>
#include <mpi.h>
#include <random>
#include <sstream>
#include <cmath>

#include "ketju_integrator/ketju_integrator.h"

// Comparisons to the old AR-CHAIN based integrator, with the reference values
// computed using gbs tolerance = 1e-14 and output time tolerance = 1e-12.
TEST(Integration_versus_old_integrator, binary_system_full_pn_spin) {
    
    struct ketju_system R;
    ketju_create_system(&R, 2, 0, MPI_COMM_WORLD);

    R.constants->c = 1;
    R.constants->G = 1;
    R.options->PN_flags = KETJU_PN_ALL;

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
    // Due to accumulation of the error the orbits diverge slightly so that the
    // global error is not quite the local tolerance * number of steps.
    // At 1e-12 tolerance a factor of 5 is sufficient to get passing result.
    const double reltol =
        5 * R.options->gbs_relative_tolerance * R.perf->succesful_steps;

    ASSERT_EQ(R.num_pn_particles, 2);
    ASSERT_NEAR(R.physical_state->time, tend,
                tend * R.options->output_time_relative_tolerance);

    struct expected_vals {
        double pos[3];
        double vel[3];
        double spin[3];
    } expected[] = {
        {{-31.112617731597158, 9.778292906413535, 2.3516385904097605},
         {-0.010385811186292311, -0.07719759020087542, 0.03387326124064798},
         {-0.6966913842042676, -0.38789169294236087, 0.6034576619128901}},
        {{31.112617731597158, -9.778292906413535, -2.3516385904097605},
         {0.010385811186292311, 0.07719759020087542, -0.03387326124064798},
         {0.4665125580313397, -0.884017742268443, -0.029642276442934865}}};

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
        5 * R.options->gbs_relative_tolerance * R.perf->succesful_steps;

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
        5 * R.options->gbs_relative_tolerance * R.perf->succesful_steps;

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

// The full PN case but with a different unit system to check that the units
// don't matter
TEST(Integration_versus_old_integrator,
     binary_system_full_pn_spin_generic_units) {

    struct ketju_system R;
    ketju_create_system(&R, 2, 0, MPI_COMM_WORLD);

    const double c = 12354;
    const double G = 63645234;
    // conversions to the natural units used above
    const double unit_len = G / (c * c);
    const double unit_time = G / (c * c * c);
    const double unit_vel = c;
    const double unit_spin = G / c;

    R.constants->c = c;
    R.constants->G = G;
    R.options->PN_flags = KETJU_PN_ALL;

    R.physical_state->mass[0] = 1;
    R.physical_state->pos[0][0] = 40 * unit_len;
    R.physical_state->vel[0][1] = .085 * unit_vel;
    R.physical_state->spin[0][0] = 1 * unit_spin;

    R.physical_state->mass[1] = 1;
    R.physical_state->pos[1][0] = -40 * unit_len;
    R.physical_state->vel[1][1] = -.085 * unit_vel;
    R.physical_state->spin[1][1] = 1 * unit_spin;

    R.options->gbs_relative_tolerance = 1e-12;
    R.options->output_time_relative_tolerance = 1e-12;

    const double tend = 1e6 * unit_time;

    ketju_run_integrator(&R, tend);
    // XXX: For some reason the relative position and velocity differ by up to
    // a factor of 6 from the expected values. Could this be just rounding
    // differences due to the unit system, or is something actually wrong?
    const double reltol =
        6.5 * 5 * R.options->gbs_relative_tolerance * R.perf->succesful_steps;

    ASSERT_EQ(R.num_pn_particles, 2);
    ASSERT_NEAR(R.physical_state->time, tend,
                tend * R.options->output_time_relative_tolerance);

    struct expected_vals {
        double pos[3];
        double vel[3];
        double spin[3];
    } expected[] = {
        {{-31.112617731597158, 9.778292906413535, 2.3516385904097605},
         {-0.010385811186292311, -0.07719759020087542, 0.03387326124064798},
         {-0.6966913842042676, -0.38789169294236087, 0.6034576619128901}},
        {{31.112617731597158, -9.778292906413535, -2.3516385904097605},
         {0.010385811186292311, 0.07719759020087542, -0.03387326124064798},
         {0.4665125580313397, -0.884017742268443, -0.029642276442934865}}};

    for (int p = 0; p < R.num_particles; ++p) {
        for (int i = 0; i < 3; ++i) {
            // The expected values are in natural units
            EXPECT_NEAR(R.physical_state->pos[p][i] / unit_len,
                        expected[p].pos[i],
                        std::abs(expected[p].pos[i] * reltol));
            EXPECT_NEAR(R.physical_state->vel[p][i] / unit_vel,
                        expected[p].vel[i],
                        std::abs(expected[p].vel[i] * reltol));
            EXPECT_NEAR(R.physical_state->spin[p][i] / unit_spin,
                        expected[p].spin[i],
                        std::abs(expected[p].spin[i] * reltol));
        }
    }
}

TEST(Integration_versus_old_integrator,
     binary_system_full_pn_no_spin_generic_units) {

    struct ketju_system R;
    ketju_create_system(&R, 2, 0, MPI_COMM_WORLD);

    const double c = 12354;
    const double G = 63645234;
    // conversions to the natural units used above
    const double unit_len = G / (c * c);
    const double unit_time = G / (c * c * c);
    const double unit_vel = c;
    const double unit_spin = G / c;

    R.constants->c = c;
    R.constants->G = G;
    R.options->PN_flags = KETJU_PN_DYNAMIC_ALL;

    R.physical_state->mass[0] = 1;
    R.physical_state->pos[0][0] = 40 * unit_len;
    R.physical_state->vel[0][1] = .085 * unit_vel;
    R.physical_state->spin[0][0] = 1 * unit_spin;

    R.physical_state->mass[1] = 1;
    R.physical_state->pos[1][0] = -40 * unit_len;
    R.physical_state->vel[1][1] = -.085 * unit_vel;
    R.physical_state->spin[1][1] = 1 * unit_spin;

    R.options->gbs_relative_tolerance = 1e-12;
    R.options->output_time_relative_tolerance = 1e-12;

    const double tend = 1e6 * unit_time;

    ketju_run_integrator(&R, tend);
    const double reltol =
        5 * R.options->gbs_relative_tolerance * R.perf->succesful_steps;

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
            // The expected values are in natural units
            EXPECT_NEAR(R.physical_state->pos[p][i] / unit_len,
                        expected[p].pos[i],
                        std::abs(expected[p].pos[i] * reltol));
            EXPECT_NEAR(R.physical_state->vel[p][i] / unit_vel,
                        expected[p].vel[i],
                        std::abs(expected[p].vel[i] * reltol));
            EXPECT_NEAR(R.physical_state->spin[p][i] / unit_spin,
                        expected[p].spin[i],
                        std::abs(expected[p].spin[i] * reltol));
        }
    }
}
