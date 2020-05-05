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
    // Due to accumulation of the error the orbits diverge so that the
    // global error is not quite the local tolerance * number of steps.
    // Inclusion of spin makes the problem worse due to the chaotic (?) dynamics
    // introduced.
    // The passing tolerance is still good enough to show that the integrator
    // works as expected.
    const double reltol =
        40 * R.options->gbs_relative_tolerance * R.perf->successful_steps;

    ASSERT_EQ(R.num_pn_particles, 2);
    ASSERT_NEAR(R.physical_state->time, tend,
                tend * R.options->output_time_relative_tolerance);

    for (int i = 0; i < 3; ++i) {
        ASSERT_NEAR(R.CoM_pos[i], 0, 1e-10);
        ASSERT_NEAR(R.CoM_vel[i], 0, 1e-10);
    }

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


TEST(Integration_versus_old_integrator, binary_system_full_pn_spin_backwards_in_time) {

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

    const double tend = -1e6;

    ketju_run_integrator(&R, tend);
    // Due to accumulation of the error the orbits diverge so that the
    // global error is not quite the local tolerance * number of steps.
    // Inclusion of spin makes the problem worse due to the chaotic (?) dynamics
    // introduced.
    // The passing tolerance is still good enough to show that the integrator
    // works as expected.
    const double reltol =
        40 * R.options->gbs_relative_tolerance * R.perf->successful_steps;

    ASSERT_EQ(R.num_pn_particles, 2);
    ASSERT_NEAR(R.physical_state->time, tend,
                std::abs(tend * R.options->output_time_relative_tolerance));

    for (int i = 0; i < 3; ++i) {
        ASSERT_NEAR(R.CoM_pos[i], 0, 1e-10);
        ASSERT_NEAR(R.CoM_vel[i], 0, 1e-10);
    }

    struct expected_vals {
        double pos[3];
        double vel[3];
        double spin[3];
    } expected[] = {
        {{57.465936676606205, -15.129432263530019, -15.590635820831748},
         {0.032766037751664645, 0.05443487932345067, -0.008614496491778749},
         {-0.9888028454741536, 0.12993116006745103, 0.07339500273075822}},
        {{-57.465936676606205, 15.129432263530019, 15.590635820831748},
         {-0.032766037751664645, -0.05443487932345067, 0.008614496491778749},
         {0.1402556819564971, 0.9771458861410306, 0.1597318405220674}}};

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
        5 * R.options->gbs_relative_tolerance * R.perf->successful_steps;

    ASSERT_EQ(R.num_pn_particles, 2);
    ASSERT_NEAR(R.physical_state->time, tend,
                tend * R.options->output_time_relative_tolerance);
    for (int i = 0; i < 3; ++i) {
        ASSERT_NEAR(R.CoM_pos[i], 0, 1e-10);
        ASSERT_NEAR(R.CoM_vel[i], 0, 1e-10);
    }

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
        5 * R.options->gbs_relative_tolerance * R.perf->successful_steps;

    ASSERT_EQ(R.num_pn_particles, 2);
    ASSERT_NEAR(R.physical_state->time, tend,
                tend * R.options->output_time_relative_tolerance);
    for (int i = 0; i < 3; ++i) {
        ASSERT_NEAR(R.CoM_pos[i], 0, 1e-10);
        ASSERT_NEAR(R.CoM_vel[i], 0, 1e-10);
    }

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
    // XXX: For some reason the relative position and velocity have larger
    // errors here than in the natural unit test. Could this be just rounding
    // differences due to the unit system, or is something actually wrong?
    const double reltol =
        50 * R.options->gbs_relative_tolerance * R.perf->successful_steps;

    ASSERT_EQ(R.num_pn_particles, 2);
    ASSERT_NEAR(R.physical_state->time, tend,
                tend * R.options->output_time_relative_tolerance);
    for (int i = 0; i < 3; ++i) {
        ASSERT_NEAR(R.CoM_pos[i], 0, 1e-10);
        ASSERT_NEAR(R.CoM_vel[i], 0, 1e-10);
    }

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
        15 * R.options->gbs_relative_tolerance * R.perf->successful_steps;

    ASSERT_EQ(R.num_pn_particles, 2);
    ASSERT_NEAR(R.physical_state->time, tend,
                tend * R.options->output_time_relative_tolerance);
    for (int i = 0; i < 3; ++i) {
        ASSERT_NEAR(R.CoM_pos[i], 0, 1e-10);
        ASSERT_NEAR(R.CoM_vel[i], 0, 1e-10);
    }

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

TEST(Integration_versus_old_integrator, small_system_full_pn_spin) {

    struct ketju_system R;
    ketju_create_system(&R, 2, 5, MPI_COMM_WORLD);

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

    double stars_pos[][3] = {
        {-48.326033420447295, -1221.7898082077652, -671.9631822729768},
        {629.0495108004391, 442.86538951248383, 1497.072826450301},
        {-291.59188391692373, 529.0324019146867, -212.38493398328478},
        {-1546.3996176783937, -880.396174204024, -43.77073976798657},
        {660.8791030303855, -1416.3188502846733, 623.7188904068939}};
    double stars_vel[][3] = {
        {0.00198941674187, -0.000493671996258, -0.033106928825273},
        {-0.027720855975409, 0.001158866493797, 0.003267364712021},
        {-0.04364648881292, -0.003242247833245, 0.037844419217374},
        {0.012157000206118, -0.023255419566347, 0.00223477021713},
        {-0.017022248123934, -0.013880205632499, 0.011190497668332}};
    for (int i = 0; i < 5; ++i) {
        int p = 2 + i;
        R.physical_state->mass[p] = 0.1;
        for (int k = 0; k < 3; ++k) {
            R.physical_state->pos[p][k] = stars_pos[i][k];
            R.physical_state->vel[p][k] = stars_vel[i][k];
        }
    }

    R.options->gbs_relative_tolerance = 1e-12;
    R.options->output_time_relative_tolerance = 1e-12;

    const double tend = 1e5;

    ketju_run_integrator(&R, tend);
    const double reltol =
        5 * R.options->gbs_relative_tolerance * R.perf->successful_steps;

    ASSERT_EQ(R.num_pn_particles, 2);
    ASSERT_EQ(R.num_particles, 7);
    ASSERT_NEAR(R.physical_state->time, tend,
                tend * R.options->output_time_relative_tolerance);
    double ref_com_pos[3] = {-23.85555684739761, -101.86428165077166,
                             47.706914433317856};
    double ref_com_vel[3] = {-0.002969727038571, -0.001588507141382,
                             0.000857204919583};
    for (int i = 0; i < 3; ++i) {
        ASSERT_NEAR(R.CoM_pos[i], ref_com_pos[i] + ref_com_vel[i] * tend,
                    1e-12 * tend);
        ASSERT_NEAR(R.CoM_vel[i], ref_com_vel[i], 1e-10);
    }

    struct expected_vals {
        double pos[3];
        double vel[3];
        double spin[3];
    } expected[] = {
        {{-389.2718796547892, -302.6588210124032, 94.61725696925683},
         {0.013043337823897876, 0.06336040873322542, -0.015837331689557272},
         {-0.5108300510736187, 0.775933126158424, 0.3701086903196324}},
        {{-475.8194904864262, -246.91226643883323, 118.23523290771712},
         {-0.019755183479327885, -0.06361701339451332, 0.011144273185840867},
         {-0.8135061369875893, -0.5754987951619258, 0.0837191844848903}},
        {{-141.24917187430896, 523.4312556215625, -496.946058439069},
         {-0.00901419384482613, 0.0066014202150411205, 0.047724690447785725},
         {0.0, 0.0, 0.0}},
        {{835.444524286353, -393.1323417828477, 705.5356590999108},
         {-0.0029143166946387976, 0.003872061390989223, 0.03291369722417577},
         {0.0, 0.0, 0.0}},
        {{-554.4890085189111, -814.3864783233964, 596.3729813814434},
         {0.03689859414721689, -0.034468337275903496, -0.015662792624645098},
         {0.0, 0.0, 0.0}},
        {{-8.442740471611785, 523.0677485470239, 125.06547141928493},
         {-0.045095085937192254, 0.02556016328909521, -0.009946734209419165},
         {0.0, 0.0, 0.0}},
        {{498.9435803781269, -861.1442042741921, 277.13220755994143},
         {0.013000282919464087, -0.03871193954089035, 0.013331847188848403},
         {0.0, 0.0, 0.0}}};

    for (int p = 0; p < R.num_particles; ++p) {
        for (int i = 0; i < 3; ++i) {
            EXPECT_NEAR(R.physical_state->pos[p][i] + R.CoM_pos[i],
                        expected[p].pos[i],
                        std::abs(expected[p].pos[i] * reltol));
            EXPECT_NEAR(R.physical_state->vel[p][i] + R.CoM_vel[i],
                        expected[p].vel[i],
                        std::abs(expected[p].vel[i] * reltol));
            if (p < R.num_pn_particles) {
                EXPECT_NEAR(R.physical_state->spin[p][i], expected[p].spin[i],
                            std::abs(expected[p].spin[i] * reltol));
            }
        }
    }
}

