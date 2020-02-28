#include <gtest/gtest.h>
#include <mpi.h>
#include <random>

#include "ketju_integrator/internal.h"
#include "ketju_integrator/ketju_integrator.h"

// Since this is C++, the struct keyword before ketju_system is not necessary,
// but feels nicer to have it for consistency with the rest of the codebase.

template <typename RNG>
static void init_random_positions(struct ketju_system *R, int N, RNG &&gen) {
    // init N random particle positions. masses etc don't matter since we're
    // only interested in the tree construction here.
    ketju_create_system(R, 0, N, MPI_COMM_WORLD);
    std::uniform_real_distribution<> dist(-1, 1);
    for (int i = 0; i < N; ++i) {
        R->physical_state->mass[i] = 1; // need to be non-zero
        for (int k = 0; k < 3; ++k) {
            R->physical_state->pos[i][k] = dist(gen);
        }
    }
    ketju_into_CoM_frame(R);
}

static bool connected_to_root(struct ketju_system *R, int i) {
    auto *mst = R->internal_state->mst;
    int n = 0;
    while (mst->vertex[i].level > 0 && n < R->num_particles) {
        i = mst->vertex[i].parent;
        ++n;
        if (i < 0 || i >= R->num_particles) return false;
    }
    return mst->vertex[i].level == 0;
}

TEST(mst_construction, random_prim) {
    struct ketju_system R;
    std::mt19937 gen(0x5eed);
    // TODO turn into a parametrized test with different particle counts
    int N = 100;
    init_random_positions(&R, N, gen);
    R.options->mst_algorithm = KETJU_MST_PRIM;
    ketju_build_mst(&R);

    auto *mst = R.internal_state->mst;
    for (int i = 0; i < N; ++i) {
        ASSERT_GE(mst->vertex[i].level, 0);
        ASSERT_LT(mst->vertex[i].level, N);
        ASSERT_TRUE(connected_to_root(&R, i));
    }

    // Check edges
    for (int i = 0; i < N - 1; ++i) {
        auto &edge = mst->edge_in_mst[i];
        ASSERT_NE(edge.vertex1, edge.vertex2);
        ASSERT_GE(edge.vertex1, 0);
        ASSERT_LT(edge.vertex1, N);
        ASSERT_GE(edge.vertex2, 0);
        ASSERT_LT(edge.vertex2, N);
        if (mst->vertex[edge.vertex1].level < mst->vertex[edge.vertex2].level) {
            ASSERT_EQ(mst->vertex[edge.vertex2].parent, edge.vertex1);
        } else {
            ASSERT_EQ(mst->vertex[edge.vertex1].parent, edge.vertex2);
        }
    }
}
