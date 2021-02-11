// Main integration and force calculation functions

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#include "ketju_integrator/internal.h"
#include "ketju_integrator/ketju_integrator.h"

// This is needed only in this file
enum velocity_type { PHYSICAL, AUXILIARY };


// Gadget softening kernels
static inline double softened_inverse_r(double r, double h) {
    if (h <= 0) return 1. / r;

    const double hinv = 1. / h;
    const double u = r * hinv;

    if (u >= 1.) {
        return 1. / r;
    } else if (u < 0.5) {
        return -hinv *
               (-2.8 + u * u * (5.333333333333 + u * u * (6.4 * u - 9.6)));
    } else {
        return -hinv * (-3.2 + 0.066666666667 / u +
                        u * u *
                            (10.666666666667 +
                             u * (-16.0 + u * (9.6 - 2.133333333333 * u))));
    }
}

static inline double softened_inverse_r3(double r, double h) {
    if (h <= 0) return 1. / (r * r * r);

    const double hinv = 1. / h;
    const double u = r * hinv;

    if (u >= 1) {
        return 1.0 / (r * r * r);
    } else if (u < 0.5) {
        return hinv * hinv * hinv *
               (10.666666666667 + u * u * (32.0 * u - 38.4));
    } else {
        return hinv * hinv * hinv *
               (21.333333333333 - 48.0 * u + 38.4 * u * u -
                10.666666666667 * u * u * u - 0.066666666667 / (u * u * u));
    }
}

// computes the force function for logh
static void compute_U(struct ketju_system *R) {

    tick(&R->perf->time_force);

    struct ketju_system_internal_state *is = R->internal_state;
    struct spanning_tree_coordinate_system *mst = is->mst;
    struct ketju_system_physical_state *ps = R->physical_state;

    is->U = 0;
    int istart = 0;
    int CumNumEdge, ThisBlock, jstart;
    // TODO: we might want the option to parallelize this over all of the tasks
    // instead of just the gbs group, since this function is mainly for
    // computing the energy between steps.
    loop_scheduling_N2(mst->num_vertices, is->task_info->gbs_group_size,
                       is->task_info->gbs_group_rank, &istart, &jstart,
                       &ThisBlock, &CumNumEdge);
    int c = 0;

    const int Nd = R->options->max_tree_distance;
    int d;
    int path[Nd];
    int sign[Nd];

    const double ss_soft = R->options->star_star_softening;

    for (int i = istart; i < mst->num_vertices; i++) {

        double mi = ps->mass[i];
        int Li = mst->vertex[i].level;
        if (i > istart) { jstart = i + 1; }
        for (int j = jstart; j < mst->num_vertices; j++) {
            if (c >= ThisBlock) {
                goto after_loops;
            }
            if (R->options->no_star_star_gravity && i >= R->num_pn_particles &&
                j >= R->num_pn_particles) {
                continue;
            }

            double mj = ps->mass[j];
            int Lj = mst->vertex[j].level;
            double r2 = 0;
            bool proximity = false;

            if (abs(Li - Lj) < Nd) {
                proximity =
                    ketju_check_relative_proximity(j, i, Nd, R, &d, path, sign);
            }

            // Lifting this tree traversal code into its own (inline) function
            // appeared to be slower, so keep it like this even though it is
            // duplicated a couple of times.
            if (proximity) {
                for (int k = 0; k < 3; k++) {
                    double dr = 0;
                    for (int l = 0; l < d; l++) {
                        int index = path[l];
                        dr += sign[l] * mst->edge_rel_pos[index][k];
                    }
                    r2 += dr * dr;
                }
            } else {
                for (int k = 0; k < 3; k++) {
                    double dr = ps->pos[j][k] - ps->pos[i][k];
                    r2 += dr * dr;
                }
            }
            double soft = i < R->num_pn_particles || j < R->num_pn_particles
                              ? 0.
                              : ss_soft;
            const double invr = softened_inverse_r(sqrt(r2), soft);

            is->U += mi * mj * invr;
            c++;
        }
    }
    after_loops:
    tock(&R->perf->time_force);

    tick(&R->perf->time_force_comm);
    if (is->task_info->gbs_group_size > 1) {
        MPI_Allreduce(MPI_IN_PLACE, &is->U, 1, MPI_DOUBLE, MPI_SUM,
                      is->task_info->gbs_comm);
    }
    tock(&R->perf->time_force_comm);

    if (!isfinite(is->U)) {
        if (R->internal_state->task_info->main_group_rank == 0) {
            ketju_debug_dump_integrator_state(
                R, "ketju_integrator_infinite_U_state_dump.bin");
        }
        terminate("U is not finite! State dumped to file.");
    }

    is->U *= R->constants->G;
}

// To support optional softening and ignoring of star-star interactions we need
// to have slightly different force loops depending on the options.
// The force loop is the most time consuming part of the integration, and any
// extra logic in the loop leads to significant performance penalties (~10-20%).
// So instead have a different function for each case.
static inline void comp_U_acc_loop_standard(struct ketju_system *R) {
    struct ketju_system_internal_state *is = R->internal_state;
    struct spanning_tree_coordinate_system *mst = is->mst;
    struct ketju_system_physical_state *ps = R->physical_state;

    const double G = R->constants->G;

    int istop = mst->num_vertices, jstop = mst->num_vertices;
    int ThisBlock, CumNumEdge, jstart, istart;
    loop_scheduling_N2(mst->num_vertices, is->task_info->gbs_group_size,
                       is->task_info->gbs_group_rank, &istart, &jstart,
                       &ThisBlock, &CumNumEdge);
    int c = 0;
    const int Nd = R->options->max_tree_distance;
    int d;
    int path[Nd];
    int sign[Nd];

    double dr[3];

    for (int i = istart; i < istop; i++) {
        const double mi = ps->mass[i];
        const int Li = mst->vertex[i].level;
        if (i > istart) jstart = i + 1;

        for (int j = jstart; j < jstop; j++) {
            if (c >= ThisBlock) {
                return;
            }
            double r2 = 0;
            const double mj = ps->mass[j];
            const int Lj = mst->vertex[j].level;

            bool proximity = false;
            if (abs(Li - Lj) <= Nd) {
                proximity =
                    ketju_check_relative_proximity(j, i, Nd, R, &d, path, sign);
            }

            if (proximity) {
                for (int k = 0; k < 3; k++) {
                    dr[k] = 0;
                    for (int l = 0; l < d; l++) {
                        dr[k] += sign[l] * mst->edge_rel_pos[path[l]][k];
                    }
                    r2 += dr[k] * dr[k];
                }
            } else {
                for (int k = 0; k < 3; k++) {
                    dr[k] = ps->pos[j][k] - ps->pos[i][k];
                    r2 += dr[k] * dr[k];
                }
            }

            const double invr = 1. / sqrt(r2);
            is->U += G * mi * mj * invr;

            const double Ginvr3 = G * invr * invr * invr;

            for (int k = 0; k < 3; k++) {
                double drGinvr3 = dr[k] * Ginvr3;
                is->acc[i][k] += mj * drGinvr3;
                is->acc[j][k] += -mi * drGinvr3;
            }
            c++;
        }
    }
}

// Force without star-star interactions
static inline void comp_U_acc_loop_no_star_star(struct ketju_system *R) {
    struct ketju_system_internal_state *is = R->internal_state;
    struct spanning_tree_coordinate_system *mst = is->mst;
    struct ketju_system_physical_state *ps = R->physical_state;

    const double G = R->constants->G;

    // Assume there are only a few pn particles/BHs and parallelize only
    // over the inner loop.
    const int istart = 0, istop = R->num_pn_particles;

    const int rank = is->task_info->gbs_group_rank;
    int group_size = is->task_info->gbs_group_size;

    if (group_size > R->num_particles) {
        // in case we for some reason have more tasks than particles
        group_size = R->num_particles;
        if (rank > group_size) return;
    }

    const int part_per_task = R->num_particles / group_size;
    const int part_per_task_remainder =
        R->num_particles % is->task_info->gbs_group_size;
    int jstart = rank * part_per_task;
    if (rank < part_per_task_remainder) {
        jstart += rank; // the non-evenly split particles go to the first tasks
    } else {
        jstart += part_per_task_remainder;
    }

    int jstop =
        jstart + part_per_task + (rank < part_per_task_remainder ? 1 : 0);

    const int Nd = R->options->max_tree_distance;
    int d;
    int path[Nd];
    int sign[Nd];

    double dr[3];

    for (int i = istart; i < istop; i++) {
        const double mi = ps->mass[i];
        const int Li = mst->vertex[i].level;

        if (jstart <= i) jstart = i + 1; // avoid double counting interactions
        for (int j = jstart; j < jstop; j++) {
            double r2 = 0;
            const double mj = ps->mass[j];
            const int Lj = mst->vertex[j].level;

            bool proximity = false;
            if (abs(Li - Lj) <= Nd) {
                proximity =
                    ketju_check_relative_proximity(j, i, Nd, R, &d, path, sign);
            }

            if (proximity) {
                for (int k = 0; k < 3; k++) {
                    dr[k] = 0;
                    for (int l = 0; l < d; l++) {
                        dr[k] += sign[l] * mst->edge_rel_pos[path[l]][k];
                    }
                    r2 += dr[k] * dr[k];
                }
            } else {
                for (int k = 0; k < 3; k++) {
                    dr[k] = ps->pos[j][k] - ps->pos[i][k];
                    r2 += dr[k] * dr[k];
                }
            }

            const double invr = 1. / sqrt(r2);
            is->U += G * mi * mj * invr;

            const double Ginvr3 = G * invr * invr * invr;

            for (int k = 0; k < 3; k++) {
                double drGinvr3 = dr[k] * Ginvr3;
                is->acc[i][k] += mj * drGinvr3;
                is->acc[j][k] += -mi * drGinvr3;
            }
        }
    }
}

// Force with softened star-star interactions, doesn't include the pn-particles
static inline void comp_U_acc_loop_softened_star_star(struct ketju_system *R) {
    struct ketju_system_internal_state *is = R->internal_state;
    struct spanning_tree_coordinate_system *mst = is->mst;
    struct ketju_system_physical_state *ps = R->physical_state;

    int istop = mst->num_vertices, jstop = mst->num_vertices;
    int ThisBlock, CumNumEdge, jstart, istart;
    loop_scheduling_N2(mst->num_vertices - R->num_pn_particles,
                       is->task_info->gbs_group_size,
                       is->task_info->gbs_group_rank, &istart, &jstart,
                       &ThisBlock, &CumNumEdge);
    istart += R->num_pn_particles;
    jstart += R->num_pn_particles;
    int c = 0;
    const int Nd = R->options->max_tree_distance;
    int d;
    int path[Nd];
    int sign[Nd];

    double dr[3];
    const double ss_soft = R->options->star_star_softening;
    const double G = R->constants->G;

    for (int i = istart; i < istop; i++) {
        const double mi = ps->mass[i];
        const int Li = mst->vertex[i].level;
        if (i > istart) jstart = i + 1;

        for (int j = jstart; j < jstop; j++) {
            if (c >= ThisBlock) { return; }
            double r2 = 0;
            const double mj = ps->mass[j];
            const int Lj = mst->vertex[j].level;

            bool proximity = false;
            if (abs(Li - Lj) <= Nd) {
                proximity =
                    ketju_check_relative_proximity(j, i, Nd, R, &d, path, sign);
            }

            if (proximity) {
                for (int k = 0; k < 3; k++) {
                    dr[k] = 0;
                    for (int l = 0; l < d; l++) {
                        dr[k] += sign[l] * mst->edge_rel_pos[path[l]][k];
                    }
                    r2 += dr[k] * dr[k];
                }
            } else {
                for (int k = 0; k < 3; k++) {
                    dr[k] = ps->pos[j][k] - ps->pos[i][k];
                    r2 += dr[k] * dr[k];
                }
            }

            const double r = sqrt(r2);

            const double invr = softened_inverse_r(r, ss_soft);

            is->U += G * mi * mj * invr;

            const double Ginvr3 = G * softened_inverse_r3(r, ss_soft);

            for (int k = 0; k < 3; k++) {
                double drGinvr3 = dr[k] * Ginvr3;
                is->acc[i][k] += mj * drGinvr3;
                is->acc[j][k] += -mi * drGinvr3;
            }
            c++;
        }
    }
}

// computes the force function and newtonian acceleration
static void compute_U_and_Newtonian_Acc(struct ketju_system *R) {

    tick(&R->perf->time_force);

    struct ketju_system_internal_state *is = R->internal_state;
    struct spanning_tree_coordinate_system *mst = is->mst;

    memset(is->acc, 0, sizeof(double[R->num_particles][3]));
    is->U = 0;

    // select the force loop based on options
    if (R->options->no_star_star_gravity) {
        comp_U_acc_loop_no_star_star(R);
    } else if (R->options->star_star_softening > 0) {
        comp_U_acc_loop_no_star_star(R);       // non-softened part
        comp_U_acc_loop_softened_star_star(R); // softened part
    } else {
        comp_U_acc_loop_standard(R);
    }

    tock(&R->perf->time_force);

    tick(&R->perf->time_force_comm);
    if (is->task_info->gbs_group_size > 1) {
        MPI_Allreduce(MPI_IN_PLACE, is->acc, 3 * R->num_particles, MPI_DOUBLE,
                      MPI_SUM, is->task_info->gbs_comm);
        MPI_Allreduce(MPI_IN_PLACE, &is->U, 1, MPI_DOUBLE, MPI_SUM,
                      is->task_info->gbs_comm);
    }
    tock(&R->perf->time_force_comm);

    tick(&R->perf->time_force);
    int hi, lo;
    for (int i = 0; i < mst->num_vertices - 1; i++) {

        int v1, v2;
        get_v1_v2_fast(R, i, &v1, &v2);

        if (mst->vertex[v1].level > mst->vertex[v2].level) {
            hi = v1;
            lo = v2;
        } else {
            hi = v2;
            lo = v1;
        }
        for (int k = 0; k < 3; k++) {
            mst->edge_rel_acc[i][k] = is->acc[hi][k] - is->acc[lo][k];
        }
    }

    tock(&R->perf->time_force);
}

// TODO the following few functions could maybe go in mst.c, but maybe easier to
// have as few externally visible symbols as possible
static int get_root_index(const struct ketju_system *R) {
    const struct spanning_tree_coordinate_system *mst = R->internal_state->mst;

    int v1, v2;
    get_v1_v2_fast(R, 0, &v1, &v2);

    return mst->vertex[v1].level == 0 ? v1 : v2;
}

static void into_Cartesian_from_MST(struct ketju_system *R) {

    tick(&R->perf->time_coord);

    const struct spanning_tree_coordinate_system *mst = R->internal_state->mst;
    struct ketju_system_physical_state *ps = R->physical_state;

    int first = get_root_index(R);
    for (int k = 0; k < 3; k++) {
        ps->pos[first][k] = mst->root_pos[k];
        ps->vel[first][k] = mst->root_vel[k];
    }
    for (int i = 0; i < mst->num_vertices - 1; i++) {

        int v1, v2;
        get_v1_v2_fast(R, i, &v1, &v2);

        int lo, hi;
        if (mst->vertex[v1].level > mst->vertex[v2].level) {
            lo = v2;
            hi = v1;
        } else {
            lo = v1;
            hi = v2;
        }
        for (int k = 0; k < 3; k++) {
            ps->pos[hi][k] = ps->pos[lo][k] + mst->edge_rel_pos[i][k];
            ps->vel[hi][k] = ps->vel[lo][k] + mst->edge_rel_vel[i][k];
        }
    }

    tock(&R->perf->time_coord);
}

static void compute_T(struct ketju_system *R) {
    struct ketju_system_internal_state *is = R->internal_state;
    struct ketju_system_physical_state *ps = R->physical_state;
    is->T = 0;
    // TODO: this might benefit from parallelization for large particle numbers
    for (int i = 0; i < R->num_particles; i++) {
        double dv, v2 = 0;
        for (int k = 0; k < 3; k++) {
            dv = ps->vel[i][k];
            v2 += dv * dv;
        }
        is->T += 0.5 * ps->mass[i] * v2;
    }
}

static void update_pos(struct ketju_system *R) {

    tick(&R->perf->time_coord);

    const struct spanning_tree_coordinate_system *mst = R->internal_state->mst;
    struct ketju_system_physical_state *ps = R->physical_state;


    int first = get_root_index(R);
    for (int k = 0; k < 3; k++) {
        ps->pos[first][k] = mst->root_pos[k];
    }
    for (int i = 0; i < mst->num_vertices - 1; i++) {

        int v1, v2;
        get_v1_v2_fast(R, i, &v1, &v2);

        int lo, hi;
        if (mst->vertex[v1].level > mst->vertex[v2].level) {
            lo = v2;
            hi = v1;
        } else {
            lo = v1;
            hi = v2;
        }
        for (int k = 0; k < 3; k++) {
            ps->pos[hi][k] = ps->pos[lo][k] + mst->edge_rel_pos[i][k];
        }
    }

    tock(&R->perf->time_coord);
}

// TODO this and the above update_pos are very similar to
// into_Cartesian_from_MST, can they be deduplicated?
static void update_vel(struct ketju_system *R,
                       enum velocity_type propagated_velocity) {

    tick(&R->perf->time_coord);
    const struct spanning_tree_coordinate_system *mst = R->internal_state->mst;

    double(*vel_to_update)[3] = propagated_velocity == PHYSICAL
                                    ? R->physical_state->vel
                                    : R->internal_state->aux_vel;

    int first = get_root_index(R);
    for (int k = 0; k < 3; k++) {
        vel_to_update[first][k] = propagated_velocity == PHYSICAL
                                      ? mst->root_vel[k]
                                      : mst->root_aux_vel[k];
    }
    for (int i = 0; i < mst->num_vertices - 1; i++) {

        int v1, v2;
        get_v1_v2_fast(R, i, &v1, &v2);

        int lo, hi;
        if (mst->vertex[v1].level > mst->vertex[v2].level) {
            lo = v2;
            hi = v1;
        } else {
            lo = v1;
            hi = v2;
        }
        for (int k = 0; k < 3; k++) {
            if (propagated_velocity == PHYSICAL) {
                vel_to_update[hi][k] =
                    vel_to_update[lo][k] + mst->edge_rel_vel[i][k];
            } else {
                vel_to_update[hi][k] =
                    vel_to_update[lo][k] + mst->edge_rel_aux_vel[i][k];
            }
        }
    }

    tock(&R->perf->time_coord);
}

static void drift(double ds, struct ketju_system *R) {

    struct ketju_system_internal_state *is = R->internal_state;
    struct spanning_tree_coordinate_system *mst = is->mst;
    struct ketju_system_physical_state *ps = R->physical_state;

    compute_T(R);
     // T + B = U, but depends only on velocities
    double dt = ds / (is->T + is->B);
    ps->time += dt;

    for (int i = 0; i < mst->num_vertices - 1; i++) {
        for (int k = 0; k < 3; k++) {
            mst->edge_rel_pos[i][k] += dt * mst->edge_rel_vel[i][k];
        }
    }
    for (int k = 0; k < 3; k++) {
        mst->root_pos[k] += dt * mst->root_vel[k];
    }
    update_pos(R);
}

static void set_auxiliary_variables(struct ketju_system *R) {

    struct ketju_system_internal_state *is = R->internal_state;
    struct spanning_tree_coordinate_system *mst = is->mst;
    struct ketju_system_physical_state *ps = R->physical_state;

    memcpy(is->aux_vel, ps->vel, sizeof(double[R->num_particles][3]));
    memcpy(mst->edge_rel_aux_vel, mst->edge_rel_vel,
           sizeof(double[mst->num_vertices-1][3]));
    memcpy(mst->root_aux_vel, mst->root_vel, sizeof(double[3]));
    memcpy(is->aux_spin, ps->spin, sizeof(double[R->num_pn_particles][3]));
}

static void physical_kick(double dt, struct ketju_system *R) {

    struct ketju_system_internal_state *is = R->internal_state;
    struct spanning_tree_coordinate_system *mst = is->mst;
    struct ketju_system_physical_state *ps = R->physical_state;

    update_vel(R, AUXILIARY);
    ketju_compute_PN_acc(R, is->aux_vel, is->aux_spin);

    double dB = 0;
    for (int i = 0; i < R->num_particles; i++) {
        for (int k = 0; k < 3; k++) {
            dB += ps->mass[i] * is->aux_vel[i][k] * is->acc_vd[i][k];
        }
    }
    is->B -= dB * dt;

    for (int i = 0; i < mst->num_vertices - 1; i++) {
        for (int k = 0; k < 3; k++) {
            mst->edge_rel_vel[i][k] +=
                dt * (mst->edge_rel_acc[i][k] + mst->edge_rel_acc_vd[i][k]);
        }
    }

    int root = get_root_index(R);
    for (int k = 0; k < 3; k++) {
        mst->root_vel[k] +=
            dt * (is->acc[root][k] + is->acc_vd[root][k]);
    }

    for (int i = 0; i < R->num_pn_particles; ++i) {
        for (int k = 0; k < 3; k++) {
            ps->spin[i][k] += dt * is->spin_derivative[i][k];
        }
    }
}

static void auxiliary_kick(double dt, struct ketju_system *R) {

    struct ketju_system_internal_state *is = R->internal_state;
    struct spanning_tree_coordinate_system *mst = is->mst;
    struct ketju_system_physical_state *ps = R->physical_state;

    update_vel(R, PHYSICAL);
    ketju_compute_PN_acc(R, ps->vel, ps->spin);

    for (int i = 0; i < mst->num_vertices - 1; i++) {
        for (int k = 0; k < 3; k++) {
            mst->edge_rel_aux_vel[i][k] +=
                dt * (mst->edge_rel_acc[i][k] + mst->edge_rel_acc_vd[i][k]);
        }
    }

    int root = get_root_index(R);
    for (int k = 0; k < 3; k++) {
        mst->root_aux_vel[k] +=
            dt * (is->acc[root][k] + is->acc_vd[root][k]);
    }

    for (int i = 0; i < R->num_pn_particles; ++i) {
        for (int k = 0; k < 3; k++) {
            is->aux_spin[i][k] += dt * is->spin_derivative[i][k];
        }
    }
}

static void kick(double ds, struct ketju_system *R) {

    struct ketju_system_internal_state *is = R->internal_state;
    struct spanning_tree_coordinate_system *mst = is->mst;

    compute_U_and_Newtonian_Acc(R);
    double dt = ds / is->U;

    if (!R->options->PN_flags) {
        // no velocity dependent forces (only PN for now), just the newtonian
        // part needed
        for (int i = 0; i < mst->num_vertices - 1; i++) {
            for (int k = 0; k < 3; k++) {
                mst->edge_rel_vel[i][k] += dt * mst->edge_rel_acc[i][k];
            }
        }
        int root = get_root_index(R);
        for (int k = 0; k < 3; k++) {
            mst->root_vel[k] += dt * is->acc[root][k];
        }
    } else {
        auxiliary_kick(dt / 2, R);
        physical_kick(dt, R);
        auxiliary_kick(dt / 2, R);
    }
    update_vel(R, PHYSICAL);
}

static void compute_total_energy(struct ketju_system *R) {

    struct ketju_system_internal_state *is = R->internal_state;

    compute_T(R);
    compute_U(R);
    is->B = is->U - is->T;
}

static void mst_leapfrog(struct ketju_system *R, double Hstep,
                         int NumberOfSubsteps) {

    if (R->options->PN_flags) { set_auxiliary_variables(R); }

    double h = Hstep / NumberOfSubsteps;

    drift(h / 2, R);
    for (int step = 0; step < NumberOfSubsteps - 1; step++) {
        kick(h, R);
        drift(h, R);
    }
    kick(h, R);
    drift(h / 2, R);
}

static int num_gbs_substeps(int k) { return 2 * k; }

static int argmin_int(int *arr, int size) {
    int min = arr[0];
    int ind = 0;
    for (int i = 1; i < size; ++i) {
        if (arr[i] < min) {
            min = arr[i];
            ind = i;
        }
    }
    return ind;
}

// Get the values of k that are done by the gbs groups.
// The values are in descending order
static void get_gbs_ks_for_groups(struct ketju_system *R,
                                  int (*gbs_ks_for_groups)[]) {
    const int kmax = R->options->gbs_kmax;
    const int num_groups = R->internal_state->group_info->num_gbs_groups;
    memset(gbs_ks_for_groups, 0, sizeof(int[num_groups][kmax]));
    int(*gbs_ks)[kmax] = (int(*)[kmax])gbs_ks_for_groups;

    int group_workloads[num_groups];
    memset(group_workloads, 0, sizeof group_workloads);
    int k_ind[num_groups];
    memset(k_ind, 0, sizeof k_ind);
    // TODO could take into account group sizes also, or even divide the groups
    // based on the most even division of gbs steps/task
    for (int k = kmax; k > 0; --k) {
        int cost = num_gbs_substeps(k);
        int group_ind = argmin_int(group_workloads, num_groups);
        group_workloads[group_ind] += cost;
        gbs_ks[group_ind][k_ind[group_ind]++] = k;
    }
}

struct state_backup {
    double time, B, U, T;
    double root_pos[3];
    double root_vel[3];
    double (*pos)[3];
    double (*vel)[3];
    double (*spin)[3];
    double (*edge_rel_pos)[3];
    double (*edge_rel_vel)[3];
};

static void allocate_state_backup(struct state_backup *sb, int num_particles,
                                  int num_pn_particles) {
    sb->pos = malloc(sizeof(double[num_particles][3]));
    sb->vel = malloc(sizeof(double[num_particles][3]));
    sb->spin = malloc(sizeof(double[num_pn_particles][3]));
    sb->edge_rel_pos = malloc(sizeof(double[num_particles-1][3]));
    sb->edge_rel_vel = malloc(sizeof(double[num_particles-1][3]));
}

static void store_state_backup(struct state_backup *sb,
                               const struct ketju_system *R) {
    sb->time = R->physical_state->time;
    sb->B = R->internal_state->B;
    sb->U = R->internal_state->U;
    sb->T = R->internal_state->T;

    memcpy(sb->root_pos, R->internal_state->mst->root_pos, sizeof(double[3]));
    memcpy(sb->root_vel, R->internal_state->mst->root_vel, sizeof(double[3]));

    const int num_particles = R->num_particles;
    memcpy(sb->pos, R->physical_state->pos, sizeof(double[num_particles][3]));
    memcpy(sb->vel, R->physical_state->vel, sizeof(double[num_particles][3]));
    memcpy(sb->spin, R->physical_state->spin, sizeof(double[R->num_pn_particles][3]));
    memcpy(sb->edge_rel_pos, R->internal_state->mst->edge_rel_pos,
           sizeof(double[num_particles - 1][3]));
    memcpy(sb->edge_rel_vel, R->internal_state->mst->edge_rel_vel,
           sizeof(double[num_particles - 1][3]));
}

static void restore_state_backup(struct state_backup *sb,
                                 const struct ketju_system *R) {
    R->physical_state->time = sb->time;
    R->internal_state->B = sb->B;
    R->internal_state->U = sb->U;
    R->internal_state->T = sb->T;

    memcpy(R->internal_state->mst->root_pos, sb->root_pos, sizeof(double[3]));
    memcpy(R->internal_state->mst->root_vel, sb->root_vel, sizeof(double[3]));

    const int num_particles = R->num_particles;
    memcpy(R->physical_state->pos, sb->pos, sizeof(double[num_particles][3]));
    memcpy(R->physical_state->vel, sb->vel, sizeof(double[num_particles][3]));
    memcpy(R->physical_state->spin, sb->spin, sizeof(double[R->num_pn_particles][3]));
    memcpy(R->internal_state->mst->edge_rel_pos, sb->edge_rel_pos,
           sizeof(double[num_particles-1][3]));
    memcpy(R->internal_state->mst->edge_rel_vel, sb->edge_rel_vel,
           sizeof(double[num_particles-1][3]));
}

static void free_state_backup(struct state_backup *sb) {
    free(sb->pos);
    free(sb->vel);
    free(sb->spin);
    free(sb->edge_rel_pos);
    free(sb->edge_rel_vel);
}

struct gbs_storage {
    int num_vars;
    int num_rows;
    double *data;
};

static void allocate_gbs_storage(struct gbs_storage *gs, int num_particles,
                                 int num_pn_particles, int kmax) {
    gs->num_vars = (1 /*time*/ + 3 /*root_pos*/ + 3 /*root_vel*/ +
                    (num_particles - 1) * (3 /*edge_pos*/ + 3 /*edge_vel*/) +
                    num_pn_particles * (3 /*spin*/));
    gs->num_rows = kmax + 1; // row for extrapolation result also
    gs->data = calloc(1, sizeof(double[gs->num_rows][gs->num_vars]));
}

static void free_gbs_storage(struct gbs_storage *gs) { free(gs->data); }

// store the data for this k for extrapolation
static void store_gbs_data(struct gbs_storage *gs, const struct ketju_system *R,
                           int k) {
    double(*data)[gs->num_vars] = (double(*)[gs->num_vars])gs->data;
    const int k_ind = k - 1;
    data[k_ind][0] = R->physical_state->time;
    int offset = 1;
    const struct spanning_tree_coordinate_system *mst = R->internal_state->mst;
    memcpy(&data[k_ind][offset], mst->root_pos, sizeof(double[3]));
    offset += 3;
    memcpy(&data[k_ind][offset], mst->root_vel, sizeof(double[3]));
    offset += 3;
    const size_t edge_vec_size = sizeof(double[mst->num_vertices-1][3]);
    memcpy(&data[k_ind][offset], mst->edge_rel_pos, edge_vec_size);
    offset += 3*(mst->num_vertices-1);
    memcpy(&data[k_ind][offset], mst->edge_rel_vel, edge_vec_size);
    offset += 3*(mst->num_vertices-1);
    memcpy(&data[k_ind][offset], R->physical_state->spin,
           sizeof(double[R->num_pn_particles][3]));
}

// Make the non-blocking receiving calls for the rows of data that need to be
// received. Called for all tasks
static void post_gbs_receives(MPI_Request *requests, int *request_index,
                              struct gbs_storage *gs,
                              int (*gbs_ks_for_groups)[],
                              const struct ketju_system *R) {
    tick(&R->perf->time_gbscomm);

    const struct task_info *ti = R->internal_state->task_info;
    const int kmax = R->options->gbs_kmax;
    double(*data)[gs->num_vars] = (double(*)[gs->num_vars])gs->data;
    int(*gbs_ks)[kmax] = (int(*)[kmax])gbs_ks_for_groups;
    for (int i = 0; i < R->internal_state->group_info->num_gbs_groups; ++i) {
        // This task will be computing the data, so doesn't receive
        if (i == ti->gbs_group_index) continue;

        for (int j = 0; j < kmax && gbs_ks[i][j] > 0; ++j) {
            int data_row = gbs_ks[i][j] - 1;
            MPI_Ibcast(data[data_row], gs->num_vars, MPI_DOUBLE, 0,
                       ti->gbs_data_comms[i], &requests[(*request_index)++]);
        }
    }
    tock(&R->perf->time_gbscomm);
}

// Make the non-blocking send call for a row of data corresponding to the given
// gbs k. Only does anything on the root task of the gbs group,
// but safe to call for all tasks in the gbs group handling the gbs k substeps.
static void post_gbs_send(MPI_Request *requests, int *request_index,
                          struct gbs_storage *gs, const struct ketju_system *R,
                          int k) {

    const struct task_info *ti = R->internal_state->task_info;
    double(*data)[gs->num_vars] = (double(*)[gs->num_vars])gs->data;
    const int i = ti->gbs_group_index;
    if (ti->gbs_data_comms[i] == MPI_COMM_NULL) return;

    tick(&R->perf->time_gbscomm);
    int data_row = k - 1;
    MPI_Ibcast(data[data_row], gs->num_vars, MPI_DOUBLE, 0,
               ti->gbs_data_comms[i], &requests[(*request_index)++]);
    tock(&R->perf->time_gbscomm);
}

// Wait for the communications to finish
static void finish_gbs_comm(MPI_Request *requests, int request_index,
                            const struct ketju_system *R) {
    tick(&R->perf->time_gbscomm);
    if (request_index > 0) {
        MPI_Waitall(request_index, requests, MPI_STATUSES_IGNORE);
    }
    tock(&R->perf->time_gbscomm);
}

static double get_n_double(int i) { return (2.0 * (i + 1.0)); }
static void extrapolate_single_variable(double *y, double yscal, const int N,
                                        double *result, double *error) {
    double gbsS[N][N];
    for (int i = 0; i < N; i++) {
        gbsS[i][0] = y[i];
    }
    for (int j = 1; j < N; j++) {
        for (int i = 0; i < N - j; i++) {
            double nterm = pow(get_n_double(i + j) / get_n_double(i), 2);
            gbsS[i][j] =
                (nterm * gbsS[i + 1][j - 1] - gbsS[i][j - 1]) / (nterm - 1.0);
        }
    }
    *result = gbsS[0][N - 1];
    *error = fabs(gbsS[0][N - 1] - gbsS[0][N - 2]) / yscal;
}
// Extrapolate a subset of variables on each task, comm the results.
// Updates the fictitious timestep Hstep with the new estimate.
static bool gbs_extrapolate(struct gbs_storage *gs, double *Hstep,
                            const struct ketju_system *R) {
    tick(&R->perf->time_gbsextr);
    // figure out which variables each task handles
    // TODO this should stay the same during the integration (except mergers)
    // so could be moved out of the inner loop
    const int num_tasks = R->internal_state->task_info->main_group_size;
    const int vars_per_task = gs->num_vars / num_tasks;
    const int remainder =
        gs->num_vars % R->internal_state->task_info->main_group_size;

    int start_inds[num_tasks];
    int var_counts[num_tasks];
    for (int i = 0; i < num_tasks; ++i) {
        start_inds[i] = i > 0 ? start_inds[i - 1] + var_counts[i - 1] : 0;
        var_counts[i] = vars_per_task;
        // handle the remainder on the last (remainder) tasks, one extra var
        // each
        if (i >= num_tasks - remainder) { var_counts[i] += 1; }
    }

    const int rank = R->internal_state->task_info->main_group_rank;
    int start_ind = start_inds[rank];
    int end_ind = start_ind + var_counts[rank];

    // run the extrapolation
    const int kmax = R->options->gbs_kmax;
    double y[kmax];
    double yscal;
    double error;
    double max_error = 0;
    double result;
    double(*data)[gs->num_vars] = (double(*)[gs->num_vars])gs->data;

    // TODO make configurable
    const double abs_tol = 1e-100;

    for (int i = start_ind; i < end_ind; ++i) {
        for (int j = 0; j < kmax; ++j) {
            y[j] = data[j][i];
        }
        yscal = fabs(y[kmax-1]) + abs_tol;

        if (i == 0) {
            yscal += R->options->output_time_absolute_tolerance;
        }

        extrapolate_single_variable(y, yscal, kmax, &result, &error);

        // Explicit nan/inf handling in case they are produced.
        // The code after this can then assume finite values.
        if (!isfinite(result)) {
            // Set the max error so that the step size gets cut by a factor of
            // ~10, and then bail as the step has failed anyway.
            max_error = R->options->gbs_relative_tolerance * 0.65 /
                        pow(0.1, 2 * kmax + 1);
            break;
        }

        data[kmax][i] = result;
        if (error > max_error) {
            max_error = error;
        }
    }
    tock(&R->perf->time_gbsextr);

    tick(&R->perf->time_gbscomm);
    // check convergence across all task
    MPI_Allreduce(MPI_IN_PLACE, &max_error, 1, MPI_DOUBLE, MPI_MAX,
                  R->internal_state->task_info->main_comm);

    double Hnextfac =
        0.94 * pow(0.65 / (max_error / R->options->gbs_relative_tolerance),
                   1.0 / (2 * kmax + 1));
    const bool converged = max_error < R->options->gbs_relative_tolerance;
    if (converged) {
        // success, communicate data and set new timestep
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, data[kmax],
                       var_counts, start_inds, MPI_DOUBLE,
                       R->internal_state->task_info->main_comm);

        // Avoid increasing the stepsize too aggressively
        if (Hnextfac > 5.0) { Hnextfac = 5.0; }
        *Hstep *= Hnextfac;
    } else {
        // Ensure that the stepsize is decreased enough if the step failed,
        if (Hnextfac > 0.7) { Hnextfac = 0.7; }
        // but not excessively.
        if (Hnextfac < 1e-3) { Hnextfac = 1e-3; }
        *Hstep *= Hnextfac;
    }

    tock(&R->perf->time_gbscomm);

    return converged;
}

static void update_system_from_gbs_result(struct ketju_system *R,
                                          struct gbs_storage *gs) {
    double *data = ((double(*)[gs->num_vars])gs->data)[gs->num_rows - 1];
    R->physical_state->time = data[0];
    int offset = 1;
    struct spanning_tree_coordinate_system *mst = R->internal_state->mst;
    memcpy(mst->root_pos, &data[offset], sizeof(double[3]));
    offset += 3;
    memcpy(mst->root_vel, &data[offset], sizeof(double[3]));
    offset += 3;
    const size_t edge_vec_size = sizeof(double[mst->num_vertices-1][3]);
    memcpy(mst->edge_rel_pos, &data[offset], edge_vec_size);
    offset += 3*(mst->num_vertices-1);
    memcpy(mst->edge_rel_vel, &data[offset], edge_vec_size);
    offset += 3*(mst->num_vertices-1);
    memcpy(R->physical_state->spin, &data[offset],
           sizeof(double[R->num_pn_particles][3]));

    into_Cartesian_from_MST(R);
}


// Estimates a safe initial timestep that is sufficiently small compared to the
// smallest two-body timescale in the system.
static double estimate_initial_timestep(struct ketju_system *R) {
    struct ketju_system_internal_state *is = R->internal_state;
    struct spanning_tree_coordinate_system *mst = is->mst;
    struct ketju_system_physical_state *ps = R->physical_state;

    int istart, jstart;
    int CumNumEdge, ThisBlock;
    loop_scheduling_N2(mst->num_vertices, is->task_info->main_group_size,
                       is->task_info->main_group_rank, &istart, &jstart,
                       &ThisBlock, &CumNumEdge);

    const int Nd = R->options->max_tree_distance;
    int d;
    int path[Nd];
    int sign[Nd];

    const double ss_soft = R->options->star_star_softening;
    double t2min = DBL_MAX;

    int c = 0;
    for (int i = istart; i < mst->num_vertices; i++) {

        double mi = ps->mass[i];
        int Li = mst->vertex[i].level;
        if (i > istart) { jstart = i + 1; }
        for (int j = jstart; j < mst->num_vertices; j++) {
            if (c >= ThisBlock) { goto after_loops; }
            if (R->options->no_star_star_gravity && i >= R->num_pn_particles &&
                j >= R->num_pn_particles) {
                continue;
            }

            double mj = ps->mass[j];
            int Lj = mst->vertex[j].level;
            double r2 = 0;
            double v2 = 0;
            bool proximity = false;

            if (abs(Li - Lj) < Nd) {
                proximity =
                    ketju_check_relative_proximity(j, i, Nd, R, &d, path, sign);
            }

            if (proximity) {
                for (int k = 0; k < 3; k++) {
                    double dr = 0;
                    double dv = 0;
                    for (int l = 0; l < d; l++) {
                        int index = path[l];
                        dr += sign[l] * mst->edge_rel_pos[index][k];
                        dv += sign[l] * mst->edge_rel_vel[index][k];
                    }
                    r2 += dr * dr;
                    v2 += dv * dv;
                }
            } else {
                for (int k = 0; k < 3; k++) {
                    double dr = ps->pos[j][k] - ps->pos[i][k];
                    r2 += dr * dr;
                    double dv = ps->vel[j][k] - ps->vel[i][k];
                    v2 += dv * dv;
                }
            }
            double soft = i < R->num_pn_particles || j < R->num_pn_particles
                              ? 0.
                              : ss_soft;
            const double invr = softened_inverse_r(sqrt(r2), soft);

            // Estimate the square of the approximate two-body time scale from a
            // combination of free propagation and keplerian timescales.
            double t2 =
                1. / (invr * (invr * v2 + R->constants->G * (mi + mj)));
            t2min = fmin(t2min, t2);

            c++;
        }
    }
after_loops:

    MPI_Allreduce(MPI_IN_PLACE, &t2min, 1, MPI_DOUBLE, MPI_MIN,
                  is->task_info->main_comm);

    const double safetyfac = 0.05;
    return safetyfac * sqrt(t2min);
}

static void trivial_propagation(struct ketju_system *R, double time_interval) {
    // Only makes sense when there's just a single particle, but this guards
    // against the zero particle case (altohugh it shouldn't really happen)
    for (int i = 0; i < R->num_particles; ++i) {
        for (int k = 0; k < 3; ++k) {
            R->physical_state->pos[i][k] +=
                R->physical_state->vel[i][k] * time_interval;
        }
    }
    R->physical_state->time += time_interval;
}

void ketju_run_integrator(struct ketju_system *R, double time_interval) {
    // TODO return status/error code

    // special case for when no actual integration is needed
    if (R->num_particles < 2) {
        tick(&R->perf->time_total);
        trivial_propagation(R, time_interval);
        tock(&R->perf->time_total);
        return;
    }

    tick(&R->perf->time_total);
    // Initialize necessary comms if not done already by the caller
    if (!R->internal_state->task_info->gbs_comms_initialized) {
        ketju_init_gbs_comms(R);
    }

    ketju_build_mst(R);
    int steps_since_reconstruction = 0;

    compute_total_energy(R);

    double dt = copysign(1, time_interval) *
                fmin(estimate_initial_timestep(R), fabs(time_interval) / 10);
    double Hstep = dt * R->internal_state->U;
    const double t_start = R->physical_state->time;
    const double initial_B = R->internal_state->B;

    const int kmax = R->options->gbs_kmax;
    int gbs_ks_for_groups[R->internal_state->group_info->num_gbs_groups][kmax];
    get_gbs_ks_for_groups(R, gbs_ks_for_groups);
    int *gbs_ks_for_this_group =
        gbs_ks_for_groups[R->internal_state->task_info->gbs_group_index];

    struct state_backup old_state;
    allocate_state_backup(&old_state, R->num_particles, R->num_pn_particles);
    store_state_backup(&old_state, R);

    struct gbs_storage gbs_storage;
    allocate_gbs_storage(&gbs_storage, R->num_particles, R->num_pn_particles,
                         R->options->gbs_kmax);

    bool finished = false;
    do {
        // Check mergers between BHs (PN particles)
        // Need to be propagating forwards in time for mergers to make sense.
        if (R->options->enable_bh_mergers && Hstep > 0) {
            int num_merged = ketju_do_bh_mergers(R);
            if (num_merged) {
                // We had mergers, need to redo tree and other things
                if (R->num_particles < 2) {
                    // Particle motion becomes trivial.
                    // Need to only propagate the CoM
                    trivial_propagation(R, t_start + time_interval -
                                               R->physical_state->time);
                    break;
                }
                ketju_build_mst(R);
                steps_since_reconstruction = 0;
                compute_total_energy(R);
                store_state_backup(&old_state, R);
                gbs_storage.num_vars -= num_merged * 9; // pos, vel, spin
            }
        }
        // Run the gbs substeps, communicating results in the background.
        int request_index = 0;
        MPI_Request requests[kmax];
        post_gbs_receives(requests, &request_index, &gbs_storage,
                          gbs_ks_for_groups, R);
        for (int i = 0; i < kmax && gbs_ks_for_this_group[i]; ++i) {
            if (i) {
                // not needed on the first round
                restore_state_backup(&old_state, R);
            }
            int k = gbs_ks_for_this_group[i];
            mst_leapfrog(R, Hstep, num_gbs_substeps(k));
            store_gbs_data(&gbs_storage, R, k);
            post_gbs_send(requests, &request_index, &gbs_storage, R, k);
        }
        finish_gbs_comm(requests, request_index, R);

        bool converged = gbs_extrapolate(&gbs_storage, &Hstep, R);

        if (converged) {
            update_system_from_gbs_result(R, &gbs_storage);
            compute_total_energy(R);
            R->perf->successful_steps += 1;
            steps_since_reconstruction += 1;

            double remaining_time =
                time_interval - (R->physical_state->time - t_start);
            if (fabs(remaining_time / time_interval) <
                    R->options->output_time_relative_tolerance ||
                fabs(remaining_time) <
                    R->options->output_time_absolute_tolerance) {
                finished = true;
            } else {
                // Check that we're not overstepping and alter timestep if
                // required
                // Should work also for negative time intervals
                // TODO this should be improved to allow for better accuracy in
                // time
                if (fabs(remaining_time) < fabs(Hstep / R->internal_state->U) ||
                    copysign(1, time_interval) * remaining_time < 0) {
                    Hstep = R->internal_state->U * remaining_time;
                }
                // new stepsize set by gbs_extrapolate otherwise
                if (steps_since_reconstruction >=
                    R->options->steps_between_mst_reconstruction) {
                    ketju_build_mst(R);
                    steps_since_reconstruction = 0;
                }

                store_state_backup(&old_state, R);
            }
        } else {
            restore_state_backup(&old_state, R);
            // stepsize is updated by the extrapolation function

            R->perf->failed_steps += 1;
        }

        if (R->options->max_step_count > 0 &&
            R->perf->failed_steps + R->perf->successful_steps >
                R->options->max_step_count) {
            if (R->internal_state->task_info->main_group_rank == 0) {
                ketju_debug_dump_integrator_state(
                    R, "ketju_integrator_max_steps_state_dump.bin");
            }
            terminate("Exceeded options->max_step_count steps! State dumped.");
        }

    } while (!finished);


    // free temp storage
    free_state_backup(&old_state);
    free_gbs_storage(&gbs_storage);

    // Note that this is not really the error when dissipative PN forces are
    // enabled
    R->perf->relative_energy_error =
        (R->internal_state->B - initial_B) / initial_B;

    tock(&R->perf->time_total);
}
