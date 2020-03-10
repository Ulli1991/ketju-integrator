// Main integration and force calculation functions

#include <assert.h>
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
            //const double invr = 1./sqrt(r2);
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

    if (!isfinite(is->U)) { terminate("U is not finite!"); }

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
            is->U += mi * mj * invr;

            const double invr3 = invr * invr * invr;

            for (int k = 0; k < 3; k++) {
                double drinvr3 = dr[k] * invr3;
                is->acc[i][k] += mj * drinvr3;
                is->acc[j][k] += -mi * drinvr3;
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
            is->U += mi * mj * invr;

            const double invr3 = invr * invr * invr;

            for (int k = 0; k < 3; k++) {
                double drinvr3 = dr[k] * invr3;
                is->acc[i][k] += mj * drinvr3;
                is->acc[j][k] += -mi * drinvr3;
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

            const double r = sqrt(r2);

            const double invr = softened_inverse_r(r, ss_soft);

            is->U += mi * mj * invr;

            const double invr3 = softened_inverse_r3(r, ss_soft);

            for (int k = 0; k < 3; k++) {
                double drinvr3 = dr[k] * invr3;
                is->acc[i][k] += mj * drinvr3;
                is->acc[j][k] += -mi * drinvr3;
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
        comp_U_acc_loop_no_star_star(R); // non-softened part
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
            mst->edge_rel_acc[i][k] =
                R->constants->G * (is->acc[hi][k] - is->acc[lo][k]);
        }
    }

    is->U *= R->constants->G;

    tock(&R->perf->time_force);
}

// TODO this could maybe go in mst.c, but maybe easier to have as few externally
// visible symbols as possible
static void into_Cartesian_from_MST(struct ketju_system *R) {

    tick(&R->perf->time_coord);

    const struct spanning_tree_coordinate_system *mst = R->internal_state->mst;
    struct ketju_system_physical_state *ps = R->physical_state;

    int v1, v2;
    get_v1_v2_fast(R, 0, &v1, &v2);

    int first;
    if (mst->vertex[v1].level == 0) {
        first = v1;
    } else {
        first = v2;
    }
    for (int k = 0; k < 3; k++) {
        ps->pos[first][k] = 0.0;
        ps->vel[first][k] = 0.0;
    }
    for (int i = 0; i < mst->num_vertices - 1; i++) {

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

    // The coordinate system is defined to have the CoM at the origin, but at
    // this point the root particle is at the origin. Fix the situation.
    double M = 0, CoM[3] = {0}, CoV[3] = {0};

    for (int i = 0; i < R->num_particles; i++) {
        M += ps->mass[i];
        for (int k = 0; k < 3; k++) {
            CoM[k] += ps->mass[i] * ps->pos[i][k];
            CoV[k] += ps->mass[i] * ps->vel[i][k];
        }
    }
    for (int k = 0; k < 3; k++) {
        CoM[k] /= M;
        CoV[k] /= M;
    }

    for (int i = 0; i < R->num_particles; i++) {
        for (int k = 0; k < 3; k++) {
            ps->pos[i][k] -= CoM[k];
            ps->vel[i][k] -= CoV[k];
        }
    }

    tock(&R->perf->time_coord);
}

// Puts the initial data to the CoM frame and shifts the CoM pos and vel.
// CoM pos and vel are only shifted, so safe to call multiple times.
void ketju_into_CoM_frame(struct ketju_system *R) {

    tick(&R->perf->time_coord);
    struct ketju_system_physical_state *ps = R->physical_state;
    double CoM_pos[3] = {0};
    double CoM_vel[3] = {0};
    double M = 0;

    for (int i = 0; i < R->num_particles; i++) {
        M += ps->mass[i];
        for (int k = 0; k < 3; k++) {
            CoM_pos[k] += ps->mass[i] * ps->pos[i][k];
            CoM_vel[k] += ps->mass[i] * ps->vel[i][k];
        }
    }
    for (int k = 0; k < 3; k++) {
        CoM_pos[k] /= M;
        CoM_vel[k] /= M;
    }
    for (int i = 0; i < R->num_particles; i++) {
        for (int k = 0; k < 3; k++) {
            ps->pos[i][k] -= CoM_pos[k];
            ps->vel[i][k] -= CoM_vel[k];
        }
    }
    for (int k = 0; k < 3; k++) {
        R->CoM_pos[k] += CoM_pos[k];
        R->CoM_vel[k] += CoM_vel[k];
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

    int v1, v2;
    get_v1_v2_fast(R, 0, &v1, &v2);

    int first;
    if (mst->vertex[v1].level == 0) {
        first = v1;
    } else {
        first = v2;
    }
    for (int k = 0; k < 3; k++) {
        ps->pos[first][k] = 0.0;
    }
    for (int i = 0; i < mst->num_vertices - 1; i++) {

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

    double M = 0, CoM[3] = {0};
    for (int i = 0; i < R->num_particles; i++) {
        M += ps->mass[i];
        for (int k = 0; k < 3; k++) {
            CoM[k] += ps->mass[i] * ps->pos[i][k];
        }
    }
    for (int k = 0; k < 3; k++) {
        CoM[k] /= M;
    }
    for (int i = 0; i < R->num_particles; i++) {
        for (int k = 0; k < 3; k++) {
            ps->pos[i][k] -= CoM[k];
        }
    }

    tock(&R->perf->time_coord);
}

// TODO this and the above update_pos are very similar to
// into_Cartesian_from_MST, can they be deduplicated?
static void update_vel(struct ketju_system *R, double (*vel_to_update)[3],
                enum velocity_type propagated_velocity) {

    tick(&R->perf->time_coord);
    const struct spanning_tree_coordinate_system *mst = R->internal_state->mst;
    struct ketju_system_physical_state *ps = R->physical_state;

    int v1, v2;
    get_v1_v2_fast(R, 0, &v1, &v2);

    int first;
    if (mst->vertex[v1].level == 0) {
        first = v1;
    } else {
        first = v2;
    }
    for (int k = 0; k < 3; k++) {
        vel_to_update[first][k] = 0.0;
    }
    for (int i = 0; i < mst->num_vertices - 1; i++) {

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

    double M = 0, CoV[3] = {0};
    for (int i = 0; i < R->num_particles; i++) {
        M += ps->mass[i];
        for (int k = 0; k < 3; k++) {
            CoV[k] += ps->mass[i] * vel_to_update[i][k];
        }
    }
    for (int k = 0; k < 3; k++) {
        CoV[k] /= M;
    }
    for (int i = 0; i < R->num_particles; i++) {
        for (int k = 0; k < 3; k++) {
            vel_to_update[i][k] -= CoV[k];
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
    update_pos(R);
}

static void set_auxiliary_variables(struct ketju_system *R) {

    struct ketju_system_internal_state *is = R->internal_state;
    struct spanning_tree_coordinate_system *mst = is->mst;
    struct ketju_system_physical_state *ps = R->physical_state;

    memcpy(is->aux_vel, ps->vel, sizeof(double[R->num_particles][3]));
    memcpy(mst->edge_rel_aux_vel, mst->edge_rel_vel,
           sizeof(double[mst->num_vertices-1][3]));
    memcpy(is->aux_spin, ps->spin, sizeof(double[R->num_pn_particles][3]));
}

// TODO: move declarations / reorganize
static void compute_Post_Newtonian_Acc(struct ketju_system *,
                                       double (*vel_to_use)[3],
                                       double (*spin_to_use)[3]);

static void physical_kick(double dt, struct ketju_system *R) {

    struct ketju_system_internal_state *is = R->internal_state;
    struct spanning_tree_coordinate_system *mst = is->mst;
    struct ketju_system_physical_state *ps = R->physical_state;

    update_vel(R, is->aux_vel, AUXILIARY);
    compute_Post_Newtonian_Acc(R, is->aux_vel, is->aux_spin);

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

    update_vel(R, ps->vel, PHYSICAL);
    compute_Post_Newtonian_Acc(R, ps->vel, ps->spin);

    for (int i = 0; i < mst->num_vertices - 1; i++) {
        for (int k = 0; k < 3; k++) {
            mst->edge_rel_aux_vel[i][k] +=
                dt * (mst->edge_rel_acc[i][k] + mst->edge_rel_acc_vd[i][k]);
        }
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
    struct ketju_system_physical_state *ps = R->physical_state;

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
    } else {
        auxiliary_kick(dt / 2, R);
        physical_kick(dt, R);
        auxiliary_kick(dt / 2, R);
    }
    update_vel(R, ps->vel, PHYSICAL);
}

static void compute_total_energy(struct ketju_system *R) {

    struct ketju_system_internal_state *is = R->internal_state;

    compute_T(R);
    compute_U(R);
    is->B = is->U - is->T;
    // TODO is there some reason that the B variable and the B value in the old
    // state are different like below?
    // R->State[R->BIndex] = R->B;
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

// Get the values of k that are done by this gbs group.
// The values are in descending order
static void get_gbs_ks_for_this_group(struct ketju_system *R,
                                      int *gbs_ks_for_this_group) {
    const int kmax = R->options->gbs_kmax;
    memset(gbs_ks_for_this_group, 0, sizeof(int[kmax]));

    const int num_groups = R->internal_state->group_info->num_gbs_groups;
    const int this_group_ind = R->internal_state->task_info->gbs_group_index;
    int group_workloads[num_groups];
    memset(group_workloads, 0, sizeof group_workloads);
    int k_ind = 0;
    // TODO could take into account group sizes also, or even divide the groups
    // based on the most even division of gbs steps/task
    for (int k = kmax; k > 0; --k) {
        int cost = num_gbs_substeps(k);
        int group_ind = argmin_int(group_workloads, num_groups);
        group_workloads[group_ind] += cost;
        if (group_ind == this_group_ind) { gbs_ks_for_this_group[k_ind++] = k; }
    }
}

struct state_backup {
    double time, B, U, T;
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
    gs->num_vars = (1 /*time*/ + 1 /*B*/ +
                    (num_particles - 1) * (3 /*edge_pos*/ + 3 /*edge_vel*/) +
                    num_pn_particles * (3 /*spin*/));
    gs->num_rows = kmax + 1; // row for extrapolation result also
    gs->data = calloc(1, sizeof(double[gs->num_vars * gs->num_rows]));
}

static void free_gbs_storage(struct gbs_storage *gs) { free(gs->data); }

// store the data for this k for extrapolation
static void store_gbs_data(struct gbs_storage *gs, const struct ketju_system *R,
                           int k) {
    double(*data)[gs->num_vars] = (double(*)[gs->num_vars])gs->data;
    const int k_ind = k - 1;
    data[k_ind][0] = R->physical_state->time;
    data[k_ind][1] = R->internal_state->B;
    int offset = 2;
    const struct spanning_tree_coordinate_system *mst = R->internal_state->mst;
    const size_t edge_vec_size = sizeof(double[mst->num_vertices-1][3]);
    memcpy(&data[k_ind][offset], mst->edge_rel_pos, edge_vec_size);
    offset += 3*(mst->num_vertices-1);
    memcpy(&data[k_ind][offset], mst->edge_rel_vel, edge_vec_size);
    offset += 3*(mst->num_vertices-1);
    memcpy(&data[k_ind][offset], R->physical_state->spin,
           sizeof(double[R->num_pn_particles][3]));
}

static void comm_gbs_data(struct gbs_storage *gs, int *gbs_ks_for_this_group,
                          const struct ketju_system *R) {
    // TODO: can this be made more efficient? Do we need to?
    tick(&R->perf->time_gbscomm);
    // first gather all the data on the root (rank 0) task of the main comm from
    // the root tasks of the gbs groups
    double(*data)[gs->num_vars] = (double(*)[gs->num_vars])gs->data;
    int k_ind = 0;
    struct task_info *ti = R->internal_state->task_info;
    for (int i = gs->num_rows - 2; i >= 0; --i) {
        if (ti->main_group_rank == 0) {
            // Receive if we don't already have the data
            // TODO this will give some extra comms which could be avoided by
            // figuring out in which gbs group the main group root is
            if (ti->gbs_group_rank == 0 &&
                gbs_ks_for_this_group[k_ind] == i + 1) {
                k_ind += 1;
            } else {
                MPI_Status status;
                // figuring out where the data will come seems complicated, so
                // take it from any source with the right tag
                MPI_Recv(data[i], gs->num_vars, MPI_DOUBLE, MPI_ANY_SOURCE, i,
                         ti->main_comm, &status);
            }
        } else if (ti->gbs_group_rank == 0 &&
                   gbs_ks_for_this_group[k_ind] == i + 1) {
            k_ind += 1;
            MPI_Send(data[i], gs->num_vars, MPI_DOUBLE, 0, i, ti->main_comm);
        }
    }
    // Broadcast the collected data to all tasks
    MPI_Bcast(data, gs->num_vars * (gs->num_rows - 1), MPI_DOUBLE, 0,
              ti->main_comm);

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
// extrapolate a subset of variables on each
// task, comm the results
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

        if (Hnextfac > 5.0) { Hnextfac = 5.0; }
        *Hstep *= Hnextfac;
    } else {
        // TODO: this looks weird, should it be Hnextfac < 0.7 instead?
        // Or is it so that we want to reduce the stepsize more aggressively in
        // case of failed steps?
        if (Hnextfac > 0.7) { Hnextfac = 0.7; }
        *Hstep *= Hnextfac;
    }

    tock(&R->perf->time_gbscomm);

    return converged;
}

static void update_system_from_gbs_result(struct ketju_system *R,
                                                 struct gbs_storage *gs) {
    double *data = ((double(*)[gs->num_vars])gs->data)[gs->num_rows - 1];
    R->physical_state->time = data[0];
    R->internal_state->B = data[1];
    int offset = 2;
    struct spanning_tree_coordinate_system *mst = R->internal_state->mst;
    const size_t edge_vec_size = sizeof(double[mst->num_vertices-1][3]);
    memcpy(mst->edge_rel_pos, &data[offset], edge_vec_size);
    offset += 3*(mst->num_vertices-1);
    memcpy(mst->edge_rel_vel, &data[offset], edge_vec_size);
    offset += 3*(mst->num_vertices-1);
    memcpy(R->physical_state->spin, &data[offset],
           sizeof(double[R->num_pn_particles][3]));

    into_Cartesian_from_MST(R);
}

static void propagate_CoM(struct ketju_system *R, double dt) {
    for (int k = 0; k < 3; ++k) {
        R->CoM_pos[k] += dt * R->CoM_vel[k];
    }
}

// Merge BH (PN particle) j into i
static void merge_bhs(struct ketju_system *R, int i, int j) {
    assert(j < R->num_pn_particles);
    assert(j != i);
    if (j < i) {
        int temp = i;
        i = j;
        j = temp;
    }
    // Conserve total momentum, angular momentum and mass
    // TODO: GR effects on mass loss, spin etc.
    struct ketju_system_physical_state *ps = R->physical_state;
    double mi = ps->mass[i];
    double mj = ps->mass[j];
    double M = mi + mj;

    double rxv[3];
    double rr[3], vv[3];
    for (int k = 0; k < 3; ++k) {
        rr[k] = ps->pos[i][k] - ps->pos[j][k];
        vv[k] = ps->vel[i][k] - ps->vel[j][k];
    }
    vector_cross(rr, vv, rxv);
    for (int k = 0; k < 3; ++k) {
        ps->pos[i][k] = (ps->pos[i][k] * mi + ps->pos[j][k] * mj) / M;
        ps->vel[i][k] = (ps->vel[i][k] * mi + ps->vel[j][k] * mj) / M;
        ps->spin[i][k] += mi * mj / M * rxv[k] + ps->spin[j][k];
    }
    ps->mass[i] = M;

    // Shift everything right of particle j left by one place to overwrite
    // j's data.
    memmove(&ps->mass[j], &ps->mass[j + 1],
            sizeof(double) * (R->num_particles - (j + 1)));
    memmove(&ps->pos[j], &ps->pos[j + 1],
            sizeof(double[3]) * (R->num_particles - (j + 1)));
    memmove(&ps->vel[j], &ps->vel[j + 1],
            sizeof(double[3]) * (R->num_particles - (j + 1)));
    memmove(&ps->spin[j], &ps->spin[j + 1],
            sizeof(double[3]) * (R->num_pn_particles - (j + 1)));

    // The extra data of the merged particle is moved to the end of the array
    size_t elem_size = R->particle_extra_data_elem_size;
    char *extra_data = R->particle_extra_data;
    if (extra_data) {
        char rem_extra_data[elem_size];
        memmove(rem_extra_data, extra_data + j * elem_size, elem_size);
        memmove(extra_data + j * elem_size, extra_data + (j + 1) * elem_size,
                elem_size * (R->num_particles - (j + 1)));
        memmove(extra_data + (R->num_particles - 1) * elem_size, rem_extra_data,
                elem_size);
    }

    R->num_particles -= 1;
    R->num_pn_particles -= 1;
    // The mst is rebuilt after all the mergers are done, only need to change
    // the size for now.
    R->internal_state->mst->num_vertices -= 1;
}

static int do_bh_mergers(struct ketju_system *R) {
    int num_merged = 0;

    for (int i = 0; i < R->num_pn_particles; ++i) {
        for (int j = i + 1; j < R->num_pn_particles; ++j) {
            double r2 = 0;
            for (int k = 0; k < 3; ++k) {
                double dr =
                    R->physical_state->pos[i][k] - R->physical_state->pos[j][k];
                r2 += dr * dr;
            }
            // Simple condition: merge if particles within 6 times their
            // combined Schwarzschild radii.
            double fac =
                12 * R->constants->G / (R->constants->c * R->constants->c);
            if (sqrt(r2) < fac * (R->physical_state->mass[i] +
                                  R->physical_state->mass[j])) {
                merge_bhs(R, i, j);
                num_merged += 1;
            }
        }
    }

    return num_merged;
}

void ketju_run_integrator(struct ketju_system *R, double time_interval) {
    // TODO return status/error code

    // special case for when no actual integration is needed
    if (R->num_particles < 2) {
        tick(&R->perf->time_total);
        ketju_into_CoM_frame(R);
        propagate_CoM(R, time_interval);
        R->physical_state->time += time_interval;
        tock(&R->perf->time_total);
        return;
    }

    tick(&R->perf->time_total);
    // Initialize necessary comms if not done already by the caller
    if (!R->internal_state->task_info->gbs_comms_initialized) {
        ketju_init_gbs_comms(R);
    }
    // ensure correct CoM frame
    ketju_into_CoM_frame(R);

    ketju_build_mst(R);
    int steps_since_reconstruction = 0;

    compute_total_energy(R);

    double dt = time_interval/10; // TODO better guess for initial step
    double Hstep = dt * R->internal_state->U;
    const double t_start = R->physical_state->time;
    const double initial_B = R->internal_state->B;

    const int kmax = R->options->gbs_kmax;
    int gbs_ks_for_this_group[kmax];
    get_gbs_ks_for_this_group(R, gbs_ks_for_this_group);

    struct state_backup old_state;
    allocate_state_backup(&old_state, R->num_particles, R->num_pn_particles);
    store_state_backup(&old_state, R);

    struct gbs_storage gbs_storage;
    allocate_gbs_storage(&gbs_storage, R->num_particles, R->num_pn_particles,
                         R->options->gbs_kmax);

    bool finished = false;
    do {
        // Check mergers between BHs (PN particles)
        if (R->options->enable_bh_mergers) {
            int num_merged = do_bh_mergers(R);
            if (num_merged) {
                // We had mergers, need to redo tree and other things
                if (R->num_particles < 2) {
                    // Particle motion becomes trivial.
                    // Need to only propagate the CoM, this is done after the
                    // loop.
                    R->physical_state->time = t_start + time_interval;
                    break;
                }
                ketju_build_mst(R);
                steps_since_reconstruction = 0;
                compute_total_energy(R);
                store_state_backup(&old_state, R);
                gbs_storage.num_vars -= num_merged * 9; // pos, vel, spin
            }
        }

        for (int i = 0; i < kmax && gbs_ks_for_this_group[i]; ++i) {
            if (i) {
                // not needed on the first round
                restore_state_backup(&old_state, R);
            }
            int k = gbs_ks_for_this_group[i];
            mst_leapfrog(R, Hstep, num_gbs_substeps(k));
            store_gbs_data(&gbs_storage, R, k);
        }
        comm_gbs_data(&gbs_storage, gbs_ks_for_this_group, R);
        bool converged = gbs_extrapolate(&gbs_storage, &Hstep, R);

        if (converged) {
            update_system_from_gbs_result(R, &gbs_storage);
            R->perf->succesful_steps += 1;
            steps_since_reconstruction += 1;

            double remaining_time =
                time_interval - (R->physical_state->time - t_start);
            if (fabs(remaining_time / time_interval) <
                    R->options->output_time_relative_tolerance ||
                fabs(remaining_time) <
                    R->options->output_time_absolute_tolerance) {
                finished = true;
            } else {
                compute_total_energy(R);
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

    } while (!finished);

    propagate_CoM(R, R->physical_state->time - t_start);

    // free temp storage
    free_state_backup(&old_state);
    free_gbs_storage(&gbs_storage);
    
    R->perf->relative_energy_error = (R->internal_state->B-initial_B) /initial_B;
    tock(&R->perf->time_total);
}

// XXX this PN stuff probably is good to put in this file as well for now at
// least

// relative velocity of particle 1 seen from particle 2 = v_1 - v_2. return
// v_rel^2.
inline static double relative_velocity(const double *v1, const double *v2,
                                       double *v12) {
    double vrel2 = 0;
    for (int k = 0; k < 3; k++) {
        v12[k] = v1[k] - v2[k];
        vrel2 += v12[k] * v12[k];
    }
    return vrel2;
}

// Spin independent terms from Mora & Will (2004), Phys. Rev. D 69, 104021.
// Spin terms from Barker & O'Connel (1975), Phys. Rev D 12, 329 and
// Kidder (1995), Phys. rev. D 52, 821 (better notation).
// The quadrupole terms from Barker & O'connel have been specialized for black
// holes.
static void compute_PN_twobody_Acc(struct ketju_system *R,
                                   double (*vel)[3],
                                   double (*spin)[3], double (*AccPN)[3],
                                   double (*spin_derivative)[3], int i, int j) {

    struct ketju_system_internal_state *is = R->internal_state;
    struct spanning_tree_coordinate_system *mst = is->mst;
    struct ketju_system_physical_state *ps = R->physical_state;

    const double c = R->constants->c;
    const double c2 = c * c;
    const double c3 = c * c2;
    const double c4 = c2 * c2;
    const double c5 = c * c4;
    const double c6 = c3 * c3;
    const double c7 = c * c6;
    const double pi2 = M_PI * M_PI;

    double acc[3] = {0};

    // mass constants
    const double m1 = ps->mass[i];
    const double m2 = ps->mass[j];
    const double m = m1 + m2;
    const double eta = m1 * m2 / (m * m);
    const double eta2 = eta * eta;
    const double eta3 = eta2 * eta;
    const double G = R->constants->G;
    const double gm = G * m;

    // relative separation & velocity & unit separation
    double rv[3] = {0};
    double vv[3] = {0};
    double nv[3] = {0};

    // calculate relative quantities and derived quantities.
    // every relative quantity x = x_1 - x_2, where x_1 corresponds to
    // index i and x_2 to index j

    const int Nd = R->options->max_tree_distance;
    int path[Nd];
    int sign[Nd];
    int d;
    bool proximity = ketju_check_relative_proximity(i, j, Nd, R, &d, path, sign);

    if (proximity) {
        for (int k = 0; k < 3; k++) {
            for (int l = 0; l < d; l++) {
                int index = path[l];
                rv[k] += sign[l] * mst->edge_rel_pos[index][k];
            }
        }
    } else {
        for (int k = 0; k < 3; k++) {
            rv[k] = ps->pos[i][k] - ps->pos[j][k];
        }
    }

    // relative velocity is also v_1 - v_2
    const double v2 = relative_velocity(vel[i], vel[j], vv);

    double r2 = 0;
    // rdot = (rv dot vv) / r
    double rdot = 0;

    for (int k = 0; k < 3; k++) {
        rdot += rv[k] * vv[k];
        r2 += rv[k] * rv[k];
    }
    const double r = sqrt(r2);
    rdot /= r;
    const double r3 = r*r2;
    const double r4 = r2*r2;

    for (int k = 0; k < 3; k++) {
        nv[k] = rv[k] / r;
    }

    // powers of v, rdot and G(m1+m2)/r
    const double v4 = v2 * v2;
    const double v6 = v4 * v2;

    const double rdot2 = rdot * rdot;
    const double rdot4 = rdot2 * rdot2;
    const double rdot6 = rdot2 * rdot4;

    const double gmpr = gm / r;
    const double gmpr2 = gmpr * gmpr;
    const double gmpr3 = gmpr * gmpr2;

    // the PN acceleration coefficients
    double A1 = 0, A2 = 0, A2p5 = 0, A3 = 0, A3p5 = 0, Atot = 0;
    double B1 = 0, B2 = 0, B2p5 = 0, B3 = 0, B3p5 = 0, Btot = 0;

    // Spin contribution to acceleration
    double acc_spin[3] = {0, 0, 0};

    // Compute the enabled terms
    unsigned int flags = R->options->PN_flags;

    if (flags & KETJU_PN_1_0_ACC) {
        A1 = 2 * (2 + eta) * gmpr - (1 + 3 * eta) * v2 + 1.5e0 * eta * rdot2;
        B1 = 2 * (2 - eta) * rdot;
    }
    if (flags & KETJU_PN_2_0_ACC) {
        A2 = -(3. / 4) * (12 + 29 * eta) * gmpr2 - eta * (3 - 4 * eta) * v4 -
             (15. / 8) * eta * (1 - 3 * eta) * rdot4 +
             (1. / 2) * eta * (13 - 4 * eta) * gmpr * v2 +
             (2 + 25 * eta + 2 * eta2) * gmpr * rdot2 +
             (3. / 2) * eta * (3 - 4 * eta) * v2 * rdot2;
        B2 = -(1. / 2) * rdot *
             ((4 + 41 * eta + 8 * eta2) * gmpr - eta * (15 + 4 * eta) * v2 +
              3 * eta * (3 + 2 * eta) * rdot2);
    }

    if (flags & KETJU_PN_2_5_ACC) {
        A2p5 = 8. / 5 * eta * gmpr * rdot * (17. / 3 * gmpr + 3 * v2);
        B2p5 = -8. / 5. * eta * gmpr * (3 * gmpr + v2);
    }

    if (flags & KETJU_PN_3_0_ACC) {
        A3 = (16 + (1399. / 12 - 41. / 16 * pi2) * eta + 71. / 2 * eta2) *
                 gmpr3 +
             eta * (20827. / 840 + 123. / 64 * pi2 - eta2) * gmpr2 * v2 -
             (1 + (22717. / 168 + 615. / 64 * pi2) * eta + 11. / 8 * eta2 -
              7 * eta3) *
                 gmpr2 * rdot2 -
             .25 * eta * (11 - 49 * eta + 52 * eta2) * v6 +
             35. / 16 * eta * (1 - 5 * eta + 5 * eta2) * rdot6 -
             .25 * eta * (75 + 32 * eta - 40 * eta2) * gmpr * v2 * v2 -
             .5 * eta * (158 - 69 * eta - 60 * eta2) * gmpr * rdot4 +
             eta * (121 - 16 * eta - 20 * eta2) * gmpr * v2 * rdot2 +
             3. / 8 * eta * (20 - 79 * eta + 60 * eta2) * v2 * v2 * rdot2 -
             15. / 8 * eta * (4 - 18 * eta + 17 * eta2) * v2 * rdot4;
        B3 = rdot *
             ((4 + (5849. / 840. + 123. / 32. * pi2) * eta - 25 * eta2 -
               8 * eta3) *
                  gmpr2 +
              1. / 8. * eta * (65 - 152 * eta - 48 * eta2) * v2 * v2 +
              15 / 8. * eta * (3 - 8 * eta - 2 * eta2) * rdot4 +
              eta * (15 + 27 * eta + 10 * eta2) * gmpr * v2 -
              1. / 6. * eta * (329 + 177 * eta + 108 * eta2) * gmpr * rdot2 -
              .75 * eta * (16 - 37 * eta - 16 * eta2) * v2 * rdot2);
    }
    if (flags & KETJU_PN_3_5_ACC) {
        A3p5 = -8. / 5 * eta * gmpr * rdot *
               (23. / 14 * (43 + 14 * eta) * gmpr2 +
                3. / 28 * (61 + 70 * eta) * v2 * v2 + 70 * rdot4 +
                1. / 42 * (519 - 1267 * eta) * gmpr * v2 +
                .25 * (147 + 188 * eta) * gmpr * rdot2 -
                15 / 4. * (19 + 2 * eta) * v2 * rdot2);
        B3p5 = 8. / 5. * eta * gmpr *
               (1. / 42. * (1325 + 546 * eta) * gmpr2 +
                1. / 28. * (313 + 42 * eta) * v2 * v2 + 75 * rdot4 -
                1. / 42. * (205 + 777 * eta) * gmpr * v2 +
                1. / 12. * (205 + 424 * eta) * gmpr * rdot2 -
                .75 * (113 + 2 * eta) * v2 * rdot2);
    }

    if (flags & KETJU_PN_SPIN_ALL) {
        // Compute all the supplementary quantities if any of spin terms are
        // enabled
        double mu           = (m1*m2)/(m1+m2);
        double sigma[3]     = {0,0,0};      // m2/m1 * S1 + m1/m2 * S2
        double S[3]         = {0,0,0};      // S1 + S2
        double n_cross_v[3] = {0,0,0};
        double S1_2         = 0;
        double S2_2         = 0;
        double S1_dot_S2    = 0;
        double n_dot_S1     = 0;
        double n_dot_S2     = 0;

        const double *S_1 = spin[i];
        const double *S_2 = spin[j];


        // n x v
        vector_cross(nv, vv, n_cross_v);

        // supplementary quantities & dot products
        for (int k = 0; k < 3; k++) {
            S[k]        = S_1[k] + S_2[k];
            // sigma = S + dm/m*Delta in Kidder's notation
            sigma[k]    = m2/m1 * S_1[k] + m1/m2 * S_2[k];
            S1_dot_S2  += S_1[k] * S_2[k];
            n_dot_S1   += nv[k] * S_1[k];
            n_dot_S2   += nv[k] * S_2[k];
            S1_2       += S_1[k] * S_1[k];
            S2_2 += S_2[k] * S_2[k];
        }

        // spin contributions to acceleration
        if (flags & KETJU_PN_1_5_SPIN_ACC) {
            double A_SO_term1    = 0;
            double A_SO_term2[3] = {0,0,0};
            double A_SO_term3[3] = {0,0,0};
            double buffer1[3]    = {0,0,0};
            double buffer2[3]    = {0,0,0};
            for (int k = 0; k < 3; k++) {
                A_SO_term1 += 6 * n_cross_v[k] * (S[k] + sigma[k]);
                buffer1[k] = 4 * S[k] + 3 * sigma[k];
                buffer2[k] = 3 * rdot * (2 * S[k] + sigma[k]);
            }

            vector_cross(vv, buffer1, A_SO_term2);
            vector_cross(nv, buffer2, A_SO_term3);

            for (int k = 0; k < 3; k++) {
                // Spin-orbit term
                acc_spin[k] +=
                    G / (c2 * r3) *
                    (nv[k] * A_SO_term1 - A_SO_term2[k] + A_SO_term3[k]);
            }
        }

        if (flags & KETJU_PN_2_0_SPIN_ACC) {

            for (int k = 0; k < 3; ++k) {
                // Spin-spin term
                acc_spin[k] +=
                    -3 * G / (mu * c2 * r4) *
                    (S1_dot_S2 * nv[k] - 5 * n_dot_S1 * n_dot_S2 * nv[k] +
                     n_dot_S1 * S_2[k] + n_dot_S2 * S_1[k]);
                // Quadrupole term
                acc_spin[k] += -3. / 2 * G * m / (r4 * c2) *
                               ((1 / (m1 * m1) *
                                 ((S1_2 - 5 * n_dot_S1 * n_dot_S1) * nv[k] +
                                  2 * n_dot_S1 * S_1[k])) +
                                (1 / (m2 * m2) *
                                 ((S2_2 - 5 * n_dot_S2 * n_dot_S2) * nv[k] +
                                  2 * n_dot_S2 * S_2[k])));
            }
        }

        // Spin derivatives
        // The derivatives of the bodies are just index reversed versions of
        // each other.
        double buffer1[3]    = {0,0,0};
        double buffer2[3]    = {0,0,0};

        if (flags & KETJU_PN_1_0_SPIN_EVOL) {
            for (int k = 0; k < 3; ++k) {
                // Omega_SO
                buffer1[k] = 2 * G * mu / (c2 * r2) * (1 + 3 * m2 / (4 * m1)) *
                             n_cross_v[k];
                buffer2[k] = 2 * G * mu / (c2 * r2) * (1 + 3 * m1 / (4 * m2)) *
                             n_cross_v[k];
            }
        }

        if (flags & KETJU_PN_1_5_SPIN_EVOL) {
            for (int k = 0; k < 3; ++k) {
                // Omega_SS
                buffer1[k] += G / (c2 * r3) * (3 * n_dot_S2 * nv[k] - S_2[k]);
                buffer2[k] += G / (c2 * r3) * (3 * n_dot_S1 * nv[k] - S_1[k]);
                // Omega_Q
                buffer1[k] +=
                    G * m2 / (c2 * r3 * m1) * (3 * n_dot_S1 * nv[k] - S_1[k]);
                buffer2[k] +=
                    G * m1 / (c2 * r3 * m2) * (3 * n_dot_S2 * nv[k] - S_2[k]);
            }
        }
        double dSpin1[3];
        double dSpin2[3];
        vector_cross(buffer1, S_1, dSpin1);
        vector_cross(buffer2, S_2, dSpin2);
        for (int k=0; k<3; ++k) {
            spin_derivative[i][k] += dSpin1[k];   
            spin_derivative[j][k] += dSpin2[k];   
        }
    }
    
    // sum all terms with correct powers of c
    Atot = A1 / c2 + A2 / c4 + A2p5 / c5 + A3 / c6 + A3p5 / c7;
    Btot = B1 / c2 + B2 / c4 + B2p5 / c5 + B3 / c6 + B3p5 / c7;

    for (int k = 0; k < 3; k++) {
        acc[k] = gm / r2 * (nv[k] * Atot + vv[k] * Btot) + acc_spin[k];
    }

    // Divide up the acceleration assuming action = reaction.
    // XXX: This is not true for PN terms,
    // but is correct to 1PN (Mora & Will 2004), _for the CoM_.
    // Relative accelerations are correct, and the CoM motion corrections are
    // likely insignificant for our purposes.
    for (int k = 0; k < 3; k++) {
        AccPN[i][k] += (m2 / m) * acc[k];
        AccPN[j][k] -= (m1 / m) * acc[k];
    }
}

static void compute_Post_Newtonian_Acc(struct ketju_system *R,
                                       double (*vel_to_use)[3],
                                       double (*spin_to_use)[3]) {
    if (!R->options->PN_flags) return;

    double(*acc_pn)[3] = R->internal_state->acc_vd;
    memset(acc_pn, 0, sizeof(double[R->num_particles][3]));
    double(*spin_derivative)[3] = R->internal_state->spin_derivative;
    memset(spin_derivative, 0, sizeof(double[R->num_pn_particles][3]));

    // Compute the PN accelerations for SMBHs
    for (int i = 0; i < R->num_pn_particles; i++) {
        for (int j = i + 1; j < R->num_pn_particles; j++) {
            compute_PN_twobody_Acc(R, vel_to_use, spin_to_use, acc_pn,
                                   spin_derivative, i, j);
        }
    }

    // Finally the chained PN accelerations
    struct spanning_tree_coordinate_system *mst = R->internal_state->mst;
    int hi, lo;
    for (int i = 0; i < mst->num_edges; i++) {
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
            mst->edge_rel_acc_vd[i][k] = acc_pn[hi][k] - acc_pn[lo][k];
        }
    }
}
