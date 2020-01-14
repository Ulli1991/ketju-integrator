// Main integration and force calculation functions

#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "ketju_integrator/ketju_integrator.h"
#include "ketju_integrator/internal.h"

// This is needed only in this file
enum velocity_type { PHYSICAL, AUXILIARY };

void loop_scheduling_N2(int Nrow, int Nt, int *istart, int *jstart,
                        int *ThisBlock, int *CumNum, int task) {
    int Np = (Nrow * (Nrow - 1)) / 2;
    int block = (int)floor(Np / Nt);
    int limit = (block + 1) * Nt - Np;
    int ThisCumNum;
    if (task < limit) {
        ThisCumNum = task * block;
        *ThisBlock = block;
    } else {
        ThisCumNum = limit * block + (task - limit) * (block + 1);
        *ThisBlock = block + 1;
    }

    double drow =
        1.0 * Nrow - 0.5 -
        0.5 * sqrt((1.0 - 2.0 * Nrow) * (1.0 - 2.0 * Nrow) - 8.0 * ThisCumNum);
    int row = (int)floor(drow);
    int rowstart =
        (int)round(-0.5 * (1.0 * row + 1.0) * (row - 2.0 * Nrow) - 1.0 * Nrow);
    *istart = row;
    *jstart = ThisCumNum - rowstart + row + 1;
    *CumNum = ThisCumNum;
}

// computes the force function for logh
void compute_U(struct ketju_system *R) {

    tick(&R->perf->time_force);
    
    struct ketju_system_internal_state *is = R->internal_state;
    struct spanning_tree_coordinate_system *mst = is->mst;
    struct ketju_system_physical_state *ps = R->physical_state;

    is->U = 0;
    int istart = 0;
    int CumNumEdge, ThisBlock, jstart;
    loop_scheduling_N2(mst->num_vertex, is->task_info->gbs_group_size,
                       &istart, &jstart, &ThisBlock, &CumNumEdge,
                       is->task_info->gbs_group_rank);
    int loop = 1;
    int c = 0;

    const int Nd = R->options->max_tree_distance;
    int d;
    int path[2];
    int sign[2];

    for (int i = istart; i < mst->num_vertex; i++) {

        double mi = ps->mass[i];
        int Li = mst->vertex[i].level;
        if (i > istart) { jstart = i + 1; }
        for (int j = jstart; j < mst->num_vertex; j++) {
            double mj = ps->mass[j];
            int Lj = mst->vertex[j].level;
            double r2 = 0;
            int proximity = 0;

            if (fabs(Li - Lj) < Nd) {
                proximity =
                    ketju_check_relative_proximity(j, i, Nd, R, &d, path, sign);
            }

            if (proximity == 1) {
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

            const double invr = 1.0 / sqrt(r2);
            is->U += mi * mj * invr;
            c++;
            if (c == ThisBlock) {
                loop = 0;
                break;
            }
        }
        if (loop == 0) break;
    }

    tick(&R->perf->time_comm);
    if (is->task_info->gbs_group_size > 1) {
        MPI_Allreduce(MPI_IN_PLACE, &is->U, 1, MPI_DOUBLE, MPI_SUM,
                      is->task_info->gbs_comm);
    }
    tock(&R->perf->time_comm);

    if (!isfinite(is->U)) {
        terminate("U is not finite!");
    }

    is->U *= R->constants->G;

    tock(&R->perf->time_force);
}

// computes the force function and newtonian acceleration
void compute_U_and_Newtonian_Acc(struct ketju_system *R) {

    tick(&R->perf->time_force);

    struct ketju_system_internal_state *is = R->internal_state;
    struct spanning_tree_coordinate_system *mst = is->mst;
    struct ketju_system_physical_state *ps = R->physical_state;


    int istart, istop = mst->num_vertex, jstop = mst->num_vertex;
    int ThisBlock, CumNumEdge, jstart;
    loop_scheduling_N2(mst->num_vertex, is->task_info->gbs_group_size, &istart,
                       &jstart, &ThisBlock, &CumNumEdge,
                       is->task_info->gbs_group_rank);
    int loop = 1;
    int c = 0;
    const int Nd = R->options->max_tree_distance;
    int d;
    int path[Nd];
    int sign[Nd];

    memset(is->acc, 0, sizeof(double[mst->num_vertex][3]));

    is->U = 0;
    double dr[3], r2;


    for (int i = istart; i < istop; i++) {
        const double mi = ps->mass[i];
        const int Li = mst->vertex[i].level;
        if (i > istart) jstart = i + 1;

        for (int j = jstart; j < jstop; j++) {
            r2 = 0;
            const double mj = ps->mass[j];
            const int Lj = mst->vertex[j].level;

            int proximity = 0;
            if (abs(Li - Lj) <= Nd) {
                proximity =
                    ketju_check_relative_proximity_ND_2(j, i, R, &d, path, sign);
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

            const double invr = 1.0 / sqrt(r2);
            is->U += mi * mj * invr;

            const double invr3 = invr * invr * invr;

            for (int k = 0; k < 3; k++) {
                double drinvr3 = dr[k] * invr3;
                is->acc[i][k] += mj * drinvr3;
                is->acc[j][k] += -mi * drinvr3;
            }
            c++;
            if (c == ThisBlock) {
                loop = 0;
                break;
            }
        }
        if (loop == 0) break;
    }

    tick(&R->perf->time_comm);
    if (is->task_info->gbs_group_size > 1) {
        MPI_Allreduce(MPI_IN_PLACE, is->acc, 3 * mst->num_vertex, MPI_DOUBLE,
                      MPI_SUM, is->task_info->gbs_comm);
        MPI_Allreduce(MPI_IN_PLACE, &is->U, 1, MPI_DOUBLE, MPI_SUM,
                      is->task_info->gbs_comm);
    }
    tock(&R->perf->time_comm);

    int hi, lo;
    for (int i = 0; i < mst->num_vertex - 1; i++) {

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


// TODO this could maybe go in mst.c
void into_Cartesian_from_MST(struct ketju_system *R) {

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
    for (int i = 0; i < mst->num_vertex - 1; i++) {

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

    // TODO maybe we do not want to force the CoM to stay at the origin, e.g. if
    // there are some extra forces
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


// puts the initial data to the CoM frame and sets the initial CoM pos and vel
void into_CoM_frame(struct ketju_system *R) {

    tick(&R->perf->time_coord);
    struct ketju_system_physical_state *ps = R->physical_state;

    for (int k = 0; k < 3; k++) {
        R->initial_CoM_pos[k] = 0;
        R->initial_CoM_vel[k] = 0;
    }
    double M = 0;
    for (int i = 0; i < R->num_particles; i++) {
        M += ps->mass[i];
        for (int k = 0; k < 3; k++) {
            R->initial_CoM_pos[k] += ps->mass[i] * ps->pos[i][k];
            R->initial_CoM_vel[k] += ps->mass[i] * ps->vel[i][k];
        }
    }
    for (int k = 0; k < 3; k++) {
        R->initial_CoM_pos[k] /= M;
        R->initial_CoM_vel[k] /= M;
    }
    for (int i = 0; i < R->num_particles; i++) {
        for (int k = 0; k < 3; k++) {
            ps->pos[i][k] -= R->initial_CoM_pos[k];
            ps->vel[i][k] -= R->initial_CoM_vel[k];
        }
    }

    tock(&R->perf->time_coord);
}

void compute_T(struct ketju_system *R) {
    struct ketju_system_internal_state *is = R->internal_state;
    struct ketju_system_physical_state *ps = R->physical_state;
    is->T = 0;
    for (int i = 0; i < R->num_particles; i++) {
        double dv, v2 = 0;
        for (int k = 0; k < 3; k++) {
            dv = ps->vel[i][k];
            v2 += dv * dv;
        }
        is->T += 0.5 * ps->mass[i] * v2;
    }
}

void update_pos(struct ketju_system *R) {

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
    for (int i = 0; i < mst->num_vertex - 1; i++) {

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

    // TODO: maybe we don't want this forcing CoM to the origin constantly?
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
void update_vel(struct ketju_system *R, double (*vel_to_update)[3],
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
    for (int i = 0; i < mst->num_vertex - 1; i++) {

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

    // TODO: maybe we don't want this forcing CoM to the origin constantly?
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

void drift(double ds, struct ketju_system *R) {

    struct ketju_system_internal_state *is = R->internal_state;
    struct spanning_tree_coordinate_system *mst = is->mst;
    struct ketju_system_physical_state *ps = R->physical_state;

    compute_T(R);
    double dt = ds / (is->T + is->B); //TODO: isn't this just is->U?
    ps->time += dt;

    for (int i = 0; i < mst->num_vertex - 1; i++) {
        for (int k = 0; k < 3; k++) {
            mst->edge_rel_pos[i][k] += dt * mst->edge_rel_vel[i][k];
        }
    }
    update_pos(R);
}


void set_auxiliary_variables(struct ketju_system *R) {

    struct ketju_system_internal_state *is = R->internal_state;
    struct spanning_tree_coordinate_system *mst = is->mst;
    struct ketju_system_physical_state *ps = R->physical_state;

    memcpy(is->aux_vel, ps->vel, sizeof(double[R->num_particles][3]));
    memcpy(mst->edge_rel_aux_vel, mst->edge_rel_vel,
           sizeof(double[mst->num_vertex][3]));
}

// TODO: move declarations / reorganize
void compute_Post_Newtonian_Acc(struct ketju_system *, double (*vel_to_use)[3]);

void physical_kick(double dt, struct ketju_system *R) {

    struct ketju_system_internal_state *is = R->internal_state;
    struct spanning_tree_coordinate_system *mst = is->mst;
    struct ketju_system_physical_state *ps = R->physical_state;

    update_vel(R, is->aux_vel, AUXILIARY);
    compute_Post_Newtonian_Acc(R, is->aux_vel);

    double dB = 0;
    for (int i = 0; i < R->num_particles; i++) {
        for (int k = 0; k < 3; k++) {
            dB += ps->mass[i] * is->aux_vel[i][k] * is->acc_vd[i][k];
        }
    }
    is->B -= dB * dt;

    for (int i = 0; i < mst->num_vertex - 1; i++) {
        for (int k = 0; k < 3; k++) {
            mst->edge_rel_vel[i][k] +=
                dt * (mst->edge_rel_acc[i][k] + mst->edge_rel_acc_vd[i][k]);
        }
    }
}

void auxiliary_kick(double dt, struct ketju_system *R) {

    struct ketju_system_internal_state *is = R->internal_state;
    struct spanning_tree_coordinate_system *mst = is->mst;
    struct ketju_system_physical_state *ps = R->physical_state;

    update_vel(R, ps->vel, PHYSICAL);
    compute_Post_Newtonian_Acc(R, ps->vel);

    for (int i = 0; i < mst->num_vertex - 1; i++) {
        for (int k = 0; k < 3; k++) {
            mst->edge_rel_aux_vel[i][k] +=
                dt * (mst->edge_rel_acc[i][k] + mst->edge_rel_acc_vd[i][k]);
        }
    }
}

void kick(double ds, struct ketju_system *R) {

    struct ketju_system_internal_state *is = R->internal_state;
    struct spanning_tree_coordinate_system *mst = is->mst;
    struct ketju_system_physical_state *ps = R->physical_state;

    compute_U_and_Newtonian_Acc(R);
    double dt = ds / is->U;

    if (!R->options->PN_flags) {
        // no velocity dependent forces (only PN for now), just the newtonian
        // part needed
        for (int i = 0; i < mst->num_vertex - 1; i++) {
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

double get_n_double(int i) { return (2.0 * (i + 1.0)); }
int get_n_int(int i) { return (2.0 * (i + 1.0)); }

double get_error(double a, double b) { return fabs(b - a); }

int gbs_group_with_min_workload(int *GbsGroupWorkLoad, int num_gbs_group) {
    int min_workload = INT_MAX;
    int index = 0;
    for (int i = 0; i < num_gbs_group; i++) {
        if (GbsGroupWorkLoad[i] < min_workload) {
            min_workload = GbsGroupWorkLoad[i];
            index = i;
        }
    }
    return index;
}

// TODO port this
#if 0
void divide_computational_load(struct ketju_system *R) {

    int *GbsGroupWorkLoad = calloc(NumGbsGroup, sizeof(int));
    int counter = 0;

    const int KMAX = 8; // TODO make configurable
    for (int i = KMAX - 1; i >= 0; i--) {
        int ThisSubdivisionCost = get_n_int(i);
        int GroupIndex = gbs_group_with_min_workload(GbsGroupWorkLoad);
        GbsGroupWorkLoad[GroupIndex] += ThisSubdivisionCost;
        ComputationToDoList.ComputationalTask[i].GroupToPerformTask =
            GroupIndex;
        ComputationToDoList.ComputationalTask[i].ThisR = R;
        ComputationToDoList.ComputationalTask[i].NumberOfSubsteps =
            get_n_int(i);
        counter++;
    }

    ComputationToDoList.NumberOfComputationalTasks = counter;

    free(GbsGroupWorkLoad);
}
#endif

void compute_total_energy(struct ketju_system *R) {

    struct ketju_system_internal_state *is = R->internal_state;

    compute_T(R);
    compute_U(R);
    is->H = is->T - is->U;
    is->B = -is->H;
    // TODO is there some reason that the B variable and the B value in the old
    // state are different like below?
    //R->State[R->BIndex] = R->B;
}

void mst_leapfrog(struct ketju_system *R, double Hstep,
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

// TODO this section
#if 0

// TODO we only want to copy the physical and internal state for temp variables
void copy_regularized_region(struct ketju_system *Orig,
                             struct ketju_system *Copy) {

    Copy->gbs_tolerance = Orig->gbs_tolerance;
    Copy->output_time_tolerance = Orig->output_time_tolerance;
    Copy->NumVertex = Orig->NumVertex;
    Copy->NumEdge = Orig->NumEdge;
    Copy->NumDynVariables = Orig->NumDynVariables;
    Copy->TimeIndex = Orig->TimeIndex;
    Copy->BIndex = Orig->BIndex;
    Copy->Hstep = Orig->Hstep;

    for (int j = 0; j < Orig->NumVertex; j++) {
        Copy->Vertex[j].id = Orig->Vertex[j].id;
        Copy->Vertex[j].degree = Orig->Vertex[j].degree;
        Copy->Vertex[j].type = Orig->Vertex[j].type;
        Copy->Vertex[j].parent = Orig->Vertex[j].parent;
        Copy->Vertex[j].edgetoparent = Orig->Vertex[j].edgetoparent;
        Copy->Vertex[j].level = Orig->Vertex[j].level;
        Copy->Vertex[j].inMST = Orig->Vertex[j].inMST;
    }
    for (int j = 0; j < Orig->NumVertex - 1; j++) {
        Copy->EdgeInMST[j].vertex1 = Orig->EdgeInMST[j].vertex1;
        Copy->EdgeInMST[j].vertex2 = Orig->EdgeInMST[j].vertex2;
    }

    for (int j = 0; j < Orig->NumDynVariables; j++) {
        Copy->State[j] = Orig->State[j];
    }
    for (int j = 0; j < 3 * Orig->NumVertex; j++) {
        Copy->Pos[j] = Orig->Pos[j];
        Copy->Vel[j] = Orig->Vel[j];
        Copy->Acc[j] = Orig->Acc[j];
        Copy->AuxVel[j] = Orig->AuxVel[j];
        Copy->AccPN[j] = Orig->AccPN[j];
    }
    for (int j = 0; j < 3 * (Orig->NumVertex - 1); j++) {
        Copy->MSTedgeAcc[j] = Orig->MSTedgeAcc[j];
        Copy->AuxEdgeVel[j] = Orig->AuxEdgeVel[j];
        Copy->MSTedgeAcc_PN[j] = Orig->MSTedgeAcc_PN[j];
    }
    for (int j = 0; j < Orig->NumVertex; j++) {
        Copy->Mass[j] = Orig->Mass[j];
    }
}

void extrapolate_single_variable(double *y, const int N, double *result,
                                 double *error, double *yscal) {

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
    for (int i = 1; i < N; i++) {
        result[i - 1] = gbsS[0][i];
        error[i - 1] = fabs(gbsS[0][i] - gbsS[0][i - 1]) / yscal[i];
    }
}

int extrapolate_all_variables(int korder) {

    int status = -1;

    struct ketju_system *ThisR = NULL;
    int *SubStepDivision = calloc(korder, sizeof(int));

    int counter = 0;
    int NumDynVariables = 0;
    for (int i = 0; i < korder; i++) {
        ThisR = ComputationToDoList.ComputationalTask[i].ThisR;
        NumDynVariables = ThisR->NumDynVariables;
        SubStepDivision[counter] = i;
        counter++;
    }

    int block = (int)floor(NumDynVariables / Ntask);
    int loop_start = block * ThisTask;
    int loop_end = block * (ThisTask + 1);
    if (ThisTask == Ntask - 1) { loop_end = NumDynVariables; }

    for (int v = loop_start; v < loop_end; v++) {
        for (int i = 0; i < korder; i++) {
            y[i] = ComputationToDoList.CopyOfSingleRegion[SubStepDivision[i]]
                       .State[v];
            yscal[i] = fabs(y[i]);
        }
        extrapolate_single_variable(y, korder, result[v], error[v], yscal);
    }

    for (int i = 0; i < NumDynVariables; i++) {
        errorbuf[i] = 0;
        resultbuf[i] = 0;
    }

    for (int i = loop_start; i < loop_end; i++) {
        errorbuf[i] = error[i][korder - 2];
        resultbuf[i] = result[i][korder - 2];
    }

#ifdef PARALLEL
    MPI_Allreduce(MPI_IN_PLACE, errorbuf, NumDynVariables, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, resultbuf, NumDynVariables, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
#endif

    double gbs_error = -1;
    double gbs_tolerance =
        ComputationToDoList.ComputationalTask[SubStepDivision[0]]
            .ThisR->gbs_tolerance;

    for (int j = 0; j < NumDynVariables; j++) {
        if (errorbuf[j] >= gbs_error) { gbs_error = errorbuf[j]; }
    }

    double Hnextfac =
        0.94 * pow(0.65 / (gbs_error / gbs_tolerance), 1.0 / (2 * korder + 1));

    if (gbs_error <= gbs_tolerance) {

        status = 1;

        for (int v = 0; v < NumDynVariables; v++) {
            ThisR->State[v] = resultbuf[v];
        }
        if (Hnextfac > 5.0) { Hnextfac = 5.0; }

        ThisR->Hstep *= Hnextfac;
        into_Cartesian_from_MST(ThisR);

        ok_steps++;

    } else {
        status = 0;
        failed_steps++;

        if (Hnextfac > 0.7) { Hnextfac = 0.7; }
        ThisR->Hstep *= Hnextfac;
    }
    free(SubStepDivision);

    return status;
}

// Integrate the subsystems using AR-MST for the desired time interval
void run_integrator(struct ketju_system *R, double time_interval) {

    // Needed variables
    double dt = 1e-7;
    double Hstep;

    // Initialize the system to integrate

    into_CoM_frame(R);

#ifdef PRIM
    PrimMST(R);
#else
    DivideAndConquerPrim(R);
#endif

    compute_total_energy(R);
    R->Hstep = dt * R->U;
    R->State[R->TimeIndex] = 0;

    double percent = time_interval / 50;
    int epoch = -1;

    // Now proceed to integration
    int not_finished = 1;
    int redo = 1;
    int status;

    do {

        do {

            redo = 0;

            // Divide the systems and substep divisions for tasks
            divide_computational_load(R);

            // Perform the leapfrog tasks
            for (int ctask = 0;
                 ctask < ComputationToDoList.NumberOfComputationalTasks;
                 ctask++) {
                if (ThisGbsGroup == ComputationToDoList.ComputationalTask[ctask]
                                        .GroupToPerformTask) {

                    // copy the system in order not to overwrite the original
                    // one
                    struct ketju_system *ThisR =
                        ComputationToDoList.ComputationalTask[ctask].ThisR;

                    copy_regularized_region(
                        ThisR, &ComputationToDoList.CopyOfSingleRegion[ctask]);

                    // leapfrog the substep division
                    Hstep = ComputationToDoList.CopyOfSingleRegion[ctask].Hstep;
                    mst_leapfrog(&ComputationToDoList.CopyOfSingleRegion[ctask],
                                 Hstep,
                                 ComputationToDoList.ComputationalTask[ctask]
                                     .NumberOfSubsteps);
                }
            }

        } while (redo);

        my_barrier();
        clock_t aika0 = clock(), aika1;

#ifdef PARALLEL
        // We need to communicate the results
        for (int ctask = 0;
             ctask < ComputationToDoList.NumberOfComputationalTasks; ctask++) {
            int root = ComputationToDoList.ComputationalTask[ctask]
                           .GroupToPerformTask *
                       NumTaskPerGbsGroup;
            double *commbuf =
                ComputationToDoList.CopyOfSingleRegion[ctask].State;

            int count =
                ComputationToDoList.CopyOfSingleRegion[ctask].NumDynVariables;
            if (ThisTask != root) {
                for (int i = 0; i < count; i++) {
                    commbuf[i] = 0;
                }
            }
            MPI_Bcast(commbuf, count, MPI_DOUBLE, root, MPI_COMM_WORLD);
        }
        aika1 = clock();

        time_gbscomm += (double)(aika1 - aika0) / CLOCKS_PER_SEC;
#endif

        // Now preform the GBS extrapolation, check convergence and obtain next
        // timestep

        my_barrier();
        aika0 = clock();
        status = extrapolate_all_variables(KMAX);
        aika1 = clock();
        time_gbs += (double)(aika1 - aika0) / CLOCKS_PER_SEC;

        if (status == 1) {

            // Check whether we are ready
            double time = R[0].State[R[0].TimeIndex];
            if (fabs(time - time_interval) / time_interval <
                R->output_time_tolerance) {
                not_finished = 0;
            } else {

                compute_total_energy(R);
                if (time > time_interval) {
                    R->Hstep = -R->U * (time - time_interval);
                } else if (R->Hstep / R->U + time > time_interval) {
                    R->Hstep = R->U * (time_interval - time);
                }

                int old_epoch = epoch;
                epoch = (int)floor(time / percent);
                if (epoch != old_epoch) {
#ifdef PRIM
                    PrimMST(&R[0]);
#else
                    DivideAndConquerPrim(&R[0]);
#endif
                }
            }
        }

    } while (not_finished);
}

#endif


// XXX this PN stuff probably is good to put in this file as well for now at
// least

// relative velocity of particle 1 seen from particle 2 = v_1 - v_2. return
// v_rel^2.
inline static double relative_velocity(double *v1, double *v2, double *v12) {
    double vrel2 = 0;
    for (int k = 0; k < 3; k++) {
        v12[k] = v1[k] - v2[k];
        vrel2 += v12[k] * v12[k];
    }
    return vrel2;
}

// Based on Mora & Will, Phys. Rev. D 69, 104021
// TODO add spin terms
void compute_PN_twobody_Acc(struct ketju_system *R, double (*Vel)[3],
                            double (*AccPN)[3], int i, int j) {

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
    double velvec_i[3], velvec_j[3];
    for (int k = 0; k < 3; k++) {
        velvec_i[k] = Vel[i][k];
        velvec_j[k] = Vel[j][k];
    }
    double rv[3] = {0};
    double vv[3] = {0};
    double nv[3] = {0};
    // rdot = (rv dot vv) / r
    double rdot = 0;
    // powers of r, v, rdot and G(m1+m2)/r
    double r2 = 0, r;
    double rdot2, rdot4, rdot6;
    double v2 = 0, v4 = 0, v6 = 0;
    double gmpr = 0, gmpr2 = 0, gmpr3 = 0;

    // the PN acceleration coefficients
    double A1 = 0, A2 = 0, A2p5 = 0, A3 = 0, A3p5 = 0, Atot = 0;
    double B1 = 0, B2 = 0, B2p5 = 0, B3 = 0, B3p5 = 0, Btot = 0;

    int k;

    // calculate relative quantities and derived quantities.
    // every relative quantity x = x_1 - x_2, where x_1 corresponds to
    // index i and x_2 to index j

    const int Nd = R->options->max_tree_distance;
    int path[Nd];
    int sign[Nd];
    int d;
    int proximity = ketju_check_relative_proximity(i, j, Nd, R, &d, path, sign);

    for (k = 0; k < 3; k++) {
        if (proximity == 1) {
            for (int l = 0; l < d; l++) {
                int index = path[l];
                rv[k] += sign[l] * mst->edge_rel_pos[index][k];
            }
        } else {
            rv[k] = ps->pos[i][k] - ps->pos[j][k];
        }
    }

    // relative velocity is also v_1 - v_2
    v2 = relative_velocity(velvec_i, velvec_j, vv);

    for (k = 0; k < 3; k++) {
        rdot += rv[k] * vv[k];
        r2 += rv[k] * rv[k];
    }
    r = sqrt(r2);

    rdot /= r;
    rdot2 = rdot * rdot;
    rdot4 = rdot2 * rdot2;
    rdot6 = rdot2 * rdot4;

    v4 = v2 * v2;
    v6 = v4 * v2;

    gmpr = gm / r;
    gmpr2 = gmpr * gmpr;
    gmpr3 = gmpr * gmpr2;

    for (k = 0; k < 3; k++) {
        nv[k] = rv[k] / r;
    }

    A1 = 2 * (2 + eta) * gmpr - (1 + 3 * eta) * v2 + 1.5e0 * eta * rdot2;
    B1 = 2 * (2 - eta) * rdot;
    A2 = -(3. / 4) * (12 + 29 * eta) * gmpr2 - eta * (3 - 4 * eta) * v4 -
         (15. / 8) * eta * (1 - 3 * eta) * rdot4 +
         (1. / 2) * eta * (13 - 4 * eta) * gmpr * v2 +
         (2 + 25 * eta + 2 * eta2) * gmpr * rdot2 +
         (3. / 2) * eta * (3 - 4 * eta) * v2 * rdot2;
    B2 = -(1. / 2) * rdot *
         ((4 + 41 * eta + 8 * eta2) * gmpr - eta * (15 + 4 * eta) * v2 +
          3 * eta * (3 + 2 * eta) * rdot2);

    A2p5 = 8. / 5 * eta * gmpr * rdot * (17. / 3 * gmpr + 3 * v2);
    B2p5 = -8. / 5. * eta * gmpr * (3 * gmpr + v2);

    A3 = (16 + (1399. / 12 - 41. / 16 * pi2) * eta + 71. / 2 * eta2) * gmpr3 +
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
         ((4 + (5849. / 840. + 123. / 32. * pi2) * eta - 25 * eta2 - 8 * eta3) *
              gmpr2 +
          1. / 8. * eta * (65 - 152 * eta - 48 * eta2) * v2 * v2 +
          15 / 8. * eta * (3 - 8 * eta - 2 * eta2) * rdot4 +
          eta * (15 + 27 * eta + 10 * eta2) * gmpr * v2 -
          1. / 6. * eta * (329 + 177 * eta + 108 * eta2) * gmpr * rdot2 -
          .75 * eta * (16 - 37 * eta - 16 * eta2) * v2 * rdot2);
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

    // sum all terms with correct powers of c
    Atot = A1 / c2 + A2 / c4 + A2p5 / c5 + A3 / c6 + A3p5 / c7;
    Btot = B1 / c2 + B2 / c4 + B2p5 / c5 + B3 / c6 + B3p5 / c7;

    for (k = 0; k < 3; k++) {
        acc[k] = gm / r2 * (nv[k] * Atot + vv[k] * Btot);
    }

    for (k = 0; k < 3; k++) {
        AccPN[i][k] += (m2 / m) * acc[k];
        AccPN[j][k] -= (m1 / m) * acc[k];
    }
}

// TODO
// The plan is to go with all the PN particles at the start of the particle
// arrays, so that this can be simplified somewhat
#if 0
void compute_Post_Newtonian_Acc(struct ketju_system *R, double *Vel) {

    // Init
    for (int i = 0; i < 3 * R->NumVertex; i++) {
        R->AccPN[i] = 0;
    }

    return;

    // Find the SMBHs and their indexes
    int Nbh = 0;
    for (int i = 0; i < R->NumVertex; i++) {
        if (R->Vertex[i].type == 5) { Nbh++; }
    }
    if (Nbh < 2) { return; }

    int c = 0;
    int bhindex[Nbh];
    for (int i = 0; i < R->NumVertex; i++) {
        if (R->Vertex[i].type == 5) {
            bhindex[c] = i;
            c++;
        }
    }

    // Compute the PN accelerations for SMBHs

    for (int i = 0; i < Nbh; i++) {
        for (int j = i + 1; j < Nbh; j++) {
            compute_PN_twobody_Acc(R, Vel, R->AccPN, bhindex[i], bhindex[j]);
        }
    }

    // Finally the chained PN accelerations

    int hi, lo;
    for (int i = 0; i < R->NumVertex - 1; i++) {
        int index = R->MSTedgeList[i];
        int v1 = R->Edge[index].vertex1;
        int v2 = R->Edge[index].vertex2;
        if (R->Vertex[v1].level > R->Vertex[v2].level) {
            hi = v1;
            lo = v2;
        } else {
            hi = v2;
            lo = v1;
        }
        for (int k = 0; k < 3; k++) {
            R->MSTedgeAcc_PN[3 * i + k] =
                R->AccPN[3 * hi + k] - R->AccPN[3 * lo + k];
        }
    }
}
#endif
