// Header for internal implementation details

#ifndef KETJU_INTEGRATOR_INTERNAL_H
#define KETJU_INTEGRATOR_INTERNAL_H

#include <stddef.h>
#include <stdio.h>
#include <stdbool.h>

#include <mpi.h>

#include "ketju_integrator.h"

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

// TODO rename
struct octree_node {
    bool child_exists[8]; // TODO replace with checks for NULL?
    struct octree_node *children[8];
    struct octree_node *parent;
    double center[3];
    double half_size;
    int *object_list;
    int Npart;
    int level;
    bool is_leaf;
    int ID;
};

struct octree_data {
    struct octree_node *root_node, **particle_in_node;
    int **indexlist;
};

struct weight_index {
    double weight;
    int index;
};
struct weight_index_2 {
    double weight;
    int index1;
    int index2;
    int id1;
    int id2;
};

struct graph_edge {
    int id;
    int vertex1;
    int vertex2;
    float weight;
};

struct graph_vertex {
    int id;
    int degree;
    int type;
    int parent;
    int edgetoparent;
    int level;
    bool inMST;
    int cell[3];
};

struct spanning_tree_coordinate_system {
    int num_vertices;
    int num_edges;    

    struct graph_vertex *vertex;

    // TODO check if some of these can be removed
    struct graph_edge *local_edge, *local_edge_subset, *edge_in_mst;

    double (*edge_rel_pos)[3];
    double (*edge_rel_vel)[3];
    double (*edge_rel_aux_vel)[3];
    double (*edge_rel_acc)[3];
    double (*edge_rel_acc_vd)[3];
};

struct group_info {
    int num_gbs_groups;
    int *gbs_group_sizes;
};

// info on MPI groups this task is in
struct task_info {
    // main task group = all tasks participating in the integration
    int main_group_size;
    int main_group_rank;
    MPI_Comm main_comm;
    
    // gbs task group = tasks participating in a given set of substep levels
    // i.e. force calculations done in this group
    // TODO maybe could be named as force_group, since mostly appears in force
    // calculations
    int gbs_group_size;
    int gbs_group_rank;
    int gbs_group_index; //< which of the gbs groups this task belongs in
    MPI_Comm gbs_comm;
    bool gbs_comms_initialized;
    
    int global_rank; // <rank in MPI_COMM_WORLD
};

// A wrapper struct for all the internal state needed by the integrator
struct ketju_system_internal_state {

    struct task_info *task_info;
    struct group_info *group_info;

    struct spanning_tree_coordinate_system *mst;
    struct octree_data *octree_data;

    double U; // potential energy
    double T;
    double B;

    double (*aux_vel)[3]; 
    double (*acc)[3]; // velocity independent acceleration
    double (*acc_vd)[3]; // velocity dependent acceleration (for all particles at least for now)
};


// Small helper functions.
// Inline definitions allow using them in different
// compilation units without introducing extra public symbols.

// timer functions for basic profiling, used as
//   tick(timer)
//   ... code ...
//   tock(timer)
// the value of *timer is not meaningful between the calls.
//
static inline void tick(double *timer) {
    *timer -= MPI_Wtime();   
}
static inline void tock(double *timer) {
    *timer += MPI_Wtime();   
}

// for exiting on catastrophic errors
// TODO most uses now could be replaced with error codes allowing potentially
// recovering from errors during integration / saving the state so far.
static inline void terminate(const char *msg) {
    printf("%s\n", msg);
    fflush(stdout);
    MPI_Abort(MPI_COMM_WORLD, 1);
}

static inline void get_v1_v2_fast(struct ketju_system *R, int i, int *v1,
                                  int *v2) {
    *v1 = R->internal_state->mst->edge_in_mst[i].vertex1;
    *v2 = R->internal_state->mst->edge_in_mst[i].vertex2;
}

// declarations of internal functions used between compilation units


int ketju_check_relative_proximity(int v1, int v2, const int Nd,
                             const struct ketju_system *R, int *d, int *path,
                             int *sign);
int ketju_check_relative_proximity_ND_2(int v1, int v2, const struct ketju_system *R,
                                  int *d, int *path, int *sign);

void ketju_init_gbs_comms(struct ketju_system *R);

void ketju_build_mst(struct ketju_system *R);

// TODO rename these
void loop_scheduling_N2(int Nrow, int Nt, int *istart, int *jstart,
                        int *ThisBlock, int *CumNum, int task);

#endif // KETJU_INTEGRATOR_INTERNAL_H
