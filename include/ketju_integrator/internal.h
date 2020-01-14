// Header for internal implementation details

#ifndef KETJU_INTEGRATOR_INTERNAL_H
#define KETJU_INTEGRATOR_INTERNAL_H

#include <stddef.h>
#include <stdio.h>

#include <mpi.h>

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

// TODO rename
struct OctreeNode {
    int ChildExists[8];
    struct OctreeNode *Children[8];
    struct OctreeNode *Parent;
    double Center[3];
    double HalfSize;
    int *ObjectList;
    int Npart;
    int level;
    int IsLeaf;
    int ID;
} * OctreeRootNode, **ParticleInNode;

struct WeightIndex {
    double weight;
    int index;
};
struct WeightIndex2 {
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
    int inMST;
    int cell[3];
};

struct spanning_tree_coordinate_system {
    int num_vertex;
    int num_edge;    

    struct graph_vertex *vertex;

    // TODO check if some of these can be removed
    struct graph_edge *edge, *local_edge, *local_edge_subset, *edge_in_mst;

    double (*edge_rel_pos)[3];
    double (*edge_rel_vel)[3];
    double (*edge_rel_aux_vel)[3];
    double (*edge_rel_acc)[3];
    double (*edge_rel_acc_vd)[3];
};

struct task_info {
    int gbs_group_size;
    int gbs_group_rank;
    MPI_Comm gbs_comm;

    int global_rank;
};

// A wrapper struct for all the internal state needed by the integrator
struct ketju_system_internal_state {

    struct spanning_tree_coordinate_system *mst;
    struct task_info *task_info;

    double U; // potential energy
    double T;
    double B;
    double H;
    double (*aux_vel)[3]; 
    double (*acc)[3]; // velocity independent acceleration
    double (*acc_vd)[3]; // velocity dependent acceleration
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
inline void tick(double *timer) {
    *timer -= MPI_Wtime();   
}
inline void tock(double *timer) {
    *timer += MPI_Wtime();   
}

// for exiting on catastrophic errors
// TODO most uses now could be replaced with error codes allowing potentially
// recovering from errors during integration / saving the state so far.
inline void terminate(const char *msg) {
    printf("%s", msg);
    MPI_Abort(MPI_COMM_WORLD, 1);
}


inline void get_v1_v2_fast(struct ketju_system *R, int i, int *v1, int *v2) {
    *v1 = R->internal_state->mst->edge_in_mst[i].vertex1;
    *v2 = R->internal_state->mst->edge_in_mst[i].vertex2;
}


// declarations of internal functions used between compilation units


int ketju_check_relative_proximity(int v1, int v2, const int Nd,
                             const struct ketju_system *R, int *d, int *path,
                             int *sign);
int ketju_check_relative_proximity_ND_2(int v1, int v2, const struct ketju_system *R,
                                  int *d, int *path, int *sign);

#endif // KETJU_INTEGRATOR_INTERNAL_H
