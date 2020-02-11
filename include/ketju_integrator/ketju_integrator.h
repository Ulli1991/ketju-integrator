// The public interface header with all the user-facing types and function
// prototypes.

#ifndef KETJU_INTEGRATOR_H
#define KETJU_INTEGRATOR_H

#include <stdbool.h>

// Types and enums
// Using bitflags for the PN terms allows more fine-grained control of the used
// terms (e.g. only PN2.5 radiation reaction or only conservative terms).
// XXX: this could end up being too complicated, but should be fairly easy to
// use especially if all commonly used combinations are pre-defined.
enum ketju_PN_flags {
    KETJU_PN_NONE = 0,
    KETJU_PN_1_0_ACC = 1 << 0,
    KETJU_PN_2_0_ACC = 1 << 2,
    KETJU_PN_2_5_ACC = 1 << 3,
    KETJU_PN_3_0_ACC = 1 << 4,
    KETJU_PN_3_5_ACC = 1 << 5,
    KETJU_PN_CONSERVATIVE = KETJU_PN_1_0_ACC | KETJU_PN_2_0_ACC | KETJU_PN_3_0_ACC,
    KETJU_PN_DYNAMIC_ALL = 0xFF, // all the spin independent terms, is this a good name?
    

    KETJU_PN_1_5_SPIN_ACC = 1 << 8,
    KETJU_PN_2_0_SPIN_ACC = 1 << 9,
    KETJU_PN_1_0_SPIN_EVOL = 1 << 10,
    KETJU_PN_1_5_SPIN_EVOL = 1 << 11,
    KETJU_PN_SPIN_ALL = 0xF00,

    KETJU_PN_ALL = KETJU_PN_DYNAMIC_ALL | KETJU_PN_SPIN_ALL,
};

enum ketju_mst_algorithm {
    KETJU_MST_PRIM,
    KETJU_MST_DIVIDE_AND_CONQUER_PRIM,
};

struct ketju_integration_options {
    int gbs_kmax;

    double gbs_relative_tolerance;

    double output_time_relative_tolerance;
    //< Maximum size of relative error in time iteration
    double output_time_absolute_tolerance;
    //< Minimum absolute size of error in time iteration

    unsigned int PN_flags; //< Bitflags for enabling different PN terms

    int steps_between_mst_reconstruction;
    int max_tree_distance;
    //< Maximum separation in the tree structure before using CoM coordinate
    //distances

    enum ketju_mst_algorithm mst_algorithm;

    double star_star_softening;
    //< force softening length between non-pn particles, disabled if <= 0
    bool no_star_star_gravity;
    //< Disable gravity between stars (non-pn particles), only interactions
    // including pn-particles are computed.
};

static const struct ketju_integration_options
    ketju_default_integration_options = {
        .gbs_kmax = 8,
        .gbs_relative_tolerance = 1e-9,
        .output_time_relative_tolerance = 1e-6,
        .output_time_absolute_tolerance = 1e-300,
        .PN_flags = KETJU_PN_NONE,
        .steps_between_mst_reconstruction = 10,
        .max_tree_distance = 2,
        .mst_algorithm = KETJU_MST_PRIM,
        .star_star_softening = 0.,
        .no_star_star_gravity = false,
};

struct ketju_natural_constants {
    double G;
    double c;
};
// Gadget default values for now
static const struct ketju_natural_constants ketju_default_natural_constants = {
    .G = 43007.1, .c = 299792.0};

struct ketju_system_physical_state {
    double time;
    double *mass;
    double (*pos)[3];
    double (*vel)[3];
    double (*spin)[3]; //spin only for pn particles
};

struct ketju_performance_counters {
    double time_force;
    double time_force_comm;
    double time_coord;
    double time_mst;
    double time_gbscomm;
    double time_gbsextr;
    double time_total;

    // check what these actually mean
    double tprim1, tprim2, tprim3, tprim4, tprim5;

    int succesful_steps;
    int failed_steps;
    double relative_energy_error;
};

struct ketju_system {
    int num_particles; //< Total number of particles
    int num_pn_particles; //< Number of PN enabled particles

    struct ketju_system_physical_state *physical_state;
    //< Physical state of the system (as a pointer for making temporary copies)
    // TODO looking like we're not going to use the pointer nature for making
    // copies, maybe make into a direct member?

    struct ketju_system_internal_state *internal_state; 
    //< Internal state (implementation detail)

    // TODO is it better to have this stuff (which doesn't need to be copied
    // into temp variables) as direct members to avoid extra pointer derefs, or
    // as pointers for consistency with the above and perhaps some cache
    // behaviour benefits?
    struct ketju_performance_counters *perf; 
    struct ketju_integration_options *options;

    struct ketju_natural_constants *constants;
    //< Values of natural constants which define the unit system
    
    //
    // XXX: the integration works only in the chained coordinates, forcing the
    // CoM to stay at the origin. Not compatible with perturbations that don't
    // conserve the CoM right now
    double CoM_pos[3];
    //< Position of the center of mass in the input coordinates
    // (origin of physical_state.pos).
    // The CoM of the input state is added to this at the start of the
    // integration, and the position is drifted based on CoM_vel.
    double CoM_vel[3];
    //< Velocity of the center of mass in the input coordinates
    // (origin of physical_state.vel)

    void *particle_extra_data;
    //< Optional extra data (e.g. IDs) associated with the particles, which is
    // reordered in case of mergers etc.
    // The caller is responsible for managing this memory.

    size_t particle_extra_data_elem_size;
    //< Size of the extra data for a single particle.
};

// Function prototypes
void ketju_create_system(struct ketju_system *R, int num_pn_particles,
                         int num_other_particles, MPI_Comm comm);
void ketju_free_system(struct ketju_system *R);
void ketju_into_CoM_frame(struct ketju_system *R);
void ketju_run_integrator(struct ketju_system *R, double time_interval);

#endif // KETJU_INTEGRATOR_H
