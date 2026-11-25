// Ketju-integrator (MSTAR)
// Copyright (C) 2020-2022 Antti Rantala, Matias Mannerkoski, and contributors.
// Licensed under the GPLv3 license.
// See LICENSE.md or <https://www.gnu.org/licenses/> for details.
// Modified by Ulrich Steinwandel, 2026: rewritten in C++ with Kokkos support (GPU).
// Copyright (C) 2026 Ulrich Steinwandel. Licensed under the GPLv3 license.
//
// Public interface of the C++/Kokkos MSTAR library (target mstar::mstar). It follows the KETJU C interface
// (include/ketju_integrator/ketju_integrator.h): a system with num_pn PN particles (BHs, first) and num_other other
// particles, its physical state, options, constants, performance counters and merger records; System::run is
// ketju_run_integrator. The algorithm is KETJU's (Rantala et al. 2020, MNRAS 492, 4131; Mannerkoski et al. 2023,
// MNRAS 524, 4062): logarithmic-Hamiltonian leapfrog in minimum-spanning-tree relative coordinates, GBS
// extrapolation with adaptive order, PN 1-3.5 + spin terms, BH mergers. Differences from the C library are listed in
// kokkos/README.md. The header does not include Kokkos; the caller initialises Kokkos (Kokkos::ScopeGuard, or
// mstar::ScopeGuard below) before creating a System and destroys every System before Kokkos::finalize.

#pragma once

#include <vector>

namespace mstar {

// PN flags: the values of KETJU's enum ketju_PN_flags
enum PNFlags : unsigned {
    PN_NONE = 0,
    PN_1_0 = 1u << 0,
    PN_THREEBODY = 1u << 1,   // 1PN N-body cross terms (Thorne & Hartle 1985)
    PN_2_0 = 1u << 2,
    PN_2_5 = 1u << 3,
    PN_3_0 = 1u << 4,
    PN_3_5 = 1u << 5,
    PN_1_5_SPIN = 1u << 8,
    PN_2_0_SPIN = 1u << 9,
    PN_SPIN_EVOL = 1u << 10,
    PN_DYNAMIC_ALL = PN_1_0 | PN_THREEBODY | PN_2_0 | PN_2_5 | PN_3_0 | PN_3_5,
    PN_SPIN_ALL = PN_1_5_SPIN | PN_2_0_SPIN | PN_SPIN_EVOL,
    PN_ALL = PN_DYNAMIC_ALL | PN_SPIN_ALL
};

// Extra forces (port of KETJU's experimental extra-force branch, Mannerkoski 2022, b60d16e), constant during one
// run() call, evaluated on the device. Velocity-independent part (added to the Newtonian accelerations): a linear
// field a = A (x - x0) and static Plummer-softened point masses. Velocity-dependent part (through the
// auxiliary-velocity scheme as PN; its work enters B like the PN work): a = D v.
struct ExtraForces {
    static constexpr int MAX_PM = 4;
    bool vindep = false;
    double A[9] = {};            // row-major 3x3
    double x0[3] = {};
    int n_pm = 0;
    double pm_m[MAX_PM] = {};
    double pm_x[MAX_PM][3] = {};
    double pm_eps2 = 0;
    bool track_work = true;      // B follows their work (false: the branch's behaviour, B unchanged)
    bool remove_mean = false;    // the mass-weighted mean (of all velocity-independent extra accelerations, the
                                 // external tide included) is left out: the caller moves the group's c.o.m.
    bool vdep = false;
    double D[9] = {};            // row-major 3x3
};

// KETJU's ketju_integration_options with the same defaults, plus the options of this library
struct Options {
    // ---- ketju_integration_options
    int gbs_kmax = 10;                              // 4..16 here
    double gbs_relative_tolerance = 1e-9;
    double gbs_absolute_tolerance = 1e-100;
    double output_time_relative_tolerance = 1e-6;
    double output_time_absolute_tolerance = 1e-300;
    unsigned PN_flags = PN_ALL;
    int steps_between_mst_reconstruction = 10;
    int max_tree_distance = 2;                      // <= 8 here
    long max_step_count = -1;                       // > 0: run() stops (returns false) after this many attempted steps
    double star_star_softening = 0;                 // GADGET spline softening between non-PN particles (<= 0: off)
    bool softening_kink_landing = true;             // land GBS steps on the softening kernel's break points (u = r / h
                                                    // = 0.5, 1) when a softened pair crosses one inside a step
    double softening_kink_eta = 5e-4;               // a crossing within this fraction of a step from an end is accepted
    double softening_kink_error_ratio = 1;          // crossings with an estimated local error (J dt^6 / 6!) below this
                                                    // ratio x the tolerance on the separation are not landed
    bool contact_landing = true;                    // collision_radius: land GBS steps on the first contact r = R1 + R2
                                                    // of a pair (event location, same eta as the kink landing) and merge
                                                    // there; false: the C library's rule, contact found at step ends only
    bool no_star_star_gravity = false;              // no forces between non-PN particles
    bool enable_bh_mergers = true;
    double bh_merger_distance_in_Rs = 12;
    bool enable_bh_merger_kicks = true;
    bool enable_external_tide = false;              // linear tide a_i = T x_i in the system's frame (its work enters
                                                    // B, not U; always tracked)
    double external_tidal_tensor[6] = {0, 0, 0, 0, 0, 0};   // T: xx, yy, zz, xy, yz, xz
    // ---- this library
    ExtraForces extra;                              // extra forces (off)
    bool host_engine = false;                       // run on the host execution space (small groups in GPU builds)
    int team_rows_max_n = 96;                       // host engine: N <= this runs every GBS row inside one team
    int team_rows_max_n_device = 0;                 // device engine: the same (A100: the staged path is faster)
    int device_tree_min_n = 256;                    // device engine: the MST on the device from this size
    bool device_tree_everywhere = false;            // tests: the team MST on the host engine too
    int gbs_error_norm = 0;                         // GBS error of a variable relative to 0: its own magnitude (the C
                                                    // library's rule: a component near zero at the step end sets the
                                                    // error), 1: the norm of its 3-vector (root / edge position or
                                                    // velocity, spin)
    bool gbs_error_diag = false;                    // count the failed steps the other rule would have accepted
                                                    // (PerformanceCounters::failed_steps_other_norm)
    bool tree_sweep = false;                        // vertex positions and velocities from the tree by a parents-first
                                                    // sweep, x_v = x_parent + edge: the additions of the root-to-vertex
                                                    // path sum in the same order (bitwise the same result), O(N) on one
                                                    // thread instead of O(N depth) over the team's threads
    int gbs_growth_limit = 0;                       // largest growth of the step after an accepted step: 0: 5 x (the C
                                                    // library), 1: (1 / gbs_growth_fac1)^(1 / (2 k - 1)), the limit of
                                                    // Hairer's extrapolation code ODEX (1.35 at k = 7 with 0.02); the
                                                    // work ratio of an order increase comes on top as there
    double gbs_growth_fac1 = 0.02;
    bool log_steps = false;                         // diagnostics: System::step_log gets every GBS step attempt
    bool mpi_time_detail = false;                   // diagnostics: a barrier before every force exchange, so that
                                                    // PerformanceCounters::t_wait (waiting for the slowest task) and
                                                    // the transfer (t_comm - t_wait) can be told apart
    int mpi_min_n = -1;                             // MPI builds, System with a communicator: systems with at least this
                                                    // many particles at entry share their O(N^2) loops over the tasks
                                                    // (< 0: the library default, 128; see set_mpi_fcomm)
    // Experimental constant-factor SDAR for one nominated Newtonian binary. Off by default; no automatic
    // binary discovery or changing-factor prescription. Unsupported force/collision models retain factor 1.
    bool experimental_slowdown = false;
    int slowdown_i = 0, slowdown_j = 1;
    double slowdown_factor = 1;
};

// ketju_natural_constants, with KETJU's defaults (ketju_default_natural_constants: GADGET units, 1e10 Msun, kpc, km/s)
struct Constants {
    double G = 43007.1, c = 299792.0;
};

// ketju_performance_counters (+ timers of this library)
struct PerformanceCounters {
    long successful_steps = 0, failed_steps = 0;
    long failed_steps_other_norm = 0;   // Options::gbs_error_diag: failed steps within tolerance by the other
                                        // gbs_error_norm rule
    double applied_slowdown_factor = 1; // actual factor of the last run, 1 for the ordinary/fallback path
    long kink_landings = 0;             // steps repeated to land on a softening-kernel break point
    long contact_landings = 0;          // steps repeated to land on a contact (Options::contact_landing)
    double total_work = 0;              // ~ number of force evaluations (KETJU's step_total_work sum)
    double relative_energy_error = 0;   // experimental slowdown: conserved H_sd, not physical energy; otherwise
                                        // over the last run(): energy T - U - U_tide (U softened as integrated), the
                                        // collision energies dE_collision removed; not the error with PN forces, extra
                                        // forces other than the tide, or after BH mergers
    double t_mst = 0, t_total = 0, t_sync = 0;   // wall seconds
    long launches = 0;                  // kernel launches issued
    double t_force = 0, t_comm = 0;     // wall seconds in the force launches (messages included) and in MPI messages
    double t_wait = 0;                  // Options::mpi_time_detail: the part of t_comm spent waiting for the slowest task
    double mean_tree_depth = 0;         // mean number of edges from the root, last tree built
    int mpi_tasks = 1;                  // tasks that shared the last run() (1: not shared)
};

// ketju_bh_merger_info. BH mergers and (with collision_radius) contact collisions of non-PN particles.
struct MergerInfo {
    double t_merger = 0;                // absolute time of the merger (KETJU: time since the start of the run call)
    double m1 = 0, m2 = 0, m_remnant = 0;
    double chi1 = 0, chi2 = 0, chi_remnant = 0;
    double v_kick = 0;
    int extra_index1 = -1, extra_index2 = -1;   // System::extra_index values of the survivor / the merged-away particle
    int is_collision = 0;               // 1: non-PN contact collision (collision_radius), 0: BH merger
    double dE_collision = 0;            // energy change of a collision: minus the pair's internal energy, plus the
                                        // change of its potential energy with the other particles and in the external
                                        // tide as it moves to its CoM (star-star pairs softened as integrated)
};

// Collision rule that stops the call (alternative to the in-integrator merging with collision_radius): after every
// accepted step, a pair closer than its collision distance, or approaching with osculating pericentre inside it and
// reaching that pericentre within the step. Distances: star-star R1 + R2; BH-star max(1.3 (M_bh / m)^(1/3) R, R) or the
// ISCO 6 G (m1 + m2) / c^2 if larger; BH-BH 20 G (m1 + m2) / c^2, or 6 G max(m1, m2) / c^2 with bh_max_single. A hit
// stops run() at that step (time = the step's end) with i < j, the separation (or osculating pericentre) and the type
// (1 star-star, 2 TDE, 3 ISCO / BH-BH); the caller merges the pair and calls again.
struct StopCollisions {
    std::vector<double> radius;         // [n] (empty: off)
    std::vector<char> is_bh;            // [n] (empty: none)
    double c = 0;                       // speed of light of the BH distances (0: none)
    bool bh_max_single = false;
    // out
    int i = -1, j = -1, type = 0;
    double r_sep = 0, r_coll = 0;
};

class System {
  public:
    // ketju_create_system
    System(int num_pn_particles, int num_other_particles);
    ~System();   // ketju_free_system
    System(const System&) = delete;
    System& operator=(const System&) = delete;
    // Reuse for another system (device buffers are kept when the sizes match): resizes the state arrays, clears the
    // counters, mergers, collision data and perturbers, keeps options and constants
    void reset(int num_pn_particles, int num_other_particles);

    // ---- physical state (ketju_system_physical_state; PN particles first), filled by the caller
    int num_particles = 0, num_pn_particles = 0;
    double time = 0;
    std::vector<double> mass;           // [n]
    std::vector<double> pos, vel;       // [3n], xyz per particle
    std::vector<double> spin;           // [3 n_pn] spin angular momenta (length x mass x velocity)
    // caller indices (particle_extra_data): permuted as KETJU permutes the extra data on mergers, merged-away
    // particles at the end
    std::vector<int> extra_index;
    // Contact collisions of non-PN particles (KETJU GIZMO addition, collision_radius): empty = off; else [n] radii.
    // Two non-PN particles with radius > 0 merge at their CoM (mass, momentum conserved; the heavier keeps its slot
    // and the larger radius) inside run() when r < R1 + R2 on an accepted step, or when the pair passed its
    // osculating pericentre during the step with a two-body pericentre below R1 + R2. Reordered on mergers.
    std::vector<double> collision_radius;
    StopCollisions stop_collisions;
    // Perturbers (perturbed subsystems): 10 values each, m, x0[3], v[3], a[3]; their pull at the predicted positions
    // x0 + v t + a t^2 / 2 (t from the start of the call, Plummer-softened by extra.pm_eps2) is a velocity-independent
    // extra force
    std::vector<double> perturbers;

    Options options;
    Constants constants;
    PerformanceCounters perf;
    std::vector<MergerInfo> mergers;    // all mergers since creation / reset
    // Work ledger (this library): sum over accepted steps of the extrapolated change of B = U - T by the PN and
    // extra-force kicks. The Newtonian T + U changed by -pn_dB through the velocity-dependent forces and the
    // velocity-independent extra forces together; ext_work is the work of the velocity-independent extra forces (the
    // external tide and the perturbers included) alone. Zero without such forces.
    double pn_dB = 0, ext_work = 0;
    // Options::log_steps: per GBS step attempt of the last run() 5 values: order k, fictitious step H, error / tolerance
    // (e0 of the rule in use), accepted (1 / 0), the order chosen for the next step
    std::vector<double> step_log;

    // ketju_run_integrator: advance by time_interval. Returns false if max_step_count was exceeded (state left at
    // the last accepted step).
    bool run(double time_interval);

    // MPI (library built with MSTAR_ENABLE_MPI; otherwise set_mpi_fcomm throws): the communicator as a Fortran handle
    // (MPI_Comm_c2f), so this header needs no mpi.h. With a communicator of more than one task, run() is collective:
    // every task calls it on a System with the same state and options. Each task runs the whole integration; for
    // systems of at least Options::mpi_min_n particles the O(N^2) loops (forces, collision / contact / kink searches,
    // initial time step) are divided over the tasks, each particle's force sum computed by one task in the serial
    // order, so the result is bitwise that of one task for every task count (same Kokkos threads per task). Smaller
    // systems are integrated by every task on its own without messages. The tasks compare their inputs at entry, the
    // step result after every GBS step and the final state; a difference throws std::runtime_error on every task.
    // Host memory spaces only (device engines do not share).
    void set_mpi_fcomm(int fortran_comm);
    void clear_mpi_comm();              // back to the default communicator (none, or MpiSession's)

    // ---- helpers
    void set_particle(int i, double m, const double x[3], const double v[3]);
    // ketju_into_CoM_frame: shift to the CoM frame, returning its position and velocity
    void into_com_frame(double com_pos[3], double com_vel[3]);
    // T - U - U_tide of the current state, U softened as integrated (the energy relative_energy_error refers to)
    double energy() const;
    double external_tide_potential() const;   // 1/2 sum m x.T.x (0 without the tide)

  private:
    struct Impl;
    Impl* impl_ = nullptr;
};

// Contact collisions on the current state as in run() (ketju_do_star_collisions): rv_prev holds, per non-PN pair
// i < j, r.v at the last check (row-major [n * n]) and the check time in rv_prev[n * n]; nullptr: contact only.
// Refreshes rv_prev. Returns the number of collisions. pair_rv fills rv_prev from the current state
// (ketju_pair_rv).
int do_star_collisions(System& S, double* rv_prev);
void pair_rv(const System& S, double* rv);

// MPI helpers. mpi_enabled: the library was built with MSTAR_ENABLE_MPI. MpiSession (tests, drivers; first object
// in main, before Kokkos starts): calls MPI_Init / MPI_Finalize unless MPI is already running, and makes
// MPI_COMM_WORLD the communicator of every System without its own; the environment variable MSTAR_MPI_MIN_N sets the
// library default of Options::mpi_min_n. Without MPI support it does nothing (rank 0, size 1).
bool mpi_enabled();
struct MpiSession {
    MpiSession(int& argc, char**& argv);
    ~MpiSession();
    MpiSession(const MpiSession&) = delete;
    MpiSession& operator=(const MpiSession&) = delete;
    int rank = 0, size = 1;
    bool owns_ = false;
};

// Kokkos lifetime helpers (the caller may use Kokkos directly instead)
void initialize(int& argc, char** argv);
void finalize();
bool is_initialized();
struct ScopeGuard {
    ScopeGuard(int& argc, char** argv);
    ~ScopeGuard();
    ScopeGuard(const ScopeGuard&) = delete;
    ScopeGuard& operator=(const ScopeGuard&) = delete;
    bool owns_ = false;
};
const char* execution_space_name();       // the device execution space (default execution space)
const char* host_execution_space_name();

// Launch + fence cost of an empty kernel, of a minimal reduction read back to the host and of an asynchronous
// launch (microseconds), over reps calls: the per-substep latency floor of a device-resident GBS step
void measure_launch_latency(int reps, double& us_launch_fence, double& us_reduce_readback, double& us_async_launch);

}  // namespace mstar
