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
};

// ketju_natural_constants, with KETJU's defaults (ketju_default_natural_constants: GADGET units, 1e10 Msun, kpc, km/s)
struct Constants {
    double G = 43007.1, c = 299792.0;
};

// ketju_performance_counters (+ timers of this library)
struct PerformanceCounters {
    long successful_steps = 0, failed_steps = 0;
    double total_work = 0;              // ~ number of force evaluations (KETJU's step_total_work sum)
    double relative_energy_error = 0;   // over the last run(): energy T - U - U_tide (U softened as integrated), the
                                        // collision energies dE_collision removed; not the error with PN forces, extra
                                        // forces other than the tide, or after BH mergers
    double t_mst = 0, t_total = 0, t_sync = 0;   // wall seconds
    long launches = 0;                  // kernel launches issued
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

    // ketju_run_integrator: advance by time_interval. Returns false if max_step_count was exceeded (state left at
    // the last accepted step).
    bool run(double time_interval);

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
