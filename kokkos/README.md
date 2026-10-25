# MSTAR in C++ with Kokkos (GPU)

C++ rewrite of the MSTAR regularised integrator (Rantala et al. 2020; public KETJU integrator, Mannerkoski et al.
2023) with Kokkos support for GPUs, by Ulrich Steinwandel.

The C library in `../src`, `../include` stays the reference implementation; this directory is a separate library
with the same physics and options, running on any Kokkos execution space (CUDA, HIP, OpenMP, Serial). Every feature
of the C library on `main` is here, including the GIZMO additions (contact collisions of non-PN particles with
`collision_radius`, their energy booking with the external tide and softening, the pericentre guard, the external
tide, star-star softening).

## Citation

Cite, when you publish results obtained with this library:

* **required** (KETJU's README asks for it): A. Rantala, P. Pihajoki, M. Mannerkoski, P. H. Johansson & T. Naab,
  *MSTAR - a fast parallelised algorithmically regularised integrator with minimum spanning tree coordinates*,
  MNRAS 492, 4131 (2020), arXiv:2001.03180;
* **recommended** for the released version (PN corrections, BH mergers and recoil, softening, dynamic order
  control): M. Mannerkoski et al., *KETJU - resolving small-scale supermassive black hole dynamics in GADGET-4*,
  MNRAS 524, 4062 (2023), arXiv:2306.04963;
* the physics the code uses: L. Blanchet, Living Rev. Relativ. 17, 2 (2014) (2-3.5PN); K. G. Arun et al., PRD 77,
  064035 (2008) (3PN gauge transformation); K. S. Thorne & J. B. Hartle, PRD 31, 1815 (1985) (1PN N-body and spin
  terms); Y. Zlochower & C. O. Lousto, PRD 92, 024022 (2015) (merger remnants); E. Hairer, S. P. Norsett &
  G. Wanner, *Solving Ordinary Differential Equations I*, ch. II.9 (GBS extrapolation and order control).

## Licence, origin and modifications

GPLv3, as the whole repository (`../LICENSE.md`). The library is derived from the KETJU integrator library (MSTAR),
Copyright (C) 2019-2022 Antti Rantala, Matias Mannerkoski, and contributors: this repository (the fork
https://github.com/Ulli1991/ketju-integrator of https://bitbucket.org/helsinkiextragalacticgroup/ketju-integrator),
for the extra forces the experimental branch `extra-force` (b60d16e, Mannerkoski 2022). Every ported file keeps the
KETJU header and states below it: *Modified by Ulrich Steinwandel, 2026: rewritten in C++ with Kokkos support (GPU).
Copyright (C) 2026 Ulrich Steinwandel. Licensed under the GPLv3 license.* The tests, the MPI stand-in
(`../thirdparty/serial_mpi`) and the forward-ported extra-force reference (`tests/reference/ketju-extra-force-fwd`,
the b60d16e patch on eb96fd0, modified files marked) are Copyright (C) 2026 Ulrich Steinwandel, GPLv3. The library
was developed as the optional GPLv3 module `src/mstar` of RUMI++; `docs/` keeps its notes from there.

## Layout

| path | content |
|---|---|
| `include/mstar/mstar.hpp` | public API (namespace `mstar`; no Kokkos in the header) |
| `src/engine.cpp` | integrator (from `integration.c`, `mst.c`, `pn.c`): GBS rows, forces, PN, MST, step control, collisions |
| `src/mergers.{hpp,cpp}` | BH mergers (`bh_mergers.c`, PN c.o.m. velocity of `pn.c`), contact collisions with energy booking |
| `src/kernels.hpp` | host/device helpers: softening kernels, osculating pericentre test, stop-collision rule |
| `examples/mstar_driver.cpp` | standalone driver (KETJU's `nbody` IC format, the counterpart of `../nbody/main.c`) |
| `examples/mstar_bench.cpp` | scaling benchmark (Plummer N = 16 .. 4096) against the C library, launch latency |
| `tests/` | C vs Kokkos comparisons, KETJU unit tests incl. GIZMO's additions on the C library, the same additions on this library, extra forces, embedded tide |
| `docs/` | RUMI++ notes on the module (copies) |

## Build

Kokkos (4.x or later, C++17) is found with `find_package(Kokkos)`; the library uses the default execution space
(the device one in GPU builds) and the default host execution space.

```
# from the repository top level (also builds the C library; KETJU_SERIAL_MPI=ON: no MPI needed)
cmake -S . -B build -DKETJU_KOKKOS=ON -DKETJU_SERIAL_MPI=ON -DKokkos_ROOT=<kokkos install>
# or standalone
cmake -S kokkos -B build-kokkos -DKokkos_ROOT=<kokkos install>       # CUDA: -DCMAKE_CXX_COMPILER=<kokkos>/bin/nvcc_wrapper
cmake --build build-kokkos -j 4 && (cd build-kokkos && ctest)
cmake --install build-kokkos --prefix <prefix>                        # then find_package(mstar) -> mstar::mstar
```

`MSTAR_TESTS=OFF` builds only the library and the driver.

## API

```cpp
#include <mstar/mstar.hpp>
mstar::ScopeGuard guard(argc, argv);       // or Kokkos::ScopeGuard / Kokkos::initialize by the host code
mstar::System S(num_pn, num_other);        // ketju_create_system; PN particles (BHs) first
S.constants.G = 1; S.constants.c = 1;      // ketju_natural_constants (defaults: KETJU's GADGET units)
S.options.PN_flags = mstar::PN_ALL;        // ketju_integration_options, same names and defaults
S.mass[i] = ...; S.pos[3*i+k] = ...; S.vel[3*i+k] = ...; S.spin[3*i+k] = ...;   // physical state
S.collision_radius = {...};                // optional: contact collisions of non-PN particles (GIZMO)
S.options.enable_external_tide = true;     // optional: a = T x (GIZMO)
bool ok = S.run(dt);                       // ketju_run_integrator; false: max_step_count exceeded
S.mergers;                                 // ketju_bh_merger_info (BH mergers and collisions, dE_collision)
S.perf.relative_energy_error;              // ketju_performance_counters
```

| KETJU C | `mstar::` |
|---|---|
| `ketju_create_system` / `ketju_free_system` | `System(num_pn, num_other)` / destructor; `reset()` reuses device buffers |
| `R->physical_state->{time, mass, pos, vel, spin}` | `S.time`, `S.mass`, `S.pos`, `S.vel`, `S.spin` (flat xyz arrays) |
| `num_particles`, `num_pn_particles` | the same names (shrink on mergers) |
| `particle_extra_data` | `S.extra_index` (caller indices, permuted as KETJU permutes the extra data) |
| `collision_radius` | `S.collision_radius` (empty: off) |
| `R->options`, `R->constants`, `R->perf` | `S.options`, `S.constants`, `S.perf` |
| `merger_infos` / `num_mergers` | `S.mergers` |
| `ketju_run_integrator` | `S.run(dt)` |
| `ketju_into_CoM_frame` | `S.into_com_frame(x, v)` |
| `ketju_do_star_collisions`, `ketju_pair_rv` | `mstar::do_star_collisions`, `mstar::pair_rv` |
| extra-force branch callbacks | `S.options.extra` (`ExtraForces`: linear field, static point masses, `a = D v`) |

Additions for host codes (RUMI++): `S.stop_collisions` (a collision rule with BH flags, TDE and ISCO distances that
stops the call at the colliding step instead of merging inside), `S.perturbers` (bodies with predicted motion
`x + v t + a t^2 / 2` acting inside the call), `ExtraForces::remove_mean`, the work ledger `S.pn_dB` / `S.ext_work`,
`options.host_engine` (run a small group on the host execution space in a GPU build), `S.energy()`.

## Differences from the C library

* All GBS rows (substep divisions 2, 4, ..., 2k) of a step run concurrently on the device in lockstep: per substep
  stage one force launch over the active rows (one team per row and body) and one kick+drift launch; small groups run
  every row inside one team (`team_rows_max_n`). The same engine runs on the device and on the host.
* Vertex coordinates are rebuilt from the tree by root-to-vertex path sums in KETJU's floating-point order; reductions
  are deterministic. Results agree with the C library to round-off (bitwise for few bodies on the CPU) and are
  reproducible run to run.
* One device-to-host copy per GBS step; the extrapolated state, its energies and the collision test are computed on
  the device. Contact collisions are detected on the device and merged on the host (only on a hit).
* The MST is built by dense O(N^2) Prim (KETJU's tree), on the host or in one device team (`device_tree_min_n`).
* B is carried as an extra GBS variable outside the error control, giving the work of the PN and extra forces
  (`pn_dB`, `ext_work`). B follows the work of velocity-independent extra forces (`track_work`, default; the branch
  leaves B blind to it, which put a binary in a static field 1e-2 off after a few orbits).
* `MergerInfo::t_merger` is the absolute time (KETJU: since the start of the run call); `extra_index1/2` are caller
  indices (`S.extra_index`) rather than slots of the extra-data array.
* Not ported: the divide-and-conquer MST (`mst_algorithm`), the MPI parallelisation (`ketju_create_system`'s
  communicator), the debug state dumps. `gbs_kmax` is 4..16, `max_tree_distance` <= 8.

## Tests (ctest)

* `ketju_unit_c`: KETJU's unit tests (`../unit-tests`, 34 incl. the 8 GIZMO-addition tests `gizmo_additions.cpp`) on
  the C library, on the GoogleTest stand-in `tests/gtest_shim`.
* `mstar_gizmo_additions`: the same GIZMO-addition tests on this library (+ a collision during a run with tide and
  softening).
* `mstar_compare_c`: this library against the C library, same options and state: Kepler e = 0.9 (and the analytic
  orbit), Pythagorean three-body, 100-body Plummer core (team and staged paths), host vs device MST, PN + spin binary,
  two BHs in a 60-star cusp with all PN terms, PN work ledger, stop-rule collisions, contact collisions (head-on and
  pericentre-decided, in one and ten calls), collision with tide and softening, external tide, softening, no
  star-star gravity, head-on and orbital BH mergers.
  Each comparison also checks momentum and angular momentum (Kokkos within 2x of C or at the round-off floor).
  `collision_tolscan` tabulates collision times and energies against the GBS tolerance and `softening_ensemble`
  compares energy errors over 16 perturbed initial states (below).
* `mstar_extra_force`: the extra forces against the branch patch forward-ported onto eb96fd0 and against RK4.
* `mstar_embedded_tide`: an embedded group, impulsive member kicks against c.o.m. kicks + continuous tide.
* `mstar_driver`: the driver on `../nbody/nbody_sample_ic.txt`.
* `mstar-bench [N_max] [steps] [C_N_max] [team_N]`: scaling (not a ctest).

Results (CPU: gcc 15, Kokkos OpenMP, 1 thread; A100: CUDA 13.2) are noted at each case in `tests/compare_c.cpp`.

## Accuracy notes (both libraries)

* **Collision times are resolved to one GBS step.** Both libraries detect a contact collision on the accepted state
  at the end of a step and merge there, without landing on the contact. The recorded time and the collision energy
  (the pair is already inside contact by a step-dependent amount) therefore follow the step sequence and do not
  converge with the GBS tolerance: in `collision_tolscan` the C library's own collision times at tolerances
  1e-8 .. 1e-13 differ by up to 6e-4 T, and its collision energies by up to 6e-8 relative. Kokkos (CPU and A100)
  differs from C by the same amounts; the energy including the booked collision energies is conserved to the
  tolerance level in every case, and so are momentum and angular momentum (minus the pair's internal angular momentum,
  which a merger at the CoM removes in both libraries).
* **Softened pairs defeat the GBS error estimate.** The GADGET spline (`star_star_softening`) has jumps in its
  higher derivatives at u = r / h = 0.5 and 1. A pair that moves through them breaks the smoothness that the
  extrapolation and its error estimate rely on: in the C library a softened two-body orbit through h has |dE/E|
  3e-8 (median over 16 states) at tolerance 1e-11 and 2e-9 at 1e-13, with outliers up to 7e-6 when a collision is
  involved (`softening_ensemble`), varying by orders of magnitude with 1e-9 changes of the initial state. The Kokkos
  library behaves identically (the same ensemble statistics); single softened trajectories are compared through
  these ensembles. Collisions decided before the pair reaches h (R1 + R2 > h) are not affected.
* **FMA on the GPU.** nvcc contracts multiply-adds (FMA, the default, kept on); the CPU build does not. In smooth
  problems this is round-off. In the softened and collision cases FMA rounding changes a step size; at the softening
  kernel's kinks, or where a collision is recorded at a step end, the step sequence then diverges within a few steps.
  In `mstar_compare_c`'s segment test (both libraries restarted from the same C state 80 times) the A100 states
  differ from C by up to 1.1e-6 per segment in these cases (CPU 2e-14). Collisions match and the summed energy error
  is no larger (A100 / C 0.47, 0.12). A CUDA build without FMA (`-DCMAKE_CXX_FLAGS=--fmad=false`, run as a variant
  in the GPU test job) agrees to the CPU bounds. Smooth cases keep the strict bounds.
* GPU (A100) results differ from the CPU ones by floating-point round-off (FMA contraction, team reduction order).
  The step sequence amplifies that round-off, which explains the different step counts and collision times; the
  measured CPU and A100 values are noted at each case in `tests/compare_c.cpp`.
