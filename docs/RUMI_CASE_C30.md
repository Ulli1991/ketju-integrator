# RUMI++ case C30 (copy)

Copied from RUMI++ `docs/CASES.md` (C30, 2026-10-25): development record and measurements of the C++/Kokkos MSTAR as the RUMI++ module.

## C30 MSTAR subsystems: groups above LOGH_MAX_BODIES on the GPU (optional GPLv3 module; NBODY_MSTAR, default off)

C++ rewrite of the MSTAR regularised integrator (Rantala et al. 2020; public KETJU integrator, Mannerkoski et al.
2023) with Kokkos support for GPUs, by Ulrich Steinwandel (src/mstar, `ENABLE_MSTAR`; docs/MSTAR.md).

- **System:** subsystems of more than 8 bodies (IMBH with its cusp, dense cores, runaway-collision regions). Before:
  LogH holds 8; larger groups were split along their bound pairs or left to the FSI ("exceeds LOGH_MAX_BODIES"), i.e.
  integrated orbit by orbit by kicks. Failing-first cases: `mstar_big_group`, `mstar_coll_group`, `mstar_imbh_cusp`,
  `mstar_dense_core` (all report groups left to the FSI / no MSTAR integration without the module or switch).
- **Port:** KETJU eb96fd0 (+ the extra-force branch b60d16e) rewritten in C++/Kokkos; the C library stays as a test
  reference. Agreement (`mstar_compare`): CPU bitwise for two bodies (Kepler e = 0.9, PN + spin binary, orbital BH
  merger), 5e-14 - 1.3e-10 for 3 - 100 bodies; A100 round-off level (Kepler 6.5e-12, PN binary 7.5e-11; the merger
  trigger follows the step sequence: t_merger 4 % apart). KETJU's 26 unit tests pass on the reference.
- **vs LogH** (`mstar_vs_logh`, tolerance 1e-12): Kepler 10 orbits MSTAR 2.1e-12 from the analytic orbit with 159 GBS
  steps, LogH 2.1e-12 at its own end time with 312; Pythagorean t = 10: MSTAR 7e-11, LogH 2.9e-11 against a 1e-14
  reference. Finding: LogH's `integrate_to_time` lands 1.7e-8 (Kepler) / 8.6e-8 (Pythagorean) relative short of the
  target time (its landing tolerance), larger than its integration error.
- **A100 scaling** (`mstar-bench`, Plummer, tolerance 1e-10, ms per GBS step): N = 16 0.40, 128 0.48, 256 0.78,
  512 1.4, 1024 3.3, 2048 8.3, 4096 27 (15 G pair interactions/s); serial KETJU: 1.6x slower at 64, 16x at 128, 77x
  at 512. Latency: async launch 4.8 us, launch + fence 13.8 us, reduce + read-back 23 us; ~35 launches and one sync
  per GBS step (all rows in lockstep). A thread-per-body force kernel was 5.6x slower at N = 1024 (team kernel kept).
  The O(N^2) Prim tree (every 10 steps and at each call) takes 16-24 % at N >= 1024, on the host and in one device
  team alike (open: a parallel MST, e.g. Boruvka). Device from 32 bodies (NBODY_MSTAR_CROSSOVER).
- **Energy books:** B is carried as an extra GBS variable outside the error control; its extrapolated change is the
  work of the PN (and extra) forces. Only that goes to `E_pn`; the rest of the members' T + U change minus collisions
  is integration error, left unbooked and visible (`unbooked-mstar`, NBODY_EATTR). 62-body cusp with 2 BHs, all PN:
  PN work 3.7e-4 |E0|, integration error 6.5e-14 |E0| (`pn_ledger`).
- **Collisions** inside groups: tested after every accepted step (separation, or osculating pericentre inside the
  collision distance and reached within one step); merged in the core as LogH collisions (no pericentre landing).
- **Extra forces (KETJU branch b60d16e):** ported (linear field, static point masses, velocity-dependent `D v`).
  Finding: the branch leaves B blind to the work of velocity-independent extra forces, so T + B drifts from U and the
  time transformation no longer matches the dynamics: an e = 0.5 binary in a static point-mass field was 1.2e-2 off in
  position and 1.6e-4 in energy after 4 orbits; with B following the work (`track_work`, default) 6.5e-13 and 9e-14
  against RK4. Velocity-dependent drag through the aux-velocity path: 3.6e-13 against RK4, work ledger to 1.5e-13.
- **Tide hook** (`NBODY_MSTAR_TIDE`, off): c.o.m. kicks for MSTAR members + the outside tidal tensor inside MSTAR.
  Embedded group (`mstar_embedded_tide`, 20-body regular group in a Plummer cluster): internal error 1.3e-3 against
  2.2e-2 for impulsive member kicks at kick spacing 1e-3 (~2 innermost orbits), but a 1e-3 floor from the neglected
  octupole tide that impulsive kicks undercut at spacing 1e-4 (1.6e-4). Kept off; next: the tidal-tensor gradient.
- **Cluster runs** (host build, NBODY_MSTAR=1): 16-body group 3.3e-10 |dE/E| (FSI-dominated), 1 collision inside a
  group with N conserved, IMBH binary + 40-star cusp (42-body PN group) 2.0e-9 with E_pn booked, dense core (100 stars
  in r_h = 2e-3 pc, 73-body group) 1.2e-6 (9.8e-7 with MSTAR off: FSI-dominated; MSTAR's own unbooked error 2.9e-12).
- **A100 module results (gate 2):** all module tests pass on the A100 with round-off-level bounds; bench
  N = 1024 / 2048 / 4096: 3.2 / 8.7 / 28 ms per GBS step (device MST: no gain, open).
- **200k gpudev A/B (status 2026-10-25, open):** production 200k (r_h 3.1 pc, no IMBH): no group above 8 forms, the
  arms are bitwise identical (449 steps in 12 min, |dE/E| 7.5e-10 both). 200k + IMBH binary (500 + 500 Msun,
  a = 1e-4 pc) + 300-star cusp: neither arm finished interval 1 in 12 min; off: a 26-body group left to the FSI, HHS
  down to d32; on: no MSTAR call reported before the time limit (no per-call log yet), and a scaled-down host
  cut-out (2000 stars + the same IMBH cusp) also stalled silently.
- **IMBH cusp analysis (2026-10-25, host cut-out, NBODY_MSTAR_LOG):** the group does reach MSTAR: 238 of the 300 cusp
  stars + the binary form one MSTAR group (mass-scaled r_ngb links the whole 0.01 pc cusp), ~20 ms per call on the
  host. The stall is its HHS level: NBODY_DT_ATTR shows the tidal-sampling step (FLYBY_CM_EPS) of the 293 members
  at L13 at the start, then L29 (1.9e-12 Myr calls). Tighter bound for groups > 8 (sum m |r - R| |v - V| instead of
  2 max|r| sum m|v|): still L29. A 40-body core (NBODY_MSTAR_MAX=40, the rest field): worse (3e-14 Myr), the cut
  cusp perturbs it at close range. NBODY_MSTAR_TIDE with the linear tide only: step 1 done but |dE/E| 4e-5 (linear tide
  invalid for perturbers at the group's own size). Now NBODY_MSTAR_TIDE takes BIFROST-style perturbers: bodies within
  NBODY_MSTAR_PERT_R (3) x the group radius act inside MSTAR with their exact pull on straight lines (mean over the
  members left out; the host gives c.o.m. kicks), the rest as tidal tensor; the tidal-sampling cap is off for such
  groups. Cut-out: step 1 in < 300 s (HHS L10-L14, 300-body group), signed |dE/E| 2.5e-6 per step (books: PN 2.2e-6;
  MSTAR's own unbooked error 1.3e-8): too large. Predicted perturber motion x + v t + a t^2/2 (a relative to the
  group's c.o.m., NBODY_MSTAR_PERT_ACC): 6.9e-7 per step (3.7x better), still 70x above MSTAR's own 1e-8.
- **Literature (2026-10-25):** BIFROST (Rantala et al. 2023, sec. 3.2) deliberately keeps perturbers out of the
  regularised integration: the subsystem is integrated in isolation and the perturbations act through the FSI kicks
  (symplectic split), with subsystem time steps from a perturbation criterion eta_pert on the subsystem's own E and L
  rates (sec. 3.5); FROST-CLUSTERS I keeps that and adds SDAR inside; perturbers inside the regularised step are the
  AR-CHAIN / perturbed-chain approach (Mikkola & Aarseth; Mikkola & Merritt), where the perturber motion is predicted
  and the back-reaction is impulsive, i.e. not a symplectic split - which is the coupling error measured here. Open:
  either a symmetric treatment of the member-perturber interaction (back-reaction from the same time-averaged forces),
  or the BIFROST route (no perturbers; an eta_pert-type step for large groups instead of the global FLYBY_CM_EPS budget,
  to be A/B'd under the energy rule).
  Cut-out with NBODY_MSTAR_NO_TIDCAP=1 (impulsive kicks, no tidal-sampling cap for the group): interval 1 not done in
  900 s on 2 cores (50k GBS steps per print, 300-body group) - slower than the perturbed route; not pursued yet.
- **200k + IMBH cusp on 4 A100 with perturbed groups (ab5, 8ac994b: straight-line perturbers):** 4 intervals in 12
  min (off / plain MSTAR: none), ~180 s per interval, 31868 MSTAR calls (groups up to 302 bodies, 154k GBS steps);
  |dE/E| 2.35e-6 after interval 3, 2.15e-6 after 4 (most of it in interval 1; later intervals change it by 5e-8 -
  2e-7); MSTAR's own unbooked error 4e-9 - 2e-8 per interval. Not acceptable under the energy rule yet.
