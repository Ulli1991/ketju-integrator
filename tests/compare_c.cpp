// Ketju-integrator tests: the C++/Kokkos MSTAR against the C library.
// Copyright (C) 2026 Ulrich Steinwandel.
// Licensed under the GPLv3 license.
// See LICENSE.md or <https://www.gnu.org/licenses/> for details.
//
// This library against the C library (KETJU, branch ketju-c), same options, same initial state: agreement of the final
// states to round-off growth, and accuracy against independent references (Kepler: analytic; energies). The C
// library's results are stored in reference_data/compare_c.txt (keyed by a hash of the run's inputs; the header names
// the generating commit); MSTAR_REF_GENERATE (build against the C library of branch ketju-c, define
// MSTAR_REF_COMMIT, MSTAR_REF_DATA) regenerates the file. ctest mstar_compare_c. Usage:
//   mstar-compare-c [case ...]   cases: kepler_e09 pythagorean core100 core100_staged pn_binary imbh_cusp_pn
//                                device_tree bh_merger bh_merger_orbit pn_ledger stop_collision star_collisions
//                                external_tide softening no_star_star (default: all)
// Bounds: 1.3x the larger measured value of the CPU (gcc 15 -O2, Kokkos OpenMP 1 thread; bitwise for two bodies)
// and the A100 (CUDA 13.2; FMA contraction and team reductions make the GBS step sequence differ at the
// tolerance-1e-12 round-off floor) builds, noted at each case.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "mstar/mstar.hpp"
#include "ref_store.hpp"
#ifdef MSTAR_REF_GENERATE
#include "ketju_integrator/ketju_integrator.h"
#endif

namespace {

struct Sys {
    int npn = 0, n = 0;
    std::vector<double> m, x, v, s;   // s: spins of the PN particles
};

struct Merge {
    double t, m1, m2, m_rem, dE;
    int is_coll;
    double chi_rem = 0, v_kick = 0;
};
struct Run {
    std::vector<double> x, v, s, m, R;
    int n = 0, npn = 0;
    double t = 0, dE = 0;
    long steps = 0, fails = 0;
    double wall = 0;
    std::vector<Merge> merges;
};

struct Opt {
    double tol = 1e-10, out_tol = 1e-12, G = 1, c = 1;
    unsigned pn = 0;
    bool mergers = false;
    double merge_rs = 12;
    int team_max = 96;
    double soft = 0;                 // star_star_softening
    bool no_ss = false;              // no_star_star_gravity
    bool tide = false;               // external tide
    double T[6] = {};
    std::vector<double> radius;      // collision_radius (empty: off)
    int calls = 1;                   // the interval in this many run calls
    bool kink = true;                // softening_kink_landing
};

void unit_constants(mstar::System& M) {
    M.constants.G = 1;
    M.constants.c = 1;
    M.options.PN_flags = 0;
}

double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

#ifdef MSTAR_REF_GENERATE
Run run_c_live(const Sys& sy, const Opt& o, double T) {
    struct ketju_system R;
    ketju_create_system(&R, sy.npn, sy.n - sy.npn, MPI_COMM_WORLD);
    R.constants->G = o.G;
    R.constants->c = o.c;
    R.options->PN_flags = o.pn;
    R.options->gbs_relative_tolerance = o.tol;
    R.options->output_time_relative_tolerance = o.out_tol;
    R.options->enable_bh_mergers = o.mergers;
    R.options->bh_merger_distance_in_Rs = o.merge_rs;
    R.options->star_star_softening = o.soft;
    R.options->softening_kink_landing = o.kink;
    R.options->no_star_star_gravity = o.no_ss;
    R.options->enable_external_tide = o.tide;
    for (int q = 0; q < 6; q++) R.options->external_tidal_tensor[q] = o.T[q];
    std::vector<double> radius = o.radius;
    if (!radius.empty()) R.collision_radius = radius.data();
    for (int i = 0; i < sy.n; i++) {
        R.physical_state->mass[i] = sy.m[i];
        for (int k = 0; k < 3; k++) {
            R.physical_state->pos[i][k] = sy.x[3 * i + k];
            R.physical_state->vel[i][k] = sy.v[3 * i + k];
        }
    }
    for (int i = 0; i < sy.npn; i++)
        for (int k = 0; k < 3; k++) R.physical_state->spin[i][k] = sy.s[3 * i + k];
    const double t0 = now();
    double dE_sum = 0;   // relative energy error summed over the calls
    Run r;
    for (int q = 0; q < o.calls; q++) {
        const int m0 = R.num_mergers;
        const double t_call = R.physical_state->time;
        ketju_run_integrator(&R, T / o.calls);
        dE_sum += R.perf->relative_energy_error;
        for (int p = m0; p < R.num_mergers; p++) {   // KETJU records the time since the start of the call
            const auto& mi = R.merger_infos[p];
            r.merges.push_back({t_call + mi.t_merger, mi.m1, mi.m2, mi.m_remnant, mi.dE_collision, mi.is_collision, mi.chi_remnant, mi.v_kick});
        }
    }
    r.wall = now() - t0;
    if (!radius.empty()) r.R.assign(radius.begin(), radius.begin() + R.num_particles);
    r.n = R.num_particles;
    r.npn = R.num_pn_particles;
    r.t = R.physical_state->time;
    r.dE = o.calls > 1 ? dE_sum : R.perf->relative_energy_error;
    r.steps = R.perf->successful_steps;
    r.fails = R.perf->failed_steps;
    for (int i = 0; i < r.n; i++) {
        r.m.push_back(R.physical_state->mass[i]);
        for (int k = 0; k < 3; k++) {
            r.x.push_back(R.physical_state->pos[i][k]);
            r.v.push_back(R.physical_state->vel[i][k]);
        }
    }
    for (int i = 0; i < r.npn; i++)
        for (int k = 0; k < 3; k++) r.s.push_back(R.physical_state->spin[i][k]);
    ketju_free_system(&R);
    return r;
}
#endif

// ---- the C library's results: tests/reference_data/compare_c.txt (generated by the C library, see its header), keyed
// by a hash of the run's inputs; MSTAR_REF_GENERATE (needs the C library on branch ketju-c) writes the file instead
refstore::Store g_ref;
std::string run_key(const Sys& sy, const Opt& o, double T) {
    refstore::Hash h;
    h.add((long)sy.npn); h.add((long)sy.n); h.add(sy.m); h.add(sy.x); h.add(sy.v); h.add(sy.s);
    h.add(o.tol); h.add(o.out_tol); h.add(o.G); h.add(o.c); h.add((long)o.pn); h.add((long)o.mergers); h.add(o.merge_rs);
    h.add(o.soft); h.add((long)o.no_ss); h.add((long)o.tide);
    for (double t : o.T) h.add(t);
    h.add(o.radius); h.add((long)o.calls); h.add((long)o.kink); h.add(T);
    return h.str();
}
std::vector<double> pack(const Run& r) {
    std::vector<double> v = {(double)r.n, (double)r.npn, r.t, r.dE, (double)r.steps, (double)r.fails, r.wall, (double)r.merges.size()};
    for (const auto& m : r.merges) {
        const double q[8] = {m.t, m.m1, m.m2, m.m_rem, m.dE, (double)m.is_coll, m.chi_rem, m.v_kick};
        v.insert(v.end(), q, q + 8);
    }
    for (const auto* a : {&r.R, &r.m, &r.x, &r.v, &r.s}) {
        v.push_back((double)a->size());
        v.insert(v.end(), a->begin(), a->end());
    }
    return v;
}
Run unpack(const std::vector<double>& v) {
    Run r;
    size_t p = 0;
    r.n = (int)v[p++]; r.npn = (int)v[p++]; r.t = v[p++]; r.dE = v[p++]; r.steps = (long)v[p++]; r.fails = (long)v[p++];
    r.wall = v[p++];
    const size_t nm = (size_t)v[p++];
    for (size_t i = 0; i < nm; i++, p += 8)
        r.merges.push_back({v[p], v[p + 1], v[p + 2], v[p + 3], v[p + 4], (int)v[p + 5], v[p + 6], v[p + 7]});
    for (auto* a : {&r.R, &r.m, &r.x, &r.v, &r.s}) {
        const size_t k = (size_t)v[p++];
        a->assign(v.begin() + p, v.begin() + p + k);
        p += k;
    }
    return r;
}
Run run_c(const Sys& sy, const Opt& o, double T) {
    const std::string key = run_key(sy, o, T);
#ifdef MSTAR_REF_GENERATE
    std::vector<double> v;
    if (!g_ref.get(key, v)) g_ref.put(key, pack(run_c_live(sy, o, T)));
#endif
    std::vector<double> v2;
    if (!g_ref.get(key, v2)) {
        std::printf("FAIL: no reference run %s in compare_c.txt (n=%d, T=%g); regenerate on branch ketju-c\n", key.c_str(), sy.n, T);
        std::exit(2);
    }
    return unpack(v2);
}

bool g_device_tree = false;   // force the one-team Prim tree (device_tree case)
Run run_k(const Sys& sy, const Opt& o, double T, mstar::PerformanceCounters* perf_out = nullptr) {
    mstar::System M(sy.npn, sy.n - sy.npn);
    unit_constants(M);
    if (g_device_tree) { M.options.device_tree_everywhere = true; M.options.device_tree_min_n = 0; }
    if (std::getenv("MSTAR_TEST_HOST_ENGINE")) M.options.host_engine = true;   // diagnostics: the host engine in GPU builds
    M.constants.G = o.G;
    M.constants.c = o.c;
    M.options.PN_flags = o.pn;
    M.options.gbs_relative_tolerance = o.tol;
    M.options.output_time_relative_tolerance = o.out_tol;
    M.options.enable_bh_mergers = o.mergers;
    M.options.bh_merger_distance_in_Rs = o.merge_rs;
    M.options.team_rows_max_n = M.options.team_rows_max_n_device = o.team_max;
    M.options.star_star_softening = o.soft;
    M.options.softening_kink_landing = o.kink;
    M.options.no_star_star_gravity = o.no_ss;
    M.options.enable_external_tide = o.tide;
    for (int q = 0; q < 6; q++) M.options.external_tidal_tensor[q] = o.T[q];
    M.collision_radius = o.radius;
    M.mass = sy.m;
    M.pos = sy.x;
    M.vel = sy.v;
    M.spin = sy.s;
    M.spin.resize(3 * sy.npn);
    const double t0 = now();
    double dE_sum = 0;
    for (int q = 0; q < o.calls; q++) {
        M.run(T / o.calls);
        dE_sum += M.perf.relative_energy_error;
    }
    Run r;
    r.wall = now() - t0;
    for (const auto& mi : M.mergers) r.merges.push_back({mi.t_merger, mi.m1, mi.m2, mi.m_remnant, mi.dE_collision, mi.is_collision});
    r.R = M.collision_radius;
    r.n = M.num_particles;
    r.npn = M.num_pn_particles;
    r.t = M.time;
    r.dE = o.calls > 1 ? dE_sum : M.perf.relative_energy_error;
    r.steps = M.perf.successful_steps;
    r.fails = M.perf.failed_steps;
    r.x = M.pos;
    r.v = M.vel;
    r.s = M.spin;
    r.m = M.mass;
    if (perf_out) *perf_out = M.perf;
    return r;
}

// max |a - b| / scale over the arrays (scale: max |a|)
double rel_diff(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) return 1e300;
    double mx = 0, sc = 0;
    for (size_t q = 0; q < a.size(); q++) {
        mx = std::fmax(mx, std::fabs(a[q] - b[q]));
        sc = std::fmax(sc, std::fabs(a[q]));
    }
    return sc > 0 ? mx / sc : mx;
}

int g_fail = 0;
bool check(const char* c, const char* what, double v, double bound, bool lower = false) {
    const bool ok = lower ? v >= bound : v <= bound;
    std::printf("  %s %-14s %-22s %.3e (%s %.3e)\n", ok ? "ok  " : "FAIL", c, what, v, lower ? ">=" : "<=", bound);
    if (!ok) g_fail++;
    return ok;
}

// bound_merge: |t_C - t_K| / T of every merger / collision (both are recorded at the end of the accepted step, so
// with different step sequences they differ by up to a step), bound_mdE: |dE_coll,C - dE_coll,K| / |dE_coll,C|
// Drift of the total momentum and angular momentum of r against the initial state s, relative to sum m |v| and
// sum m |x x v| (the larger of initial and final; conserved without PN, tide and extra forces; a contact collision
// conserves P and removes the pair's internal angular momentum from L, in both codes alike)
void pl_drift(const Sys& s, const Run& r, double& dP, double& dL) {
    auto sums = [](const std::vector<double>& m, const std::vector<double>& x, const std::vector<double>& v, int n,
                   double* P, double* L, double& sp, double& sl) {
        for (int k = 0; k < 3; k++) P[k] = L[k] = 0;
        sp = sl = 0;
        for (int i = 0; i < n; i++) {
            const double* q = &x[3 * i];
            const double* u = &v[3 * i];
            const double l[3] = {q[1] * u[2] - q[2] * u[1], q[2] * u[0] - q[0] * u[2], q[0] * u[1] - q[1] * u[0]};
            for (int k = 0; k < 3; k++) { P[k] += m[i] * u[k]; L[k] += m[i] * l[k]; }
            sp += m[i] * std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
            sl += m[i] * std::sqrt(l[0] * l[0] + l[1] * l[1] + l[2] * l[2]);
        }
    };
    double P0[3], L0[3], P1[3], L1[3], sp, sl, sp1, sl1;
    sums(s.m, s.x, s.v, s.n, P0, L0, sp, sl);
    sums(r.m, r.x, r.v, r.n, P1, L1, sp1, sl1);
    sp = std::fmax(sp, sp1);
    sl = std::fmax(sl, sl1);
    dP = dL = 0;
    for (int k = 0; k < 3; k++) {
        dP = std::fmax(dP, std::fabs(P1[k] - P0[k]) / sp);
        dL = std::fmax(dL, std::fabs(L1[k] - L0[k]) / sl);
    }
}

// Returned: the largest C-vs-Kokkos differences (state, merger times, collision energies) for convergence checks
struct Diff {
    double x = 0, t_merge = 0, dE_coll = 0;
};
Diff compare(const char* name, const Sys& sy, const Opt& o, double T, double bound_x, double bound_dE_c,
             double bound_dE_k, double bound_merge = 1e-12, double bound_mdE = 1e-12) {
    Diff df;
    const Run a = run_c(sy, o, T);
    const Run b = run_k(sy, o, T);
    std::printf("%s: N=%d npn=%d T=%g  C: %ld steps (%ld fail) dE %.3e %.3fs | Kokkos: %ld steps (%ld fail) dE %.3e %.3fs\n",
                name, sy.n, sy.npn, T, a.steps, a.fails, a.dE, a.wall, b.steps, b.fails, b.dE, b.wall);
    check(name, "N_C - N_K", std::fabs((double)(a.n - b.n)), 0.0);
    if (!o.pn && !o.tide) {   // momentum and angular momentum: Kokkos as good as C, or at the round-off floor 1.5e-13
                              // (C: up to 1.0e-13 for 40-100 bodies)
        double pa, la, pb, lb;
        pl_drift(sy, a, pa, la);
        pl_drift(sy, b, pb, lb);
        std::printf("  dP C %.2e K %.2e, dL C %.2e K %.2e\n", pa, pb, la, lb);
        check(name, "|dP| Kokkos (<= 2 C, 1.5e-13)", pb, std::fmax(2 * pa, 1.5e-13));
        check(name, "|dL| Kokkos (<= 2 C, 1.5e-13)", lb, std::fmax(2 * la, 1.5e-13));
    }
    check(name, "merger count C - K", std::fabs((double)a.merges.size() - (double)b.merges.size()), 0.0);
    for (size_t q = 0; q < a.merges.size() && q < b.merges.size(); q++) {
        const Merge &x = a.merges[q], &y = b.merges[q];
        std::printf("  merger %zu: coll %d/%d t %.12g / %.12g m1 %g/%g m2 %g/%g dE %.12e / %.12e\n", q, x.is_coll, y.is_coll,
                    x.t, y.t, x.m1, y.m1, x.m2, y.m2, x.dE, y.dE);
        check(name, "merger kind C - K", std::fabs((double)(x.is_coll - y.is_coll)), 0.0);
        check(name, "merger |dt| / T", std::fabs(x.t - y.t) / T, bound_merge);
        df.t_merge = std::fmax(df.t_merge, std::fabs(x.t - y.t) / T);
        if (x.is_coll) df.dE_coll = std::fmax(df.dE_coll, std::fabs(x.dE - y.dE) / std::fabs(x.dE));
        check(name, "merger masses", std::fabs(x.m1 - y.m1) + std::fabs(x.m2 - y.m2), 1e-14 * (x.m1 + x.m2));
        if (x.is_coll) check(name, "collision |ddE| / |dE|", std::fabs(x.dE - y.dE) / std::fabs(x.dE), bound_mdE);
    }
    if (!a.R.empty()) check(name, "collision radii C vs K", rel_diff(a.R, b.R), 0.0);
    check(name, "|dx|/x C vs Kokkos", rel_diff(a.x, b.x), bound_x);
    check(name, "|dv|/v C vs Kokkos", rel_diff(a.v, b.v), bound_x);
    df.x = std::fmax(rel_diff(a.x, b.x), rel_diff(a.v, b.v));
    if (sy.npn) check(name, "|dS|/S C vs Kokkos", rel_diff(a.s, b.s), bound_x);
    check(name, "|t_C - t_K| / T", std::fabs(a.t - b.t) / T, 1e-12);
    check(name, "|dE/E| C", std::fabs(a.dE), bound_dE_c);
    check(name, "|dE/E| Kokkos", std::fabs(b.dE), bound_dE_k);
    check(name, "step count diff", std::fabs((double)(a.steps - b.steps)), std::fmax(2.0, 0.25 * a.steps));   // A100: 19 of 98 (Kepler)
    return df;
}

void kepler_pair(double m1, double m2, double a, double e, double* q, double* v) {
    // apocentre on the x axis
    const double M = m1 + m2, r = a * (1 + e), vr = std::sqrt(M / a * (1 - e) / (1 + e));
    const double x[3] = {r, 0, 0}, u[3] = {0, vr, 0};
    for (int k = 0; k < 3; k++) {
        q[k] = m2 / M * x[k]; q[3 + k] = -m1 / M * x[k];
        v[k] = m2 / M * u[k]; v[3 + k] = -m1 / M * u[k];
    }
}

Sys plummer(int n, int npn, unsigned seed, double m_bh = 0) {
    // Plummer sphere (Aarseth, Henon & Wielen 1974), G = M = 1, virial units (r scale 3 pi / 16); first npn bodies
    // get mass m_bh each (added at rest near the centre when m_bh > 0)
    std::mt19937_64 g(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    Sys s;
    s.n = n;
    s.npn = npn;
    s.m.assign(n, 1.0 / (n - npn));
    s.x.assign(3 * n, 0);
    s.v.assign(3 * n, 0);
    s.s.assign(3 * npn, 0);
    const double rs = 3 * M_PI / 16;
    for (int i = npn; i < n; i++) {
        double r;
        do { r = 1.0 / std::sqrt(std::pow(U(g), -2.0 / 3.0) - 1.0); } while (r > 10);
        const double ct = 2 * U(g) - 1, ph = 2 * M_PI * U(g), st = std::sqrt(1 - ct * ct);
        s.x[3 * i] = rs * r * st * std::cos(ph);
        s.x[3 * i + 1] = rs * r * st * std::sin(ph);
        s.x[3 * i + 2] = rs * r * ct;
        double q, y;
        do { q = U(g); y = 0.1 * U(g); } while (y > q * q * std::pow(1 - q * q, 3.5));
        const double ve = q * std::sqrt(2.0) * std::pow(1 + r * r, -0.25) / std::sqrt(rs);
        const double c2 = 2 * U(g) - 1, p2 = 2 * M_PI * U(g), s2 = std::sqrt(1 - c2 * c2);
        s.v[3 * i] = ve * s2 * std::cos(p2);
        s.v[3 * i + 1] = ve * s2 * std::sin(p2);
        s.v[3 * i + 2] = ve * c2;
    }
    for (int i = 0; i < npn; i++) s.m[i] = m_bh;
    // centre of mass frame
    double M = 0, cx[3] = {0, 0, 0}, cv[3] = {0, 0, 0};
    for (int i = 0; i < n; i++) {
        M += s.m[i];
        for (int k = 0; k < 3; k++) { cx[k] += s.m[i] * s.x[3 * i + k]; cv[k] += s.m[i] * s.v[3 * i + k]; }
    }
    for (int i = 0; i < n; i++)
        for (int k = 0; k < 3; k++) { s.x[3 * i + k] -= cx[k] / M; s.v[3 * i + k] -= cv[k] / M; }
    return s;
}

// Collision times and energies against the GBS tolerance (no checks, a table): both codes record a collision at the
// end of the accepted step on which it is detected, so t_merger is resolved only to a step and does not converge with
// the tolerance; the C-vs-C spread over tolerances is the scale against which the Kokkos-vs-C differences are judged
void collision_tolscan(const char* name, const Sys& sy, Opt o, double T) {
    const double tols[4] = {1e-8, 1e-10, 1e-12, 1e-13};
    Run c[4], k[4];
    for (int q = 0; q < 4; q++) {
        o.tol = tols[q];
        c[q] = run_c(sy, o, T);
        k[q] = run_k(sy, o, T);
    }
    std::printf("%s tolerance scan (|dt|/T, |ddE|/|dE| per merger; C-vs-C: against C at 1e-13):\n", name);
    for (int q = 0; q < 4; q++) {
        std::printf("  tol %.0e  C %ld steps |dE/E| %.2e, K %ld steps |dE/E| %.2e | x K-C %.2e C-C %.2e |", tols[q], c[q].steps,
                    std::fabs(c[q].dE), k[q].steps, std::fabs(k[q].dE),
                    std::fmax(rel_diff(c[q].x, k[q].x), rel_diff(c[q].v, k[q].v)),
                    std::fmax(rel_diff(c[q].x, c[3].x), rel_diff(c[q].v, c[3].v)));
        for (size_t m = 0; m < c[q].merges.size() && m < k[q].merges.size() && m < c[3].merges.size(); m++)
            std::printf("  m%zu K-C %.2e %.2e  C-C %.2e %.2e", m, std::fabs(k[q].merges[m].t - c[q].merges[m].t) / T,
                        std::fabs(k[q].merges[m].dE - c[q].merges[m].dE) / std::fabs(c[q].merges[m].dE),
                        std::fabs(c[q].merges[m].t - c[3].merges[m].t) / T,
                        std::fabs(c[q].merges[m].dE - c[3].merges[m].dE) / std::fabs(c[3].merges[m].dE));
        std::printf("\n");
    }
}

// Segments: from every C state at t = 0, T / nseg, ... both libraries integrate T / nseg (a few steps, no chaotic
// amplification of round-off): Kokkos must give the same collisions, the same summed |dE/E| (within 1.3x) and the
// same states to round-off growth. This compares the arithmetic of the two libraries (also on the GPU) where whole
// trajectories diverge through the step sequence. bound_x: the per-segment state bound (default: round-off growth).
void segment_compare(const char* name, const Sys& s, const Opt& o, double T, int nseg, double bound_x = 1e-9) {
    Sys c = s;
    Opt oc = o;
    double sum_c = 0, sum_k = 0, dx_max = 0;
    int n_mismatch = 0;
    for (int seg = 0; seg < nseg; seg++) {
        const Run rc = run_c(c, oc, T / nseg), rk = run_k(c, oc, T / nseg);
        sum_c += std::fabs(rc.dE);
        sum_k += std::fabs(rk.dE);
        if (rc.n != rk.n) n_mismatch++;
        else dx_max = std::fmax(dx_max, std::fmax(rel_diff(rc.x, rk.x), rel_diff(rc.v, rk.v)));
        c.n = rc.n; c.m = rc.m; c.x = rc.x; c.v = rc.v;
        oc.radius = rc.R;
    }
    std::printf("%s: %d segments, sum |dE/E| C %.3e Kokkos %.3e, max |dx|/x %.3e, collision mismatches %d\n", name, nseg,
                sum_c, sum_k, dx_max, n_mismatch);
    check(name, "collision mismatches", n_mismatch, 0.0);
    check(name, "sum |dE/E| K / C", sum_k / sum_c, 1.3);
    check(name, "max segment |dx|/x", dx_max, bound_x);
}

double g_ens_max_c = 0, g_ens_max_k = 0;   // maxima of the last energy_ensemble
// |dE/E| over an ensemble of 128 initial states (x_0 shifted by q 1e-9): median and 90th percentile, C and Kokkos
void energy_ensemble(const char* name, const Sys& sy, const Opt& o, double T, double* med_c, double* p90_c, double* med_k,
                     double* p90_k) {
    std::vector<double> ec, ek;
    for (int q = 0; q < 128; q++) {
        Sys s = sy;
        s.x[0] += q * 1e-9;
        ec.push_back(std::fabs(run_c(s, o, T).dE));
        ek.push_back(std::fabs(run_k(s, o, T).dE));
    }
    std::sort(ec.begin(), ec.end());
    std::sort(ek.begin(), ek.end());
    *med_c = ec[64]; *p90_c = ec[115]; *med_k = ek[64]; *p90_k = ek[115];
    g_ens_max_c = ec[127];
    g_ens_max_k = ek[127];
    std::printf("%s: |dE/E| over 128 states: C median %.2e p90 %.2e max %.2e | Kokkos median %.2e p90 %.2e max %.2e\n", name,
                *med_c, *p90_c, ec[127], *med_k, *p90_k, ek[127]);
}

bool want(int argc, char** argv, const char* c) {
    bool any = false;
    for (int a = 1; a < argc; a++)
        if (argv[a][0] != '-') { any = true; if (!std::strcmp(argv[a], c)) return true; }
    return !any;
}

}  // namespace

int main(int argc, char** argv) {
#ifdef MSTAR_REF_GENERATE
    MPI_Init(&argc, &argv);   // KETJU needs MPI (or the serial stand-in)
#endif
    Kokkos::initialize(argc, argv);
    {
#ifdef MSTAR_REF_GENERATE
        g_ref.header = std::string("# C-library reference runs of mstar-compare-c, generated by the C library of branch ketju-c at commit ") +
                       MSTAR_REF_COMMIT + "; one line per run: key count values (hexadecimal floats)\n";
        g_ref.load(MSTAR_REF_DATA "/compare_c.txt");
#else
        if (!g_ref.load(MSTAR_REF_DATA "/compare_c.txt")) { std::printf("FAIL: cannot read %s\n", MSTAR_REF_DATA "/compare_c.txt"); return 2; }
#endif
        if (want(argc, argv, "kepler_e09")) {
            // e = 0.9, 10 orbits, apocentre to apocentre: analytic reference = the initial state
            Sys s;
            s.n = 2;
            s.m = {1.0, 0.5};
            s.x.assign(6, 0);
            s.v.assign(6, 0);
            kepler_pair(1.0, 0.5, 1.0, 0.9, s.x.data(), s.v.data());
            const double P = 2 * M_PI / std::sqrt(1.5);
            Opt o;
            o.tol = 1e-12;
            compare("kepler_e09", s, o, 10 * P, 4.9e-11, 1.8e-12, 1.8e-12);   // CPU bitwise; A100 dx 6.5e-12, dv 3.8e-11
            const Run b = run_k(s, o, 10 * P);
            check("kepler_e09", "|x - x_analytic|", rel_diff(s.x, b.x), 5.7e-12);   // CPU 2.13e-12, A100 4.4e-12
        }
        if (want(argc, argv, "pythagorean")) {
            // Burrau's problem (masses 3, 4, 5 at the corners of a 3-4-5 triangle, at rest), to t = 10 (before the
            // close approaches that amplify round-off)
            Sys s;
            s.n = 3;
            s.m = {3, 4, 5};
            s.x = {1, 3, 0, -2, -1, 0, 1, -1, 0};
            s.v.assign(9, 0);
            Opt o;
            o.tol = 1e-12;
            compare("pythagorean", s, o, 10.0, 2.4e-10, 1.8e-12, 5.6e-12);   // dv CPU 1.27e-10, A100 1.84e-10; dE A100 4.3e-12
        }
        if (want(argc, argv, "core100")) {
            Sys s = plummer(100, 0, 42);
            Opt o;
            o.tol = 1e-10;
            compare("core100", s, o, 0.25, 1.3e-12, 1.3e-13, 1.3e-13);   // measured dx 5.4e-14, dv 9.9e-13, dE 7.9e-15 / 9.9e-14
        }
        if (want(argc, argv, "core100_staged")) {
            Sys s = plummer(100, 0, 42);
            Opt o;
            o.tol = 1e-10;
            o.team_max = 0;   // staged path (separate force / kick-drift launches)
            compare("core100_staged", s, o, 0.25, 1.3e-12, 1.3e-13, 1.3e-13);
        }
        if (want(argc, argv, "device_tree")) {
            // the one-team Prim (device tree) must give the host Prim's tree: identical runs
            Sys s = plummer(100, 0, 42);
            Opt o;
            o.tol = 1e-10;
            const Run a = run_k(s, o, 0.25);
            g_device_tree = true;
            const Run b = run_k(s, o, 0.25);
            g_device_tree = false;
            check("device_tree", "|dx|/x host vs team Prim", rel_diff(a.x, b.x), 0.0);
            check("device_tree", "step count diff", std::fabs((double)(a.steps - b.steps)), 0.0);
        }
        if (want(argc, argv, "pn_binary")) {
            // KETJU unit test binary_system_full_pn_spin (G = c = 1), shortened to 1e5
            Sys s;
            s.n = 2;
            s.npn = 2;
            s.m = {0.5, 1.5};
            s.x = {40, 0, 0, -40, 0, 0};
            s.v = {0, .085, 0, 0, -.085, 0};
            s.s = {1, 0, 0, 0, 1, 0};
            Opt o;
            o.tol = 1e-12;
            o.pn = 0x7ff;   // KETJU_PN_ALL
            compare("pn_binary", s, o, 1e5, 9.8e-11, 1e300, 1e300);   // CPU bitwise; A100 dx 9.4e-13, dv 7.5e-11 (dE: PN)
        }
        if (want(argc, argv, "imbh_cusp_pn")) {
            // two BHs (0.05 each) bound at the centre of a 60-star cusp, all PN terms (c = 30), staged and team
            Sys s = plummer(62, 2, 7, 0.05);
            s.x[0] = 0.02; s.x[3] = -0.02;
            const double vb = std::sqrt(0.1 / 0.04) / 2;
            s.v[1] = vb; s.v[4] = -vb;
            s.s = {0.0005, 0, 0.001, 0, 0, -0.0008};
            Opt o;
            o.tol = 1e-10;
            o.c = 30;
            o.pn = 0x7ff;
            compare("imbh_cusp_pn", s, o, 0.05, 7.3e-13, 1e300, 1e300);   // dv CPU 1.05e-13, A100 1.67e-13, A100 without FMA 5.57e-13 (team reduction order: round-off)
            o.team_max = 0;
            compare("imbh_cusp_pn_st", s, o, 0.05, 2.2e-13, 1e300, 1e300);
        }
        if (want(argc, argv, "pn_ledger")) {
            // PN work ledger: in the 2-BH cusp the Newtonian T + U changes by the PN forces (-pn_dB) plus the
            // integration error; the error must stay at the GBS tolerance level while the PN part is much larger
            Sys s = plummer(62, 2, 7, 0.05);
            s.x[0] = 0.02; s.x[3] = -0.02;
            const double vb = std::sqrt(0.1 / 0.04) / 2;
            s.v[1] = vb; s.v[4] = -vb;
            mstar::System M(2, 60);
            unit_constants(M);
            M.constants.c = 30;
            M.options.PN_flags = 0x7ff;
            M.options.gbs_relative_tolerance = 1e-10;
            M.mass = s.m; M.pos = s.x; M.vel = s.v; M.spin = {0.0005, 0, 0.001, 0, 0, -0.0008};
            auto E = [&](const std::vector<double>& x, const std::vector<double>& v) {
                double T = 0, W = 0;
                for (int i = 0; i < s.n; i++) {
                    T += 0.5 * s.m[i] * (v[3*i]*v[3*i] + v[3*i+1]*v[3*i+1] + v[3*i+2]*v[3*i+2]);
                    for (int j = i + 1; j < s.n; j++) {
                        double r2 = 0;
                        for (int k = 0; k < 3; k++) r2 += (x[3*i+k] - x[3*j+k]) * (x[3*i+k] - x[3*j+k]);
                        W -= s.m[i] * s.m[j] / std::sqrt(r2);
                    }
                }
                return T + W;
            };
            const double E0 = E(s.x, s.v);
            M.run(0.2);
            const double dE = E(M.pos, M.vel) - E0, dE_pn = -M.pn_dB;
            std::printf("pn_ledger: dE/E0 %.3e, PN part %.3e, rest (integration error) %.3e\n", dE / std::fabs(E0),
                        dE_pn / std::fabs(E0), (dE - dE_pn) / std::fabs(E0));
            check("pn_ledger", "|dE - dE_pn| / |E0|", std::fabs(dE - dE_pn) / std::fabs(E0), 2.6e-13);   // CPU 6.46e-14, A100 1.15e-13, A100 without FMA 2.02e-13
            check("pn_ledger", "|E0| / |dE_pn| (PN visible)", std::fabs(E0) / std::fmax(std::fabs(dE_pn), 1e-300), 1e7);
        }
        if (want(argc, argv, "stop_collision")) {
            // two stars (R = 0.001) of a 40-star group heading for each other (head-on), and a second pair on a
            // grazing hyperbolic passage with osculating pericentre 0.5 r_coll: both must stop the call at the
            // first of them; the grazing one is decided on its osculating pericentre (r_sep < r_coll)
            for (int variant = 0; variant < 2; variant++) {
                Sys s = plummer(40, 0, 11);
                for (int k = 0; k < 6; k++) { s.x[k] = 0; s.v[k] = 0; }
                s.x[0] = 0.05; s.x[3] = -0.05;
                if (variant == 0) { s.v[0] = -1; s.v[3] = 1; }
                else {   // impact parameter b with pericentre 0.5 r_coll = 0.001: b^2 = rp^2 + 2 G M rp / v_inf^2
                    const double rp = 0.001, vinf = 2.0, mu = 2.0 / 40, b = std::sqrt(rp * rp + 2 * mu * rp / (vinf * vinf));
                    s.x[1] = b / 2; s.x[4] = -b / 2; s.v[0] = -1; s.v[3] = 1;
                }
                mstar::System M(0, 40);
                unit_constants(M);
                M.options.gbs_relative_tolerance = 1e-10;
                M.mass = s.m; M.pos = s.x; M.vel = s.v;
                M.stop_collisions.radius.assign(40, 1e-5);
                M.stop_collisions.radius[0] = M.stop_collisions.radius[1] = 0.001;
                M.stop_collisions.is_bh.assign(40, 0);
                M.run(1.0);
                const char* nm = variant ? "coll_graze" : "coll_headon";
                std::printf("%s: stop t=%.6f pair %d-%d r_sep=%.3e r_coll=%.3e type %d\n", nm, M.time, M.stop_collisions.i, M.stop_collisions.j,
                            M.stop_collisions.r_sep, M.stop_collisions.r_coll, M.stop_collisions.type);
                check(nm, "pair != 0-1", (M.stop_collisions.i == 0 && M.stop_collisions.j == 1) ? 0.0 : 1.0, 0.0);
                check(nm, "r_sep / r_coll", M.stop_collisions.r_sep / M.stop_collisions.r_coll, 1.0);
                check(nm, "t_stop (< 0.05)", M.time, 0.05);
            }
        }
        if (want(argc, argv, "star_collisions")) {
            // contact collisions inside MSTAR (collision_radius, GIZMO addition): in a 40-star group stars 0-1 head-on
            // and stars 2-3 on a grazing hyperbolic passage with two-body pericentre 0.5 (R2 + R3) (the pericentre test
            // decides it); in one call and in 10 host-sized calls (the r.v history restarts at every call)
            Sys s = plummer(40, 0, 11);
            for (int k = 0; k < 12; k++) { s.x[k] = 0; s.v[k] = 0; }
            const double rp = 0.001, vinf = 2.0, mu = 2.0 / 40, b = std::sqrt(rp * rp + 2 * mu * rp / (vinf * vinf));
            s.x[0] = 0.05; s.x[3] = -0.05; s.v[0] = -1; s.v[3] = 1;
            s.x[6] = 0.05; s.x[7] = 0.3 + b / 2; s.x[9] = -0.05; s.x[10] = 0.3 - b / 2; s.v[6] = -1; s.v[9] = 1;
            Opt o;
            o.tol = 1e-10;
            o.radius.assign(40, 1e-5);
            for (int i = 0; i < 4; i++) o.radius[i] = 0.001;
            // state, collision times and energies differ by the step-end recording of the collisions: bounds 1.3x the largest
            // Kokkos-vs-C or C-vs-C difference of collision_tolscan (CPU: x 7.1e-8, dt 9.2e-4, ddE 1.0e-8; A100 at 1e-10:
            // 4.6e-8, 6.7e-4, 6.6e-9); |dE/E| C 1.0e-13, CPU 6.2e-14, A100 6.7e-15, A100 without FMA 5.0e-13: it follows
            // the step sequence at the 1e-12 level (C over tolerances 1e-8..1e-13: 3.7e-14..1.2e-12), bound 1.3x that
            compare("star_collisions", s, o, 0.1, 9.3e-8, 1.4e-13, 1.55e-12, 1.2e-3, 1.3e-8);
            o.calls = 10;
            // in 10 calls the step sequences diverge as in one (A100 without FMA: dv 5.7e-9, dt 4.0e-4, ddE 1.4e-9, |dE/E|
            // 8.8e-14; CPU 5.1e-10, 4.0e-5, 1.0e-10, 3.1e-14; A100 8.7e-10, 7.1e-5, 1.7e-10, 2.7e-14): the bounds of the
            // single call
            compare("star_collisions_10", s, o, 0.1, 9.3e-8, 4.1e-14, 1.55e-12, 1.2e-3, 1.3e-8);
        }
        if (want(argc, argv, "coll_tide_soft")) {
            // a collision during the run with star_star_softening and the external tide. Well-conditioned set-up:
            // h = 0.05 below the contact distance R1 + R2 = 0.1, so no integrated pair crosses the kernel's break
            // points (u = 0.5, 1) before it merges (softening_ensemble below for pairs that do)
            Sys s;
            s.n = 3;
            s.m = {1, 0.7, 0.5};
            s.x = {1, 0, 0, -1, 0.02, 0, 0, 3, 0.5};
            s.v = {-0.3, 0, 0, 0.3, 0, 0, 0.2, 0, 0};
            Opt o;
            o.tol = 1e-11;
            o.out_tol = 1e-12;
            o.soft = 0.05;
            o.tide = true;
            const double T[6] = {-0.002, 0.001, 0.0015, 0.0004, -0.0003, 0.0006};
            for (int q = 0; q < 6; q++) o.T[q] = T[q];
            o.radius = {0.05, 0.05, 0.05};
            // bounds: |dE/E| CPU 4.1e-13 (tolerance 1e-11); the state, collision time and energy: 1.3x the C-vs-C spread over
            // tolerances 1e-8..1e-13 (collision_tolscan: x 7.1e-7, dt 9.4e-5, ddE 6.1e-8; CPU Kokkos vs C bitwise)
            // (A100: dt 1.26e-4, ddE 7.6e-8 at tolerance 1e-10 in the scan); merger time bound 1.3x the A100 value
            compare("coll_tide_soft", s, o, 20, 9.2e-7, 1e-12, 1e-12, 1.65e-4, 9.9e-8);
            o.team_max = 0;   // staged path (the device default)
            compare("coll_tide_soft_st", s, o, 20, 9.2e-7, 1e-12, 1e-12, 1.65e-4, 9.9e-8);
            o.team_max = 96;
            o.tol = 1e-13;
            // |dE/E| C and CPU 2.0e-13, A100 8.9e-13 (a single trajectory; in the tolerance scan the A100 is below C at 1e-8
            // and 1e-12, and the segments below compare the arithmetic directly): bound 1.3x the A100 value
            compare("coll_tide_soft_1e-13", s, o, 20, 9.2e-7, 1.2e-12, 1.2e-12, 1.65e-4, 9.9e-8);
            // per-segment states: CPU 2.2e-14, A100 without FMA 2.2e-14, A100 (FMA, the nvcc default) 2.9e-7: FMA rounding
            // changes a step size, and the segment holding the collision then merges the pair at a different step
            // end (collisions are recorded at step ends); collisions and energy are unaffected (sum |dE/E| A100 4.1e-13,
            // C 8.7e-13). Bound 1.3x the A100 value
            segment_compare("coll_tide_soft_segments", s, o, 20, 80, 3.8e-7);
            if (want(argc, argv, "collision_tolscan")) collision_tolscan("coll_tide_soft", s, o, 20);
        }
        if (want(argc, argv, "softening_ensemble")) {
            // A pair moving through the softening kernel's break points (u = 0.5 and 1; the spline's 4th force
            // derivative jumps there) defeats the GBS error estimate unless the step is landed on the crossing
            // (softening_kink_landing, both libraries). Without landing |dE/E| was 1e2 - 1e6 x the tolerance and varied
            // by orders of magnitude with 1e-9 changes of the initial state (C: max 8e-7 at tolerance 1e-11, 8e-6 at
            // 1e-13 over 128 states); with landing it is at the tolerance level for every state.
            Sys s;
            s.n = 3;
            s.m = {1, 0.7, 0.5};
            s.x = {1, 0, 0, -1, 0.02, 0, 0, 3, 0.5};
            s.v = {-0.3, 0, 0, 0.3, 0, 0, 0.2, 0, 0};
            Opt o;
            o.out_tol = 1e-12;
            o.soft = 0.3;
            o.tide = true;
            const double T[6] = {-0.002, 0.001, 0.0015, 0.0004, -0.0003, 0.0006};
            for (int q = 0; q < 6; q++) o.T[q] = T[q];
            o.radius = {0.05, 0.05, 0.05};
            for (double tol : {1e-11, 1e-13}) {
                o.tol = tol;
                // (a) segments (segment_compare): the same arithmetic must give the same per-segment errors
                // per-segment states: CPU 2.3e-14 / 7.6e-15, A100 without FMA the same, A100 (FMA, the nvcc default)
                // 1.1e-6 / 7.4e-7: FMA rounding meets the GBS error estimate at the softening-kernel kinks (u = 0.5, 1),
                // the step sequence diverges inside a segment; collisions match and the summed |dE/E| is lower on the
                // A100 (Kokkos / C 0.47 / 0.12). Bounds 1.3x the A100 values
                segment_compare(tol > 1e-12 ? "softening_segments_1e-11" : "softening_segments_1e-13", s, o, 20, 80,
                                tol > 1e-12 ? 1.5e-6 : 9.7e-7);
                // (b) ensembles over 128 initial states (x_0 + q 1e-9). Measured (CPU, C = Kokkos): max |dE/E| 1.04e-12
                // (1e-11), 6.70e-13 (1e-13); without landing 7.95e-7 / 8.10e-6
                double mc, pc, mk, pk;
                energy_ensemble(tol > 1e-12 ? "softening_ensemble_1e-11" : "softening_ensemble_1e-13", s, o, 20, &mc, &pc, &mk, &pk);
                check("softening_ens", "Kokkos / C median |dE/E| (128)", mk / mc, 3.0);
                check("softening_ens", "Kokkos / C p90 |dE/E| (128)", pk / pc, 3.0);
                check("softening_ens", "max |dE/E| C (128)", g_ens_max_c, tol > 1e-12 ? 1.35e-12 : 8.7e-13);
                check("softening_ens", "max |dE/E| Kokkos (128)", g_ens_max_k, tol > 1e-12 ? 1.35e-12 : 8.7e-13);
                Opt off = o;
                off.kink = false;
                energy_ensemble(tol > 1e-12 ? "softening_ensemble_off_1e-11" : "softening_ensemble_off_1e-13", s, off, 20, &mc, &pc, &mk, &pk);
                check("softening_ens", "landing off: max C (lower)", g_ens_max_c, 1e3 * (tol > 1e-12 ? 1.35e-12 : 8.7e-13), true);
            }
        }
        if (want(argc, argv, "collision_tolscan")) {
            Sys s = plummer(40, 0, 11);
            for (int k = 0; k < 12; k++) { s.x[k] = 0; s.v[k] = 0; }
            const double rp = 0.001, vinf = 2.0, mu = 2.0 / 40, b = std::sqrt(rp * rp + 2 * mu * rp / (vinf * vinf));
            s.x[0] = 0.05; s.x[3] = -0.05; s.v[0] = -1; s.v[3] = 1;
            s.x[6] = 0.05; s.x[7] = 0.3 + b / 2; s.x[9] = -0.05; s.x[10] = 0.3 - b / 2; s.v[6] = -1; s.v[9] = 1;
            Opt o;
            o.radius.assign(40, 1e-5);
            for (int i = 0; i < 4; i++) o.radius[i] = 0.001;
            collision_tolscan("star_collisions", s, o, 0.1);
            Sys t = plummer(50, 0, 5);
            for (int i = 0; i < 50; i++) t.x[3 * i] += 0.3;
            for (int k = 0; k < 6; k++) t.v[k] = 0;
            t.x[0] = 0.35; t.x[1] = t.x[2] = 0; t.x[3] = 0.25; t.x[4] = t.x[5] = 0; t.v[0] = -1; t.v[3] = 1;
            Opt p;
            p.tide = true;
            const double TT[6] = {-0.5, 0.2, 0.3, 0.1, -0.05, 0.08};
            for (int q = 0; q < 6; q++) p.T[q] = TT[q];
            p.radius.assign(50, 0.0);
            p.radius[0] = p.radius[1] = 0.001;
            collision_tolscan("external_tide_coll", t, p, 0.25);
        }
        if (want(argc, argv, "external_tide")) {
            // linear external tide a = T x (GIZMO addition) on a 50-body Plummer group off the origin
            Sys s = plummer(50, 0, 5);
            for (int i = 0; i < 50; i++) s.x[3 * i] += 0.3;
            Opt o;
            o.tol = 1e-10;
            o.tide = true;
            const double T[6] = {-0.5, 0.2, 0.3, 0.1, -0.05, 0.08};
            for (int q = 0; q < 6; q++) o.T[q] = T[q];
            compare("external_tide", s, o, 0.25, 3.9e-13, 1.2e-13, 1.2e-13);   // dv CPU 1.3e-13, A100 3.0e-13 (round-off); |dE/E| C 8.9e-14, CPU 8.0e-14, A100 1.5e-14
            // with contact collisions and the tide: the collision energy includes the tide term
            for (int k = 0; k < 6; k++) s.v[k] = 0;
            s.x[0] = 0.35; s.x[1] = s.x[2] = 0; s.x[3] = 0.25; s.x[4] = s.x[5] = 0; s.v[0] = -1; s.v[3] = 1;
            o.radius.assign(50, 0.0);
            o.radius[0] = o.radius[1] = 0.001;
            // bounds 1.3x the C-vs-C spread of collision_tolscan (x 3.9e-7, dt 2.8e-4, ddE 3.7e-9; CPU Kokkos at 1e-10: 6.6e-9,
            // 6.9e-6, 6.7e-11 (nearly the same step sequence); A100 3.4e-7, 2.6e-4, 3.2e-9); |dE/E| C 6.0e-14, CPU 5.2e-14,
            // A100 4.6e-14, A100 without FMA 1.5e-13 (Kokkos over tolerances 1e-8..1e-13: 4.6e-15..6.5e-12): bound 2.0e-13
            compare("external_tide_coll", s, o, 0.25, 5.0e-7, 7.8e-14, 2.0e-13, 3.7e-4, 4.8e-9);
        }
        if (want(argc, argv, "softening")) {
            // star_star_softening (GADGET spline between non-PN particles): a 100-body core with h = 0.05, and the
            // two-BH PN cusp with h = 0.01 (BH-star and BH-BH pairs unsoftened)
            Sys s = plummer(100, 0, 42);
            Opt o;
            o.tol = 1e-10;
            o.soft = 0.05;
            compare("softening_core", s, o, 0.25, 8.4e-11, 1.5e-12, 9.1e-12);   // tolerance 1e-10; with landing |dE/E| C 1.3e-14, CPU 3.0e-15, A100 1.6e-15, A100 without FMA 7.0e-12 (one realization; dv 6.5e-11)
            Sys c = plummer(62, 2, 7, 0.05);
            c.x[0] = 0.02; c.x[3] = -0.02;
            const double vb = std::sqrt(0.1 / 0.04) / 2;
            c.v[1] = vb; c.v[4] = -vb;
            c.s = {0.0005, 0, 0.001, 0, 0, -0.0008};
            Opt p;
            p.tol = 1e-10;
            p.c = 30;
            p.pn = 0x7ff;
            p.soft = 0.01;
            compare("softening_cusp_pn", c, p, 0.05, 6.1e-12, 1e300, 1e300);   // CPU dv 4.7e-12 (dE: PN)
        }
        if (want(argc, argv, "no_star_star")) {
            // no_star_star_gravity: the stars of the two-BH PN cusp feel only the BHs
            Sys c = plummer(62, 2, 7, 0.05);
            c.x[0] = 0.02; c.x[3] = -0.02;
            const double vb = std::sqrt(0.1 / 0.04) / 2;
            c.v[1] = vb; c.v[4] = -vb;
            Opt p;
            p.tol = 1e-10;
            p.c = 30;
            p.pn = 0x7ff;
            p.no_ss = true;
            compare("no_star_star", c, p, 0.05, 1.3e-13, 1e300, 1e300);   // CPU dv 9.8e-14 (dE: PN)
        }
        if (want(argc, argv, "bh_merger")) {
            // KETJU unit test head_on_merger: two BHs from rest at +-40 (G = c = 1) merge; remnant at the origin
            Sys s;
            s.n = 2;
            s.npn = 2;
            s.m = {1, 1};
            s.x = {40, 0, 0, -40, 0, 0};
            s.v.assign(6, 0);
            s.s.assign(6, 0);
            Opt o;
            o.tol = 1e-12;
            o.pn = 0x7ff;
            o.mergers = true;
            const Run a = run_c(s, o, 1e6), b = run_k(s, o, 1e6);
            std::printf("bh_merger: C n=%d m=%.6f | Kokkos n=%d m=%.6f\n", a.n, a.m[0], b.n, b.m[0]);
            check("bh_merger", "n_K - 1", std::fabs(b.n - 1.0), 0);
            check("bh_merger", "|m_C - m_K|", std::fabs(a.m[0] - b.m[0]), 1e-12);
            check("bh_merger", "|x_K|", std::fabs(b.x[0]) + std::fabs(b.x[1]), 1e-8);
        }
        if (want(argc, argv, "bh_merger_orbit")) {
            // inspiral and merger of a spinning unequal-mass pair (masses 1, 0.6, chi 0.5 and 0.7 misaligned; KETJU
            // unit test merger_from_orbit set-up): remnant from the C++ port of bh_mergers.c against the reference
            const double m[2] = {1.0, 0.6}, x[6] = {40, 0, 0, -40, 0, 0}, v[6] = {0, 0.05, 0, 0, 0, 0};
            const double S[6] = {0.5 * 0.6, 0, 0.5 * 0.8, 0, 0.7 * 0.36 * 0.6, -0.7 * 0.36 * 0.8};
            Sys sy;
            sy.n = sy.npn = 2;
            sy.m = {m[0], m[1]};
            sy.x.assign(x, x + 6); sy.v.assign(v, v + 6); sy.s.assign(S, S + 6);
            Opt o;
            o.tol = 1e-12; o.out_tol = 1e-12; o.pn = 0x7ff; o.mergers = true;
            const Run ra = run_c(sy, o, 1e6);
            mstar::System M(2, 0);
            unit_constants(M);
            M.constants.c = 1; M.options.PN_flags = 0x7ff; M.options.gbs_relative_tolerance = 1e-12;
            M.options.output_time_relative_tolerance = 1e-12;
            M.mass = {m[0], m[1]};
            M.pos.assign(x, x + 6); M.vel.assign(v, v + 6); M.spin.assign(S, S + 6);
            M.run(1e6);
            const bool both = ra.merges.size() == 1 && M.mergers.size() == 1;
            check("bh_merger_orbit", "one merger each", both ? 0.0 : 1.0, 0.0);
            if (both) {
                const Merge& a = ra.merges[0];
                const auto& b = M.mergers[0];
                std::printf("bh_merger_orbit: t %.10g / %.10g, m_rem %.12f / %.12f, chi_rem %.10f / %.10f, v_kick %.6e / %.6e\n",
                            a.t, b.t_merger, a.m_rem, b.m_remnant, a.chi_rem, b.chi_remnant, a.v_kick, b.v_kick);
                check("bh_merger_orbit", "|dt_merger| / t", std::fabs(a.t - b.t_merger) / a.t, 0.06);   // CPU identical; A100 4.2e-2: the trigger is tested at
                // step ends, so its time and configuration follow the step sequence
                check("bh_merger_orbit", "|dm_rem| / m_rem", std::fabs(a.m_rem - b.m_remnant) / a.m_rem, 7.6e-6);   // A100 5.8e-6
                check("bh_merger_orbit", "|dchi_rem|", std::fabs(a.chi_rem - b.chi_remnant), 2.4e-4);   // A100 1.8e-4
                check("bh_merger_orbit", "|dv_kick| / v_kick", std::fabs(a.v_kick - b.v_kick) / a.v_kick, 0.33);   // A100 0.26 (kick direction-sensitive)
            }
        }
        std::printf("%s (%d failed checks)\n", g_fail ? "FAILED" : "PASSED", g_fail);
#ifdef MSTAR_REF_GENERATE
        g_ref.save(MSTAR_REF_DATA "/compare_c.txt");
#endif
    }
    Kokkos::finalize();
#ifdef MSTAR_REF_GENERATE
    MPI_Finalize();
#endif
    return g_fail ? 1 : 0;
}
