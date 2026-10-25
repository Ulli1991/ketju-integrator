// Ketju-integrator tests: the C++/Kokkos MSTAR against the C library.
// Copyright (C) 2026 Ulrich Steinwandel.
// Licensed under the GPLv3 license.
// See LICENSE.md or <https://www.gnu.org/licenses/> for details.
//
// The Kokkos library (kokkos/src) against the C library of this repository (src/, built with the single-task MPI
// stand-in), same options, same initial state: agreement of the final states to round-off growth, and accuracy
// against independent references (Kepler: analytic; energies). ctest mstar_compare_c. Usage:
//   mstar-compare-c [case ...]   cases: kepler_e09 pythagorean core100 core100_staged pn_binary imbh_cusp_pn
//                                device_tree bh_merger bh_merger_orbit pn_ledger stop_collision star_collisions
//                                external_tide softening no_star_star (default: all)
// Bounds: 1.3x the larger measured value of the CPU (gcc 15 -O2, Kokkos OpenMP 1 thread; bitwise for two bodies)
// and the A100 (CUDA 13.2; FMA contraction and team reductions make the GBS step sequence differ at the
// tolerance-1e-12 round-off floor) builds, noted at each case.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "ketju_integrator/ketju_integrator.h"
#include "mstar/mstar.hpp"

namespace {

struct Sys {
    int npn = 0, n = 0;
    std::vector<double> m, x, v, s;   // s: spins of the PN particles
};

struct Merge {
    double t, m1, m2, m_rem, dE;
    int is_coll;
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
};

void unit_constants(mstar::System& M) {
    M.constants.G = 1;
    M.constants.c = 1;
    M.options.PN_flags = 0;
}

double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

Run run_c(const Sys& sy, const Opt& o, double T) {
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
            r.merges.push_back({t_call + mi.t_merger, mi.m1, mi.m2, mi.m_remnant, mi.dE_collision, mi.is_collision});
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

bool g_device_tree = false;   // force the one-team Prim tree (device_tree case)
Run run_k(const Sys& sy, const Opt& o, double T, mstar::PerformanceCounters* perf_out = nullptr) {
    mstar::System M(sy.npn, sy.n - sy.npn);
    unit_constants(M);
    if (g_device_tree) { M.options.device_tree_everywhere = true; M.options.device_tree_min_n = 0; }
    M.constants.G = o.G;
    M.constants.c = o.c;
    M.options.PN_flags = o.pn;
    M.options.gbs_relative_tolerance = o.tol;
    M.options.output_time_relative_tolerance = o.out_tol;
    M.options.enable_bh_mergers = o.mergers;
    M.options.bh_merger_distance_in_Rs = o.merge_rs;
    M.options.team_rows_max_n = M.options.team_rows_max_n_device = o.team_max;
    M.options.star_star_softening = o.soft;
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
bool check(const char* c, const char* what, double v, double bound) {
    const bool ok = v <= bound;
    std::printf("  %s %-14s %-22s %.3e (<= %.3e)\n", ok ? "ok  " : "FAIL", c, what, v, bound);
    if (!ok) g_fail++;
    return ok;
}

// bound_merge: |t_C - t_K| / T of every merger / collision (both are recorded at the end of the accepted step, so
// with different step sequences they differ by up to a step), bound_mdE: |dE_coll,C - dE_coll,K| / |dE_coll,C|
void compare(const char* name, const Sys& sy, const Opt& o, double T, double bound_x, double bound_dE_c,
             double bound_dE_k, double bound_merge = 1e-12, double bound_mdE = 1e-12) {
    const Run a = run_c(sy, o, T);
    const Run b = run_k(sy, o, T);
    std::printf("%s: N=%d npn=%d T=%g  C: %ld steps (%ld fail) dE %.3e %.3fs | Kokkos: %ld steps (%ld fail) dE %.3e %.3fs\n",
                name, sy.n, sy.npn, T, a.steps, a.fails, a.dE, a.wall, b.steps, b.fails, b.dE, b.wall);
    check(name, "N_C - N_K", std::fabs((double)(a.n - b.n)), 0.0);
    check(name, "merger count C - K", std::fabs((double)a.merges.size() - (double)b.merges.size()), 0.0);
    for (size_t q = 0; q < a.merges.size() && q < b.merges.size(); q++) {
        const Merge &x = a.merges[q], &y = b.merges[q];
        std::printf("  merger %zu: coll %d/%d t %.12g / %.12g m1 %g/%g m2 %g/%g dE %.12e / %.12e\n", q, x.is_coll, y.is_coll,
                    x.t, y.t, x.m1, y.m1, x.m2, y.m2, x.dE, y.dE);
        check(name, "merger kind C - K", std::fabs((double)(x.is_coll - y.is_coll)), 0.0);
        check(name, "merger |dt| / T", std::fabs(x.t - y.t) / T, bound_merge);
        check(name, "merger masses", std::fabs(x.m1 - y.m1) + std::fabs(x.m2 - y.m2), 1e-14 * (x.m1 + x.m2));
        if (x.is_coll) check(name, "collision |ddE| / |dE|", std::fabs(x.dE - y.dE) / std::fabs(x.dE), bound_mdE);
    }
    if (!a.R.empty()) check(name, "collision radii C vs K", rel_diff(a.R, b.R), 0.0);
    check(name, "|dx|/x C vs Kokkos", rel_diff(a.x, b.x), bound_x);
    check(name, "|dv|/v C vs Kokkos", rel_diff(a.v, b.v), bound_x);
    if (sy.npn) check(name, "|dS|/S C vs Kokkos", rel_diff(a.s, b.s), bound_x);
    check(name, "|t_C - t_K| / T", std::fabs(a.t - b.t) / T, 1e-12);
    check(name, "|dE/E| C", std::fabs(a.dE), bound_dE_c);
    check(name, "|dE/E| Kokkos", std::fabs(b.dE), bound_dE_k);
    check(name, "step count diff", std::fabs((double)(a.steps - b.steps)), std::fmax(2.0, 0.25 * a.steps));   // A100: 19 of 98 (Kepler)
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

bool want(int argc, char** argv, const char* c) {
    bool any = false;
    for (int a = 1; a < argc; a++)
        if (argv[a][0] != '-') { any = true; if (!std::strcmp(argv[a], c)) return true; }
    return !any;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);   // KETJU needs MPI (or the serial stand-in)
    Kokkos::initialize(argc, argv);
    {
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
            compare("imbh_cusp_pn", s, o, 0.05, 2.2e-13, 1e300, 1e300);   // dv CPU 1.05e-13, A100 1.67e-13
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
            check("pn_ledger", "|dE - dE_pn| / |E0|", std::fabs(dE - dE_pn) / std::fabs(E0), 1.5e-13);   // CPU 6.46e-14, A100 1.15e-13
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
            compare("star_collisions", s, o, 0.1, 3.1e-8, 1.4e-13, 1.4e-13, 5.2e-4, 4.4e-9);   // CPU dv 2.3e-8, dt 4.0e-4, ddE 3.4e-9 (the collisions are recorded at step ends)
            o.calls = 10;
            compare("star_collisions_10", s, o, 0.1, 6.6e-10, 4.1e-14, 4.1e-14, 5.2e-5, 1.4e-10);   // CPU dv 5.1e-10, dt 4.0e-5, ddE 1.0e-10
        }
        if (want(argc, argv, "coll_tide_soft")) {
            // a collision during the run with star_star_softening and the external tide (gizmo_additions_kokkos
            // Gizmo_collision_energy.run_with_tide_and_softening)
            Sys s;
            s.n = 3;
            s.m = {1, 0.7, 0.5};
            s.x = {1, 0, 0, -1, 0.02, 0, 0, 3, 0.5};
            s.v = {-0.3, 0, 0, 0.3, 0, 0, 0.2, 0, 0};
            Opt o;
            o.tol = 1e-11;
            o.out_tol = 1e-12;
            o.soft = 0.3;
            o.tide = true;
            const double T[6] = {-0.002, 0.001, 0.0015, 0.0004, -0.0003, 0.0006};
            for (int q = 0; q < 6; q++) o.T[q] = T[q];
            o.radius = {0.05, 0.05, 0.05};
            compare("coll_tide_soft", s, o, 20, 1e-12, 1.9e-10, 1.9e-10);   // CPU bitwise, dE 1.48e-10 (tolerance 1e-11)
            o.tol = 1e-13;
            compare("coll_tide_soft_1e-13", s, o, 20, 1e-12, 1.25e-12, 1.25e-12);   // CPU bitwise, dE 9.6e-13
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
            compare("external_tide", s, o, 0.25, 1.7e-13, 1.2e-13, 1.2e-13);   // CPU dv 1.3e-13, dE 8.9e-14 / 8.0e-14
            // with contact collisions and the tide: the collision energy includes the tide term
            for (int k = 0; k < 6; k++) s.v[k] = 0;
            s.x[0] = 0.35; s.x[1] = s.x[2] = 0; s.x[3] = 0.25; s.x[4] = s.x[5] = 0; s.v[0] = -1; s.v[3] = 1;
            o.radius.assign(50, 0.0);
            o.radius[0] = o.radius[1] = 0.001;
            compare("external_tide_coll", s, o, 0.25, 8.5e-9, 7.8e-14, 7.8e-14, 9.0e-6, 8.6e-11);   // CPU dv 6.6e-9, dE 6.0e-14, dt 6.9e-6, ddE 6.7e-11
        }
        if (want(argc, argv, "softening")) {
            // star_star_softening (GADGET spline between non-PN particles): a 100-body core with h = 0.05, and the
            // two-BH PN cusp with h = 0.01 (BH-star and BH-BH pairs unsoftened)
            Sys s = plummer(100, 0, 42);
            Opt o;
            o.tol = 1e-10;
            o.soft = 0.05;
            compare("softening_core", s, o, 0.25, 4.6e-11, 6.5e-12, 6.5e-12);   // CPU dv 3.5e-11, dE 5.0e-12 / 3.1e-12
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
            struct ketju_system R;
            ketju_create_system(&R, 2, 0, MPI_COMM_WORLD);
            R.constants->c = 1; R.constants->G = 1;
            R.options->PN_flags = KETJU_PN_ALL;
            R.options->gbs_relative_tolerance = 1e-12;
            R.options->output_time_relative_tolerance = 1e-12;
            const double m[2] = {1.0, 0.6}, x[6] = {40, 0, 0, -40, 0, 0}, v[6] = {0, 0.05, 0, 0, 0, 0};
            const double S[6] = {0.5 * 0.6, 0, 0.5 * 0.8, 0, 0.7 * 0.36 * 0.6, -0.7 * 0.36 * 0.8};
            for (int i = 0; i < 2; i++) {
                R.physical_state->mass[i] = m[i];
                for (int k = 0; k < 3; k++) {
                    R.physical_state->pos[i][k] = x[3 * i + k];
                    R.physical_state->vel[i][k] = v[3 * i + k];
                    R.physical_state->spin[i][k] = S[3 * i + k];
                }
            }
            ketju_run_integrator(&R, 1e6);
            mstar::System M(2, 0);
            unit_constants(M);
            M.constants.c = 1; M.options.PN_flags = 0x7ff; M.options.gbs_relative_tolerance = 1e-12;
            M.options.output_time_relative_tolerance = 1e-12;
            M.mass = {m[0], m[1]};
            M.pos.assign(x, x + 6); M.vel.assign(v, v + 6); M.spin.assign(S, S + 6);
            M.run(1e6);
            const bool both = R.num_mergers == 1 && M.mergers.size() == 1;
            check("bh_merger_orbit", "one merger each", both ? 0.0 : 1.0, 0.0);
            if (both) {
                const auto& a = R.merger_infos[0];
                const auto& b = M.mergers[0];
                std::printf("bh_merger_orbit: t %.10g / %.10g, m_rem %.12f / %.12f, chi_rem %.10f / %.10f, v_kick %.6e / %.6e\n",
                            a.t_merger, b.t_merger, a.m_remnant, b.m_remnant, a.chi_remnant, b.chi_remnant, a.v_kick, b.v_kick);
                check("bh_merger_orbit", "|dt_merger| / t", std::fabs(a.t_merger - b.t_merger) / a.t_merger, 0.06);   // CPU identical; A100 4.2e-2: the trigger is tested at
                // step ends, so its time and configuration follow the step sequence
                check("bh_merger_orbit", "|dm_rem| / m_rem", std::fabs(a.m_remnant - b.m_remnant) / a.m_remnant, 7.6e-6);   // A100 5.8e-6
                check("bh_merger_orbit", "|dchi_rem|", std::fabs(a.chi_remnant - b.chi_remnant), 2.4e-4);   // A100 1.8e-4
                check("bh_merger_orbit", "|dv_kick| / v_kick", std::fabs(a.v_kick - b.v_kick) / a.v_kick, 0.33);   // A100 0.26 (kick direction-sensitive)
            }
            ketju_free_system(&R);
        }
        std::printf("%s (%d failed checks)\n", g_fail ? "FAILED" : "PASSED", g_fail);
    }
    Kokkos::finalize();
    MPI_Finalize();
    return g_fail ? 1 : 0;
}
