// Ketju-integrator tests (C++/Kokkos MSTAR).
// Copyright (C) 2026 Ulrich Steinwandel.
// Licensed under the GPLv3 license.
// See LICENSE.md or <https://www.gnu.org/licenses/> for details.
//
// Extra forces of the C++/Kokkos MSTAR (ExtraForces, port of KETJU's extra-force branch b60d16e, Mannerkoski 2022).
// The C reference is the forward port of the branch patch onto eb96fd0 (the C library, branch ketju-c); its results
// are stored in reference_data/extra_force.txt (header: generating commit; MSTAR_REF_GENERATE regenerates). Cases:
//  ketju_example  the branch's nbody example (sample IC: 2 BHs + 100 stars, all PN, a_x += 1e10 x, a_x += 1e5 v_x):
//                 forward port vs C++ to round-off
//  static_field   e = 0.5 binary in the field of a static point mass and in a linear tidal field, against an
//                 independent RK4 integration; energy with the external potential conserved to the integration error,
//                 the work ledger equal to the external work booked from the states. With KETJU's branch semantics
//                 (track_work = false: B blind to the external work) the same runs were off by 1.2e-2 / 2.6e-1 in x
//                 and 1.6e-4 / 7.3e-2 in energy (the time transformation T + B drifts from U)
//  drag           velocity-dependent toy force a = -gamma v on a binary (auxiliary-velocity path): converges to RK4 as
//                 the GBS tolerance shrinks; the work ledger (-pn_dB) equals the RK4 drag work
// Bounds 1.3x measured (CPU gcc 15 -O2, Kokkos OpenMP 1 thread).

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "mstar/mstar.hpp"
#include "ref_store.hpp"
#ifdef MSTAR_REF_GENERATE
#include "ketju_integrator/ketju_integrator.h"
#endif

#ifndef MSTAR_SAMPLE_IC
#define MSTAR_SAMPLE_IC "nbody_sample_ic.txt"
#endif

namespace {

int g_fail = 0;
void check(const char* c, const char* what, double v, double bound, bool lower = false) {
    const bool ok = lower ? v >= bound : v <= bound;
    std::printf("  %s %-14s %-36s %.3e (%s %.3e)\n", ok ? "ok  " : "FAIL", c, what, v, lower ? ">=" : "<=", bound);
    if (!ok) g_fail++;
}
double rel_diff(const std::vector<double>& a, const std::vector<double>& b) {
    double mx = 0, sc = 0;
    for (size_t q = 0; q < a.size(); q++) {
        mx = std::fmax(mx, std::fabs(a[q] - b[q]));
        sc = std::fmax(sc, std::fabs(a[q]));
    }
    return mx / sc;
}

refstore::Store g_ref;

#ifdef MSTAR_REF_GENERATE
// ---- the branch's example forces (the C library with the extra-force patch forward-ported onto eb96fd0)
double g_ex_A = 1e10;
void ex_vindep(const struct ketju_system* R, double (*acc)[3]) {
    for (int i = 0; i < R->num_particles; ++i) acc[i][0] += R->physical_state->pos[i][0] * g_ex_A;
}
void ex_vdep(const struct ketju_system* R, double (*vel)[3], double (*)[3], double (*acc)[3]) {
    for (int i = 0; i < R->num_particles; ++i) acc[i][0] += vel[i][0] * 1e5;
}

// the C reference: {n, steps, fails, x (3n), v (3n)} for the example IC
std::vector<double> ketju_example_c(double A_x, int npn, int nst, FILE* fp) {
    g_ex_A = A_x;
    struct ketju_system R;
    ketju_create_system(&R, npn, nst, MPI_COMM_WORLD);
    for (int i = 0; i < npn + nst; i++) {
        double* x = R.physical_state->pos[i];
        double* v = R.physical_state->vel[i];
        int got = std::fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf", &R.physical_state->mass[i], &x[0], &x[1], &x[2],
                              &v[0], &v[1], &v[2]);
        if (i < npn) got += std::fscanf(fp, "%lf %lf %lf", &R.physical_state->spin[i][0], &R.physical_state->spin[i][1],
                                        &R.physical_state->spin[i][2]);
        (void)got;
    }
    double cp[3], cv[3];
    ketju_into_CoM_frame(&R, cp, cv);
    R.options->gbs_relative_tolerance = 1e-9;
    R.options->mst_algorithm = KETJU_MST_PRIM;
    R.options->max_tree_distance = 2;
    R.options->PN_flags = KETJU_PN_ALL;
    R.extra_vindep_force = ex_vindep;
    R.extra_vdep_force = ex_vdep;
    const double T = 2e-6;
    const int sub = 4;
    for (int q = 0; q < sub; q++) ketju_run_integrator(&R, T / sub);
    std::vector<double> out = {(double)R.num_particles, (double)R.perf->successful_steps, (double)R.perf->failed_steps};
    for (int i = 0; i < R.num_particles; i++) for (int k = 0; k < 3; k++) out.push_back(R.physical_state->pos[i][k]);
    for (int i = 0; i < R.num_particles; i++) for (int k = 0; k < 3; k++) out.push_back(R.physical_state->vel[i][k]);
    ketju_free_system(&R);
    return out;
}
#endif

void ketju_example(double A_x, double bx, double bv) {
    FILE* fp = std::fopen(MSTAR_SAMPLE_IC, "r");
    if (!fp) { std::printf("FAIL: cannot open %s\n", MSTAR_SAMPLE_IC); g_fail++; return; }
    int npn, nst;
    if (std::fscanf(fp, "%d %d", &npn, &nst) != 2) { g_fail++; std::fclose(fp); return; }
    char key[64];
    std::snprintf(key, sizeof key, "ketju_example_A%.0e", A_x);
#ifdef MSTAR_REF_GENERATE
    g_ref.put(key, ketju_example_c(A_x, npn, nst, fp));
    std::rewind(fp);
    if (std::fscanf(fp, "%d %d", &npn, &nst) != 2) { g_fail++; std::fclose(fp); return; }
#endif
    std::vector<double> ref;
    if (!g_ref.get(key, ref)) { std::printf("FAIL: no reference %s in extra_force.txt\n", key); g_fail++; std::fclose(fp); return; }
    // the C++ integrator with the same state (the example IC in the c.o.m. frame), options and constants
    mstar::System M(npn, nst);
    M.options.PN_flags = mstar::PN_ALL;   // constants: the library defaults (kpc, km/s, Msun), as the C example
    M.options.gbs_relative_tolerance = 1e-9;
    M.options.extra.vindep = true;
    M.options.extra.track_work = false;   // KETJU's branch: B ignores the extra work
    M.options.extra.A[0] = A_x;
    M.options.extra.vdep = true;
    M.options.extra.D[0] = 1e5;
    const int n = npn + nst;
    for (int i = 0; i < n; i++) {
        double* x = &M.pos[3 * i];
        double* v = &M.vel[3 * i];
        int got = std::fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf", &M.mass[i], &x[0], &x[1], &x[2], &v[0], &v[1], &v[2]);
        if (i < npn) got += std::fscanf(fp, "%lf %lf %lf", &M.spin[3 * i], &M.spin[3 * i + 1], &M.spin[3 * i + 2]);
        (void)got;
    }
    std::fclose(fp);
    double cp[3], cv[3];
    M.into_com_frame(cp, cv);
    const double T = 2e-6;   // a few hundred GBS steps
    const int sub = 4;       // as the example: run in substeps
    for (int q = 0; q < sub; q++) M.run(T / sub);
    std::vector<double> xr(ref.begin() + 3, ref.begin() + 3 + 3 * (size_t)ref[0]), vr(ref.begin() + 3 + 3 * (size_t)ref[0], ref.end());
    std::printf("ketju_example: N=%d, reference %d steps (%d fail), C++ %ld steps (%ld fail)\n", n, (int)ref[1], (int)ref[2],
                M.perf.successful_steps, M.perf.failed_steps);
    if ((int)ref[0] != M.num_particles) { check("ketju_example", "particle count", 1, 0); return; }
    char lab[64];
    std::snprintf(lab, sizeof lab, "|dx|/x (a_x += %.0e x)", A_x);
    check("ketju_example", lab, rel_diff(xr, M.pos), bx);
    std::snprintf(lab, sizeof lab, "|dv|/v (a_x += %.0e x)", A_x);
    check("ketju_example", lab, rel_diff(vr, M.vel), bv);
}

// ---- independent RK4 reference for a few bodies with an external acceleration ext(x, v, a)
struct Rk4Out { std::vector<double> x, v; double work_vdep; };
Rk4Out rk4(const std::vector<double>& m, std::vector<double> x, std::vector<double> v, double T, long nstep,
           const std::function<void(const double*, const double*, double*)>& ext_vindep, const double* D) {
    const int n = (int)m.size();
    auto acc = [&](const std::vector<double>& X, const std::vector<double>& Vv, std::vector<double>& A) {
        A.assign(3 * n, 0.0);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (j == i) continue;
                double d[3], r2 = 0;
                for (int k = 0; k < 3; k++) { d[k] = X[3 * j + k] - X[3 * i + k]; r2 += d[k] * d[k]; }
                const double f = m[j] / (r2 * std::sqrt(r2));
                for (int k = 0; k < 3; k++) A[3 * i + k] += f * d[k];
            }
            if (ext_vindep) ext_vindep(&X[3 * i], &Vv[3 * i], &A[3 * i]);
            if (D)
                for (int r = 0; r < 3; r++)
                    A[3 * i + r] += D[3 * r] * Vv[3 * i] + D[3 * r + 1] * Vv[3 * i + 1] + D[3 * r + 2] * Vv[3 * i + 2];
        }
    };
    const double h = T / nstep;
    double work = 0;
    std::vector<double> k1x, k1v, k2x, k2v, k3x, k3v, k4x, k4v, xt(3 * n), vt(3 * n);
    auto dpower = [&](const std::vector<double>& Vv) {   // drag power sum m v . D v
        double p = 0;
        if (!D) return 0.0;
        for (int i = 0; i < n; i++)
            for (int r = 0; r < 3; r++)
                p += m[i] * Vv[3 * i + r] * (D[3 * r] * Vv[3 * i] + D[3 * r + 1] * Vv[3 * i + 1] + D[3 * r + 2] * Vv[3 * i + 2]);
        return p;
    };
    for (long s = 0; s < nstep; s++) {
        k1x = v; acc(x, v, k1v);
        for (int q = 0; q < 3 * n; q++) { xt[q] = x[q] + 0.5 * h * k1x[q]; vt[q] = v[q] + 0.5 * h * k1v[q]; }
        k2x = vt; acc(xt, vt, k2v);
        for (int q = 0; q < 3 * n; q++) { xt[q] = x[q] + 0.5 * h * k2x[q]; vt[q] = v[q] + 0.5 * h * k2v[q]; }
        k3x = vt; acc(xt, vt, k3v);
        for (int q = 0; q < 3 * n; q++) { xt[q] = x[q] + h * k3x[q]; vt[q] = v[q] + h * k3v[q]; }
        k4x = vt; acc(xt, vt, k4v);
        const double p1 = dpower(v), p2 = dpower(k2x), p3 = dpower(k3x), p4 = dpower(k4x);
        work += h / 6 * (p1 + 2 * p2 + 2 * p3 + p4);
        for (int q = 0; q < 3 * n; q++) {
            x[q] += h / 6 * (k1x[q] + 2 * k2x[q] + 2 * k3x[q] + k4x[q]);
            v[q] += h / 6 * (k1v[q] + 2 * k2v[q] + 2 * k3v[q] + k4v[q]);
        }
    }
    return {x, v, work};
}

double newton_energy(const std::vector<double>& m, const std::vector<double>& x, const std::vector<double>& v) {
    double E = 0;
    const int n = (int)m.size();
    for (int i = 0; i < n; i++) {
        E += 0.5 * m[i] * (v[3 * i] * v[3 * i] + v[3 * i + 1] * v[3 * i + 1] + v[3 * i + 2] * v[3 * i + 2]);
        for (int j = i + 1; j < n; j++) {
            double r2 = 0;
            for (int k = 0; k < 3; k++) r2 += (x[3 * i + k] - x[3 * j + k]) * (x[3 * i + k] - x[3 * j + k]);
            E -= m[i] * m[j] / std::sqrt(r2);
        }
    }
    return E;
}

void binary(std::vector<double>& m, std::vector<double>& x, std::vector<double>& v, const double* X0, const double* V0) {
    m = {1.0, 0.5};
    const double M = 1.5, e = 0.5, r = 1 + e, vr = std::sqrt(M * (1 - e) / (1 + e));
    x = {0.5 / M * r, 0, 0, -1.0 / M * r, 0, 0};
    v = {0, 0.5 / M * vr, 0.1 * 0.5 / M * vr, 0, -1.0 / M * vr, -0.1 / M * vr};
    for (int i = 0; i < 2; i++)
        for (int k = 0; k < 3; k++) { x[3 * i + k] += X0[k]; v[3 * i + k] += V0[k]; }
}

void static_field() {
    // (a) point mass M = 100 at the origin, binary c.o.m. at 20 on a circular orbit; (b) linear tidal tensor
    for (int variant = 0; variant < 2; variant++) {
        const char* nm = variant ? "tidal_tensor" : "point_mass";
        const double Mp = 100, R0 = 20, vc = std::sqrt((Mp) / R0);
        const double X0[3] = {R0, 0, 0}, V0[3] = {0, vc, 0};
        std::vector<double> m, x, v;
        binary(m, x, v, X0, V0);
        // tidal tensor of the point mass at X0 (a = A (x - X0) to first order, the c.o.m. kept fixed: V0 = 0)
        double A[9] = {2 * Mp / (R0 * R0 * R0), 0, 0, 0, -Mp / (R0 * R0 * R0), 0, 0, 0, -Mp / (R0 * R0 * R0)};
        if (variant) {
            const double Z[3] = {0, 0, 0};
            binary(m, x, v, Z, Z);
        }
        std::function<void(const double*, const double*, double*)> ext;
        std::function<double(const double*)> phi;   // external potential per unit mass
        if (variant == 0) {
            ext = [&](const double* xi, const double*, double* a) {
                const double r2 = xi[0] * xi[0] + xi[1] * xi[1] + xi[2] * xi[2], f = -Mp / (r2 * std::sqrt(r2));
                for (int k = 0; k < 3; k++) a[k] += f * xi[k];
            };
            phi = [&](const double* xi) { return -Mp / std::sqrt(xi[0] * xi[0] + xi[1] * xi[1] + xi[2] * xi[2]); };
        } else {
            ext = [&](const double* xi, const double*, double* a) {
                for (int r = 0; r < 3; r++) a[r] += A[3 * r] * xi[0] + A[3 * r + 1] * xi[1] + A[3 * r + 2] * xi[2];
            };
            phi = [&](const double* xi) {
                double s = 0;
                for (int r = 0; r < 3; r++)
                    for (int c = 0; c < 3; c++) s += xi[r] * A[3 * r + c] * xi[c];
                return -0.5 * s;
            };
        }
        const double T = 20.0;
        const Rk4Out ref = rk4(m, x, v, T, 400000, ext, nullptr);
        const Rk4Out ref2 = rk4(m, x, v, T, 800000, ext, nullptr);
        mstar::System M(0, 2);
        M.constants.G = M.constants.c = 1;
        M.options.PN_flags = 0;
        M.options.gbs_relative_tolerance = 1e-12;
        M.options.output_time_relative_tolerance = 1e-13;
        M.mass = m; M.pos = x; M.vel = v;
        M.options.extra.vindep = true;
        if (variant == 0) {
            M.options.extra.n_pm = 1;
            M.options.extra.pm_m[0] = Mp;
        } else {
            std::memcpy(M.options.extra.A, A, sizeof(A));
        }
        M.run(T);
        auto Etot = [&](const std::vector<double>& X, const std::vector<double>& Vv) {
            double E = newton_energy(m, X, Vv);
            for (int i = 0; i < 2; i++) E += m[i] * phi(&X[3 * i]);
            return E;
        };
        const double E0 = Etot(x, v), E1 = Etot(M.pos, M.vel);
        const double W_ext = newton_energy(m, M.pos, M.vel) - newton_energy(m, x, v);   // booked from states
        const double ledger = -M.pn_dB;   // the integrator's own record of the external work
        std::printf("%s: RK4 self-check %.3e; C++ %ld steps; dE_N/E0 %.3e (external work from states), "
                    "dE_total/E0 %.3e (integration error)\n", nm, rel_diff(ref2.x, ref.x), M.perf.successful_steps,
                    W_ext / std::fabs(E0), (E1 - E0) / std::fabs(E0));
        check(nm, "|x - x_RK4|", rel_diff(ref2.x, M.pos), variant ? 2.6e-12 : 2e-12);   // measured 2.0e-12 / 6.5e-13 (RK4 self-check 1.1e-12 / 1.3e-12)
        check(nm, "|dE_total/E0| (error visible)", std::fabs(E1 - E0) / std::fabs(E0), variant ? 1.5e-13 : 1.2e-13);
        check(nm, "|W_ext/E0| (external work)", std::fabs(W_ext / E0), 1e-6, true);
        check(nm, "|ledger - W_ext| / |W_ext|", std::fabs(ledger - W_ext) / std::fabs(W_ext), variant ? 5.2e-11 : 2.1e-10);   // tidal: CPU 1.19e-11, A100 4.0e-11
        check(nm, "|ext_work - W_ext| / |W_ext|", std::fabs(M.ext_work - W_ext) / std::fabs(W_ext), variant ? 6.7e-11 : 1.4e-10);   // measured 5.2e-11 / 1.06e-10
    }
}

void drag() {
    const double gamma = 0.01;
    const double D[9] = {-gamma, 0, 0, 0, -gamma, 0, 0, 0, -gamma};
    const double Z[3] = {0, 0, 0};
    std::vector<double> m, x, v;
    binary(m, x, v, Z, Z);
    const double T = 10.0;
    const Rk4Out ref = rk4(m, x, v, T, 400000, nullptr, D);
    double prev = 1e300;
    for (double tol : {1e-8, 1e-10, 1e-12}) {
        mstar::System M(0, 2);
        M.constants.G = M.constants.c = 1;
        M.options.PN_flags = 0;
        M.options.gbs_relative_tolerance = tol;
        M.options.output_time_relative_tolerance = 1e-13;
        M.mass = m; M.pos = x; M.vel = v;
        M.options.extra.vdep = true;
        std::memcpy(M.options.extra.D, D, sizeof(D));
        M.run(T);
        const double err = rel_diff(ref.x, M.pos);
        const double dE = newton_energy(m, M.pos, M.vel) - newton_energy(m, x, v);
        std::printf("drag tol %.0e: %ld steps, |x - x_RK4| %.3e, dE_N %.6e, ledger -pn_dB %.6e, RK4 drag work %.6e\n",
                    tol, M.perf.successful_steps, err, dE, -M.pn_dB, ref.work_vdep);
        if (tol == 1e-12) {
            check("drag", "|x - x_RK4| at tol 1e-12", err, 7.9e-13);   // CPU 3.6e-13, A100 6.0e-13
            check("drag", "|ledger - RK4 work| / |work|", std::fabs(-M.pn_dB - ref.work_vdep) / std::fabs(ref.work_vdep), 2.8e-13);
            check("drag", "|dE_N - ledger| / |dE_N|", std::fabs(dE + M.pn_dB) / std::fabs(dE), 1.4e-12);   // A100 1.03e-12
        }
        check("drag", "error shrinks with the tolerance", err / prev, 1.0);
        prev = err;
    }
}

}  // namespace

int main(int argc, char** argv) {
#ifdef MSTAR_REF_GENERATE
    MPI_Init(&argc, &argv);
    g_ref.header = std::string("# C-library reference runs of mstar-extra-force (extra-force patch forward-ported onto eb96fd0), generated by the C library of branch ketju-c at commit ") + MSTAR_REF_COMMIT + "; key count values (hexadecimal floats)\n";
#else
    if (!g_ref.load(MSTAR_REF_DATA "/extra_force.txt")) { std::printf("FAIL: cannot read extra_force.txt\n"); return 2; }
#endif
    Kokkos::initialize(argc, argv);
    {
        // forward port, same algorithm: round-off for a moderate field (measured 2.9e-15 at 1e5); the example's
        // 1e10 x makes KETJU's time transformation (B blind to the extra work) ill-conditioned and round-off grows
        // with the field: measured 2.6e-12 at 1e8, 3.56e-8 / 2.04e-6 at 1e10
        ketju_example(1e5, 1e-14, 1e-13);
        ketju_example(1e10, 1.2e-7, 6.6e-6);   // A100 8.8e-8 / 5.1e-6
        static_field();
        drag();
        std::printf("%s (%d failed checks)\n", g_fail ? "FAILED" : "PASSED", g_fail);
#ifdef MSTAR_REF_GENERATE
        g_ref.save(MSTAR_REF_DATA "/extra_force.txt");
#endif
    }
    Kokkos::finalize();
#ifdef MSTAR_REF_GENERATE
    MPI_Finalize();
#endif
    return g_fail ? 1 : 0;
}
