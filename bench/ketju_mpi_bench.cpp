// MPI scaling benchmark of the KETJU C library (real MPI), same Plummer groups as kokkos/examples/mstar_bench.cpp.
// Copyright (C) 2026 Ulrich Steinwandel. Licensed under the GPLv3 license.
//
// Plummer spheres N = 16 .. N_max (G = M = 1, seed 1234+N, PN off, gbs_relative_tolerance 1e-10) through the
// KETJU C API on all MPI tasks of MPI_COMM_WORLD, default task split. The integration interval is calibrated
// (re-running from the IC) so that about `steps` GBS steps (successful + failed) are attempted; the timed run
// reports ms per attempted step on rank 0. Usage: ketju-mpi-bench [N_min] [N_max] [steps]
#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include "ketju_integrator/ketju_integrator.h"

static void plummer(int n, unsigned seed, std::vector<double>& m, std::vector<double>& x, std::vector<double>& v) {
    std::mt19937_64 g(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    m.assign(n, 1.0 / n); x.assign(3 * n, 0); v.assign(3 * n, 0);
    const double rs = 3 * M_PI / 16;
    for (int i = 0; i < n; i++) {
        double r;
        do { r = 1.0 / std::sqrt(std::pow(U(g), -2.0 / 3.0) - 1.0); } while (r > 10);
        const double ct = 2 * U(g) - 1, ph = 2 * M_PI * U(g), st = std::sqrt(1 - ct * ct);
        x[3 * i] = rs * r * st * std::cos(ph); x[3 * i + 1] = rs * r * st * std::sin(ph); x[3 * i + 2] = rs * r * ct;
        double q, y;
        do { q = U(g); y = 0.1 * U(g); } while (y > q * q * std::pow(1 - q * q, 3.5));
        const double ve = q * std::sqrt(2.0) * std::pow(1 + r * r, -0.25) / std::sqrt(rs);
        const double c2 = 2 * U(g) - 1, p2 = 2 * M_PI * U(g), s2 = std::sqrt(1 - c2 * c2);
        v[3 * i] = ve * s2 * std::cos(p2); v[3 * i + 1] = ve * s2 * std::sin(p2); v[3 * i + 2] = ve * c2;
    }
}

// one run from the IC over `T`; returns attempted steps, wall time (max over ranks, after a barrier) in *wall
static long run(int n, const std::vector<double>& m, const std::vector<double>& x, const std::vector<double>& v,
                double T, double* wall) {
    struct ketju_system R;
    ketju_create_system(&R, 0, n, MPI_COMM_WORLD);
    R.options->PN_flags = 0;
    R.constants->G = 1;
    R.options->gbs_relative_tolerance = 1e-10;
    for (int i = 0; i < n; i++) {
        R.physical_state->mass[i] = m[i];
        for (int k = 0; k < 3; k++) { R.physical_state->pos[i][k] = x[3 * i + k]; R.physical_state->vel[i][k] = v[3 * i + k]; }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();
    ketju_run_integrator(&R, T);
    MPI_Barrier(MPI_COMM_WORLD);
    *wall = MPI_Wtime() - t0;
    const long st = R.perf->successful_steps + R.perf->failed_steps;
    ketju_free_system(&R);
    return st;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const int nmin = argc > 1 ? std::atoi(argv[1]) : 16, nmax = argc > 2 ? std::atoi(argv[2]) : 4096;
    const int steps = argc > 3 ? std::atoi(argv[3]) : 20;
    if (rank == 0) std::printf("# ranks %d\n# %6s %6s %12s %12s\n", size, "N", "steps", "T", "C_ms/st");
    for (int n = nmin; n <= nmax; n *= 2) {
        std::vector<double> m, x, v;
        plummer(n, 1234 + n, m, x, v);
        double T = 1e-3, w = 0;
        long st = run(n, m, x, v, T, &w);   // also warms up
        for (int it = 0; it < 6 && (st < steps - 2 || st > steps + 4); it++) {
            T *= (double)steps / std::max(st, 1L);
            st = run(n, m, x, v, T, &w);
        }
        st = run(n, m, x, v, T, &w);   // timed
        if (rank == 0) { std::printf("  %6d %6ld %12.4e %12.4f\n", n, st, T, 1e3 * w / st); std::fflush(stdout); }
    }
    MPI_Finalize();
    return 0;
}
