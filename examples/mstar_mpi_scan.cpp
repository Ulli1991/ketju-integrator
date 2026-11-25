// Ketju-integrator (MSTAR), C++/Kokkos library. Copyright (C) 2026 Ulrich Steinwandel. Licensed under the GPLv3 license.
//
// One point of the MPI sharing scan (scripts/bench_mpi_scan.sh): wall time per GBS step of a system of N bodies on the
// tasks the program is started on, in one of three modes that mirror what a caller such as GIZMO's KETJU coupling does:
//   single   one task (rank 0) integrates, the others do nothing; path sums            (the single-task path)
//   default  shared by all tasks with the tree sweep if N >= the library's mpi_min_n,  (the default)
//            else as single
//   shared   shared by all tasks with the tree sweep whatever N                        (to place the threshold)
//   mstar-mpi-scan N attempts a_bin mode [repeats = 3] [detail = 0]
// a_bin: semi-major axis of the binary of bodies 0 and 1 in cluster radii (1e-4: hard, 0.05: none to speak of).
// detail = 1 puts a barrier before every force exchange so the MPI time splits into waiting for the slowest task and
// the transfer itself. The fastest of the repeats is reported. Rank 0 prints one line.

#include "mstar/mstar.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

int main(int argc, char** argv) {
    mstar::MpiSession mpi(argc, argv);
    mstar::ScopeGuard guard(argc, argv);
    if (argc < 5) {
        std::printf("usage: mstar-mpi-scan N attempts a_bin single|default|shared [repeats] [detail]\n");
        return 2;
    }
    const int n = std::atoi(argv[1]);
    const long steps = std::atol(argv[2]);
    const double a_bin = std::atof(argv[3]);
    const std::string mode = argv[4];
    const int repeats = argc > 5 ? std::atoi(argv[5]) : 3;
    const bool detail = argc > 6 && std::atoi(argv[6]) != 0;
    const int default_min_n = 128;   // Options::mpi_min_n < 0
    const bool share = mpi.size > 1 && mstar::mpi_enabled() && (mode == "shared" || (mode == "default" && n >= default_min_n));
    double best = 1e300, force = 0, comm = 0, wait = 0;
    long ok = 0, failed = 0;
    int tasks = 1;
    std::uint64_t h = 0;
    for (int rep = 0; rep < repeats; rep++) {
        if (!share && mpi.rank != 0) break;   // the other tasks of an unshared region wait
        mstar::System S(0, n);
        S.constants.G = 1;
        S.options.PN_flags = mstar::PN_NONE;
        S.options.gbs_relative_tolerance = 1e-9;
        S.options.max_step_count = steps;
        S.options.mpi_min_n = share ? 2 : (1 << 30);
        S.options.tree_sweep = share;
        S.options.mpi_time_detail = detail;
        std::uint64_t seed = 4242;
        auto u = [&]() {
            seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
            return (double)(seed >> 11) / 9007199254740992.0;
        };
        for (int i = 0; i < n; i++) {
            double x[3], v[3], r2;
            do {
                r2 = 0;
                for (int k = 0; k < 3; k++) { x[k] = 2 * u() - 1; r2 += x[k] * x[k]; }
            } while (r2 > 1);
            for (int k = 0; k < 3; k++) v[k] = 0.4 * (2 * u() - 1);
            S.set_particle(i, (0.5 + u()) / n, x, v);
        }
        const double mb = 2.0 / n, vc = std::sqrt(2 * mb / a_bin);
        const double x0[3] = {0.1 - 0.5 * a_bin, 0.05, -0.02}, x1[3] = {0.1 + 0.5 * a_bin, 0.05, -0.02};
        const double v0[3] = {0.01, -0.5 * vc, 0}, v1[3] = {0.01, 0.5 * vc, 0};
        S.set_particle(0, mb, x0, v0);
        S.set_particle(1, mb, x1, v1);
        double cp[3], cv[3];
        S.into_com_frame(cp, cv);
        S.collision_radius.assign(n, 1e-9);
        const auto t0 = std::chrono::steady_clock::now();
        S.run(1e3);   // ends at max_step_count
        const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (wall < best) { best = wall; force = S.perf.t_force; comm = S.perf.t_comm; wait = S.perf.t_wait; }
        ok = S.perf.successful_steps; failed = S.perf.failed_steps; tasks = S.perf.mpi_tasks;
        h = 1469598103934665603ULL;
        for (double q : S.pos) { std::uint64_t b; std::memcpy(&b, &q, 8); h = (h ^ b) * 1099511628211ULL; }
        for (double q : S.vel) { std::uint64_t b; std::memcpy(&b, &q, 8); h = (h ^ b) * 1099511628211ULL; }
    }
    if (mpi.rank == 0) {
        const long st = ok + failed > 0 ? ok + failed : 1;
        std::printf("SCAN N %d a_bin %g tasks %d mode %s sharing %d attempts %ld wall %.6f ms_per_step %.6f force_pct %.1f mpi_pct %.1f wait_pct %.1f hash %016llx\n",
                    n, a_bin, mpi.size, mode.c_str(), tasks, st, best, 1e3 * best / st, 100 * force / best, 100 * comm / best, 100 * wait / best,
                    (unsigned long long)h);
    }
    return 0;
}
