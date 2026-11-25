// Ketju-integrator (MSTAR), C++/Kokkos library. Copyright (C) 2026 Ulrich Steinwandel. Licensed under the GPLv3 license.
//
// Wall time per GBS step of one large system with a hard binary, shared over the MPI tasks the program is started on
// (1 task: the serial library). Contact collisions are on (radii too small to collide), tolerance 1e-9.
//   mstar-mpi-bench [N = 770] [GBS step attempts = 200] [binary semi-major axis / cluster radius = 1e-4]
//                   [gbs_error_norm = 0] [gbs_error_diag = 0] [time interval = 0: until the step limit; > 0: no step limit]
// Rank 0 prints: tasks, N, steps (accepted + failed), wall s, ms per step, share of the force launches and of the MPI
// messages in the wall time, mean tree depth, and a hash of the final state (the same for every task count).

#include "mstar/mstar.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char** argv) {
    mstar::MpiSession mpi(argc, argv);
    mstar::ScopeGuard guard(argc, argv);
    const int n = argc > 1 ? std::atoi(argv[1]) : 770;
    const long steps = argc > 2 ? std::atol(argv[2]) : 200;
    const double a_bin = argc > 3 ? std::atof(argv[3]) : 1e-4;
    const int norm = argc > 4 ? std::atoi(argv[4]) : 0, diag = argc > 5 ? std::atoi(argv[5]) : 0;
    const double T = argc > 6 ? std::atof(argv[6]) : 0;
    {
        mstar::System S(0, n);
        S.constants.G = 1;
        S.options.PN_flags = mstar::PN_NONE;
        S.options.gbs_relative_tolerance = 1e-9;
        S.options.max_step_count = T > 0 ? -1 : steps;
        S.options.gbs_error_norm = norm;
        S.options.gbs_error_diag = diag != 0;
        S.options.mpi_min_n = 2;
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
        S.run(T > 0 ? T : 1.0);   // T = 0: ends at max_step_count
        const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::uint64_t h = 1469598103934665603ULL;
        for (double q : S.pos) { std::uint64_t b; std::memcpy(&b, &q, 8); h = (h ^ b) * 1099511628211ULL; }
        for (double q : S.vel) { std::uint64_t b; std::memcpy(&b, &q, 8); h = (h ^ b) * 1099511628211ULL; }
        const long st = S.perf.successful_steps + S.perf.failed_steps;
        if (mpi.rank == 0)
            std::printf("BENCH tasks %d (shared by %d) N %d steps %ld + %ld failed wall %.3f s, %.3f ms/step, force %.1f%%, messages %.1f%%, "
                        "tree depth %.1f, t %.6e, norm %d, failed that the other norm accepts %ld, dE/E %.3e, hash %016llx\n",
                        mpi.size, S.perf.mpi_tasks, n, S.perf.successful_steps, S.perf.failed_steps, wall, 1e3 * wall / (st > 0 ? st : 1),
                        100 * S.perf.t_force / wall, 100 * S.perf.t_comm / wall, S.perf.mean_tree_depth, S.time, norm, S.perf.failed_steps_other_norm, S.perf.relative_energy_error, (unsigned long long)h);
    }
    return 0;
}
