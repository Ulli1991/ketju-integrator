// Ketju-integrator (MSTAR), C++/Kokkos library. Copyright (C) 2026 Ulrich Steinwandel. Licensed under the GPLv3 license.
//
// Speed regression test of the MPI sharing, meant for 4 tasks: N = 512 bodies with a hard binary, a fixed number of GBS
// step attempts, integrated unshared (every task integrates the whole system on its own core: the one-task time) and
// shared by the tasks. Bar, deliberately generous: the shared run is at least 2 x faster on 4 tasks (measured: see the
// README; an ideal split would be 4 x). The two runs must also end in the same state bit for bit. On one task, or
// without MPI support, the test only reports.

#include "mstar/mstar.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {
double run(bool share, std::vector<double>& state, int& tasks) {
    const int n = 512;
    mstar::System S(0, n);
    S.constants.G = 1;
    S.options.PN_flags = mstar::PN_NONE;
    S.options.max_step_count = 24;
    S.options.mpi_min_n = share ? 2 : (1 << 30);
    S.options.tree_sweep = share;
    std::uint64_t seed = 99;
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
    const double a = 1e-4, mb = 2.0 / n, vc = std::sqrt(2 * mb / a);
    const double x0[3] = {0.1 - 0.5 * a, 0.05, -0.02}, x1[3] = {0.1 + 0.5 * a, 0.05, -0.02};
    const double v0[3] = {0.01, -0.5 * vc, 0}, v1[3] = {0.01, 0.5 * vc, 0};
    S.set_particle(0, mb, x0, v0);
    S.set_particle(1, mb, x1, v1);
    double cp[3], cv[3];
    S.into_com_frame(cp, cv);
    const auto t0 = std::chrono::steady_clock::now();
    S.run(1e3);   // ends at max_step_count
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    state = S.pos;
    state.insert(state.end(), S.vel.begin(), S.vel.end());
    state.push_back(S.time);
    tasks = S.perf.mpi_tasks;
    return wall;
}
}  // namespace

int main(int argc, char** argv) {
    mstar::MpiSession mpi(argc, argv);
    mstar::ScopeGuard guard(argc, argv);
    std::vector<double> a, b;
    int ta = 1, tb = 1;
    double t_one = 1e300, t_shared = 1e300;
    for (int rep = 0; rep < 2; rep++) {   // the faster of two, against a busy node
        const double w1 = run(false, a, ta), w2 = run(true, b, tb);
        if (w1 < t_one) t_one = w1;
        if (w2 < t_shared) t_shared = w2;
    }
    const bool same = a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
    const bool shared = tb == mpi.size && mpi.size > 1;
    const double ratio = t_one / t_shared;
    const bool judged = shared && mpi.size >= 4;
    const bool ok = same && ta == 1 && (!judged || ratio >= 2.0);
    if (mpi.rank == 0 || !ok)
        std::printf("%s rank %d: N 512, 24 step attempts: one task %.3f s, shared by %d tasks %.3f s, ratio %.2f (bar >= 2 on 4 tasks%s), states %s\n",
                    ok ? "PASS" : "FAIL", mpi.rank, t_one, tb, t_shared, ratio, judged ? "" : "; not judged on this task count", same ? "equal" : "DIFFERENT");
    return ok ? 0 : 1;
}
