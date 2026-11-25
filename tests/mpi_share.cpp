// Ketju-integrator (MSTAR), C++/Kokkos library. Copyright (C) 2026 Ulrich Steinwandel. Licensed under the GPLv3 license.
//
// MPI sharing of one system over the tasks of a communicator (System::set_mpi_fcomm, Options::mpi_min_n): every case
// is integrated twice by every task, once without sharing (each task on its own) and once with the O(N^2) loops
// divided over the tasks, and the two results must agree bit for bit (state, time, step counts, merger records,
// energy error, work ledgers). The same holds for Options::tree_sweep (vertex coordinates by a parents-first sweep
// instead of path sums), alone and together with sharing. As the unshared run does not depend on the task count, this is also the statement
// that results do not depend on the task count. Run on 1, 2 and 4 tasks (1 task: sharing is off, the test is then
// only a repeatability check). Rank 0 prints a hash of each final state for comparison between task counts.

#include "mstar/mstar.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace {

struct Lcg {
    std::uint64_t s;
    double operator()() {   // uniform in [0, 1)
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return (double)(s >> 11) / 9007199254740992.0;
    }
};

// n bodies of total mass ~1 in a unit sphere, velocity dispersion ~0.4; bodies 0 and 1 a circular binary of
// semi-major axis a_bin (masses m_bin each)
void cluster(mstar::System& S, std::uint64_t seed, double a_bin, double m_bin) {
    const int n = S.num_particles;
    Lcg u{seed};
    for (int i = 0; i < n; i++) {
        double x[3], v[3], r2;
        do {
            r2 = 0;
            for (int k = 0; k < 3; k++) { x[k] = 2 * u() - 1; r2 += x[k] * x[k]; }
        } while (r2 > 1);
        for (int k = 0; k < 3; k++) v[k] = 0.4 * (2 * u() - 1);
        S.set_particle(i, (0.5 + u()) / n, x, v);
    }
    const double vc = std::sqrt(S.constants.G * 2 * m_bin / a_bin);
    const double x0[3] = {0.1 - 0.5 * a_bin, 0.05, -0.02}, x1[3] = {0.1 + 0.5 * a_bin, 0.05, -0.02};
    const double v0[3] = {0.01, -0.5 * vc, 0}, v1[3] = {0.01, 0.5 * vc, 0};
    S.set_particle(0, m_bin, x0, v0);
    S.set_particle(1, m_bin, x1, v1);
    double cp[3], cv[3];
    S.into_com_frame(cp, cv);
}

// bodies i and j on a head-on course, separation d, closing speed vrel
void head_on(mstar::System& S, int i, int j, double d, double vrel) {
    const double xi[3] = {-0.4, 0.3, 0.2}, xj[3] = {-0.4 + d, 0.3, 0.2};
    const double vi[3] = {0.5 * vrel, 0, 0}, vj[3] = {-0.5 * vrel, 0, 0};
    S.set_particle(i, S.mass[i], xi, vi);
    S.set_particle(j, S.mass[j], xj, vj);
}

std::uint64_t fnv(std::uint64_t h, const void* p, size_t n) {
    const unsigned char* c = (const unsigned char*)p;
    for (size_t q = 0; q < n; q++) { h ^= c[q]; h *= 1099511628211ULL; }
    return h;
}

struct Result {
    std::vector<double> d;   // every double of the outcome
    std::vector<long> l;     // every integer of the outcome
    int tasks = 1;
    bool same(const Result& o) const {
        return d.size() == o.d.size() && l == o.l && (d.empty() || std::memcmp(d.data(), o.d.data(), d.size() * sizeof(double)) == 0);
    }
    std::uint64_t hash() const {
        std::uint64_t h = fnv(1469598103934665603ULL, d.data(), d.size() * sizeof(double));
        return fnv(h, l.data(), l.size() * sizeof(long));
    }
};

using Setup = std::function<void(mstar::System&)>;

Result integrate(int npn, int nother, const Setup& setup, double T, int min_n, int team_rows_max_n, bool sweep) {
    mstar::System S(npn, nother);
    S.constants.G = 1;
    S.constants.c = 1e3;
    S.options.PN_flags = mstar::PN_NONE;
    setup(S);
    S.options.mpi_min_n = min_n;
    S.options.tree_sweep = sweep;
    if (team_rows_max_n >= 0) S.options.team_rows_max_n = team_rows_max_n;
    const bool ok = S.run(T);
    Result r;
    r.tasks = S.perf.mpi_tasks;
    r.d = S.mass;
    r.d.insert(r.d.end(), S.pos.begin(), S.pos.end());
    r.d.insert(r.d.end(), S.vel.begin(), S.vel.end());
    r.d.insert(r.d.end(), S.spin.begin(), S.spin.end());
    r.d.insert(r.d.end(), S.collision_radius.begin(), S.collision_radius.end());
    for (double x : {S.time, S.perf.relative_energy_error, S.perf.total_work, S.pn_dB, S.ext_work, S.stop_collisions.r_sep, S.stop_collisions.r_coll}) r.d.push_back(x);
    for (const auto& m : S.mergers)
        for (double x : {m.t_merger, m.m1, m.m2, m.m_remnant, m.v_kick, m.dE_collision}) r.d.push_back(x);
    r.l = {ok, S.num_particles, S.num_pn_particles, S.perf.successful_steps, S.perf.failed_steps, S.perf.kink_landings,
           S.perf.contact_landings, (long)S.mergers.size(), S.stop_collisions.i, S.stop_collisions.j, S.stop_collisions.type};
    for (const auto& m : S.mergers) { r.l.push_back(m.extra_index1); r.l.push_back(m.extra_index2); r.l.push_back(m.is_collision); }
    for (int i = 0; i < S.num_particles; i++) r.l.push_back(S.extra_index[i]);
    return r;
}

int g_fail = 0;

// alone: mpi_min_n above N (no sharing); shared: mpi_min_n = 2. team_alone: team_rows_max_n of the unshared run (-1:
// the default, i.e. the path a serial run takes); expect: checks on the outcome (collisions happened, ...)
void check(const mstar::MpiSession& mpi, const char* name, int npn, int nother, const Setup& setup, double T, int team_alone,
           const std::function<bool(const Result&)>& expect) {
    const Result a = integrate(npn, nother, setup, T, 1 << 30, team_alone, false);
    const Result b = integrate(npn, nother, setup, T, 2, -1, false);
    const Result c = integrate(npn, nother, setup, T, 1 << 30, team_alone, true);   // sweep, unshared
    const Result d = integrate(npn, nother, setup, T, 2, -1, true);                 // sweep, shared
    const bool shared = !mstar::mpi_enabled() || mpi.size == 1 ? b.tasks == 1 : b.tasks == mpi.size;
    const bool ok = a.same(b) && a.same(c) && a.same(d) && a.tasks == 1 && shared && d.tasks == b.tasks && expect(a);
    if (!ok) g_fail++;
    if (mpi.rank == 0 || !ok)
        std::printf("[%s] rank %d %-14s N %d steps %ld + %ld failed, mergers %ld, kink %ld, contact %ld, tasks %d, bitwise shared %s, sweep %s, both %s, HASH %s %016llx\n",
                    ok ? "  OK  " : "FAILED", mpi.rank, name, npn + nother, a.l[3], a.l[4], a.l[7], a.l[5], a.l[6], b.tasks,
                    a.same(b) ? "equal" : "DIFFERENT", a.same(c) ? "equal" : "DIFFERENT", a.same(d) ? "equal" : "DIFFERENT", name,
                    (unsigned long long)a.hash());
    std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
    mstar::MpiSession mpi(argc, argv);
    mstar::ScopeGuard guard(argc, argv);
    const auto any = [](const Result&) { return true; };

    // Newtonian cluster with a hard binary (a = 1e-3 of the cluster radius)
    check(mpi, "newton200", 0, 200, [](mstar::System& S) { cluster(S, 11, 1e-3, 0.005); }, 0.02, -1, any);

    // contact collisions with landing: one pair on a head-on course reaches R1 + R2 during the run
    check(mpi, "contact150", 0, 150, [](mstar::System& S) {
        cluster(S, 12, 1e-3, 0.005);
        head_on(S, 2, 3, 0.02, 2.0);
        S.collision_radius.assign(150, 1e-5);
        S.collision_radius[2] = S.collision_radius[3] = 2e-3;
    }, 0.02, -1, [](const Result& r) { return r.l[7] >= 1 && r.l[1] < 150; });

    // softened star-star forces (kink landings) and an external tide (its acceleration is applied by every task)
    check(mpi, "soft_tide160", 0, 160, [](mstar::System& S) {
        cluster(S, 13, 4e-3, 0.005);
        S.options.star_star_softening = 5e-3;
        S.options.enable_external_tide = true;
        const double T[6] = {-0.3, -0.3, 0.6, 0.05, 0, 0};
        for (int k = 0; k < 6; k++) S.options.external_tidal_tensor[k] = T[k];
    }, 0.02, -1, [](const Result& r) { return r.l[5] >= 1; });

    // extra forces: linear field with its mean removed, a static point mass and a moving perturber
    check(mpi, "extra130", 0, 130, [](mstar::System& S) {
        cluster(S, 14, 2e-3, 0.005);
        S.options.extra.vindep = true;
        S.options.extra.remove_mean = true;
        S.options.extra.A[0] = 0.2; S.options.extra.A[4] = -0.1; S.options.extra.A[8] = -0.1; S.options.extra.A[1] = 0.03;
        S.options.extra.n_pm = 1;
        S.options.extra.pm_m[0] = 0.5; S.options.extra.pm_x[0][0] = 3; S.options.extra.pm_eps2 = 1e-4;
        S.perturbers = {0.3, -2.5, 0.5, 0, 0.2, 0, 0, 0, 0, 0.01};
    }, 0.02, -1, any);

    // two black holes with all PN terms among the stars
    check(mpi, "pn130", 2, 128, [](mstar::System& S) {
        cluster(S, 15, 1e-3, 0.05);
        S.options.PN_flags = mstar::PN_ALL;
    }, 0.004, -1, any);

    // stop rule: the call ends at the first collision
    check(mpi, "stop140", 0, 140, [](mstar::System& S) {
        cluster(S, 16, 1e-3, 0.005);
        head_on(S, 5, 6, 0.02, 2.0);
        S.stop_collisions.radius.assign(140, 1e-5);
        S.stop_collisions.radius[5] = S.stop_collisions.radius[6] = 2e-3;
    }, 0.02, -1, [](const Result& r) { return r.l[8] == 5 && r.l[9] == 6; });

    // below team_rows_max_n: the shared run takes the staged path, compared with the staged path of one task ...
    check(mpi, "small40_staged", 0, 40, [](mstar::System& S) { cluster(S, 17, 1e-3, 0.02); }, 0.05, 0, any);
    // ... and with the path a serial run takes there (all rows in one team)
    check(mpi, "small40_team", 0, 40, [](mstar::System& S) { cluster(S, 17, 1e-3, 0.02); }, 0.05, -1, any);
    // fewer bodies than tasks can share evenly
    check(mpi, "tiny3", 0, 3, [](mstar::System& S) { cluster(S, 18, 1e-2, 0.3); }, 0.5, 0, any);

    if (mpi.rank == 0) std::printf("%s (%d task%s, MPI support %s)\n", g_fail ? "FAIL" : "PASS", mpi.size, mpi.size == 1 ? "" : "s", mstar::mpi_enabled() ? "on" : "off");
    return g_fail ? 1 : 0;
}
