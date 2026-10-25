// Ketju-integrator sample N-body driver program.
// Copyright (C) 2020-2022 Antti Rantala, Matias Mannerkoski, and contributors.
// Licensed under the GPLv3 license.
// See LICENSE.md or <https://www.gnu.org/licenses/> for details.
// Modified by Ulrich Steinwandel, 2026: rewritten in C++ with Kokkos support (GPU).
// Copyright (C) 2026 Ulrich Steinwandel. Licensed under the GPLv3 license.
//
// Standalone driver of the C++/Kokkos MSTAR (the counterpart of nbody/main.c): reads KETJU's IC format
// (header "num_bhs num_stars", rows "m x y z vx vy vz Sx Sy Sz" for the BHs, "m x y z vx vy vz" for the stars;
// GADGET units by default), integrates in the CoM frame with all PN terms and writes nbody_output.txt in the format of
// nbody/main.c.
//   mstar-driver ic_file integration_time substeps [tolerance=1e-9] [host|device] [G c]

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "mstar/mstar.hpp"

namespace {

bool read_ic(const char* name, mstar::System*& S) {
    FILE* fp = std::fopen(name, "r");
    if (!fp) return false;
    int npn = 0, nst = 0;
    if (std::fscanf(fp, "%d %d", &npn, &nst) != 2) { std::fclose(fp); return false; }
    S = new mstar::System(npn, nst);
    for (int i = 0; i < npn + nst; ++i) {
        double* x = &S->pos[3 * i];
        double* v = &S->vel[3 * i];
        if (std::fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf", &S->mass[i], &x[0], &x[1], &x[2], &v[0], &v[1], &v[2]) != 7) {
            std::fclose(fp);
            return false;
        }
        if (i < npn && std::fscanf(fp, "%lf %lf %lf", &S->spin[3 * i], &S->spin[3 * i + 1], &S->spin[3 * i + 2]) != 3) {
            std::fclose(fp);
            return false;
        }
    }
    std::fclose(fp);
    return true;
}

void write_output(FILE* fp, const mstar::System& S) {
    std::fprintf(fp, "%g %d %d\n", S.time, S.num_particles, S.num_pn_particles);
    for (int i = 0; i < S.num_particles; ++i) std::fprintf(fp, "%g ", S.mass[i]);
    std::fprintf(fp, "\n");
    for (int i = 0; i < 3 * S.num_particles; ++i) std::fprintf(fp, "%g ", S.pos[i]);
    std::fprintf(fp, "\n");
    for (int i = 0; i < 3 * S.num_particles; ++i) std::fprintf(fp, "%g ", S.vel[i]);
    std::fprintf(fp, "\n");
    for (int i = 0; i < 3 * S.num_pn_particles; ++i) std::fprintf(fp, "%g ", S.spin[i]);
    std::fprintf(fp, "\n");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::printf("Usage:\nmstar-driver ic_file integration_time substeps [tolerance=1e-9] [host|device] [G c]\n");
        return 2;
    }
    mstar::ScopeGuard guard(argc, argv);
    int rc = 0;
    {
        mstar::System* S = nullptr;
        if (!read_ic(argv[1], S)) {
            std::printf("Error reading %s\n", argv[1]);
            delete S;
            return 1;
        }
        double cp[3], cv[3];
        S->into_com_frame(cp, cv);   // integrate in the CoM frame
        S->options.gbs_relative_tolerance = argc > 4 ? std::strtod(argv[4], nullptr) : 1e-9;
        S->options.max_tree_distance = 2;
        S->options.PN_flags = mstar::PN_ALL;
        S->options.host_engine = argc > 5 && !std::strcmp(argv[5], "host");
        if (argc > 7) {
            S->constants.G = std::strtod(argv[6], nullptr);
            S->constants.c = std::strtod(argv[7], nullptr);
        }
        FILE* out = std::fopen("nbody_output.txt", "w");
        const double T = std::strtod(argv[2], nullptr);
        const long sub = std::strtol(argv[3], nullptr, 0);
        for (long i = 0; i < sub; ++i) {
            if (!S->run(T / sub)) { rc = 1; break; }
            if (out) write_output(out, *S);
        }
        if (out) std::fclose(out);
        std::printf("MSTAR (%s engine, %s): N = %d (%d PN), t = %g, %ld steps (%ld failed), %d mergers, last |dE/E| %.3e, "
                    "wall %.3f s (MST %.3f s, sync %.3f s), %ld launches\n",
                    S->options.host_engine ? "host" : "default", S->options.host_engine ? mstar::host_execution_space_name()
                                                                                        : mstar::execution_space_name(),
                    S->num_particles, S->num_pn_particles, S->time, S->perf.successful_steps, S->perf.failed_steps,
                    (int)S->mergers.size(), std::abs(S->perf.relative_energy_error), S->perf.t_total, S->perf.t_mst,
                    S->perf.t_sync, S->perf.launches);
        delete S;
    }
    return rc;
}
