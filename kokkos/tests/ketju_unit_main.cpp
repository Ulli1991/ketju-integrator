// Ketju-integrator tests: runner on the GoogleTest stand-in.
// Copyright (C) 2026 Ulrich Steinwandel.
// Licensed under the GPLv3 license.
// See LICENSE.md or <https://www.gnu.org/licenses/> for details.
//
// Runner for the unmodified KETJU unit tests (unit-tests/{integration,mergers,mst_construction,gizmo_additions}.cpp)
// on the GoogleTest stand-in (gtest_shim/gtest/gtest.h) and the single-task MPI stand-in; replaces their main.cpp
// (gtest-mpi-listener). ctest: ketju_unit_c.
// Optional argument: substring filter on the test names.

#include <gtest/gtest.h>
#include <mpi.h>
#include <cstdio>
#include <cstring>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    const char *filter = argc > 1 ? argv[1] : nullptr;
    int run = 0, failed = 0;
    for (auto &t : testing::registry()) {
        if (filter && !std::strstr(t.name.c_str(), filter)) continue;
        const int before = testing::failures();
        t.run();
        ++run;
        const bool ok = testing::failures() == before;
        if (!ok) ++failed;
        std::printf("[%s] %s\n", ok ? "  OK  " : "FAILED", t.name.c_str());
        std::fflush(stdout);
    }
    std::printf("%d tests, %d failed\n", run, failed);
    MPI_Finalize();
    return failed ? 1 : 0;
}
