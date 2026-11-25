#!/usr/bin/env python3
"""Tables from the SCAN lines of scripts/bench_mpi_scan.sh, and the bar: with the default threshold no (N, tasks) point
is slower than one task integrating alone (mode single on the same number of started tasks) by more than 2 %.
usage: bench_mpi_table.py <scan output file>; exit status 1 if a point misses the bar."""
import sys


def parse(line):
    w = line.split()
    d = {}
    i = 1
    while i + 1 < len(w):
        d[w[i]] = w[i + 1]
        i += 2
    return d


rows = [parse(l) for l in open(sys.argv[1]) if l.startswith("SCAN")]
scan = [r for r in rows if "threads" not in r and "detail" not in r]
key = lambda r: (float(r["a_bin"]), int(r["N"]), int(r["tasks"]), r["mode"])
ms = {key(r): float(r["ms_per_step"]) for r in scan}
info = {key(r): r for r in scan}
hashes = {}
for r in scan:
    hashes.setdefault((float(r["a_bin"]), int(r["N"])), set()).add(r["hash"])
tasks = sorted({k[2] for k in ms})
sizes = sorted({k[1] for k in ms})
miss = []
for a in sorted({k[0] for k in ms}, reverse=False):
    print(f"\n== binary a = {a:g} cluster radii ({'hard binary' if a < 1e-2 else 'no hard binary'})")
    print("ms per GBS step, mode single (one task integrates; reference at each task count)")
    print("      N " + "".join(f"{t:>10d}" for t in tasks))
    for n in sizes:
        print(f"{n:7d} " + "".join(f"{ms.get((a, n, t, 'single'), float('nan')):10.4f}" for t in tasks))
    for mode, title in (("default", "default threshold: time relative to single at the same task count (bar <= 1.02); * = shared"),
                        ("shared", "always shared: time relative to single (places the threshold)")):
        print(title)
        print("      N " + "".join(f"{t:>10d}" for t in tasks))
        for n in sizes:
            cells = []
            for t in tasks:
                ref = ms.get((a, n, t, "single"))
                v = ref if t == 1 else ms.get((a, n, t, mode))
                if ref is None or v is None:
                    cells.append(f"{'-':>10s}")
                    continue
                shared = t > 1 and int(info[(a, n, t, mode)]["sharing"]) > 1
                cells.append(f"{v / ref:9.3f}{'*' if shared else ' '}")
                if mode == "default" and v / ref > 1.02:
                    miss.append((a, n, t, v / ref))
            print(f"{n:7d} " + "".join(cells))
    print("shared runs (default threshold): speed-up over single, efficiency = speed-up / tasks, MPI share of the wall time")
    print("      N " + "".join(f"{t:>18d}" for t in tasks if t > 1))
    for n in sizes:
        cells = []
        for t in tasks:
            if t == 1:
                continue
            k = (a, n, t, "default")
            if k not in ms or int(info[k]["sharing"]) <= 1:
                cells.append(f"{'-':>18s}")
                continue
            s = ms[(a, n, t, "single")] / ms[k]
            cells.append(f"{s:6.1f}x {100 * s / t:3.0f}% {float(info[k]['mpi_pct']):4.0f}%")
        print(f"{n:7d} " + "".join(cells))
bad_hash = {k: v for k, v in hashes.items() if len(v) > 1}
print(f"\nfinal state hashes: {'the same in every mode and task count for each (a, N)' if not bad_hash else 'DIFFER for ' + str(sorted(bad_hash))}")
for r in rows:
    if "threads" in r:
        print(f"threads: {r['tasks']:>4s} tasks x {r['threads']} threads on {r['cores_per_task']} core(s) each: {float(r['ms_per_step']):8.3f} ms per step, MPI {r['mpi_pct']}%")
    if "detail" in r:
        print(f"detail {r['detail']}: {r['tasks']:>4s} tasks: {float(r['ms_per_step']):8.3f} ms per step, force launches {r['force_pct']}%, MPI {r['mpi_pct']}%, "
              f"of which waiting for the slowest task {r['wait_pct']}%")
print(f"\nBAR default <= 1.02 x single: {'met at every point' if not miss else 'MISSED at ' + ', '.join(f'a={a:g} N={n} tasks={t}: {v:.3f}' for a, n, t, v in miss)}")
sys.exit(1 if miss or bad_hash else 0)
