#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 112
#SBATCH -t 03:00:00
#SBATCH -J mstar_scan
# MPI sharing scan of the MSTAR library, one Slurm job on one node (set partition, memory and output on the sbatch
# line). Needs a build with -DMSTAR_ENABLE_MPI=ON; the modules of that build must be loaded by the submitting shell
# or by MSTAR_SCAN_SETUP (a file sourced here).
#   sbatch [-p ...] scripts/bench_mpi_scan.sh <build directory> <output file> [parts = scan,threads,detail]
# Parts:
#   scan     N = 8 .. 1500, with a hard binary (a = 1e-4) and without (a = 0.05), tasks 1 .. 112, modes single (one
#            task integrates, the others wait), default (shared from the library's threshold), shared (always shared)
#   threads  N = 770 shared: tasks x Kokkos threads at a fixed number of cores, and two threads on one core
#   detail   N = 770 shared with a barrier before each exchange: waiting for the slowest task against transfer
# Lines "SCAN ..." go to the output file; scripts/bench_mpi_table.py turns them into tables and checks the bar
# (default not slower than single by more than 2 % at any point).
set -u
BUILD=$1; OUT=$2; PARTS=${3:-scan,threads,detail}
[ -n "${MSTAR_SCAN_SETUP:-}" ] && source "$MSTAR_SCAN_SETUP"
X=$BUILD/mstar-mpi-scan; [ -x $X ] || { echo "no $X"; exit 1; }
LAUNCH=${MSTAR_SCAN_LAUNCH:-srun --cpu-bind=cores}
export OMP_PROC_BIND=false
NT=${SLURM_NTASKS:-112}
attempts() { awk -v n=$1 'BEGIN{a=int(2000/(3.8e-4*n*n)); if(a<12)a=12; if(a>20000)a=20000; print a}'; }
: > $OUT
echo "# $(date) host $(hostname) tasks $NT build $BUILD" >> $OUT
case ,$PARTS, in *,scan,*)
  export OMP_NUM_THREADS=1
  for a in 1e-4 0.05; do for n in 8 16 32 64 128 256 512 770 1500; do for np in 1 2 4 8 16 32 64 112; do
    [ $np -le $NT ] || continue
    for mode in single default shared; do
      [ $np = 1 ] && [ $mode != single ] && continue
      $LAUNCH -n $np -c 1 $X $n $(attempts $n) $a $mode 3 2>&1 | grep "^SCAN" >> $OUT
    done
  done; done; done;;
esac
case ,$PARTS, in *,threads,*)
  # the same cores, divided into tasks x threads; then every task with two threads on its one core
  for spec in "$NT 1" "$((NT/2)) 2" "$((NT/4)) 4" "$((NT/8)) 8"; do set -- $spec
    OMP_NUM_THREADS=$2 $LAUNCH -n $1 -c $2 $X 770 16 1e-4 shared 3 2>&1 | grep "^SCAN" | sed "s/^SCAN/SCAN threads $2 cores_per_task $2/" >> $OUT
  done
  OMP_NUM_THREADS=2 $LAUNCH -n $NT -c 1 $X 770 16 1e-4 shared 3 2>&1 | grep "^SCAN" | sed "s/^SCAN/SCAN threads 2 cores_per_task 1/" >> $OUT;;
esac
case ,$PARTS, in *,detail,*)
  export OMP_NUM_THREADS=1
  for np in 8 32 64 112; do [ $np -le $NT ] || continue; for d in 0 1; do
    $LAUNCH -n $np -c 1 $X 770 16 1e-4 shared 3 $d 2>&1 | grep "^SCAN" | sed "s/^SCAN/SCAN detail $d/" >> $OUT
  done; done;;
esac
echo "# end $(date)" >> $OUT
