#!/usr/bin/env bash
#SBATCH --partition=all
#SBATCH --nodelist=simcl1n[1-4]
#SBATCH --nodes=4                 # 4 nodes in total
#SBATCH --ntasks-per-node=1       # 1 MPI rank per node
#SBATCH --exclusive
#SBATCH --time=2-00:00:00
#SBATCH --job-name=benchmark_S2
#SBATCH --output=benchmark_S2.out
#SBATCH --error=benchmark_S2.err

set -euo pipefail
set -x

############ user-supplied (or default) parameters ############
file=${1:-"../data/state_vectors_csvs/scenario2_300149.csv"}
dt=${2:-"1h"}
t_end=${3:-"10d"}
vs=${4:-"12h"}
theta_start=${5:-"1.05"}          # used only for the plots’ label

############ bookkeeping ######################################
root_dir=$(pwd)                   # remember where we started

sanitized=$(basename "$file" .csv | tr -c 'A-Za-z0-9' _)
benchmark_dir="$root_dir/benchmark_S2_0/${sanitized}_dt_${dt}_tend_${t_end}_vs_${vs}_theta_${theta_start}"
mkdir -p "$benchmark_dir/vs_outputs"

data_bodies="$benchmark_dir/data_bodies.dat"
data_threads="$benchmark_dir/data_threads.dat"
data_nodes="$benchmark_dir/data_nodes.dat"
data_theta="$benchmark_dir/data_theta.dat"

echo 'bodies nodes threads total_time'     >"$data_bodies"
echo 'nodes  threads total_time'           >"$data_threads"
echo 'threads nodes  total_time'           >"$data_nodes"
echo 'theta  total_time summed_dist_error' >"$data_theta"

############ compile / enter build dir ########################
cd "$root_dir/build"
simulate=./simulate
[[ -x $simulate ]] || { echo "simulate not found"; exit 1; }

############ helper ###########################################
run_simulation () {
    local bodies=$1 nodes=$2 threads=$3 theta=$4 ref=${5:-false}
    export OMP_NUM_THREADS=$threads

    tmp=$(mktemp --tmpdir="$benchmark_dir" sim.XXXX)
    # --wait=0 keeps Slurm from throwing “step creation disabled”
    srun --exclusive --wait=0 -N "$nodes" --cpus-per-task="$threads" \
         "$simulate" \
         --file "$file" --dt "$dt" --t_end "$t_end" --vs "$vs" \
         --vs_dir "$benchmark_dir/vs_outputs" \
         --theta "$theta" --bodies "$bodies" ${ref:+-r} 2>&1 | tee "$tmp"

    # grab the numbers printed by simulate
    total=$(grep -m1 'TOTAL_TIME:'  "$tmp" | awk '{print $4}')
    dist=$(  grep -m1 'SUMMED_DIST_ERROR:' "$tmp" | awk '{print $4}')
    rm -f "$tmp"
    echo "${total:-na} ${dist:-na}"
}

############ experiment ranges ################################
body_list=(1000 10000 25000 50000 100000 200000)
thread_list=(1 2 4 8 12 24 48 96)
node_list=(1 2 3 4)
theta_list=(0.1 0.3 0.5 0.7 1.0 1.2 1.5 2.0)

fixed_nodes=4
fixed_threads=48
fixed_theta=1.05
fixed_bodies=100000

############ Phase 1: bodies ##################################
for b in "${body_list[@]}"; do
    read -r t _ < <(run_simulation "$b" "$fixed_nodes" "$fixed_threads" "$fixed_theta")
    echo "$b $fixed_nodes $fixed_threads $t" >>"$data_bodies"
done

############ Phase 2: threads (1 node) ########################
for th in "${thread_list[@]}"; do
    read -r t _ < <(run_simulation "$fixed_bodies" 1 "$th" "$fixed_theta")
    echo "1 $th $t" >>"$data_threads"
done

############ Phase 3: nodes ###################################
for n in "${node_list[@]}"; do
    read -r t _ < <(run_simulation "$fixed_bodies" "$n" "$fixed_threads" "$fixed_theta")
    echo "$fixed_threads $n $t" >>"$data_nodes"
done

############ Phase 4: theta scan ##############################
read -r t_ref _ < <(run_simulation "$fixed_bodies" "$fixed_nodes" "$fixed_threads" 0.1 true)
echo "0.1 $t_ref 0" >>"$data_theta"
for th in "${theta_list[@]:1}"; do
    read -r t d < <(run_simulation "$fixed_bodies" "$fixed_nodes" "$fixed_threads" "$th")
    echo "$th $t $d" >>"$data_theta"
done

############ plotting (run from root) #########################
cd "$root_dir"

gnuplot <<EOF
set term png size 1000,800
set grid
# ---- bodies ----
set output "$benchmark_dir/plot_bodies.png"
set title "Runtime vs. #Bodies"
set xlabel "Bodies"; set ylabel "Time [s]"
plot "$data_bodies" using 1:4 with linespoints title "time"
# ---- threads ----
set output "$benchmark_dir/plot_threads.png"
set title "Runtime vs. Threads (1 node)"
set xlabel "Threads"; set ylabel "Time [s]"
plot "$data_threads" using 2:3 with linespoints title "time"
# ---- nodes ----
set output "$benchmark_dir/plot_nodes.png"
set title "Runtime vs. Nodes ($fixed_threads threads/rank)"
set xlabel "Nodes"; set ylabel "Time [s]"
plot "$data_nodes" using 2:3 with linespoints title "time"
# ---- theta ----
set output "$benchmark_dir/plot_theta.png"
set title "Runtime & Error vs. Theta"
set xlabel "Theta"; set ylabel "Time [s]"; set y2label "Error"
set ytics nomirror; set y2tics
plot "$data_theta" using 1:2 w lp title "time", \
     "$data_theta" using 1:3 axes x1y2 w lp title "error"
EOF

echo "✔  All results are in:  $benchmark_dir"
