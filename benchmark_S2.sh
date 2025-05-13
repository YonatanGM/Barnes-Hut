#!/usr/bin/env bash
#SBATCH --partition=all
#SBATCH --nodelist=simcl1n[1-4]
#SBATCH --job-name=benchmark_S2
#SBATCH --output=benchmark_S2.out
#SBATCH --error=benchmark_S2.err
#SBATCH --time=2-00:00:00
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1        # one MPI rank per node

set -euo pipefail
set -x

# ---------- user-tunable defaults -------------------------------------------
file=${1:-"../data/state_vectors_csvs/scenario2_300149.csv"}
dt=${2:-"1h"}
t_end=${3:-"10d"}
vs=${4:-"12h"}
theta=${5:-"1.05"}

# ---------- bookkeeping ------------------------------------------------------
sanitized_file=$(basename "$file" .csv | tr -c 'A-Za-z0-9' '_')
timestamp=$(date +%Y%m%d_%H%M%S)
benchmark_dir="benchmark_S2_0/${sanitized_file}_dt_${dt}_tend_${t_end}_vs_${vs}_theta_${theta}"
mkdir -p "${benchmark_dir}/vs_outputs"

data_bodies_file="${benchmark_dir}/data_bodies.dat"
data_threads_file="${benchmark_dir}/data_threads.dat"
data_nodes_file="${benchmark_dir}/data_nodes.dat"
data_theta_file="${benchmark_dir}/data_theta.dat"

plot_bodies_file="${benchmark_dir}/plot_bodies.png"
plot_threads_file="${benchmark_dir}/plot_threads.png"
plot_nodes_file="${benchmark_dir}/plot_nodes.png"
plot_theta_file="${benchmark_dir}/plot_theta.png"

echo "bodies nodes threads total_time"       >  "$data_bodies_file"
echo "nodes threads total_time"              >  "$data_threads_file"
echo "threads nodes total_time"              >  "$data_nodes_file"
echo "theta total_time summed_dist_error"    >  "$data_theta_file"

# ---------- sweep parameters -------------------------------------------------
body_counts=(1000 10000 25000 50000 100000 200000)
node_counts=(1 2 3 4)
thread_counts=(1 2 4 8 12 24 48 96)
theta_values=(0.1 0.3 0.5 0.7 1.0 1.2 1.5 2.0)

# ---------- build & binary ---------------------------------------------------
BUILD_DIR="./build"
if [[ ! -d $BUILD_DIR ]]; then
    mkdir -p "$BUILD_DIR"
fi
cd "$BUILD_DIR"

simulate_binary=./simulate
[[ -x $simulate_binary ]] || { echo "simulate binary missing"; exit 1; }

# ---------- helper -----------------------------------------------------------
run_simulation () {
    local bodies=$1  nodes=$2  threads=$3  theta_now=$4
    local is_ref=${5:-false}

    export OMP_NUM_THREADS=$threads
    local tmp_out
    tmp_out=$(mktemp "sim_${bodies}_${threads}.XXXX")

    srun --exclusive -N "$nodes" \
         --cpus-per-task="$threads" \
         "$simulate_binary" \
         --file  "$file" \
         --dt    "$dt" \
         --t_end "$t_end" \
         --vs    "$vs" \
         --vs_dir "${OLDPWD}/${benchmark_dir}/vs_outputs" \
         --theta "$theta_now" \
         --bodies "$bodies" \
         ${is_ref:+-r} 2>&1 | tee "$tmp_out"

    local total_time summed_dist
    total_time=$(grep    "TOTAL_TIME"         "$tmp_out" | awk '{print $4}')
    summed_dist=$(grep   "SUMMED_DIST_ERROR"  "$tmp_out" | awk '{print $4}')
    rm -f "$tmp_out"

    echo "${total_time:-na} ${summed_dist:-na}"
}

# ---------- Phase 1 – bodies -------------------------------------------------
fixed_nodes=4
fixed_threads=48
fixed_theta=1.05
fixed_bodies=100000   # for later phases

for b in "${body_counts[@]}"; do
    read -r t _ < <(run_simulation "$b" "$fixed_nodes" "$fixed_threads" "$fixed_theta")
    echo "$b $fixed_nodes $fixed_threads $t" >> "${OLDPWD}/${data_bodies_file}"
done

# ---------- Phase 2 – threads ------------------------------------------------
fixed_nodes_thr=1
for th in "${thread_counts[@]}"; do
    read -r t _ < <(run_simulation "$fixed_bodies" "$fixed_nodes_thr" "$th" "$fixed_theta")
    echo "$fixed_nodes_thr $th $t" >> "${OLDPWD}/${data_threads_file}"
done

# ---------- Phase 3 – nodes --------------------------------------------------
for n in "${node_counts[@]}"; do
    read -r t _ < <(run_simulation "$fixed_bodies" "$n" "$fixed_threads" "$fixed_theta")
    echo "$fixed_threads $n $t" >> "${OLDPWD}/${data_nodes_file}"
done

# ---------- Phase 4 – theta --------------------------------------------------
sorted_theta_values=($(printf "%s\n" "${theta_values[@]}" | sort -n))
reference_theta=${sorted_theta_values[0]}

read -r t_ref _ < <(run_simulation "$fixed_bodies" "$fixed_nodes" \
                    "$fixed_threads" "$reference_theta" true)
echo "$reference_theta $t_ref 0" >> "${OLDPWD}/${data_theta_file}"

for th in "${sorted_theta_values[@]:1}"; do
    read -r t dist < <(run_simulation "$fixed_bodies" "$fixed_nodes" \
                       "$fixed_threads" "$th")
    echo "$th $t $dist" >> "${OLDPWD}/${data_theta_file}"
done

cd "${OLDPWD}"

# ---------- plotting (unchanged gnuplot blocks) ------------------------------
# … keep your gnuplot code exactly as before …


# =======================
# Generate Plots using GNUplot
# =======================

echo "Generating Plots..."

# Return to the original directory to access data files
cd "${OLDPWD}"

# 1. Runtime vs. Number of Bodies
gnuplot <<EOF
set terminal png size 1000,800
set output "${plot_bodies_file}"
set title "Runtime vs. Number of Bodies (MPI Nodes: $fixed_nodes, OpenMP Threads: $fixed_threads)"
set xlabel "Number of Bodies"
set ylabel "Total Time (s)"
set grid
set key outside

set label "Simulation Parameters:\nFILE: $file\nDT: $dt\nT_END: $t_end\nVS: $vs\nTHETA: $fixed_theta" at graph 0.02, 0.95 left

plot "${data_bodies_file}" using 1:4 with linespoints title "Runtime" lc rgb "blue"
EOF
echo "Plot generated: $plot_bodies_file"

# 2. Runtime vs. OpenMP Threads
gnuplot <<EOF
set terminal png size 1000,800
set output "${plot_threads_file}"
set title "Runtime vs. OpenMP Threads (MPI Nodes: $fixed_nodes, Bodies: $fixed_bodies)"
set xlabel "OpenMP Threads"
set ylabel "Total Time (s)"
set grid
set key outside

set label "Simulation Parameters:\nFILE: $file\nDT: $dt\nT_END: $t_end\nVS: $vs\nTHETA: $fixed_theta" at graph 0.02, 0.95 left

plot "${data_threads_file}" using 2:3 with linespoints title "Runtime" lc rgb "green"
EOF
echo "Plot generated: $plot_threads_file"

# 3. Runtime vs. MPI Nodes
gnuplot <<EOF
set terminal png size 1000,800
set output "${plot_nodes_file}"
set title "Runtime vs. MPI Nodes (OpenMP Threads: $fixed_threads, Bodies: $fixed_bodies)"
set xlabel "MPI Nodes"
set ylabel "Total Time (s)"
set grid
set key outside

set label "Simulation Parameters:\nFILE: $file\nDT: $dt\nT_END: $t_end\nVS: $vs\nTHETA: $fixed_theta" at graph 0.02, 0.95 left

plot "${data_nodes_file}" using 2:3 with linespoints title "Runtime" lc rgb "red"
EOF
echo "Plot generated: $plot_nodes_file"

# 4. Runtime and Distance Sum Difference vs. Theta
gnuplot <<EOF
set terminal png size 1000,800
set output "${plot_theta_file}"
set title "Runtime and Distance Sum Difference vs. Theta (θ)"
set xlabel "Theta (θ)"
set ylabel "Runtime (s)"
set y2label "Distance Sum Difference"
set grid
set key outside

# Align the y-axes
set ytics nomirror
set y2tics

# Position labels
set label "Simulation Parameters:\nFILE: $file\nDT: $dt\nT_END: $t_end\nVS: $vs" at graph 0.02, 0.95 left

# Plot both data series
plot "${data_theta_file}" using 1:2 with linespoints title "Runtime" lc rgb "blue", \
     "${data_theta_file}" using 1:3 axes x1y2 with linespoints title "Distance Sum Diff" lc rgb "red"
EOF
echo "Plot generated: $plot_theta_file"

echo "All plots have been generated in the '$benchmark_dir' directory."
