#!/bin/bash

#SBATCH --partition=all
#SBATCH --nodelist=simcl1n[1-4]
#SBATCH --job-name="nbody_sim"
#SBATCH --output=job_output.out
#SBATCH --error=job_error.err
#SBATCH --time=02:00:00
#SBATCH --nodes=4
#SBATCH --exclusive

# Exit immediately if a command exits with a non-zero status
set -e
set -x
# Default Parameters
file=${1:-"../data/scenario2_306051.csv"}
dt=${2:-"1h"}
t_end=${3:-"1d"}
vs=${4:-"1h"}
theta=${5:-"1.05"}

# Setup Benchmark Directory
sanitized_file=$(basename "$file" .csv | tr -c 'A-Za-z0-9' '_')
timestamp=$(date +%Y%m%d_%H%M%S)
benchmark_dir="benchmark/${sanitized_file}_dt_${dt}_tend_${t_end}_vs_${vs}_theta_${theta}"
mkdir -p "$benchmark_dir/vs_outputs"

# Data and Plot Files
data_bodies_file="${benchmark_dir}/data_bodies.dat"
plot_bodies_file="${benchmark_dir}/plot_bodies.png"

data_threads_file="${benchmark_dir}/data_threads.dat"
plot_threads_file="${benchmark_dir}/plot_threads.png"

data_nodes_file="${benchmark_dir}/data_nodes.dat"
plot_nodes_file="${benchmark_dir}/plot_nodes.png"

data_theta_file="${benchmark_dir}/data_theta.dat"
plot_theta_file="${benchmark_dir}/plot_theta.png"

# Initialize data files with headers
echo "bodies nodes threads total_time" > "$data_bodies_file"
echo "nodes threads total_time" > "$data_threads_file"
echo "threads nodes total_time" > "$data_nodes_file"
# θ-data now holds total_time and summed_dist_error
echo "theta total_time summed_dist_error" > "$data_theta_file"

# Parameter Ranges
body_counts=(1000 10000 25000 50000 100000)
node_counts=(1 2 3 4)
thread_counts=(2 8 16 32 48)
theta_values=(0.1 0.5 1.05 1.2 1.5 2.0)

# Build Directory and Simulation Binary
BUILD_DIR="./build"
if [ ! -d "$BUILD_DIR" ]; then
    mkdir -p "$BUILD_DIR"
    # Optionally, compile the simulation binary here
    # e.g., make -C "$BUILD_DIR"
fi

cd "$BUILD_DIR"

simulate_binary="./simulate"

# Check if the simulation binary exists and is executable
if [ ! -x "$simulate_binary" ]; then
    echo "Error: Simulation binary '$simulate_binary' not found or not executable."
    exit 1
fi

# Function to Run Simulation
# Arguments:
#   1. bodies
#   2. nodes
#   3. threads
#   4. theta
#   5. is_reference, default "false"
# Returns:
#   total_time summed_dist_error
run_simulation() {
    local bodies=$1
    local nodes=$2
    local threads=$3
    local current_theta=$4
    local is_ref=${5:-false}

    export OMP_NUM_THREADS=$threads
    # # Run the simulation and capture output
    # simulation_output=$(srun --exclusive -N "$nodes" "$simulate_binary" \
    #     --file "$file" --dt "$dt" --t_end "$t_end" --vs "$vs" \
    #     --vs_dir "${OLDPWD}/${benchmark_dir}/vs_outputs" --theta "$current_theta" --bodies "$bodies" --log 2>&1)
    # build the command array
    # mpirun -np, srun --exclusive -N
    # build the command array
    local cmd=(mpirun -np "$nodes" "$simulate_binary"
                --file "$file"
                --dt   "$dt"
                --t_end "$t_end"
                --vs   "$vs"
                --vs_dir "${OLDPWD}/${benchmark_dir}/vs_outputs"
                --theta "$current_theta"
                --bodies "$bodies")
    if [ "$is_ref" = "true" ]; then
        cmd+=( -r )
    fi

    simulation_output=$("${cmd[@]}" 2>&1)

    # Extract TOTAL_TIME and SUMMED_DIST_ERROR components
    total_time=$(echo "$simulation_output" | grep "TOTAL_TIME" | awk '{print $4}')
    summed_dist_error=$(echo "$simulation_output" | grep "SUMMED_DIST_ERROR" | awk '{print $4}')

    # fallback to "na" if missing
    total_time=${total_time:-na}
    summed_dist_error=${summed_dist_error:-na}

    echo "$total_time $summed_dist_error"
}

# =======================
# Phase 1: Runtime vs. Number of Bodies
# =======================
fixed_nodes=4
fixed_threads=32
fixed_theta=1.05
fixed_bodies=5000  # Adjust as needed

echo "Starting Phase 1: Runtime vs. Number of Bodies"
for bodies in "${body_counts[@]}"; do
    metrics=$(run_simulation "$bodies" "$fixed_nodes" "$fixed_threads" "$fixed_theta")
    total_time=$(echo "$metrics" | awk '{print $1}')
    echo "$bodies $fixed_nodes $fixed_threads $total_time" >> "${OLDPWD}/${data_bodies_file}"
done
echo "Phase 1 completed."

# =======================
# Phase 2: Runtime vs. OpenMP Threads
# =======================
echo "Starting Phase 2: Runtime vs. OpenMP Threads"
for threads in "${thread_counts[@]}"; do
    metrics=$(run_simulation "$fixed_bodies" "$fixed_nodes" "$threads" "$fixed_theta")
    total_time=$(echo "$metrics" | awk '{print $1}')
    echo "$fixed_nodes $threads $total_time" >> "${OLDPWD}/${data_threads_file}"
done
echo "Phase 2 completed."

# =======================
# Phase 3: Runtime vs. MPI Nodes
# =======================
echo "Starting Phase 3: Runtime vs. MPI Nodes"
for nodes in "${node_counts[@]}"; do
    metrics=$(run_simulation "$fixed_bodies" "$nodes" "$fixed_threads" "$fixed_theta")
    total_time=$(echo "$metrics" | awk '{print $1}')
    echo "$fixed_threads $nodes $total_time" >> "${OLDPWD}/${data_nodes_file}"
done
echo "Phase 3 completed."

# =======================
# Phase 4: Runtime and Distance Sum vs. Theta
# =======================
echo "Starting Phase 4: Runtime and Distance Sum vs. Theta"

# Sort theta_values in ascending order to identify the smallest theta
sorted_theta_values=($(printf "%s\n" "${theta_values[@]}" | sort -n))

# Extract the smallest theta as the reference
reference_theta=${sorted_theta_values[0]}
echo "Reference theta: $reference_theta"

# Run the reference simulation *with* the -r flag
metrics_ref=$(run_simulation \
    "$fixed_bodies" "$fixed_nodes" "$fixed_threads" \
    "$reference_theta" "true")

# parse out the two fields
total_time_ref=$(echo "$metrics_ref" | awk '{print $1}')
dist_ref=$(echo "$metrics_ref" | awk '{print $2}')

# Validate reference metrics
if [ "$total_time_ref" = "na" ] || [ "$dist_ref" = "na" ]; then
    echo "Error: Missing metrics in reference run with Theta: $reference_theta. Aborting."
    exit 1
fi

echo "Reference run completed: Theta=$reference_theta, TOTAL_TIME=$total_time_ref, SUMMED_DIST_ERROR=$dist_ref"

# Record reference metrics (error = 0 by definition)
echo "$reference_theta $total_time_ref 0" >> "${OLDPWD}/${data_theta_file}"

# Now loop the other thetas
for current_theta in "${sorted_theta_values[@]:1}"; do
    echo "Running θ = $current_theta …"
    metrics=$(run_simulation \
        "$fixed_bodies" "$fixed_nodes" "$fixed_threads" \
        "$current_theta" "false")

    tot_time=$(echo "$metrics" | awk '{print $1}')
    dist=$(echo "$metrics" | awk '{print $2}')

    echo "θ=$current_theta, TOTAL_TIME=$tot_time, SUMMED_DIST_ERROR=$dist"
    echo "$current_theta $tot_time $dist" >> "${OLDPWD}/${data_theta_file}"
done

echo "Phase 4 completed."

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
