#!/bin/bash
#SBATCH --partition=all             # Use the "all" partition
#SBATCH --nodelist=simcl1n[1-4]     # Node list: simcl1n1, simcl1n2, simcl1n3, simcl1n4
#SBATCH --job-name="nbody_sim"      # Job name
#SBATCH --output=job_output.out     # Standard output
#SBATCH --error=job_error.err       # Standard error
#SBATCH --time=02:00:00             # Maximum run time (2 hours)
#SBATCH --nodes=4                   # Total number of nodes to allocate (adjust as needed)
#SBATCH --exclusive                 # Exclusive access to nodes

# Load necessary modules
# module load gcc/10.2.0
module load openmpi/3.1.6-gcc-10.2
# module load gnuplot

# Define the output CSV file and gnuplot file
output_file="benchmark_results.csv"
plot_file="performance_plot.png"

# Initialize the CSV file with headers
echo "MPI_Nodes,OMP_Threads,SIMULATION_TIME(s),TOTAL_TIME(s)" > $output_file

# Define the range of MPI nodes and OpenMP threads
mpi_nodes=(1 2 3 4)                # Number of MPI nodes
omp_threads=(8 16 24 32 48)        # Number of OpenMP threads per MPI node

# Simulation parameters (modify as needed)

cd ./build

file=${1:-"../data/scenario1.csv"}
dt=${2:-"1h"}
t_end=${3:-"1y"}
vs=${4:-"1d"}
vs_dir=${5:-"sim0"}
theta=${6:-"1.05"}

# Path to the simulation binary (update this path accordingly)
simulate_binary="./simulate"

# Iterate over each MPI node count and OpenMP thread count
for N in "${mpi_nodes[@]}"; do
  for OMP in "${omp_threads[@]}"; do
    # Check if the number of OpenMP threads does not exceed physical cores per node
    if (( OMP > 48 )); then
      echo "Skipping configuration: MPI Nodes=$N, OMP Threads=$OMP (exceeds 48 cores per node)"
      continue
    fi

    # Set the number of OpenMP threads
    export OMP_NUM_THREADS=$OMP

    # Log the current configuration
    echo "Running with MPI Nodes: $N, OpenMP Threads: $OMP"

    # Execute the simulation using srun and capture output
    log_output=$(srun -N $N $simulate_binary \
      --file $file --dt $dt --t_end $t_end --vs $vs \
      --vs_dir $vs_dir --theta $theta --log 2>&1)

    # Extract simulation and total time from the log output
    simulation_time=$(echo "$log_output" | grep "SIMULATION_TIME" | awk '{print $2}')
    total_time=$(echo "$log_output" | grep "TOTAL_TIME" | awk '{print $2}')

    # Set to 'NA' if values are not found
    simulation_time=${simulation_time:-NA}
    total_time=${total_time:-NA}

    # Append the results to the CSV file
    echo "$N,$OMP,$simulation_time,$total_time" >> $output_file
  done
done

# Generate a single performance plot using gnuplot
echo "Generating performance plot..."

gnuplot <<EOF
set terminal png size 1000,800
set output "$plot_file"
set title "Benchmark Results: OpenMP Threads vs Time (Colored by MPI Processes)"
set xlabel "OpenMP Threads"
set ylabel "Simulation Time (s)"
set style data linespoints
set key outside
set datafile separator ","

# Define custom line styles with unique colors
set style line 1 lc rgb '#FF0000' lt 1 lw 2 pt 7  # Red for MPI = 1
set style line 2 lc rgb '#00FF00' lt 1 lw 2 pt 7  # Green for MPI = 2
set style line 3 lc rgb '#0000FF' lt 1 lw 2 pt 7  # Blue for MPI = 3
set style line 4 lc rgb '#FF00FF' lt 1 lw 2 pt 7  # Magenta for MPI = 4

# Plot OpenMP Threads vs Time with different lines for each MPI process count
plot "$output_file" using (\$1==1 ? \$2 : 1/0):(\$1==1 ? \$3 : 1/0) title "MPI = 1" with linespoints ls 1, \
     "$output_file" using (\$1==2 ? \$2 : 1/0):(\$1==2 ? \$3 : 1/0) title "MPI = 2" with linespoints ls 2, \
     "$output_file" using (\$1==3 ? \$2 : 1/0):(\$1==3 ? \$3 : 1/0) title "MPI = 3" with linespoints ls 3, \
     "$output_file" using (\$1==4 ? \$2 : 1/0):(\$1==4 ? \$3 : 1/0) title "MPI = 4" with linespoints ls 4
EOF

echo "Performance plot generated: $plot_file"
echo "Benchmarking complete. Results saved to $output_file."
