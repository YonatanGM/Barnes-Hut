#!/bin/bash
#SBATCH --partition=all             # use the "all" partition
#SBATCH --nodelist=simcl1n[1-4]     # node list: simcl1n1, simcl1n2, simcl1n3, simcl1n4
#SBATCH --job-name="nbody_sim"      # job name
#SBATCH --output=job_output.out     # standard output
#SBATCH --error=job_error.err       # standard error
#SBATCH --time=02:00:00             # maximum run time (2 hours)
#SBATCH --nodes=4                   # total number of nodes to allocate (adjust as needed)
#SBATCH --exclusive                 # exclusive access to nodes

# Parameters for the simulation
file=${1:-"../data/scenario1.csv"}
dt=${2:-"1h"}
t_end=${3:-"1d"}
vs=${4:-"1d"}
theta=${5:-"1.05"}
vs_dir=${6:-"vs_${file//[^a-zA-Z0-9]/_}_${dt}_${t_end}_${vs}_${theta}"}

benchmark_data_file="benchmark/data_${file//[^a-zA-Z0-9]/_}_${dt}_${t_end}_${vs}_${theta}.dat"
plot_file="benchmark/plot_${file//[^a-zA-Z0-9]/_}_${dt}_${t_end}_${vs}_${theta}.png"

node_counts=(1 2 3 4)        # MPI node counts
thread_counts=(2 8 16 32 48) # OpenMP thread counts

# Move to the build directory and create the benchmark directory
cd ./build
mkdir -p benchmark "$vs_dir"
simulate_binary="./simulate"


> "$benchmark_data_file"


echo "node thread total_time" >> "$benchmark_data_file"


echo "Benchmark data file: $benchmark_data_file"
echo "Visualization plot file: $plot_file"

# Loop through MPI nodes and OpenMP threads, run simulations, and record data
for node in "${node_counts[@]}"; do
  for thread in "${thread_counts[@]}"; do
    # Set the number of OpenMP threads
    export OMP_NUM_THREADS=$thread

    echo "Running with MPI nodes: $node, OpenMP threads: $thread"

    # Run the simulation and capture output
    total_time=$(srun --exclusive -N $node $simulate_binary \
      --file $file --dt $dt --t_end $t_end --vs $vs \
      --vs_dir $vs_dir --theta $theta --log 2>&1 | \
      grep "TOTAL_TIME" | awk '{print $2}')

    # Handle cases where TOTAL_TIME isn't found
    total_time=${total_time:-na}

    # Append the results to the data file
    echo "$node $thread $total_time" >> "$benchmark_data_file"
  done

  # Add two blank lines after each node group for GNUplot index support
  echo -e "\n" >> "$benchmark_data_file"
done

# Check if there are valid data points before plotting
grep -q "^[0-9]" "$benchmark_data_file" || { echo "No valid data points to plot."; exit 1; }

# Generate the plot 
gnuplot <<EOF
set terminal png size 1000,800
set output "$plot_file"
set title "Number of Nodes vs Number of OpenMP Threads vs Time"
set xlabel "Thread Count"
set ylabel "Total Time (s)"
set key outside
set style data linespoints

# Correct label syntax with escaped newlines
set label "Simulation Parameters:\nFILE: $file\nDT: $dt\nT_END: $t_end\nVS: $vs\nTHETA: $theta" at graph 0.02, 0.98 left

# Plot data for each node block using the index with automatically generated labels
plot for [i=0:*] "$benchmark_data_file" index i using 2:3 title sprintf('Node %d', i+1) with lines lc i
EOF




echo "Plot generated: $plot_file"

