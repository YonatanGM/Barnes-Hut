#!/bin/bash
#SBATCH --partition=all             # Use the "all" partition
#SBATCH --nodelist=simcl1n[1-4]     # node list 
#SBATCH --job-name="nbody_sim"      # Job name
#SBATCH --output=job_output.out     # Standard output
#SBATCH --error=job_error.err       # Standard error
#SBATCH --time=02:00:00             # Maximum run time (2 hours)
#SBATCH --ntasks-per-node=96        # Number of tasks per node (adjust to the number of cores per node)
#SBATCH --nodes=4                   # Number of nodes to use (you can adjust this)
#SBATCH --exclusive                 # Exclusive access to nodes

# Print environment variables set by SLURM (optional)
echo "SLURM_NNODES=$SLURM_NNODES"
echo "Working directory=$SLURM_SUBMIT_DIR"

# Load necessary modules (adjust to your project requirements)

# Navigate to your project directory
cd ./build

# Run the simulation 
# scenario2.
srun -n 384 ./simulate --file ../data/scenario2.csv --dt 1h --t_end 1y --vs 1d --vs_dir sim --theta 1.05