#!/bin/bash
#SBATCH --partition=all
#SBATCH --nodelist=simcl1n[1-4]
#SBATCH --job-name="scenario2_retry"
#SBATCH --output=scenario2_output.out
#SBATCH --error=scenario2_error.err
#SBATCH --time=06:30:00
#SBATCH --nodes=4
#SBATCH --exclusive

set -e
set -x
export OMP_NUM_THREADS=48

echo "Running scenario 2 only... at $(date)"
srun --exclusive -N 4 ./build/simulate --file ./data/state_vectors_csvs/scenario2_300149.csv --dt 1h --t_end 1y --vs 7d --vs_dir sim_s2 --theta 1.05 --bodies 300000
echo "Scenario 2 complete at $(date)"
