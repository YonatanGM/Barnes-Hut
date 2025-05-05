#!/bin/bash
#SBATCH --partition=all
#SBATCH --nodelist=simcl1n[1-4]
#SBATCH --job-name="scenario_runs"
#SBATCH --output=scenario_runs_output.out
#SBATCH --error=scenario_runs_error.err
#SBATCH --time=02:00:00
#SBATCH --nodes=4
#SBATCH --exclusive

set -e
set -x
export OMP_NUM_THREADS=48

echo "Running scenario 1..."
srun --exclusive -N 4 ./build/simulate --file ../data/state_vectors_csvs/scenario1_19054.csv --dt 1h --t_end 12y --vs 2d --vs_dir sim_s1 --theta 1.05
echo "Scenario 1 complete"

echo "Running scenario 2..."
srun --exclusive -N 4 ./build/simulate --file ../data/state_vectors_csvs/scenario2_300149.csv --dt 1h --t_end 1y --vs 7d --vs_dir sim_s2 --theta 1.05 --bodies 300000
echo "Scenario 2 complete"
