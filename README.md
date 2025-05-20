# Parallel N-body Simulation

[![Pipeline Status](https://gitlab-sim.informatik.uni-stuttgart.de/mamoyn/implementation/badges/phase-1/pipeline.svg)](https://gitlab-sim.informatik.uni-stuttgart.de/mamoyn/implementation/-/pipelines)
[![Coverage](https://gitlab-sim.informatik.uni-stuttgart.de/mamoyn/implementation/badges/phase-1/coverage.svg)](https://gitlab-sim.informatik.uni-stuttgart.de/mamoyn/implementation/-/pipelines)

## How to build

To build the project, run:

```bash
mkdir build
cd build
cmake ..
make
```

This builds the executable and tests which you can run using `ctest`. To check test coverage, build using `cmake -DENABLE_COVERAGE=ON ..`, then run
`make coverage`.

## How to run

The CSV files are in data/state_vectors_csvs. Run these commands from the build folder.

Scenario 1:

```bash
srun -N 4 ./simulate --file ../data/state_vectors_csvs/scenario1_19054.csv --dt 1h --t_end 12y --vs 2d --vs_dir sim_s1 --theta 1.05
```

Scenario 2:

```bash
srun -N 4 ./simulate --file ../data/state_vectors_csvs/scenario2_300149.csv --dt 1h --t_end 1y --vs 7d --vs_dir sim_s2 --theta 1.05
```

To run with fewer bodies, add the --bodies option. For more details, use --help.

You can also use `sbatch scenario_1.sh` or `sbatch scenario_2.sh` to run the job. Timing results are based on these scripts.

## Generating the scenario files

The orbital element CSVs are in data/orbital_elements_raw_csvs.

* For scenario 2, large asteroid data (300,000 bodies) comes from the JPL database.
* Planets and moons, and small asteroid data are from Ilias.

To generate the state vector CSVs for each scenario, first convert the planet/moon and asteroid orbital element CSVs to state vectors, then combine them.

There’s a command-line tool for this:

To build:

```bash
g++ -std=c++17 -o orbital_converter src/orbital_converter.cpp src/kepler_to_cartesian.cpp -I./include
```

To run:

```bash
./orbital_converter <planets_and_moons.csv> <asteroids.csv> <output_dir>
```

This will output combined state vector CSVs in the specified folder.

## Result

Scenario 1: 19,054 bodies in 1213.48 seconds (\~20 minutes)

Scenario 2: 300,000 bodies in 1665.55 seconds (\~28 minutes)

See the `result` folder for:

* Scenario 1 animation
* Final timestep CSVs for both scenarios
* ParaView screenshots and energy plot

See [performance\_analysis.md](performance_analysis.md) for benchmark results.

## Approach

* The root process broadcasts the total number of bodies, their masses, initial positions, and velocities to all MPI ranks using `MPI_Bcast`, so all ranks start with the same global data.
* Masses, positions, velocities, names, and orbit classes are distributed to MPI ranks using `MPI_Scatterv`, giving each rank a subset of bodies to handle.
* Each rank calculates the initial accelerations for its assigned bodies using their local data and global data (like positions and masses of all bodies) before entering the simulation loop.

Inside the simulation loop:

* Each rank performs a full-step position update and a half-step velocity update for its assigned bodies.
* Updated positions are gathered globally across all ranks using `MPI_Allgatherv`, ensuring that every rank has the full updated state of all bodies needed for the next computation.
* A second half-step velocity update is performed to finalize the integration step.
* Accelerations are calculated for each body based on the updated global positions.
* Global kinetic and potential energies are calculated by summing contributions from all ranks using `MPI_Allreduce`.
* Each rank writes VTP files for its assigned bodies, while the root rank manages updates to the PVD file that aggregates all outputs.

The acceleration calculations, position, and velocity updates are parallelized with OpenMP.
