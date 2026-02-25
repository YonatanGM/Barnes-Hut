# Parallel N-body Simulation

<!-- [![Pipeline Status](https://gitlab-sim.informatik.uni-stuttgart.de/mamoyn/implementation/badges/main/pipeline.svg)](https://gitlab-sim.informatik.uni-stuttgart.de/mamoyn/implementation/-/pipelines)
[![Coverage](https://gitlab-sim.informatik.uni-stuttgart.de/mamoyn/implementation/badges/main/coverage.svg)](https://gitlab-sim.informatik.uni-stuttgart.de/mamoyn/implementation/-/pipelines) -->

[demo](https://github.com/YonatanGM/SCL-Barnes-Hut/blob/main/bh_demo.gif)

## How to build

To build the project, run:

```bash
mkdir build
cd build
cmake ..
make
```

This builds the main executable `simulate`.

## How to run

The CSV files are located in the `data/state_vectors_csvs` directory. All commands should be run from the `build` folder.

The simulation supports several force calculation strategies via the `--fc` flag:
*   `tree`: Baseline method where all ranks exchange their full local trees.
*   `let`: Default method using Locally Essential Trees (LET) where remote interactions are handled by merging trees.
*   `let_direct`: An alternative LET method where remote forces are calculated with a direct summation.


#### Scenario 1:

```bash
srun -N 4 ./simulate \
    --file ../data/state_vectors_csvs/scenario1_19054.csv \
    --dt 1h --tend 12y --vs 2d --outdir sim_s1 \
    --theta 1.05 --fc let
```

#### Scenario 2:

```bash
srun -N 4 ./simulate \
    --file ../data/state_vectors_csvs/scenario2_300149.csv \
    --dt 1h --tend 1y --vs 7d --outdir sim_s2 \
    --theta 1.05 --fc let
```

To run with fewer bodies, add the `--bodies` option. For a full list of options, use `--help`.

You can also use `sbatch scenario_1.sh` or `sbatch scenario_2.sh` to submit a job to the queue.

## Approach

This simulation uses a parallel Barnes-Hut algorithm with MPI and OpenMP for N-body problems.

The main loop has several phases:

1.  **Load Balancing:** To balance the work, particles are moved between MPI ranks every so often.
    *   It maps particles to 1D space with Morton Z-order curves.
    *   It builds a global histogram of where particles are.
    *   It calculates new boundaries to divide the particle count evenly.
    *   It moves particles to their new ranks with `MPI_Alltoallv`.

2.  **Local Octree Construction:** After balancing, each rank builds an octree with just its own particles. This is done by sorting the particles by their Morton key and building the tree from the bottom up.

3.  **Force Calculation:** The simulation has a few ways to calculate forces.
    *   **Full Tree Exchange (`--fc tree`):** A basic method where every rank sends its full local tree to everyone else. Each rank then builds the same global tree and traverses it. This is simple but sends a lot of data.
    *   **Locally Essential Tree (LET) (`--fc let` or `--fc let_direct`):** This approach sends less data.
        *   Each rank finds the nodes (pseudo-leaves) from its tree that other ranks need to know about based on the Barnes-Hut opening angle (`θ`).
        *   It sends only these "essential" nodes.
        *   The receiving rank can either **merge** these nodes into its local tree (`let`) or calculate their forces with a **direct sum** (`let_direct`).

4.  **Integration:** It uses a Kick-Drift-Kick (Leapfrog) integrator to update particle positions and velocities. OpenMP is used to parallelize the force calculations and updates on each rank.

5.  **Visualization:** The program saves output that can be opened in ParaView. Each rank writes its own particle data to `.vtp` files. The root rank also creates `.pvd` timeline files to animate the particle snapshots, the load-balancing histograms, and the exchanged LET data for debugging.


## Generating the scenario files

The orbital element CSVs are in `data/orbital_elements_raw_csvs`.

*   For scenario 2, large asteroid data (300,000 bodies) comes from the JPL database.
*   Planets, moons, and small asteroid data are provided.

To generate the state vector CSVs for each scenario, you can use the provided command-line conversion tool.

To build the tool:

```bash
g++ -std=c++17 -o orbital_converter src/orbital_converter.cpp src/kepler_to_cartesian.cpp -I./include
```

To run the tool:

```bash
./orbital_converter <planets_and_moons.csv> <asteroids.csv> <output_dir>
```

This will output the combined state vector CSVs required for the simulation.

