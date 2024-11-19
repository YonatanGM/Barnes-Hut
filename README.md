# n-body-simulation-bha

This project simulates the gravitational interactions of bodies in a solar system. To build the project, use CMake. First, create a `build` directory, navigate into it, and run the following commands:

```bash
mkdir build
cd build
cmake ..
make
```

This will generate an executable named `simulate`. Run the simulation with:

```bash
./simulate --file data.csv --dt 1h --t_end 1y --vs 1d --vs_dir sim --theta 1.05
```

- `--file`: Input CSV file containing the initial state vectors.
- `--dt`: Time step (e.g., `1h` for 1 hour).
- `--t_end`: Total simulation duration (e.g., `1y` for 1 year).
- `--vs`: Visualization interval (e.g., `1d` for 1 day).
- `--vs_dir`: Output directory for visualization files.
- `--theta`: Barnes-Hut threshold parameter.

The simulation outputs CSV files containing body positions, velocities, and other properties at each visualization step. These files can be loaded into ParaView for visualization. Writing in .vtp format is planned for future update. 