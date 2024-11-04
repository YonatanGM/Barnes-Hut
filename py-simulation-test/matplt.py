import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
data_folder = '../output'  # Folder containing the CSV files
file_prefix = 'state_step_'  # Prefix for the files (e.g., 'state_step_0.csv')
num_timesteps = 1000  # Number of timesteps/files to animate
num_bodies = 100  # Only use the first 10 bodies from each file

# Load data from the specified timesteps, using only the first 10 bodies from each file
def load_data():
    data = []
    for i in range(0, num_timesteps, 100):
        filename = os.path.join(data_folder, f"{file_prefix}{i}.csv")
        df = pd.read_csv(filename)
        positions = df[['pos_x', 'pos_y', 'pos_z']].iloc[:num_bodies].values  # First 10 bodies
        data.append(positions)
    return data

# Load the data
simulation_data = load_data()

# Calculate axis limits based on the positions of the first 10 bodies across the timesteps
all_positions = np.vstack(simulation_data)  # Combine all positions from the timesteps
x_min, x_max = all_positions[:, 0].min(), all_positions[:, 0].max()
y_min, y_max = all_positions[:, 1].min(), all_positions[:, 1].max()
z_min, z_max = all_positions[:, 2].min(), all_positions[:, 2].max()

# Adding padding to make the points more visible
padding = 0.1  # 10% padding
x_range = x_max - x_min
y_range = y_max - y_min
z_range = z_max - z_min

ax_limits = {
    "x": (x_min - padding * x_range, x_max + padding * x_range),
    "y": (y_min - padding * y_range, y_max + padding * y_range),
    "z": (z_min - padding * z_range, z_max + padding * z_range)
}

# Initialize the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set plot limits with padding
ax.set_xlim(ax_limits["x"])
ax.set_ylim(ax_limits["y"])
ax.set_zlim(ax_limits["z"])

# Scatter plot for the celestial bodies
scatter = ax.scatter([], [], [], s=20, c='blue')  # Initial empty scatter

# Update function for animation
def update(frame):
    positions = simulation_data[frame]
    scatter._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
    ax.set_title(f"Timestep {frame}")
    return scatter,

# Animation setup
ani = FuncAnimation(fig, update, frames=num_timesteps, interval=500, blit=False)

# Show the plot
plt.show()