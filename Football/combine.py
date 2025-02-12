import numpy as np
import matplotlib.pyplot as plt

# 1️⃣ Load Data
# Load velocity and deformation data
real_world_velocities = np.loadtxt("/Users/nadiajelani/Desktop/football/real_world_velocity_data.csv", delimiter=",", skiprows=1)
real_world_dimensions = np.loadtxt("/Users/nadiajelani/Desktop/football/real_world_deformation_data.csv", delimiter=",", skiprows=1)

# Extract width and height from deformation data
widths, heights = real_world_dimensions[:, 0], real_world_dimensions[:, 1]

# Create time arrays
time_velocity = np.arange(len(real_world_velocities)) * TIME_INTERVAL
time_deformation = np.arange(len(widths)) * TIME_INTERVAL

# 2️⃣ Combined Plot
plt.figure(figsize=(12, 6))

# Plot Velocity
plt.subplot(2, 1, 1)
plt.plot(time_velocity, real_world_velocities, label="Velocity (m/s)", color='blue', marker='o')
plt.axvline(x=time_velocity[-1], color='red', linestyle='--', label="Impact Point")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Football Velocity Over Time")
plt.legend()

# Plot Deformation (Width and Height)
plt.subplot(2, 1, 2)
plt.plot(time_deformation, widths, label="Width (m)", color='red', marker='o')
plt.plot(time_deformation, heights, label="Height (m)", color='green', marker='o')
plt.xlabel("Time (s)")
plt.ylabel("Size (m)")
plt.title("Football Deformation Over Time")
plt.legend()

# Save the combined plot
plt.tight_layout()
plt.savefig("/Users/nadiajelani/Desktop/football/combined_visual_representation.png")
plt.show()

print("Combined visual representation saved as 'combined_visual_representation.png'")
