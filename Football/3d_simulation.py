import pybullet as p
import pybullet_data
import numpy as np

# Load pixel deformation data (width, height) from the CSV file
pixel_dimensions = np.loadtxt("/Users/nadiajelani/Desktop/football/deformation_data.csv", delimiter=",", skiprows=1)

# Define the same pixel-to-meter conversion factor
pixels_to_meters = KNOWN_OBJECT_DIAMETER_M / KNOWN_OBJECT_DIAMETER_PIXELS

# Convert pixel dimensions (width, height) to real-world units (meters)
real_world_dimensions = pixel_dimensions * pixels_to_meters

# Save the converted deformation data to a new file
np.savetxt("/Users/nadiajelani/Desktop/football/real_world_deformation_data.csv", real_world_dimensions, delimiter=",", header="Width (m),Height (m)", comments='')

print("Real-world deformation data saved to real_world_deformation_data.csv.")

# Load real-world deformation data
real_world_dimensions = np.loadtxt("/Users/nadiajelani/Desktop/football/real_world_deformation_data.csv", delimiter=",", skiprows=1)

# Initialize PyBullet physics engine
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Create a simple football model with a soft body for deformation
football_radius = 0.11  # Standard football radius in meters
football_mass = 0.43    # Standard football mass in kilograms
football_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=football_radius)
football_body = p.createMultiBody(baseMass=football_mass, baseCollisionShapeIndex=football_shape)

# Set the environment (gravity)
p.setGravity(0, 0, -9.8)

# Simulate deformation under impact
impact_force = [0, 0, -500]  # Example force in Newtons
p.applyExternalForce(objectUniqueId=football_body, linkIndex=-1, forceObj=impact_force, posObj=[0, 0, 0], flags=p.WORLD_FRAME)

# Run simulation to visualize deformation
print("Simulating deformation...")
for i in range(240):  # Simulate for 8 seconds at 30 FPS
    p.stepSimulation()

# Get deformation results
football_position, _ = p.getBasePositionAndOrientation(football_body)
print(f"Final Football Position After Deformation: {football_position}")
p.disconnect()

# Save the deformation result
np.savetxt("/Users/nadiajelani/Desktop/football/3d_deformation_results.csv", [football_position], delimiter=",", header="X,Y,Z", comments='')
print("3D deformation results saved to 3d_deformation_results.csv.")

import matplotlib.pyplot as plt

# Load real-world deformation data
real_world_dimensions = np.loadtxt("/Users/nadiajelani/Desktop/football/real_world_deformation_data.csv", delimiter=",", skiprows=1)

# Extract width and height in meters
widths, heights = real_world_dimensions[:, 0], real_world_dimensions[:, 1]

# Create time array
time_dims = np.arange(len(widths)) * TIME_INTERVAL

# Plot deformation over time
plt.figure(figsize=(8, 4))
plt.plot(time_dims, widths, label="Width (m)", color='red')
plt.plot(time_dims, heights, label="Height (m)", color='green')
plt.xlabel("Time (s)")
plt.ylabel("Size (m)")
plt.title("Football Deformation Over Time (Real-World Units)")
plt.legend()
plt.savefig("/Users/nadiajelani/Desktop/football/real_world_deformation_plot.png")
plt.show()

print("Real-world deformation plot saved as real_world_deformation_plot.png.")
