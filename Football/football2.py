import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# System Settings
MASK_FOLDER = "/Users/nadiajelani/Desktop/football_project/masks/"  # Folder containing masks
FRAME_RATE = 30  # Frames per second
TIME_INTERVAL = 1 / FRAME_RATE  # Time between frames in seconds
DEFORMATION_THRESHOLD = 0.1  # Threshold for significant deformation (10% change)

# Results Storage
centroids = []
dimensions = []  # (width, height)

# Helper Functions
def get_centroid(mask):
    """Calculate the centroid of the football from the mask."""
    M = cv2.moments(mask)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)
    return None

def get_deformation(mask):
    """Calculate bounding box dimensions of the football for deformation analysis."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])  # Bounding box
        return (w, h)
    return None

def calculate_velocity(point1, point2, time_interval):
    """Calculate velocity between two points."""
    distance = np.linalg.norm(np.array(point2) - np.array(point1))
    velocity = distance / time_interval
    return velocity

def is_deformed(width, height, original_width, original_height, threshold=DEFORMATION_THRESHOLD):
    """Check if the football has significantly deformed."""
    width_change = abs(width - original_width) / original_width
    height_change = abs(height - original_height) / original_height
    return width_change > threshold or height_change > threshold

# Process Masks
print("Processing mask images...")
for mask_name in sorted(os.listdir(MASK_FOLDER)):  # Ensure correct order
    mask_path = os.path.join(MASK_FOLDER, mask_name)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Extract centroid and deformation
    centroid = get_centroid(mask)
    shape = get_deformation(mask)

    if centroid:
        centroids.append(centroid)
    if shape:
        dimensions.append(shape)

if len(centroids) < 2 or len(dimensions) < 2:
    print("Not enough data to calculate velocity or deformation.")
    exit()

# Calculate Velocities
print("Calculating velocities...")
velocities = []
for i in range(1, len(centroids)):
    velocity = calculate_velocity(centroids[i - 1], centroids[i], TIME_INTERVAL)
    velocities.append(velocity)

# Inbound Velocity (last velocity before impact)
pre_impact_frame = len(centroids) - 1  # Assume last frame is pre-impact
inbound_velocity = calculate_velocity(centroids[pre_impact_frame - 1], centroids[pre_impact_frame], TIME_INTERVAL)
print(f"Inbound Velocity: {inbound_velocity:.2f} pixels/second")

# Detect Significant Deformation
print("Detecting significant deformation...")
original_width, original_height = dimensions[0]
deformation_frames = []
for i, (width, height) in enumerate(dimensions):
    if is_deformed(width, height, original_width, original_height):
        deformation_frames.append(i)

# Calculate Contact Time (if deformation is detected)
if deformation_frames:
    start_frame = deformation_frames[0]
    end_frame = deformation_frames[-1]
    contact_time = (end_frame - start_frame + 1) * TIME_INTERVAL
    print(f"Contact Time: {contact_time:.2f} seconds")
else:
    print("No significant deformation detected.")

# Plot Velocity Over Time
time = np.arange(len(velocities)) * TIME_INTERVAL
plt.figure(figsize=(8, 4))
plt.plot(time, velocities, marker='o', linestyle='-', color='blue', label="Velocity (pixels/s)")
plt.axvline(x=pre_impact_frame * TIME_INTERVAL, color='r', linestyle='--', label='Impact Point')
plt.xlabel("Time (s)")
plt.ylabel("Velocity (pixels/s)")
plt.title("Football Velocity Over Time")
plt.legend()
plt.savefig("velocity_plot.png")
plt.show()

# Plot Deformation Over Time
widths, heights = zip(*dimensions)
time_dims = np.arange(len(dimensions)) * TIME_INTERVAL
plt.figure(figsize=(8, 4))
plt.plot(time_dims, widths, label="Width (pixels)", color='red')
plt.plot(time_dims, heights, label="Height (pixels)", color='green')
plt.xlabel("Time (s)")
plt.ylabel("Size (pixels)")
plt.title("Football Deformation Over Time")
plt.legend()
plt.savefig("deformation_plot.png")
plt.show()

# Save Results
print("Saving results...")
np.savetxt("/Users/nadiajelani/Desktop/football/velocity_data.csv", velocities, delimiter=",", header="Velocity (pixels/s)", comments='')
np.savetxt("/Users/nadiajelani/Desktop/football/deformation_data.csv", dimensions, delimiter=",", header="Width,Height", comments='')

# Summary Report
with open("/Users/nadiajelani/Desktop/football/summary_report.txt", "w") as f:
    f.write(f"Inbound Velocity: {inbound_velocity:.2f} pixels/second\n")
    if deformation_frames:
        f.write(f"Contact Time: {contact_time:.2f} seconds\n")
    else:
        f.write("No significant deformation detected.\n")

print("Results saved: velocity_data.csv, deformation_data.csv, velocity_plot.png, deformation_plot.png, summary_report.txt")
