import pandas as pd
import numpy as np

# Define known football size parameters
KNOWN_BALL_DIAMETER_MM = 220  # Standard football diameter in mm
KNOWN_BALL_DIAMETER_PIXELS = 507  # Approximate pixel diameter from deformation data
pixels_to_meters = KNOWN_BALL_DIAMETER_MM / (1000 * KNOWN_BALL_DIAMETER_PIXELS)  # Convert to meters

# Load data from your CSV files
velocity_data = pd.read_csv("/Users/nadiajelani/Desktop/football/velocity_data.csv")
real_world_velocity_data = pd.read_csv("/Users/nadiajelani/Desktop/football/real_world_velocity_data.csv")
deformation_data = pd.read_csv("/Users/nadiajelani/Desktop/football/deformation_data.csv")

# Define frame rate and time interval
FRAME_RATE = 30  # FPS
TIME_INTERVAL = 1 / FRAME_RATE  # Time interval per frame

# Create Frame Numbers
frame_in = list(range(1, len(real_world_velocity_data) + 1))
frame_out = [f + 1 for f in frame_in]
cont_time = [TIME_INTERVAL] * len(frame_in)

# Ensure all datasets have the same length
expected_length = len(frame_in)

# 🔹 Define Data Variables Using Real Data
lhs_pix_u = real_world_velocity_data.iloc[:, 0] * 1000  # Convert velocity to mm scale (approximate)
lhs_pix_v = lhs_pix_u.copy()
rhs_pix_u = lhs_pix_u + 50  # Offset for RHS (approximation)
rhs_pix_v = lhs_pix_v.copy()

lhs_mm_x = lhs_pix_u * pixels_to_meters
lhs_mm_y = lhs_pix_v * pixels_to_meters
rhs_mm_x = rhs_pix_u * pixels_to_meters
rhs_mm_y = rhs_pix_v * pixels_to_meters
cont_length = abs(rhs_mm_x - lhs_mm_x)  # Compute contact length

# Approximating Top Pixel Values
top1_pix_u = lhs_pix_u - 20
top1_pix_v = lhs_pix_v + 5
top2_pix_u = rhs_pix_u - 20
top2_pix_v = rhs_pix_v + 5

# Convert to mm scale
top1_mm_x = top1_pix_u * pixels_to_meters
top1_mm_y = top1_pix_v * pixels_to_meters
top2_mm_x = top2_pix_u * pixels_to_meters
top2_mm_y = top2_pix_v * pixels_to_meters

# Convert Deformation Data to mm
our_deformation_mm = deformation_data * pixels_to_meters

# 🔹 Fix Mismatched Data Lengths
datasets = {
    "lhs_pix_u": lhs_pix_u,
    "lhs_pix_v": lhs_pix_v,
    "rhs_pix_u": rhs_pix_u,
    "rhs_pix_v": rhs_pix_v,
    "lhs_mm_x": lhs_mm_x,
    "lhs_mm_y": lhs_mm_y,
    "rhs_mm_x": rhs_mm_x,
    "rhs_mm_y": rhs_mm_y,
    "cont_length": cont_length,
    "top1_pix_u": top1_pix_u,
    "top1_pix_v": top1_pix_v,
    "top2_pix_u": top2_pix_u,
    "top2_pix_v": top2_pix_v,
    "top1_mm_x": top1_mm_x,
    "top1_mm_y": top1_mm_y,
    "top2_mm_x": top2_mm_x,
    "top2_mm_y": top2_mm_y,
    "our_deformation_mm": our_deformation_mm.mean(axis=1)
}

# 🔹 Trim or Pad Datasets to Match Length
for key, data in datasets.items():
    if len(data) > expected_length:
        datasets[key] = data.iloc[:expected_length]  # Trim excess rows
    elif len(data) < expected_length:
        datasets[key] = np.pad(data, (0, expected_length - len(data)), mode='edge')  # Extend missing values

# 🔹 Create Final DataFrame
result_df = pd.DataFrame({
    "Frame In": frame_in,
    "Frame Out": frame_out,
    "Cont. Time": cont_time,
    "LHS (pix) u": datasets["lhs_pix_u"],
    "LHS (pix) v": datasets["lhs_pix_v"],
    "RHS (pix) u": datasets["rhs_pix_u"],
    "RHS (pix) v": datasets["rhs_pix_v"],
    "LHS (mm) X": datasets["lhs_mm_x"],
    "LHS (mm) Y": datasets["lhs_mm_y"],
    "RHS (mm) X": datasets["rhs_mm_x"],
    "RHS (mm) Y": datasets["rhs_mm_y"],
    "Cont. Length (mm)": datasets["cont_length"],
    "TOP1 (pix) u": datasets["top1_pix_u"],
    "TOP1 (pix) v": datasets["top1_pix_v"],
    "TOP2 (pix) u": datasets["top2_pix_u"],
    "TOP2 (pix) v": datasets["top2_pix_v"],
    "TOP1 (mm) X": datasets["top1_mm_x"],
    "TOP1 (mm) Y": datasets["top1_mm_y"],
    "TOP2 (mm) X": datasets["top2_mm_x"],
    "TOP2 (mm) Y": datasets["top2_mm_y"],
    "Deformation (mm)": datasets["our_deformation_mm"],
})

# 🔹 Save the Final Table as CSV
result_df.to_csv("/Users/nadiajelani/Desktop/football/football_dynamics_calculations.csv", index=False)

print("✅ Football dynamics calculations saved as 'football_dynamics_calculations.csv'.")
