import os
import cv2
import numpy as np
from tqdm import tqdm  # Progress bar


# Define input and output directories
input_dir = "/Users/nadiajelani/Library/CloudStorage/OneDrive-SheffieldHallamUniversity/football/Data and Videos/Ball 3"
output_dir = "/Users/nadiajelani/Desktop/football_project/processed_data/ball 3"

# Check if input folder exists
if os.path.exists(input_dir):
    print(f"✅ 'Ball 3' folder exists at: {input_dir}")
    # List contents of the Ball 3 folder
    files = os.listdir(input_dir)
    if files:
        print(f"Contents of 'Ball 3': {files}")
    else:
        print(f"'Ball 3' folder is empty.")
else:
    print(f"❌ 'Ball 3' folder does NOT exist at: {input_dir}")
    exit(1)  # Exit if the input folder doesn't exist

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# 🔹 Function to Check if Image is Blurry
def is_blurry(image, threshold=100):
    """
    Check if an image is blurry based on the Laplacian variance.
    :param image: Input image (numpy array).
    :param threshold: Variance threshold to classify as blurry.
    :return: True if blurry, False otherwise.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold  # Low variance → blurry image

# Collect all image files from subfolders
image_files = []
for root, _, files in os.walk(input_dir):
    for file in files:
        if file.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg')):  # Include .bmp files
            image_files.append(os.path.join(root, file))

print(f"✅ Found {len(image_files)} images. Starting preprocessing...")

# Process images
for img_path in tqdm(image_files, desc="Processing Images"):
    try:
        # Load the image
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Skipping corrupted image: {img_path}")
            continue

        # Resize to 512x512
        img_resized = cv2.resize(img, (512, 512))

        # Check if the image is blurry
        if is_blurry(img_resized):
            print(f"⚠️ Skipping blurry image: {img_path}")
            continue

        # Convert to grayscale (optional, depending on your use case)
        gray_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        # Normalize pixel values (0-1 range) for further processing
        normalized_img = gray_img / 255.0

        # Save the processed image to the output directory
        img_name = os.path.basename(img_path)  # Extract image name
        save_path = os.path.join(output_dir, img_name)
        cv2.imwrite(save_path, (normalized_img * 255).astype(np.uint8))  # Convert back to 0-255 for saving

    except Exception as e:
        print(f"⚠️ Error processing {img_path}: {e}")

print(f"✅ Processing Completed! Clean images saved in: {output_dir}")
