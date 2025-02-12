import os
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt

# Function to organize data into a dictionary for processing
def organize_data(base_folder):
    data_structure = {}
    
    for ball_folder in os.listdir(base_folder):
        ball_path = os.path.join(base_folder, ball_folder)
        if os.path.isdir(ball_path):
            data_structure[ball_folder] = {}
            for drop_folder in os.listdir(ball_path):
                drop_path = os.path.join(ball_path, drop_folder)
                if os.path.isdir(drop_path):
                    image_files = sorted(
                        [os.path.join(drop_path, img) for img in os.listdir(drop_path) if img.endswith('.bmp')]
                    )
                    data_structure[ball_folder][drop_folder] = image_files
    return data_structure

# Function to process images in batches and extract relevant features
def process_images_in_batches(image_paths, batch_size, output_file):
    results = []

    if len(image_paths) == 0:
        print("No images found to process.")
        return results

    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        
        for image_path in batch:
            frame = cv2.imread(image_path)
            
            if frame is None:
                print(f"Error reading image: {image_path}")
                continue

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply GaussianBlur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Apply morphological operations to remove small noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            morph = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)

             # Adaptive Thresholding
            thresh = cv2.adaptiveThreshold(
            morph, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )

            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Find the largest contour (assumed to be the football)
            if len(contours) == 0:
                print(f"No contours found in image: {image_path}")
                continue

            largest_contour = max(contours, key=cv2.contourArea)
            
            # Fit a bounding box around the football
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Store results
            results.append({
                'image_path': image_path,
                'bounding_box': (x, y, w, h)
            })

        # Save intermediate results to file
        with open(output_file, 'w') as f:
            json.dump(results, f)

        print(f"Processed batch {i // batch_size + 1} of {len(image_paths) // batch_size + 1}")

        print(f"Processing complete. Results saved to {output_file}")
        cv2.imshow("Thresholded Image", thresh)
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
        cv2.imshow("Contours", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return results

# Function to process all balls and drops with batch processing
def process_all_in_batches(data_structure, batch_size, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for ball_key, drops in data_structure.items():
        print(f"Processing {ball_key}...")
        for drop_key, image_paths in drops.items():
            print(f"  Processing {drop_key}...")
            if len(image_paths) == 0:
                print(f"    No images found in {drop_key}. Skipping.")
                continue
            output_file = os.path.join(output_folder, f"{ball_key}_{drop_key}_results.json")
            process_images_in_batches(image_paths, batch_size, output_file)
        print(f"Finished processing {ball_key}.")

# Function to visualize bounding boxes from JSON results
def visualize_results(json_file):
    if not os.path.exists(json_file):
        print(f"JSON file {json_file} not found.")
        return

    with open(json_file, 'r') as f:
        results = json.load(f)

    if len(results) == 0:
        print("No results to visualize.")
        return

    for result in results:
        image_path = result['image_path']
        bounding_box = result['bounding_box']

        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error reading image: {image_path}")
            continue

        # Draw bounding box
        x, y, w, h = bounding_box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Convert BGR to RGB for matplotlib
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display image
        plt.imshow(frame_rgb)
        plt.title(f"Bounding Box: {bounding_box}")
        plt.axis('off')
        plt.show()

# Example Usage
if __name__ == "__main__":
    base_folder = '/Users/nadiajelani/Library/CloudStorage/OneDrive-SheffieldHallamUniversity/football/Data and Videos'
    output_folder = "Processed Results"  # Folder to save results
    batch_size = 100  # Number of images per batch

    print("Organizing data for Ball 1, Drop 1...")
    
    # Organize data for Ball 1, Drop 1 only
    data_structure = {
        "Ball 1": {
            "Drop 1": sorted(
                [
                    os.path.join(base_folder, "Ball 1", "Drop 1", img)
                    for img in os.listdir(os.path.join(base_folder, "Ball 1", "Drop 1"))
                    if img.endswith('.bmp')
                ]
            )
        }
    }
    print("Data organization complete.")

    print("Starting batch processing for Ball 1, Drop 1...")
    process_all_in_batches(data_structure, batch_size, output_folder)
    print("Batch processing complete.")

    # Visualize results for Ball 1, Drop 1
    sample_json_file = os.path.join(output_folder, "Ball_1_Drop_1_results.json")
    print("Visualizing results for Ball 1, Drop 1...")
    visualize_results(sample_json_file)
    print("Visualization complete.")
