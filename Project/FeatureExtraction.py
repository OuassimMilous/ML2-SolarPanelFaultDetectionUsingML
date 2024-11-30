import os
import csv
import json
import cv2
import numpy as np
from scipy.stats import skew, kurtosis

# Define paths for the dataset
base_path = "./dataset"  # Base path to the dataset directory
annotations_path = os.path.join(base_path, "annotations")  # Path to the annotations directory
images_path = os.path.join(base_path, "images")  # Path to the images directory

# Output CSV file for storing processed data
output_file = "data.csv"

# Calculate the area of a polygon given its corner points using the shoelace formula.
def calculate_polygon_area(corners):
    n = len(corners)
    area = 0
    for i in range(n):
        x1, y1 = corners[i]['x'], corners[i]['y']
        x2, y2 = corners[(i + 1) % n]['x'], corners[(i + 1) % n]['y']
        area += x1 * y2 - x2 * y1
    return abs(area) / 2

# Prepare the final data structure for CSV rows
csv_rows = []

# Loop through each annotation file in the annotations directory
for annotation_file in os.listdir(annotations_path):
    if annotation_file.endswith(".json"):  # Only process .json files
        # Extract the name without the '.json' extension
        annotation_name = os.path.splitext(annotation_file)[0]

        # Load the annotation file
        annotation_path = os.path.join(annotations_path, annotation_file)
        with open(annotation_path, 'r') as file:
            data = json.load(file)  # Load the JSON data

        instances = data.get('instances', [])  # Extract the list of instances from the annotation data

        # Get corresponding image file based on annotation name
        image_name = annotation_file.replace(".json", ".jpg")
        image_path = os.path.join(images_path, image_name)

        # Check if the image exists
        if not os.path.exists(image_path):
            print(f"Image {image_name} not found. Skipping...")
            continue

        # Load the thermal image in grayscale mode
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Failed to load {image_name}. Skipping...")
            continue

        # Process each panel (instance) in the JSON annotation
        for instance_idx, instance in enumerate(instances):
            # Initialize a dictionary to store panel data for CSV row
            panel_data = {
                "annotation_file": annotation_name,
                "panel_index": instance_idx + 1,
                "mean_temperature": float('nan'),
                "temperature_standard_deviation": float('nan'),
                "max_temperature": float('nan'),
                "temperature_range": float('nan'),
                "temperature_skewness": float('nan'),
                "temperature_kurtosis": float('nan'),
                "faulty": 'No'  # Default status for faulty panels
            }

            try:
                # Handle missing or invalid corner data
                if 'corners' not in instance:
                    print(f"Instance {instance_idx + 1} in {annotation_file} is missing corners. Using default corners.")
                    # Use default or approximate corners if missing
                    x_coords, y_coords = [0, 0, image.shape[1], image.shape[1]], [0, image.shape[0], image.shape[0], 0]
                else:
                    # Extract corner coordinates from the instance
                    x_coords = [corner['x'] for corner in instance['corners']]
                    y_coords = [corner['y'] for corner in instance['corners']]

                # Define region of interest (ROI) based on the instance corners
                polygon_points = np.array([(x, y) for x, y in zip(x_coords, y_coords)], dtype=np.int32)

                # Create a mask for the polygon (panel region) in the image
                mask = np.zeros(image.shape, dtype=np.uint8)
                cv2.fillPoly(mask, [polygon_points], 255)

                # Extract the region corresponding to the panel using the mask
                panel_region = cv2.bitwise_and(image, image, mask=mask)

                # If panel region is empty, fall back to using the entire image
                if np.count_nonzero(panel_region) == 0:
                    print(f"Instance {instance_idx + 1} in {annotation_file} has no valid temperature data in the panel region. Using full image for fallback.")
                    panel_region = image

                # Extract the temperature data from the panel region
                panel_temperature_data = panel_region[panel_region > 0].astype(float)
                panel_data["mean_temperature"] = np.mean(panel_temperature_data)  # Mean temperature
                panel_data["temperature_standard_deviation"] = np.std(panel_temperature_data)  # Standard deviation

                # Calculate maximum temperature in the panel region
                panel_data["max_temperature"] = np.max(panel_temperature_data)

                # Calculate the temperature range (max - min)
                panel_data["temperature_range"] = np.max(panel_temperature_data) - np.min(panel_temperature_data)

                # Calculate temperature skewness (shape of the distribution)
                panel_data["temperature_skewness"] = skew(panel_temperature_data) if len(panel_temperature_data) > 1 else float('nan')

                # Calculate temperature kurtosis (peakedness of the distribution)
                panel_data["temperature_kurtosis"] = kurtosis(panel_temperature_data) if len(panel_temperature_data) > 1 else float('nan')

                # Faulty status based on 'defected_module' flag in the annotation
                panel_data["faulty"] = 'Yes' if instance.get('defected_module', False) else 'No'

            except Exception as e:
                print(f"Error processing instance {instance_idx + 1} in {annotation_file}: {e}")

            # Append the panel data to the CSV rows
            csv_rows.append(panel_data)

# Write the final data to a CSV file
with open(output_file, 'w', newline='') as csvfile:
    fieldnames = [
        "annotation_file", 
        "panel_index", 
        "mean_temperature", 
        "temperature_standard_deviation", 
        "max_temperature", 
        "temperature_range", 
        "temperature_skewness", 
        "temperature_kurtosis", 
        "faulty"
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()  # Write the header row
    writer.writerows(csv_rows)  # Write the data rows

print(f"Results saved to {output_file}")  # Notify the user that the data was saved to the CSV
