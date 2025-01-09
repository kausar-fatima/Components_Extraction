import cv2
import numpy as np
import os

# Load the image
image_path = "C:/Users/PMLS/Desktop/CVIP_A#2_2021SE25/image.jpg"
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold to binary (black and white)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# Sum pixel values vertically (histogram)
vertical_histogram = np.sum(binary, axis=1)

# Define a threshold to find colored rows
threshold = 5  # Adjust this value based on the image
regions = []  # To store (start_row, end_row) of each line
start_row = None

# Detect all start and end rows of colored regions
for r, value in enumerate(vertical_histogram):
    if value > threshold and start_row is None:
        start_row = r  # Start of a colored region
    if value <= threshold and start_row is not None:
        end_row = r  # End of the colored region
        regions.append((start_row, end_row))
        start_row = None

# If the image ends with a colored region, handle it
if start_row is not None:
    regions.append((start_row, len(vertical_histogram)))

# Create 'lines' folder if it doesn't exist
lines_folder = "lines"
if not os.path.exists(lines_folder):
    os.makedirs(lines_folder)

# Extract and save each line as a separate image
if len(regions) == 0:
    print("No colored regions found.")
else:
    for idx, (start_row, end_row) in enumerate(regions):
        # Initialize the output array to store the extracted line
        height, width, _ = image.shape
        line = np.zeros((end_row - start_row, width, 3), dtype=np.uint8)

        # Apply the given logic to copy rows
        for r in range(start_row, end_row):
            for c in range(width):
                line[r - start_row][c] = image[r][c]

        # Save the extracted line to the 'lines' folder
        output_path = os.path.join(lines_folder, f"line_{idx + 1}.jpg")
        cv2.imwrite(output_path, line)
        print(f"Saved: {output_path}")
