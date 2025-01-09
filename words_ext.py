import cv2
import numpy as np
import os

# Step 1: Extract Lines from the Original Image
def extract_lines(image_path, output_folder="lines"):
    # Create output folder for lines
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the image
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

    # Extract and save each line as a separate image
    line_paths = []
    if len(regions) == 0:
        print("No colored regions found.")
    else:
        for idx, (start_row, end_row) in enumerate(regions):
            # Initialize the output array to store the extracted line
            height, width, _ = image.shape
            line = np.zeros((end_row - start_row, width, 3), dtype=np.uint8)

            # Copy rows of the line
            for r in range(start_row, end_row):
                for c in range(width):
                    line[r - start_row][c] = image[r][c]

            # Save the extracted line
            output_path = os.path.join(output_folder, f"line_{idx + 1}.jpg")
            cv2.imwrite(output_path, line)
            line_paths.append(output_path)
            print(f"Saved line: {output_path}")
    return line_paths

# Step 2: Extract Words from Each Line
def extract_words(line_paths, output_folder="words"):
    # Create output folder for words
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each line image
    for line_idx, line_path in enumerate(line_paths):
        # Load the line image
        line = cv2.imread(line_path)

        # Convert to grayscale
        gray_line = cv2.cvtColor(line, cv2.COLOR_BGR2GRAY)

        # Threshold to binary (black and white)
        _, binary_line = cv2.threshold(gray_line, 127, 255, cv2.THRESH_BINARY_INV)

        # Sum pixel values horizontally (histogram)
        horizontal_histogram = np.sum(binary_line, axis=0)

        # Define a threshold to find spaces between words
        threshold = 5  # Adjust this value based on the image
        min_space_width = 15  # Minimum number of consecutive non-colored columns to detect spaces
        words = []  # To store (start_col, end_col) of each word
        start_col = None
        space_counter = 0  # Counter to track spaces

        # Detect all start and end columns of words
        for c, value in enumerate(horizontal_histogram):
            if value > threshold:  # Found part of a word
                if start_col is None:
                    start_col = c  # Start of a word
                space_counter = 0  # Reset space counter
            elif value <= threshold:  # Found space
                if start_col is not None:
                    space_counter += 1
                    if space_counter >= min_space_width:  # Space is wide enough to separate words
                        end_col = c - space_counter  # End of the word
                        words.append((start_col, end_col))
                        start_col = None
                        space_counter = 0

        # If the line ends with a word, handle it
        if start_col is not None:
            words.append((start_col, len(horizontal_histogram)))

        # Extract and save each word as a separate image
        if len(words) == 0:
            print(f"No words found in {line_path}.")
        else:
            for word_idx, (start_col, end_col) in enumerate(words):
                # Extract columns of the word
                word = line[:, start_col:end_col]

                # Save the extracted word
                output_path = os.path.join(output_folder, f"word_{line_idx + 1}_{word_idx + 1}.jpg")
                cv2.imwrite(output_path, word)
                print(f"Saved word: {output_path}")

# Step 3: Run the Complete Process
image_path = "C:/Users/PMLS/Desktop/CVIP_A#2_2021SE25/image.jpg"  # Input image
lines_folder = "lines"  # Folder to save line images
words_folder = "words"  # Folder to save word images

# Extract lines and then words
line_paths = extract_lines(image_path, output_folder=lines_folder)
extract_words(line_paths, output_folder=words_folder)
