import cv2
import numpy as np
import os

# Step 1: Extract Words from Each Line
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

# Step 2: Get all line images from the lines folder
lines_folder = "lines"  # Folder containing the line images
line_paths = [os.path.join(lines_folder, file) for file in os.listdir(lines_folder) if file.endswith('.jpg')]

# Step 3: Run the Process for Extracting Words
words_folder = "words"  # Folder to save word images

# Extract words from the provided lines
extract_words(line_paths, output_folder=words_folder)
