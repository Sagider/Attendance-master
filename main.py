import face_recognition
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np
import pickle
import csv # Import the csv module
from datetime import datetime # Import datetime for timestamp

def load_known_faces(known_faces_dir):
    """
    Loads known face encodings and names from the specified directory.
    If no face is found in an image, it attempts to rotate the image
    clockwise by 90 degrees, then counter-clockwise by 90 degrees (from original),
    to try and find a face.

    Args:
        known_faces_dir (str): The path to the directory containing known faces.
                               Expected structure: known_faces/{person_name}/image.jpg

    Returns:
        tuple: A tuple containing two lists:
               - known_face_encodings (list): List of face encodings for known people.
               - known_face_names (list): List of names corresponding to the encodings.
    """
    known_face_encodings = []
    known_face_names = []

    if not os.path.exists(known_faces_dir):
        print(f"Error: Directory '{known_faces_dir}' not found. Please create it and add known faces.")
        return known_face_encodings, known_face_names

    print(f"Loading known faces from: {known_faces_dir}")
    for person_name in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, person_name)
        if os.path.isdir(person_dir):
            for image_file in os.listdir(person_dir):
                if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(person_dir, image_file)
                    print(f"  Processing: {image_path} (for {person_name})")
                    try:
                        # Load the original image
                        original_image = face_recognition.load_image_file(image_path)
                        face_encodings = face_recognition.face_encodings(original_image)
                        current_image_source = "original"

                        # If no face found in original, try rotating clockwise
                        if not face_encodings:
                            print(f"    - No face found in original. Trying clockwise 90-degree rotation...")
                            pil_original_image = Image.fromarray(original_image)
                            rotated_clockwise_image_pil = pil_original_image.rotate(-90, expand=True)
                            rotated_clockwise_image_np = np.array(rotated_clockwise_image_pil)
                            face_encodings = face_recognition.face_encodings(rotated_clockwise_image_np)
                            current_image_source = "clockwise rotated"

                        # If still no face, try rotating counter-clockwise (from original)
                        if not face_encodings:
                            print(f"    - No face found after clockwise. Trying counter-clockwise 90-degree rotation from original...")
                            pil_original_image = Image.fromarray(original_image)
                            rotated_counter_clockwise_image_pil = pil_original_image.rotate(90, expand=True)
                            rotated_counter_clockwise_image_np = np.array(rotated_counter_clockwise_image_pil)
                            face_encodings = face_recognition.face_encodings(rotated_counter_clockwise_image_np)
                            current_image_source = "counter-clockwise rotated"

                        if face_encodings:
                            # Use the first face found in the image (from any successful orientation)
                            known_face_encodings.append(face_encodings[0])
                            known_face_names.append(person_name)
                            print(f"    - Face encoding loaded for {person_name} from {current_image_source} image.")
                        else:
                            print(f"    - No face found in {image_file} for {person_name} after all rotations. Skipping.")
                    except Exception as e:
                        print(f"    - Could not process {image_file} for {person_name}: {e}")
    print(f"Loaded {len(known_face_encodings)} known faces.")
    return known_face_encodings, known_face_names

def recognize_faces_in_image(input_image_path, known_face_encodings, known_face_names):
    """
    Identifies faces in an input image and draws bounding boxes with names.
    Applies orientation checks (rotations) to the input image if no faces are initially found.

    Args:
        input_image_path (str): The path to the input image to process.
        known_face_encodings (list): List of face encodings for known people.
        known_face_names (list): List of names corresponding to the encodings.

    Returns:
        list: A list of names of people identified in the picture.
    """
    if not os.path.exists(input_image_path):
        print(f"Error: Input image '{input_image_path}' not found.")
        return []

    print(f"\nProcessing input image: {input_image_path}")
    try:
        # Load the original image
        original_image = face_recognition.load_image_file(input_image_path)
    except Exception as e:
        print(f"Error loading input image '{input_image_path}': {e}")
        return []

    # Initialize current image for processing
    image_to_process = original_image
    face_locations = []
    face_encodings = []
    current_image_orientation = "original"

    # Try original orientation
    face_locations = face_recognition.face_locations(image_to_process)
    face_encodings = face_recognition.face_encodings(image_to_process, face_locations)

    # If no face found, try clockwise rotation
    if not face_locations:
        print(f"    - No face found in original input image. Trying clockwise 90-degree rotation...")
        pil_original_image = Image.fromarray(original_image)
        rotated_clockwise_image_pil = pil_original_image.rotate(-90, expand=True)
        image_to_process = np.array(rotated_clockwise_image_pil)
        face_locations = face_recognition.face_locations(image_to_process)
        face_encodings = face_recognition.face_encodings(image_to_process, face_locations)
        if face_locations:
            current_image_orientation = "clockwise rotated"

    # If still no face, try counter-clockwise rotation (from original)
    if not face_locations:
        print(f"    - No face found after clockwise rotation. Trying counter-clockwise 90-degree rotation from original...")
        pil_original_image = Image.fromarray(original_image) # Re-load original to ensure correct base for rotation
        rotated_counter_clockwise_image_pil = pil_original_image.rotate(90, expand=True)
        image_to_process = np.array(rotated_counter_clockwise_image_pil)
        face_locations = face_recognition.face_locations(image_to_process)
        face_encodings = face_recognition.face_encodings(image_to_process, face_locations)
        if face_locations:
            current_image_orientation = "counter-clockwise rotated"

    if not face_locations:
        print("    - No recognizable faces found in the input image after all rotations.")
        return []

    print(f"Found {len(face_locations)} face(s) in the input image ({current_image_orientation} orientation).")

    pil_image = Image.fromarray(image_to_process) # Create PIL image from the successfully oriented image
    draw = ImageDraw.Draw(pil_image)

    # Define a font for drawing names, dynamically sized based on average face height
    if face_locations:
        # Calculate an average face height to base font size on
        # (bottom - top) gives height
        avg_face_height = np.mean([bottom - top for (top, right, bottom, left) in face_locations])
        # Increased font size scaling factor to 0.25 and minimum size to 20
        calculated_font_size = max(20, int(avg_face_height * 0.25))
        try:
            font = ImageFont.truetype("arial.ttf", calculated_font_size)
        except IOError:
            font = ImageFont.load_default() # Fallback to default font if arial.ttf is not found
    else:
        # Fallback font size if no faces are detected in the image
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except IOError:
            font = ImageFont.load_default()

    identified_people = []

    # Loop through each face found in the input image
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare the current face with all known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # If a match is found, use the known face with the smallest distance
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        identified_people.append(name)

        # Draw a box around the face
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0), width=3)

        # Draw a label with the name below the face
        # Use font.getbbox for modern Pillow versions to get text dimensions
        try:
            text_bbox = draw.textbbox((0, 0), name, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        except AttributeError: # Fallback for older Pillow versions
            text_width, text_height = draw.textsize(name, font=font)

        # Adjust the rectangle position and size based on new font size
        padding = 10 # Padding below the box and around text
        # Set color based on whether the face is known or unknown
        box_color = (0, 255, 0) if name != "Unknown" else (255, 0, 0) # Green for known, Red for unknown
        draw.rectangle(((left, bottom - text_height - padding), (right, bottom)), fill=box_color, outline=box_color)
        draw.text((left + 6, bottom - text_height - padding + 2), name, fill=(255, 255, 255), font=font) # Pass font here

    # Display the result image
    pil_image.show()

    # Clean up the drawing
    del draw

    return identified_people

if __name__ == "__main__":
    # --- Configuration ---
    # IMPORTANT: Create this directory and place subdirectories for each person.
    # Each person's subdirectory should contain one or more images of their face.
    KNOWN_FACES_DIRECTORY = "known_faces"
    # Define where to save/load the pre-computed model
    MODEL_SAVE_PATH = "face_recognition_model.pkl"
    # -------------------

    known_face_encodings = []
    known_face_names = []

    # Attempt to load the pre-trained model
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Attempting to load known faces from saved model: {MODEL_SAVE_PATH}")
        try:
            with open(MODEL_SAVE_PATH, 'rb') as f:
                known_face_encodings, known_face_names = pickle.load(f)
            print(f"Successfully loaded {len(known_face_encodings)} known faces from saved model.")
        except Exception as e:
            print(f"Error loading saved model ({e}). Re-training from '{KNOWN_FACES_DIRECTORY}' directory.")
            # If loading fails, proceed to train from directory and then save
            known_face_encodings, known_face_names = load_known_faces(KNOWN_FACES_DIRECTORY)
            if known_face_encodings: # Only save if faces were successfully loaded
                try:
                    with open(MODEL_SAVE_PATH, 'wb') as f:
                        pickle.dump((known_face_encodings, known_face_names), f)
                    print(f"Saved newly trained model to {MODEL_SAVE_PATH}")
                except Exception as e:
                    print(f"Error saving model to '{MODEL_SAVE_PATH}': {e}")
    else:
        # If no saved model found, load from directory and save for future runs
        print(f"No saved model found. Training from '{KNOWN_FACES_DIRECTORY}' directory...")
        known_face_encodings, known_face_names = load_known_faces(KNOWN_FACES_DIRECTORY)
        if known_face_encodings: # Only save if faces were successfully loaded
            try:
                with open(MODEL_SAVE_PATH, 'wb') as f:
                    pickle.dump((known_face_encodings, known_face_names), f)
                print(f"Saved trained model to {MODEL_SAVE_PATH}")
            except Exception as e:
                print(f"Error saving model to '{MODEL_SAVE_PATH}': {e}")


    if not known_face_encodings:
        print("\nNo known faces loaded. Please ensure 'known_faces' directory is set up correctly.")
    else:
        # Loop to ask for multiple image paths
        while True:
            # 2. Ask user for input image path
            INPUT_IMAGE_PATH = input("\nPlease enter the path to the image you want to analyze (or type 'quit' to exit): ")

            if INPUT_IMAGE_PATH.lower() == 'quit':
                print("Exiting face recognition application. Goodbye!")
                break

            # 3. Recognize faces in the input image
            people_in_picture = recognize_faces_in_image(INPUT_IMAGE_PATH, known_face_encodings, known_face_names)

            # 4. Display the list of people
            print("\n--- People Present in the Picture ---")
            if people_in_picture:
                # Use a set to get unique names, then convert back to list for display
                unique_people = sorted(list(set(people_in_picture)))
                for person in unique_people:
                    print(f"- {person}")

                # Save identified people to a CSV file
                if unique_people: # Only create CSV if there are people identified
                    try:
                        # Generate timestamp for filename
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        csv_filename = f"identified_people_{timestamp}.csv"
                        with open(csv_filename, mode='w', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(["Person Name"]) # CSV header
                            for person_name in unique_people:
                                writer.writerow([person_name])
                        print(f"Successfully saved identified people to {csv_filename}")
                    except Exception as e:
                        print(f"Error saving identified people to CSV: {e}")
                else:
                    print("No identifiable people to save to CSV for this image.")
            else:
                print("No recognizable faces found, or all faces are 'Unknown'.")

