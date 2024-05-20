import os
import cv2

# Define input and output directories
input_folder = r"D:\Supermarket-shelf-dataset" # Replace with your input video folder path
output_folder = r"D:\superstore-images" # Replace with your output frame folder path

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get a list of all MP4 files in the input folder
video_files = [f for f in os.listdir(input_folder) if f.endswith('.mp4')]

# Function to extract frames from a video and save as images
def extract_frames(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Save the frame as an image
        frame_filename = f"{frame_count:04d}.jpg"  # You can change the image format if needed
        frame_path = os.path.join(output_path, frame_filename)
        cv2.imwrite(frame_path, frame)

    cap.release()

# Loop through each video file and extract frames
for video_file in video_files:
    video_path = os.path.join(input_folder, video_file)
    output_subfolder = os.path.join(output_folder, os.path.splitext(video_file)[0])

    # Create a subfolder for each video
    if not os.path.exists(output_subfolder):
        os.makedirs(output_subfolder)

    extract_frames(video_path, output_subfolder)

print("Frames extracted successfully.")
