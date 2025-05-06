import cv2
import os

# Video file path
video_path = 'perfect_leaf_segmentation_100th_try.mp4'

# Output directory to save frames
output_dir = 'extracted_frames'
os.makedirs(output_dir, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Frame counter
frame_count = 0

# Read until video is completed
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # If frame is read correctly, ret is True
    if not ret:
        print("Reached end of video or error reading frame.")
        break
    
    # Save frame (you can modify this to save every nth frame)
    frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(frame_filename, frame)
    print(f"Saved {frame_filename}")
    
    frame_count += 1

# Release the video capture object
cap.release()
print(f"Finished extracting {frame_count} frames to '{output_dir}' directory.")