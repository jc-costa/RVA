import cv2
import numpy as np
import os
from tqdm import tqdm

def select_leaves(image):
    """Your provided leaf segmentation function"""
    if image is None:
        print("Error loading image!")
        return None

    # Convert to RGB and apply blur
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    blured = cv2.GaussianBlur(image_rgb, (19,19), 0)
    
    # HSV thresholding
    hsv = cv2.cvtColor(blured, cv2.COLOR_RGB2HSV)
    lower_green = np.array([36, 0, 0])
    upper_green = np.array([53, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Post-processing
    mask_3c = np.stack((mask,)*3, axis=-1)
    result1 = cv2.bitwise_and(image, mask_3c)
    gray = cv2.cvtColor(result1, cv2.COLOR_BGR2GRAY)
    blured = cv2.medianBlur(gray, 9)
    _, thresh = cv2.threshold(blured, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    structuring_element = np.ones((3,3), np.uint8)
    eroded = cv2.erode(thresh, structuring_element, iterations=4)
    
    return eroded  # Return binary mask instead of masked image

def refine_leaf_mask(mask, img):
    """Additional refinement to remove non-leaf elements"""
    # Remove small objects
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    refined_mask = np.zeros_like(mask)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:  # Minimum leaf area (adjust as needed)
            cv2.drawContours(refined_mask, [cnt], -1, 255, -1)
    
    # Remove table edges using Canny
    edges = cv2.Canny(img, 50, 150)
    non_leaf = cv2.dilate(edges, np.ones((15,15), np.uint8), iterations=2)
    final_mask = cv2.bitwise_and(refined_mask, cv2.bitwise_not(non_leaf))
    
    return final_mask

def create_leaf_video():
    input_folder = "folhas"
    output_video = "perfect_leaf_segmentation_100th_try.mp4"
    
    # Get all plant images
    plant_images = []
    day = 1
    while True:
        img_path = os.path.join(input_folder, f"plant_day{day}.jpg")
        if not os.path.exists(img_path):
            break
        img = cv2.imread(img_path)
        if img is not None:
            plant_images.append(img)
        day += 1
    
    if not plant_images:
        print("No images found!")
        return
    
    # Process images
    frames = []
    for idx, img in enumerate(tqdm(plant_images)):
        # Get initial mask
        mask = select_leaves(img)
        if mask is None:
            continue
            
        # Refine mask
        final_mask = refine_leaf_mask(mask, img)
        
        # Save mask for verification
        cv2.imwrite(f"final_mask_day{idx+1}.png", final_mask)
        
        # Apply color variations
        for color in [(255,100,100), (100,255,100), (100,100,255)]:
            colored = img.copy()
            colored[final_mask > 0] = colored[final_mask > 0] * [color[2]/255, color[1]/255, color[0]/255]
            cv2.putText(colored, f"Day {idx+1}", (30, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)
            frames.append(colored)
    
    # Generate video
    height, width = frames[0].shape[:2]
    video = cv2.VideoWriter(output_video, 
                          cv2.VideoWriter_fourcc(*'mp4v'), 
                          5, 
                          (width, height))
    
    for frame in frames:
        video.write(frame)
    video.release()
    print(f"Video saved to {os.path.abspath(output_video)}")

if __name__ == "__main__":
    create_leaf_video()