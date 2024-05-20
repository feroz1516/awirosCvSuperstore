from ultralytics import YOLO
import torch
import cv2
import numpy as np
import os
import shutil

# Load a pretrained YOLOv8n model
class_names = {
    2: "Bounce",
    35: "Supermilk",
    18: "Malkist",
    9:"Dark Fantasy",
    20: "Marie Light",
    38: "Unibic"
    

    # Add more class names as needed
}

modelyolo = YOLO('E:\\Projects\\CV App-a-thon\\biscuitmodel\\shelfnew.pt')

# Open a video capture object
cap = cv2.VideoCapture('E:\\Projects\\CV App-a-thon\\biscuitpredict\\basketvid1.mp4')

frame_number = 0  # Initialize frame number

output_frames_dir = 'output_frames'  # Specify the output frame directory
os.makedirs(output_frames_dir, exist_ok=True)  # Create the directory if it doesn't exist
out_of_stock_count = {class_index: 0 for class_index in class_names.keys()}  # Initialize count for each class

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    results = modelyolo.predict(source=frame, save=True)  # Process each frame

    # Copy the predicted frames from the 'runs/detect/predict' directory to the output directory
    for root, dirs, files in os.walk('runs/detect/predict'):
        for file in files:
            if file.endswith('.jpg'):
                src_file = os.path.join(root, file)
                dst_file = os.path.join(output_frames_dir, f"frame_{frame_number}_{file}")
                shutil.copy(src_file, dst_file)

    boxes_cls = []  # Initialize a list for class predictions

    for result in results:
        boxes = result.boxes.cls
    # Convert the PyTorch tensor to a Python list
        boxes_list = boxes.tolist()
    # Extend the 'boxes_cls' list with the elements from 'boxes_list'
        boxes_cls.extend(boxes_list)

    boxes_cls_set = set(boxes_cls)  # Create a set of detected class indices

    out_stock = []  # Initialize a list for out-of-stock items
    

    for class_index, class_name in class_names.items():
        if class_index not in boxes_cls_set:
            out_stock.append((class_index,class_name))
    
    for class_index in class_names.keys():
        if class_index in boxes_cls_set:
            out_of_stock_count[class_index] = 0
        else:
            out_of_stock_count[class_index] += 1

    if any(count > 0 for count in out_of_stock_count.values()):
        
        for class_index, count in out_of_stock_count.items():
            class_name = class_names[class_index]
            if count > 0:
                print(f"{class_name} no detection. Count: {count}")
                if count > 26:
                    text=print(f"{class_name} is out of stock")
                else:
                    text=print("in stock")
    
                    # Load the image
                    input_image_path = 'E:\\Projects\\CV App-a-thon\\biscuitpredict\\Screenshot (69).png'
                    output_folder = 'E:\\Projects\\CV App-a-thon\\stock_output'

                    # Make sure the output folder exists
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)

                    image = cv2.imread(input_image_path)

                    # Specify the text and its position
                    
                    position = (50, 50)  # (x, y) coordinates

                    # Choose a font
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 2  # Font scale factor
                    font_color = (255, 255, 255)  # White color
                    font_thickness = 5

                    # Get the size of the text
                    # (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)

                    # # Calculate the position to center the text
                    # image_height, image_width, _ = image.shape
                    # x = (image_width - text_width) // 2+50
                    # y = (image_height + text_height) // 2

                    # Add the text to the image
                    

                    cv2.putText(image, text, position, font, font_scale, font_color, font_thickness)
                    

                    # Save the modified image to the output folder
                    output_image_path = os.path.join(output_folder, 'output_image.jpg')
                    cv2.imwrite(output_image_path, image)

                    print(f"Image with text saved to {output_image_path}")
    
                                
                            
                            


            frame_number += 1  # Increment frame number
    else:
        print("stock")
                # Release the video capture object and close the video file
cap.release()
cv2.destroyAllWindows()