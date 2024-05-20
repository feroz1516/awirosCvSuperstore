from ultralytics import YOLO
from ultralytics import RTDETR

import cv2
import imageio
import numpy as np
import time
#244, 155

from ultralytics import YOLO

areas = [
    {"points": [0, 719,4, 271, 160, 210, 541, 212, 318, 716], "name": "Area 1"},
    {"points": [160,210, 541, 212, 621, 22, 527, 23], "name": "Area 2"},
    {"points": [541, 212, 621, 22, 711, 18, 858, 201], "name": "Area 3"},
    {"points": [318, 716,  1228, 717,858, 201,541, 212], "name": "Area 4"}
]

model = YOLO('models/humandetect_openvino_model')

def draw_polygon(image, points, label=None):
    # Convert the points list to a NumPy array for easier manipulation
    points = np.array(points, dtype=np.int32)
    
    # Reshape the points to a Nx2 array, where N is the number of points
    points = points.reshape((-1, 2))
    
    # Draw the polygon
    cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)
    
    # Add a label if provided
    if label is not None:
        label_position = (points[0][0]+100, points[0][1]-40)  # Adjust the label position
        cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        
def plot_line(x1, y1, x2, y2, decide):
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    error = dx - dy
    
    x, y = x1, y1
    
    line_pixels = {}
    while True:
        if decide:
            line_pixels[y]=x
        else:
            line_pixels[x]=y
        if x == x2 and y == y2:
            break
        
        e2 = 2 * error
        if e2 > -dy:
            error -= dy
            x += sx
        if e2 < dx:
            error += dx
            y += sy
    
    return line_pixels




def check_area4(x,y):
    # print("x: ",x , "y: ", y)
    line1_pixels = plot_line( 318, 716, 541, 212, 1)
    #print("line1_pixels: ",line1_pixels)
    line1_x =  line1_pixels.get(y, False)
    # print("line1_x: ",line1_x)

    line4_pixels = plot_line( 541, 212,  858, 201, 0) 
    line4_y = line4_pixels.get(x, False) 

    # print("line4_pixels: ",line4_pixels)
    # print("line4_y: ",line4_y)

    
    line1 = False
    line4 = False

    if line1_x:
        if x > line1_x:
            line1 = True
        else:
            line1 = False   


    
    if line4_y:
        if y > line4_y:
            line4 = True
        else:
            line4= False

    if line1 and line4:
        return True
    else:
        return False

    
def check_area1(x,y):
    # print("x: ",x , "y: ", y)
    line1_pixels = plot_line( 318, 716, 541, 212, 1)
    #print("line1_pixels: ",line1_pixels)
    line1_x =  line1_pixels.get(y, False)
    # print("line1_x: ",line1_x)

    line2_pixels = plot_line(  541, 212 , 0, 210, 0)
    # print("line2_pixels: ",line2_pixels)
    line2_y = line2_pixels.get(x, False) 
    # print("line2_y: ",line2_y)

    
    line1 = False
    line2 = False
   

    if line1_x:
        if x < line1_x:
            line1 = True
        else:
            line1 = False   

    if line2_y:
        if y > line2_y:
            line2 = True
        else:
            line2= False

    # print("line1 ", line1 ,"line2 ", line2 )
    if line1 and line2 :
        return True
    else:
        return False

def check_area2(x,y):
    # print("x: ",x , "y: ", y)
    line1_pixels = plot_line( 541, 212, 621, 22, 1)
    #print("line1_pixels: ",line1_pixels)
    line1_x =  line1_pixels.get(y, False)
    # print("line1_x: ",line1_x)

    line2_pixels = plot_line(  541, 212, 0, 210 , 0)
    # print("line2_pixels: ",line2_pixels)
    line2_y = line2_pixels.get(x, False) 
    # print("line2_y: ",line2_y)

    
    line1 = False
    line2 = False
   

    if line1_x:
        if x < line1_x:
            line1 = True
        else:
            line1 = False   

    if line2_y:
        if y < line2_y:
            line2 = True
        else:
            line2= False

    # print("line1 ", line1 ,"line2 ", line2 )
    if line1 and line2 :
        return True
    else:
        return False
    
def check_area3(x,y):
    # print("x: ",x , "y: ", y)
    line1_pixels = plot_line( 541, 212, 621, 22, 1)
    #print("line1_pixels: ",line1_pixels)
    line1_x =  line1_pixels.get(y, False)
    # print("line1_x: ",line1_x)

    line2_pixels = plot_line(  541, 212,  858, 201 , 0)
    # print("line2_pixels: ",line2_pixels)
    line2_y = line2_pixels.get(x, False) 
    # print("line2_y: ",line2_y)

    
    line1 = False
    line2 = False
   

    if line1_x:
        if x > line1_x:
            line1 = True
        else:
            line1 = False   

    if line2_y:
        if y < line2_y:
            line2 = True
        else:
            line2= False

    # print("line1 ", line1 ,"line2 ", line2 )
    if line1 and line2 :
        return True
    else:
        return False

def coordinates(coords):
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    for person in coords:
        a = person[0]
        b = person[2]
        c = a+((b-a)/2)
        # count_area1 = check_area1(round(c) , round(person[1]))
       
        count_area1 = check_area1(round(c) , round(person[1]))
        if count_area1:
            count1 = count1 +1

        count_area2 = check_area2(round(c) , round(person[1]))
        if count_area2:
            count2 = count2 +1

        count_area3= check_area3(round(c) , round(person[1]))
        if count_area3:
            count3 = count3 +1

        count_area4 = check_area4(round(c) , round(person[1]))
        if count_area4:
            count4 = count4+1
        
    return count1, count2 , count3 , count4

# Define a dictionary to store the start time for each person (ID as the key)
start_times = {}

# Define a dictionary to store the dwell time for each person (ID as the key)
dwell_times = {}


def inference_video(file_name, media_path):

    # Define path to the video file
    # Define the path to the output video file
    video_file_name = 'predicted_' + file_name

    # Define the confidence threshold for object detection
    confidence_threshold = 0.4
    iou = 0.2
    # Define a dictionary to map class IDs to labels
    class_labels = {0: "Person"}

    # Open the video file
    cap = cv2.VideoCapture(media_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    output_writer = imageio.get_writer(video_file_name, fps=30, codec='vp9', quality=8, pixelformat='yuv420p')
    frame_counter = 0  # Initialize frame counter
    time_increment = 0  # Initialize time increment


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection on the frame
        results = model.track(frame, mode='track', conf=confidence_threshold, iou=iou)
        frame_counter += 1

        for result in results:
            print(result.boxes)
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            labels = result.boxes.cls.cpu().numpy()
            ids = result.boxes.id.numpy()

            color = (255, 255, 255)
            count1, count2, count3, count4 = coordinates(boxes)
            cv2.putText(frame, "Area 1: " + str(count1), (1130, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, "Area 2: " + str(count2), (1130, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, "Area 3: " + str(count3), (1130, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, "Area 4: " + str(count4), (1130, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            for box, score, label, person_id in zip(boxes, scores, labels, ids):
                x1, y1, x2, y2 = map(int, box)
                class_id = int(label)
                confidence = round(score, 2)

                # Filter out low-confidence detections
                if confidence > confidence_threshold:
                    # Get the corresponding label from the dictionary
                    class_label = class_labels.get(class_id, "Unknown")

                    # Draw bounding box and label on the frame
                    # Green color
                    label_text = f"{class_label}: {confidence:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    if person_id not in start_times:
                        start_times[person_id] = 0
                    else:
                        if frame_counter%30==0:
                            start_times[person_id]=start_times[person_id]+1
                    # Format the dwell time into HH:MM:SS
                    hours, remainder = divmod(int(start_times[person_id]), 3600)
                    minutes, seconds = divmod(remainder, 60)
                    formatted_time = f"Dwell Time: {hours:02d}:{minutes:02d}:{seconds:02d}"

                    # Display or log the formatted dwell time above the person's head
                    cv2.putText(frame, formatted_time, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            
            # Increment the frame counter

            # Write the frame with bounding boxes to the output video
        for area_info in areas:
            draw_polygon(frame, area_info["points"])
            draw_polygon(frame, area_info["points"], label=area_info["name"])
        cv2.imshow('Video with Detection', frame)
        cv2.waitKey(1)  # Adjust the delay as needed (1 millisecond for 30 FPS)

        output_writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB

    # Release video capture and writer
    
    cap.release()
    output_writer.close()

    cv2.destroyAllWindows()

    return video_file_name

def main():
    inference_video("hello.mp4","sample_video/supermarket_TRIM.mp4")
    # count1 ,count2, count3, count4 = coordinates([[698.2042, 200.1895, 808.0192, 538.6069],
    #     [378.0180, 195.8536, 466.9362, 431.3445],
    #     [466.0624, 134.2867, 521.3545, 237.3621],
    #     [602.8120, 228.4440, 696.6322, 544.3158],
    #     [662.4548,  66.5581, 705.5332, 164.8587],
    #     [203.2202, 359.9202, 321.7858, 638.3414],
    #     [582.7720, 149.6522, 642.8651, 247.5154],
    #     [119.0742, 439.9740, 268.3291, 720.0000]])
    
    # print(count1,count2,count3,count4)
# Check if the script is being run directly, not imported as a module
if __name__ == "__main__":
    main()
