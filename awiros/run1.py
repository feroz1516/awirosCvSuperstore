from fetch_data import *
import sys
from ultralytics import YOLO


import cv2
import imageio
import numpy as np
import time
#244, 155

from ultralytics import YOLO
model_person = YOLO('/home/awiros-docker/alexandria/person_detect/v1/openvino/personv8_openvino_model/')
model_inventory = YOLO('/home/awiros-docker/alexandria/shelf_monitoring/v1/openvino/shelfv8_openvino_model/')


areas = [
    {"points": [0, 719,4, 271, 160, 210, 541, 212, 318, 716], "name": "Area 1"},
    {"points": [160,210, 541, 212, 621, 22, 527, 23], "name": "Area 2"},
    {"points": [541, 212, 621, 22, 711, 18, 858, 201], "name": "Area 3"},
    {"points": [318, 716,  1228, 717,858, 201,541, 212], "name": "Area 4"}
]

# Load a pretrained YOLOv8n model
class_names = {
    2: "Bounce",
    35: "Supermilk",
    18: "Malkist",
    9: "Dark Fantasy",
    20: "Marie Light",
    28: "Nutri Choice-Herbs-",
    38: "Unibic"
    # Shelf 1 products
}


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


def person_detection(frame, confidence_threshold):

    class_labels = {0: "Person"}

   
    frame_counter = 0  
    time_increment = 0  

    results = model_person.track(frame, mode='track', conf=confidence_threshold)
    frame_counter += 1

    for result in results:
        blobs = []

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
            confidence =round(float(score),3)

            # Filter out low-confidence detections
            if confidence > confidence_threshold:
                # Get the corresponding label from the dictionary
                class_label = class_labels.get(class_id, "Unknown")
                blb = blob()
                
                blb.tx = x1
                blb.ty = y1
                blb.bx = x2
                blb.by = y2

                blb.conf = confidence
                blb.id = random.randint(0,1000000)
                blb.label = class_label
                print("blb.ty:", blb.ty)
                print("blb.by:", blb.by)
                print("blb.tx:", blb.tx)
                print("blb.bx:", blb.bx)

                blb.cropped_frame = frame[blb.ty:blb.by, blb.tx:blb.bx, :]
                
                print("blb.cropped_frame: ", blb.cropped_frame)
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

                blobs.append(blb)

        for area_info in areas:
            draw_polygon(frame, area_info["points"])
            draw_polygon(frame, area_info["points"], label=area_info["name"])
        
        return frame, blobs

def inventory_detection(stream , confidence_value):
    blb = blob()
    blobs = []
    results = model_inventory(stream , conf = confidence_value)
    boxes_cls = []
   
    for result in results:
        boxes = result.boxes.cls
        # Convert the PyTorch tensor to a Python list
        boxes_list = boxes.tolist()
        # Extend the 'boxes_cls' list with the elements from 'boxes_list'
        boxes_cls.extend(boxes_list)

    boxes_cls_set = set(boxes_cls)
    out_stock = []
    
    for class_index, class_name in class_names.items():
        if class_index not in boxes_cls_set:
            out_stock.append(class_name)
    if len(out_stock) > 0:
        print("Out of Stock Classes:")
        stock = 0 
        c = 0 
        for item in out_stock:
            text=(f"{item} is out of stock.")
            print(text)
            blb.attribs['Availability' + str(c)] = text 
            c=c+1

    else:
        stock = 1
        text = "All items are in stock."
        print(text)
        blb.attribs['Availability'] = text 


    position = (50, 50)  # (x, y) coordinates
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2  # Font scale factor
    font_color = (255, 255, 255)  # White color
    font_thickness = 5

    # Get the size of the text
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)

    # Calculate the position to center the text
    image_height, image_width, _ = stream.shape
    x = (image_width - text_width) // 2+50
    y = (image_height + text_height) // 2

    # Add the text to the image
    cv2.putText(stream, text, position, font, font_scale, font_color, font_thickness)
    blobs.append(blb)
    return stream, blobs, stock




def run(acs_url, broker_url, topic_name):
    print("Creating Meta....")
    meta_obj = meta(acs_url, broker_url, topic_name)

    print("Parsing Acs....")
    meta_obj.parse_acs()
    frame_counter = 0

    while True:
        meta_obj.run_camera()
        for stream_index,stream in enumerate(meta_obj.streams): 
            detection_conf = meta_obj.awi_detection_conf[stream_index]
            if stream_index == 0:    
                image, blob = person_detection(stream , detection_conf)  
                print("image:",image)
                eve = event()
                if (len(blob)>0):
                    for blb in blob:
                        blb.frame = stream  
                        print("blb: ",blb)
                        eve.eve_blobs.append(blb)
                    print("length of eve.eve_blobs",len(eve.eve_blobs))
                    eve.set_frame(image)
                    eve.type = "Test2"
                    event.source_type_key = "Test3"
                    event.source_entity_idx = random.randint(0,1000)
                    
                    meta_obj.push_event(eve)
                    print("eve:",eve.eve_frame)

                   
                    print("Pushing alert....")
                    meta_obj.send_event()

            if stream_index == 1:
                image, blob , stock = inventory_detection(stream, detection_conf)
                eve = event()
                if (len(blob)>0):
                    for blb in blob:
                        blb.frame = stream  
                        print("blb: ",blb)
                        eve.eve_blobs.append(blb)
                    print("length of eve.eve_blobs",len(eve.eve_blobs))
                    eve.set_frame(image)
                    eve.type = "Test2"
                    event.source_type_key = "Test3"
                    event.source_entity_idx = random.randint(0,1000)
                    
                    meta_obj.push_event(eve)
   
                    print("eve:",eve.eve_frame)
                    print("Pushing alert....")
                    meta_obj.send_event()


        frame_counter+=1
        print("frame count:", frame_counter)


if __name__ == "__main__":

    print("here")
    
    acs_url = sys.argv[1]
    broker_url = sys.argv[2]
    topic_name = sys.argv[3]
    
    # run(acs_url, broker_url, topic_name) 
    run(acs_url, broker_url, topic_name) 
# # _________________________________________________







