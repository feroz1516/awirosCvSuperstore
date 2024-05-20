from fetch_data import *
import sys
from ultralytics import YOLO
import pandas as pd
import cv2
import numpy as np
import time
#244, 155
from sklearn.neighbors import KernelDensity
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt
from ultralytics import YOLO
model_person = YOLO('/home/awiros-docker/alexandria/person_detect/v1/openvino/fp16/personv8_openvino_model/')
model_inventory = YOLO('/home/awiros-docker/alexandria/shelf_monitoring/v1/openvino/fp16/shelfnew_openvino_model/')
model_basket = YOLO('/home/awiros-docker/alexandria/basket_detection/v1/openvino/fp16/basketv8_openvino_model/')

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
    38: "Unibic"
    # Shelf 1 products
}


product_classes = ['50 50 Biscuit', 'Biscafe', 'Bounce', 'Bourbon Dark fantasy', 'Bourbon', 'Bourn Vita Biscuit', 'Chocobakes', 'Coffee Joy', 'Creme', 'Dark Fantasy', 'Digestive', 'Elite', 'Ginger', 'Good Day', 'Happy Happy', 'Hide - Seek', 'Jim Jam', 'KrackJack', 'Malkist', 'Marie Gold', 'Marie Light', 'Milk Bikis', 'Milk Short Cake', 'Mom Magic', 'Monaco', 'Nice', 'Nutri Choice', 'Nutri Choice-Crackers-', 'Nutri Choice-Herbs-', 'Nutri Choice-Sugar Free-', 'Oreo', 'Parle G', 'Potazo', 'Sunfeast green', 'Super Millets', 'Supermilk', 'Tninz', 'Treat', 'Unibic', 'Unibic-box', 'allrounder']

out_of_stock_count = {class_index: 0 for class_index in class_names.keys()} 

basket_id = {}
product_id = {}
t_id = {}
processed_products = {}  


def draw_polygon(image, points, label=None):
    points = np.array(points, dtype=np.int32)
    points = points.reshape((-1, 2))
    
    cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)
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


def person_detection(frame, confidence_threshold, frame_counter):
    frame_heatmap = frame.copy()
    class_labels = {0: "Person"}
    count1, count2, count3, count4 = 0,0,0,0
    min_distance_threshold = 50  
    max_frames_to_track = 10  
    tracked_persons = []  
    results = model_person.track(frame, mode='track', conf=confidence_threshold)
    if hasattr(results[0], 'boxes') and len(results[0].boxes.xyxy) > 0:
        bounding_boxes = results[0].boxes.xyxy.cpu().numpy()

        for box in bounding_boxes:
            centroid = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))

            # Check if the person is already being tracked
            person_tracked = False
            for person in tracked_persons:
                distance = np.sqrt((centroid[0] - person['centroid'][0]) ** 2 + (centroid[1] - person['centroid'][1]) ** 2)
                if distance < min_distance_threshold:
                    person_tracked = True
                    person['frames_since_last_detection'] = 0
                    person['centroid'] = centroid
                    break

            # If the person is not being tracked, add them to the list of tracked persons
            if not person_tracked:
                tracked_persons.append({
                    'centroid': centroid,
                    'frames_since_last_detection': 0
                })

        # Update the heatmap for each tracked person
        heatmap = np.zeros_like(frame_heatmap)
        for person in tracked_persons:
            if person['frames_since_last_detection'] < max_frames_to_track:
                bandwidth_value = 15
                kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth_value)

                data1 = np.array([person['centroid']])
                kde.fit(data1)

                x, y = np.meshgrid(np.linspace(0, frame_heatmap.shape[1], frame_heatmap.shape[1]), np.linspace(0, frame_heatmap.shape[0], frame_heatmap.shape[0]))
                grid_coords = np.vstack([x.ravel(), y.ravel()]).T

                person_heatmap = np.exp(kde.score_samples(grid_coords))
                person_heatmap = person_heatmap.reshape(frame_heatmap.shape[0], frame_heatmap.shape[1], 1)

                person_heatmap = (person_heatmap - person_heatmap.min()) / (person_heatmap.max() - person_heatmap.min()) * 255

                person_heatmap = np.repeat(person_heatmap, 3, axis=2)

                heatmap += person_heatmap.astype(np.uint8)

                person['frames_since_last_detection'] += 1

        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)

        alpha = 0.3
        heatmap_overlay = cv2.addWeighted(frame_heatmap, 1 - alpha, heatmap_color, alpha, 0)

    for result in results:
        blobs = []

     
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

        count_boxes = 0 
        for box, score, label, person_id in zip(boxes, scores, labels, ids):
            count_boxes +=1
            x1, y1, x2, y2 = map(int, box)
            class_id = int(label)
            confidence = round(float(score),3)

            # Filter out low-confidence detections
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
       

            blb.cropped_frame = frame[blb.ty:blb.by, blb.tx:blb.bx, :]
            
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
            if count_boxes == len(boxes):
                blb.attribs["Persons in area 1"] = count1
                blb.attribs["Persons in area 2"] = count2
                blb.attribs["Persons in area 3"] = count3
                blb.attribs["Persons in area 4"] = count4
            
            blobs.append(blb)
        for area_info in areas:
            draw_polygon(frame, area_info["points"])
            draw_polygon(frame, area_info["points"], label=area_info["name"])
        
        return frame, blobs , [count1, count2, count3, count4], heatmap_overlay

def inventory_detection(stream , confidence_value):

    blobs = []
    class_count = {}
    results = model_inventory(stream , conf = confidence_value, task = 'detect')
    boxes_cls = []
    color = (0,255,255)
    for result in results:
        classes = result.boxes.cls.cpu().numpy()
        # Convert the PyTorch tensor to a Python list
        predictions = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        class_list = classes.tolist()
        # Extend the 'boxes_cls' list with the elements from 'boxes_list'
        boxes_cls.extend(class_list)

        
        for class_index in boxes_cls:
            if class_index not in class_count:
                class_count[class_index] = 1
            else:
                class_count[class_index] += 1

        class_count_dict = dict(class_count)

        class_names_list = list(class_count_dict.keys())
        class_counts = list(class_count_dict.values())
        class_names_list = list(class_names.keys())
        class_counts = [class_count_dict.get(class_index, 0) for class_index in class_names_list]

        for pred , label ,score in zip(predictions, class_list,scores):
            blb = blob()
            x0 , y0 , x1, y1 = pred
            blb.tx = int(x0)
            blb.ty = int(y0)
            blb.bx = int(x1)
            blb.by = int(y1)
            class_label  = class_names[int(label)]
            blb.label = class_label
            confidence =round(float(score),3)
            blb.conf = confidence
            blb.cropped_frame = stream[blb.ty:blb.by , blb.tx:blb.bx, :]
            label_text = f"{class_label}: {confidence:.2f}"
            cv2.rectangle(stream, (int(x0), int(y0)), (int(x1),int( y1)), color, 2)
            cv2.putText(stream, label_text, (int(x0),int(y0) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            blobs.append(blb)

    blb = blob()
    blb.tx, blb.ty, blb.bx, blb.by = 0,0,0,0


    boxes_cls_set = set(boxes_cls)
    out_stock = []
    for class_index, class_name in class_names.items():
        if class_index not in boxes_cls_set:
            out_stock.append((class_index,class_name))
    
    for class_index in class_names.keys():
        if class_index in boxes_cls_set:
            out_of_stock_count[class_index] = 0
        else:
            out_of_stock_count[class_index] += 1



    print("Out of stock: ", out_of_stock_count)
    if any(count > 0 for count in out_of_stock_count.values()):
        c = 0
        for class_index, count in out_of_stock_count.items():
            class_name = class_names[class_index]
            if count > 0:
                print(f"{class_name} no detection. Count: {count}")
                if count > 2:
                    text=(f"{class_name} is out of stock")
                    print(text)
                    c = c + 1
                    blb.attribs["Availability "+str(c)+":"] = text

    
                else:
                    text="All items in stock"
                    print(text)
                    blb.attribs["Availability :"] = text
    
    else:
        text="All items in stock"
        print(text)
        blb.attribs["Availability :"] = text
                # Release the video capture object and close the video file

    blobs.append(blb)

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
    
    return stream, blobs, class_counts  


def basket_analysis(frame, confidence_value):

    data = pd.read_csv("basket_analysis_apriori.csv")
    blobs = []
    color = (0, 255, 255)
    results = model_basket.track(frame, mode='track', conf=0.4 ,iou = 0.4 )

    for result in results:
        if result.boxes.id is None:
            continue
        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy()

        for box, id in zip(boxes, ids):
            bx1, by1, bx2, by2 = map(int, box)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
            cv2.putText(frame, "Basket", (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            results_product = model_inventory.track(frame, mode='track', conf=0.1)
            products_in_basket = []
            for result_product in results_product:
                blobs = []
                if result_product.boxes.id is None:
                    continue

                boxes_product = result_product.boxes.xyxy.cpu().numpy()
                conf_product = result_product.boxes.conf.cpu().numpy()

                classes_product = result_product.boxes.cls.cpu().numpy()
                ids_product = result_product.boxes.id.cpu().numpy()

                for box_p, label_p, id_p,conf in zip(boxes_product, classes_product, ids_product,conf_product):
                    blb = blob()
                    px1, py1, px2, py2 = map(int, box_p)
                    blb.tx = px1
                    blb.ty = py1
                    blb.bx = px2
                    blb.by = py2
                    if px1 >= bx1 and px2 <= bx2 and py1 >= by1 and py2 <= by2:
                        product_name = product_classes[int(label_p)]
                        blb.conf = round(float(conf),3)
                        blb.id = random.randint(0,1000000)
                        blb.label = product_name
                        blb.cropped_frame = frame[blb.ty:blb.by, blb.tx:blb.bx, :]
                        if id in basket_id:
                            basket_id[id][id_p] = product_name
                            products_in_basket.append(product_name)  # Add product name to the list

                        else:
                            basket_id[id] = {id_p: product_name}
                            max_transaction_id = data["Transaction-ID"].max()+1
                            t_id[id] = max_transaction_id
                            next_transaction_id = max_transaction_id
                            new_row = pd.DataFrame({'Transaction-ID': [next_transaction_id]})

                            # Add columns for all other products and set them to zero
                            product_columns = data.columns[1:]  # Exclude the 'Transaction-ID' column
                            for column in product_columns:
                                new_row[column] = 0

                            # Append the new row to the existing DataFrame
                            data = data.append(new_row, ignore_index=True)

                           
                        cv2.rectangle(frame, (px1, py1), (px2, py2), (0,255,0), 2)
                        cv2.putText(frame, product_name+" "+str(id_p), (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                        blobs.append(blb)
            if products_in_basket:
                products_text = ", ".join(products_in_basket)
                cv2.putText(frame, "Products: " + products_text, (bx1, by1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    for transaction_id, products in basket_id.items():
        transaction_id = t_id[transaction_id]
        if transaction_id in data["Transaction-ID"].values:
            # Get the row index where Transaction-ID matches
            row_index = data[data["Transaction-ID"] == transaction_id].index[0]
            
            # Check if this transaction ID has been processed in the current frame
            if transaction_id not in processed_products:
                processed_products[transaction_id] = set()  # Initialize the set for this transaction ID

            # Loop through the products in the dictionary and update the corresponding columns
            for product_id, product_name in products.items():
                if product_name not in processed_products[transaction_id]:
                    # Find the corresponding product column in the transaction data based on product_name
                    column_name = data.columns[data.columns.str.contains(product_name)].item()
                    # Update the corresponding cell in the transaction data
                    data.at[row_index, column_name] += 1
                    processed_products[transaction_id].add(product_name)

    data.to_csv("basket_analysis_apriori.csv", index=False)
    basket = data.drop(columns=["Transaction-ID"])  # Exclude the Transaction-ID column
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)

    # Apply the Apriori algorithm to find frequent itemsets with a minimum support threshold (e.g., 0.2)
    product_popularity = data.drop(columns=['Transaction-ID']).sum()

    # Find the most popular product and its count
    most_popular_product_name = product_popularity.idxmax()
    most_popular_product_count = product_popularity.max()

    least_popular_product_name = product_popularity.idxmin()
    least_popular_product_count = product_popularity.min()

    frequent_itemsets = apriori(basket, min_support=0.2, use_colnames=True)

    # Generate association rules with a minimum confidence threshold (e.g., 0.5)
    association_rules_df = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

    # Sort the association rules by lift (a measure of how strongly two items are associated)
    association_rules_df = association_rules_df.sort_values(by="lift", ascending=False)

    # Print the best product recommendations and closely associated products

    best_product_recommendation = association_rules_df.head(1).iloc[0]

    # Extract antecedents and consequents
    antecedents = best_product_recommendation['antecedents']
    consequents = best_product_recommendation['consequents']

    # Extract antecedent and consequent product names
    antecedent_products = [str(product) for product in antecedents]
    consequent_products = [str(product) for product in consequents]

    # Create the recommendation description
    recommendation_description = f"That people who bought {', '.join(antecedent_products)} are likely to also buy {', '.join(consequent_products)} - Apriori Algorithm."

    blb = blob()
    blb.tx,blb.ty,blb.bx,blb.by = 0,0,0,0
    blb.attribs["Recommendation"] = recommendation_description

    blb.attribs["Most Popular Product"] = most_popular_product_name + "| Count: " + str(most_popular_product_count)
    blb.attribs["Least Popular Product"]= least_popular_product_name+ "| Count: " + str(least_popular_product_count)
    blobs.append(blb)
    cv2.putText(frame, recommendation_description, (100, 1000), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0, 0), 2)
    # Print the recommendation description
    closely_associated_products = association_rules_df.head(5)

   
    return frame , blobs


def visualization(pie_input, bar_input):
    image_shape = (1080,1930,3)
    white_pixel_image = np.ones(image_shape, dtype=np.uint8) * 255


    colors = ["#702963","#800020", "#9F2B68", "#702963", "#DA70D6", "#5D3FD3"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    #Bar chart
    ax1.bar(class_names.values(), bar_input, color= colors, width=0.8)
    ax1.set_xlabel('Product Names')
    ax1.set_ylabel('Number of Detections')
    ax1.set_title('Products In-Stock')
    ax1.tick_params(axis='x', rotation=45)  
    ax1.set_ylim(0, 30)
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
    ax1.legend(legend_handles, class_names.values(), loc='upper right', title='Product Classes', prop={'size': 6})   
    #Pie chart
    ax2.pie(pie_input, autopct='%1.1f%%', shadow=True, startangle=90,colors=["#9F2B68", "#800020", "#702963","#DA70D6"])
    ax2.set_title('People in each area')
    ax2.set_aspect('equal')  
    ax2.legend(["Area 1", "Area 2", "Area 3", "Area 4"],loc='lower right', title='Product Classes', prop={'size': 6}, bbox_to_anchor=(1.25, 0))

    # Render the Matplotlib visualization on the white pixel image
    figure_canvas = fig.canvas
    figure_canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image_from_canvas = np.frombuffer(figure_canvas.buffer_rgba(), dtype=np.uint8).reshape(int(height), int(width), 4)
    image_from_canvas = cv2.cvtColor(image_from_canvas, cv2.COLOR_RGBA2BGR)

    # Define the position for overlaying the Matplotlib visualization
    start_x, start_y = 50, 50

    # Overlay the Matplotlib visualization on the white pixel image
    white_pixel_image[start_y:start_y + int(height), start_x:start_x + int(width)] = image_from_canvas

    # Close the plots
    plt.close()

    
    return white_pixel_image



def run(acs_url, broker_url, topic_name):
    print("Creating Meta....")

    meta_obj = meta(acs_url, broker_url, topic_name)

    print("Parsing Acs....")
    meta_obj.parse_acs()
    frame_counter = 0

    while True:
        meta_obj.run_camera()
        for stream_index,stream in enumerate(meta_obj.streams): 
            if frame_counter%15==0:
                detection_conf = meta_obj.awi_detection_conf[stream_index]
                if stream_index == 0:    
                    image, blob , area_count, heatmap_overlay = person_detection(stream , detection_conf , frame_counter)  
                    
                    eve2 = event()
                    eve2.set_frame(heatmap_overlay)
                    meta_obj.push_event(eve2)
                    meta_obj.send_event()

                    eve = event()
                    if (len(blob)>0):
                        for blb in blob:
                            blb.frame = stream  
                            eve.eve_blobs.append(blb)
                        eve.set_frame(image)
                        eve.type = "Test2"
                        event.source_type_key = "Test3"
                        event.source_entity_idx = random.randint(0,1000)
                        
                        meta_obj.push_event(eve)
    
                    
                        print("Pushing alert....")
                        meta_obj.send_event()

                if stream_index == 1:
                    image, blob ,class_count  = inventory_detection(stream, detection_conf)
                    eve = event()
                    if (len(blob)>0):
                        for blb in blob:
                            blb.frame = stream  
                            eve.eve_blobs.append(blb)
                        eve.set_frame(image)
                        eve.type = "Test2"
                        event.source_type_key = "Test3"
                        event.source_entity_idx = random.randint(0,1000)
                        
                        meta_obj.push_event(eve)
                        print("Pushing alert....")
                        meta_obj.send_event()

                if stream_index == 2:
                    image , blob = basket_analysis(stream, detection_conf)
                    eve = event()
                    if (len(blob)>0):
                        for blb in blob:
                            blb.frame = stream  
                            eve.eve_blobs.append(blb)
                        eve.set_frame(image)
                        eve.type = "Test2"
                        event.source_type_key = "Test3"
                        event.source_entity_idx = random.randint(0,1000)
                        
                        meta_obj.push_event(eve)
    
                    
                        print("Pushing alert....")
                        meta_obj.send_event()
                try:
                    image  = visualization(area_count, class_count)
                    # blb = blob()
                    # eve.eve_blobs.append(blb)
                    eve = event()
                    eve.set_frame(image)
                    meta_obj.push_event(eve)
                except:
                    print("Not all streams are selected")



        frame_counter+=1
        print("frame count:", frame_counter)


if __name__ == "__main__":

    
    acs_url = sys.argv[1]
    broker_url = sys.argv[2]
    topic_name = sys.argv[3]
    
    # run(acs_url, broker_url, topic_name) 
    run(acs_url, broker_url, topic_name) 
