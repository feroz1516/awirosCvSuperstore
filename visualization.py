import matplotlib.pyplot as plt
import numpy as np
import cv2
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
class_names = {
    2: "Bounce",
    35: "Supermilk",
    18: "Malkist",
    9: "Dark Fantasy",
    20: "Marie Light",
    
    38: "Unibic"
    
    # Add more class names as needed
}

modelyolo = YOLO('E:\\Projects\\CV App-a-thon\\biscuitmodel\\best.pt')
results = modelyolo(source='E:\\Projects\\CV App-a-thon\\biscuitpredict\\Screenshot (69).png', save=True)

# Extract class predictions
boxes_cls = []

for result in results:
    boxes = result.boxes.cls
    # Convert the PyTorch tensor to a Python list
    boxes_list = boxes.tolist()
    # Extend the 'boxes_cls' list with the elements from 'boxes_list'
    boxes_cls.extend(boxes_list)

# Perform class count without using Counter
class_count = {}
for class_index in boxes_cls:
    if class_index not in class_count:
        class_count[class_index] = 1
    else:
        class_count[class_index] += 1



# Convert the class_count to a dictionary
class_count_dict = dict(class_count)
print(class_count_dict)

# Get class names and counts
class_names_list = list(class_count_dict.keys())
class_counts = list(class_count_dict.values())
print(class_names_list)
print(class_counts)

# Create a list of unique colors for each class
# Make sure class_names and class_count_dict are aligned
class_names_list = list(class_names.keys())
class_counts = [class_count_dict.get(class_index, 0) for class_index in class_names_list]

# Create a list of unique colors for each class
# Create a single figure with two subplots (1 row, 2 columns)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Create a bar chart on the first subplot (ax1)



ax1.bar(class_names.values(), class_counts, color= ["#702963","#800020", "#9F2B68", "#702963", "#DA70D6", "#5D3FD3"], width=0.8)   



ax1.set_xlabel('Product Names')
ax1.set_ylabel('Number of Detections')
ax1.set_title('Products In-Stock')
ax1.tick_params(axis='x', rotation=45)  
ax1.set_ylim(0, 30)
legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
ax1.legend(legend_handles, class_names.values(), loc='upper right', title='Product Classes', prop={'size': 6})


# Create a pie chart on the second subplot (ax2)
second_pie_data = [15, 25, 30, 20]


ax2.pie(second_pie_data, autopct='%1.1f%%', shadow=True, startangle=90,colors=["#9F2B68", "#800020", "#702963","#DA70D6"])



ax2.set_title('Customer Purchasing Behaviour')
ax2.set_aspect('equal')  # Ensure the pie chart is circular
ax2.legend(second_labels = ['Area 1', 'Area 2', 'Area 3', 'Area 4'], loc='lower right', title='Product Classes', prop={'size': 6}, bbox_to_anchor=(1.25, 0))

# Save the combined chart as an image
output_image_path = 'visualization.png'
plt.savefig(output_image_path, format='png', bbox_inches='tight', pad_inches=0.1)

# Close the plots
plt.close()

# Load the saved image using OpenCV
image = cv2.imread(output_image_path)

# Convert the image to white pixels
image[np.where((image > [200, 200, 200]).all(axis=2))] = [255, 255, 255]

# Save the modified image as a white pixel image
output_white_pixel_image_path = 'white_pixel_image.png'
cv2.imwrite(output_white_pixel_image_path, image)

# Display the modified image

cv2.imshow('White Pixel Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()



def visualization(pie_input, bar_input):
    colors = ["#702963","#800020", "#9F2B68", "#702963", "#DA70D6", "#5D3FD3"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
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
    ax2.set_title('Customer Purchasing Behaviour')
    ax2.set_aspect('equal')  
    ax2.legend( loc='lower right', title='Product Classes', prop={'size': 6}, bbox_to_anchor=(1.25, 0))
    output_image_path = 'visualization.png'
    plt.savefig(output_image_path, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close()
    image = cv2.imread(output_image_path)
    image[np.where((image > [200, 200, 200]).all(axis=2))] = [255, 255, 255]
    return image
