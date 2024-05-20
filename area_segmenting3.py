import cv2
import numpy as np

def draw_polygon(image, points, label=None):
    # Convert the points list to a NumPy array for easier manipulation
    points = np.array(points, dtype=np.int32)
    
    # Reshape the points to a Nx2 array, where N is the number of points
    points = points.reshape((-1, 2))
    
    # Draw the polygon
    cv2.polylines(image, [points], isClosed=True, color=(255, 0, 0), thickness=2)
    
    # Add a label if provided
    if label is not None:
        label_position = (points[0][0], points[0][1])  # Adjust the label position
        cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Create a blank white image for demonstration
image = cv2.imread('store_sample.png')

# Define areas with names as lists of points [x1, y1, x2, y2, ..., x5, y5]
areas = [
    {"points": [4, 271, 244, 155, 541, 212, 318, 716, 0, 719], "name": "Area 1"},
    {"points": [244, 155, 541, 212, 621, 22, 527, 23], "name": "Area 2"},
    {"points": [541, 212, 621, 22, 711, 18, 858, 201], "name": "Area 3"},
    {"points": [318, 716,  1228, 717,858, 201,541, 212], "name": "Area 4"}
]

# Draw the polygons and labels on the image
for area_info in areas:
    draw_polygon(image, area_info["points"])
    draw_polygon(image, area_info["points"], label=area_info["name"])

# Display the image with the polygons and labels
cv2.imshow("Polygon", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
