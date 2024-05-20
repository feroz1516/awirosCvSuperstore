import cv2
import numpy as np

def draw_bounding_box(image, points):
    """
    Draws a bounding box on the input image using four points.
    
    Args:
    - image: The input image on which to draw the bounding box.
    - points: A list of four points [x1, y1, x2, y2, x3, y3, x4, y4] specifying the corners of the bounding box.
    """
    # Convert the points list to a NumPy array for easier manipulation
    points = np.array(points, dtype=np.int32)
    
    # Reshape the points to a 2x4 array
    points = points.reshape((-1, 2))
    
    # Draw the bounding box as a polygon
    cv2.polylines(image, [points], isClosed=True, color=(255, 0, 0), thickness=2)

# Create a blank white image for demonstration
image = cv2.imread('store_sample.png')

# Define four points for the bounding box [x1, y1, x2, y2, x3, y3, x4, y4]
bounding_box_points = [4, 271, 244, 155, 541, 212, 318, 716]

# Draw the bounding box on the image
draw_bounding_box(image, bounding_box_points)

# Display the image with the bounding box
cv2.imshow("Bounding Box", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
