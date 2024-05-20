import cv2
import numpy as np
from sklearn.neighbors import KernelDensity

# Read the original image
image = cv2.imread('sample_images/supermarket_Moment2.jpg')

# Simulate YOLO bounding box predictions (replace with actual results)
# Each prediction should be in the format [x1, y1, x2, y2]
predictions = [
   [698.2042, 200.1895, 808.0192, 538.6069],
   [698.2042, 200.1895, 808.0192, 538.6069],
   [698.2042, 200.1895, 808.0192, 538.6069],
   [378.0180, 195.8536, 466.9362, 431.3445]
]


# Extract x and y coordinates from the bounding box predictions
x1_coords = [prediction[0] for prediction in predictions]
y1_coords = [prediction[1] for prediction in predictions]
x2_coords = [prediction[2] for prediction in predictions]
y2_coords = [prediction[3] for prediction in predictions]

# Calculate the center coordinates (x, y) for each bounding box
x_coords = [(x1 + x2) / 2 for x1, x2 in zip(x1_coords, x2_coords)]
y_coords = [(y1 + y2) / 2 for y1, y2 in zip(y1_coords, y2_coords)]

# Create a KernelDensity estimator
bandwidth_value = 30  # Adjust the bandwidth as needed
kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth_value)

# Fit the KDE estimator to the data
data = np.vstack([x_coords, y_coords]).T
kde.fit(data)

# Generate grid points for evaluation
x, y = np.meshgrid(np.linspace(0, image.shape[1], image.shape[1]), np.linspace(0, image.shape[0], image.shape[0]))
grid_coords = np.vstack([x.ravel(), y.ravel()]).T

# Evaluate the KDE at grid points
heatmap = np.exp(kde.score_samples(grid_coords))
heatmap = heatmap.reshape(image.shape[0], image.shape[1])

# Normalize the heatmap to the 0-255 range
heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) * 255

# Convert the heatmap to a color map for visualization
heatmap_color = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_HOT)

# Overlay the heatmap on the original image
alpha = 0.6  # Opacity of the heatmap overlay
heatmap_overlay = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)

# Save or display the output image with the heatmap overlay
cv2.imwrite('output_image_with_heatmap.jpg', heatmap_overlay)  # To save the result
cv2.imshow('Image with Heatmap', heatmap_overlay)  # To display the result

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
