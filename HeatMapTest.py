
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.neighbors import KernelDensity

data = pd.read_csv("transaction_data1.csv")

model_path= 'models/humandetect.pt'
media_path = 'sample_video/supermarket.mp4'
model = YOLO(model_path)
    # Define the confidence threshold for object detection
confidence_threshold = 0.1
iou = 0.2
# Define a dictionary to map class IDs to labels
class_labels = {0: "Person"}

# Open the video file
cap = cv2.VideoCapture(media_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
predictions = []
while True:
    ret, image = cap.read()
    if not ret:
        break
    
    # Perform object detection on the frame
    results = model.track(image, mode='track', conf=confidence_threshold, iou=iou)

    for result in results:
        print(result.boxes)
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        labels = result.boxes.cls.cpu().numpy()
        ids = result.boxes.id.numpy()

        color = (255, 255, 255)
        

        for box, score, label, person_id in zip(boxes, scores, labels, ids):
            x1, y1, x2, y2 = map(int, box)
            class_id = int(label)
            confidence = round(score, 2)
            predictions.append([x1,y1,x2,y2])
            # Filter out low-confidence detections
            if confidence > confidence_threshold:
                # Get the corresponding label from the dictionary
                class_label = class_labels.get(class_id, "Unknown")

                # Draw bounding box and label on the frame
                # Green color
                label_text = f"{class_label}: {confidence:.2f}"
                # cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                print(predictions)

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
    cv2.imshow('Image with Heatmap', heatmap_overlay)  # To display the result
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.release()
cv2.destroyAllWindows()