import cv2
import numpy as np
from sklearn.neighbors import KernelDensity
from ultralytics import YOLO

model = YOLO('humandetect.pt')

min_distance_threshold = 50  
max_frames_to_track = 10  

cap = cv2.VideoCapture('supermarketvideo.mp4')

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video_with_heatmap.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame_heatmap = cap.read()

    if ret:
        frame_counter = 0  

        results = model(frame_heatmap)

        tracked_persons = []  

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

                    data = np.array([person['centroid']])
                    kde.fit(data)

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

            out = cv2.VideoWriter('output_video_with_heatmap.mp3', fourcc, 60.0, (int(cap.get(3)), int(cap.get(4))))


            cv2.imshow('Image with Heatmap', heatmap_overlay)

        else:
            tracked_persons = []

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
out.release()

cv2.destroyAllWindows()