from ultralytics import YOLO

model = YOLO('models/humandetect_openvino_model/')


model.track(source='sample_video/supermarket_Trim.mp4')