from ultralytics import YOLO

model = YOLO('models/basket_pt.pt')

model.info()