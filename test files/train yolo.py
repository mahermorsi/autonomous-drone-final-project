from ultralytics import YOLO
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()
    # Load a COCO-pretrained YOLOv8n model
    model = YOLO('yolov8n.pt')

    results = model.train(data='../tree detection/data.yaml', epochs=2, imgsz=640)
    path = model.export(format='onnx')

    # Run inference with the YOLOv8n model on the 'bus.jpg' image
