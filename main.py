from ultralytics import YOLO


# Load a model
model = YOLO("yolov8s.yaml")  # build a new model from scratch

# Use the model
results = model.train(
    data="yolov8_config.yaml",  # path to the dataset YAML file
    epochs=100,           # number of epochs to train
    imgsz=640,            # image size (height, width)
    batch=16,             # batch size, you can adjust this based on your GPU memory
    workers=4,            # number of data-loading workers, adjust based on your CPU
    project="runs/train", # directory to save training results 
    name="object_detection",           # name of the experiment, results will be saved in 'runs/train/object_detection'
    cache=True            # caches images for faster training (if memory allows)
)





