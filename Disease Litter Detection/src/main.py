from ultralytics import YOLO
from dotenv import load_dotenv
from roboflow import Roboflow
import os

def main():
    load_dotenv()

    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
    project = rf.workspace("taco-t7kkz").project("taco-dataset-ql1ng")
    version = project.version(15)
    dataset = version.download("yolov8")

    data_yaml_path = os.path.join(dataset.location, "data.yaml")

    model = YOLO("yolov8n.pt")

    model.train(
    data=data_yaml_path,
    epochs=20,
    imgsz=640,
    batch=16,
    device=0,
    workers=2,
    project="runs",
    name="taco_yolov8"
    )
    
    results = model.train(...)
    print(results.save_dir)

if __name__ == "__main__":
    main()

print("Training completed.")



