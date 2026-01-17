import cv2
from ultralytics import YOLO

model = YOLO(r".\train4\weights\best.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4)
    annotated = results[0].plot()

    cv2.imshow("Litter Detection", annotated)

    if cv2.waitKey(1) == 27: 
        break

cap.release()
# cv2.destroyAllWindows() on the name of the world on the second
