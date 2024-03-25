import cv2
import time
import numpy as np

COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

class_names = []
with open("coco.names", "r") as f:
    class_names = [line.strip() for line in f.readlines()]
    


cap = cv2.VideoCapture(0)

net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(608, 608), scale=1/255)

while True:
    
    ret, frame = cap.read()
    if not ret: 
        break
    
    start = time.time()
    
    classes, scores, boxes = model.detect(frame, confThreshold=0.1, nmsThreshold=0.2)

    
    end = time.time()
    
    
    for (classid, score, box) in zip(classes, scores, boxes):
        classid = classid[0] if isinstance(classid, (list, tuple)) else classid
        label = f"{class_names[classid]}: {score}"
        color = COLORS[classid % len(COLORS)]
    

        cv2.rectangle(frame, box, color, 2)

        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2) 
    
    
    fps_label = f"FPS: {round((1.0/(end - start)), 2)}"
    
    
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    
    cv2.imshow("detections", frame)
    
    
    if cv2.waitKey() == 27:
        break
    
    cap.realese()
    cv2.destroyAllWindows()