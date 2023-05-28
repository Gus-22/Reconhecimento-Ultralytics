import cv2
import time
import numpy as np

COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

class_names = []
with open("Reconhecimento_drone\coco.names", "r") as f:
    class_names = [line.strip() for line in f.readlines()]
    


cap = cv2.VideoCapture("DJI_0051.MP4")

net = cv2.dnn.readNet("Reconhecimento_drone\yolov4.weights", "Reconhecimento_drone\yolov4.cfg")

model = cv2.dnn_DetectionModel(net)
model.setinputParams(size=(608, 608), scale=1/255)

while True:
    
    frame = cap.read()
    
    start = time.time()
    
    classes, scores, boxes = model.detect(frame, 0.1, 0.2)
    
    end = time.time()
    
    
    for (classid, score, box) in zip(classes, scores, boxes):
        
        color = COLORS[int(classid) % len(COLORS)]
        
        
        label = f"{class_names[classid[0]]}: {score}"
        
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
    
    


