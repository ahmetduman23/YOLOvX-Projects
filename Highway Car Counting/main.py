import cv2 
import numpy as np
from ultralytics import YOLO
import imutils
from collections import defaultdict


video_path = "inferance/test.mp4"
model_path = "models/yolov8n.pt"

cap = cv2.VideoCapture(video_path)
model = YOLO(model_path)

width = 1280
height = 720

fourcc = cv2.VideoWriter_fourcc(*"XVID")
writer = cv2.VideoWriter("result/video.avi", fourcc, 20.0, (width, height))


color_green = (0,255,0)
color_red = (0,0,255)
color_white = (255,255,255)

thickness=2
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale=0.5

vehicle_ids = [2, 3, 5, 7]

track_history = defaultdict(lambda: [])

down = {}
up = {}

threshold = 450


while True:
    ret, frame = cap.read()
    
    if ret == False:
        break
    
    frame = imutils.resize(frame, width=1280)
    # Verbose ekstra bilgi almak istemediğimiz için false değerini aldı
    # persist frameler arasında ki nesnenin takibini sağlar
    results = model.track(frame, persist=True, verbose=False)[0]
    
    # track_ids = results.boxes.id.int().cpu().tolist()
    bboxes = np.array(results.boxes.data.tolist(), dtype="int")
    
    #referance line
    cv2.line(frame, (0,450),(1280,450),color_green,thickness)
    
    cv2.putText(frame, "Referance Line", (620, 445), font, 0.7, color_red, thickness)

    
    for box in bboxes:
        x1, y1, x2, y2, track_id, score, class_id = box
        
        cx = int((x1 + x2)/2)
        cy = int((y1 + y2)/2)
        if class_id in vehicle_ids:
            
            class_name = results.names[int(class_id)].upper() # car => CAR

            # print("BBoxes: ", (x1, y1, x2, y2))
            # print("Class: ", class_name)
            # print("ID: ", track_id)

            track = track_history[track_id]
            track.append((cx, cy))

            if len(track) > 15:
                track.pop(0)
            
            points = np.hstack(track).astype("int32").reshape((-1,1,2))
            cv2.polylines(frame, [points], isClosed=False, color=color_green, thickness=thickness)
            
            
            text = "ID: {}  {}".format(track_id, class_name)
            cv2.putText(frame, text, (x1, y1-5), font, fontScale, color_green, thickness)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_green, thickness)

    
    
            if cy > threshold - 5 and cy < threshold + 5 and cx < 670:
                down[track_id] = x1, y1, x2, y2
            
            
            if cy > threshold - 5 and cy < threshold + 5 and cx > 670:     
                up[track_id] = x1, y1, x2, y2
    
            
        #print("Down Dictionary Keys: ", list(down.keys()))
        #print("Up Dictionary Keys: ", list(up.keys()))

        up_text = "Going:{}".format(len(list(up.keys())))
        down_text = "Coming:{}".format(len(list(down.keys())))
    
        cv2.putText(frame, up_text, (1200, threshold-5), font, fontScale, color_white, thickness)
        cv2.putText(frame, down_text, (0, threshold-5), font, fontScale, color_white, thickness)
    
    
    writer.write(frame)
    cv2.imshow("Test", frame)
    
    if cv2.waitKey(10) & 0xFF==ord("q"):
        break
    
    

cap.release()
writer.release()
cv2.destroyAllWindows()