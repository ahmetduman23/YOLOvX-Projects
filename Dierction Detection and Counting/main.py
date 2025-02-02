import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import cv2
import numpy as np 
from ultralytics import YOLO
import imutils 
from collections import defaultdict


blue = (255,0,0)
green = (0,255,0)
red = (0,0,255)
black = (0,0,0)
white = (255,255,255)

font_scale = 0.6
thickness = 2
font = cv2.FONT_HERSHEY_SIMPLEX


width = 1280
height = 720

fourcc = cv2.VideoWriter_fourcc(*"XVID")
writer = cv2.VideoWriter("result/video2.avi", fourcc, 20.0, (width, height))

up = {}
down = {}
right = {}
left = {}

vehicle_ids = [2, 3, 5, 7]

polygon_up = np.array([[467,295],[578,370],
                        [710,361],[559,283]],np.int32)

polygon_down = np.array([[770,600],[506,645],
                        [551,711],[864,707]],np.int32)

polygon_left = np.array([[107,451],[213,438],
                        [220,469],[85,481]],np.int32)

polygon_right = np.array([[990,453],[1262,404],
                        [1276,467],[1072,506]],np.int32)


video_path = "inferance/intersection.mp4"
cap = cv2.VideoCapture(video_path)

model_path = "models/yolov8n.pt"
model = YOLO(model_path)

track_history = defaultdict(lambda: [])

while True:
    ret, frame = cap.read()
    if ret==False:
        break
    
    frame = imutils.resize(frame, width=1280)

    cv2.polylines(frame, [polygon_up], isClosed=True, color=green, thickness=thickness)
    cv2.polylines(frame, [polygon_down], isClosed=True, color=green, thickness=thickness)
    cv2.polylines(frame, [polygon_left], isClosed=True, color=green, thickness=thickness)
    cv2.polylines(frame, [polygon_right], isClosed=True, color=green, thickness=thickness)

    results = model.track(frame, persist=True, verbose=False)[0]
    
    bboxes = np.array(results.boxes.data.tolist(), dtype="int")
    
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
            cv2.polylines(frame, [points], isClosed=False, color=green, thickness=thickness)

            up_result = cv2.pointPolygonTest(polygon_up, (cx,cy), measureDist=False)
            down_result = cv2.pointPolygonTest(polygon_down, (cx,cy), measureDist=False)
            left_result = cv2.pointPolygonTest(polygon_left, (cx,cy), measureDist=False)
            right_result = cv2.pointPolygonTest(polygon_right, (cx,cy), measureDist=False)

            if up_result > 0:
                #print("UP !")
                up[track_id] = x1, y1, x2, y2
    
            if down_result > 0:
                #print("DOWN !")
                down[track_id] = x1, y1, x2, y2
                
            if left_result > 0:
                #print("LEFT !")
                left[track_id] = x1, y1, x2, y2

            if right_result > 0:
                #print("RIGHT !")
                right[track_id] = x1, y1, x2, y2
    
    
    # print("Up Direction Counter:", len(list(up.keys())))
    # print("Down Direction Counter:", len(list(down.keys())))
    # print("Left Direction Counter:", len(list(left.keys())))
    # print("Right Direction Counter:", len(list(right.keys())))

    up_counter_text = "Up Direction Counter: {}".format(str(len(list(up.keys()))))
    down_counter_text = "Down Direction Counter: {}".format(str(len(list(down.keys()))))
    left_counter_text = "Left Direction Counter: {}".format(str(len(list(left.keys()))))
    right_counter_text = "Right Direction Counter: {}".format(str(len(list(right.keys()))))

    
    cv2.rectangle(frame, (0, 0), (350,120), white, -1)
    cv2.putText(frame, up_counter_text, (10, 25), font, 0.8, black, thickness)
    cv2.putText(frame, down_counter_text, (10, 50), font, 0.8, black, thickness)
    cv2.putText(frame, left_counter_text, (10, 75), font, 0.8, black, thickness)
    cv2.putText(frame, right_counter_text, (10, 100), font, 0.8, black, thickness)

    writer.write(frame)
    cv2.imshow("Test", frame)
    
    if cv2.waitKey(10) & 0xFF==ord("q"):
        break
    
cap.release()
writer.release()
cv2.destroyAllWindows()