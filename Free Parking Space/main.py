import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import cv2

import numpy as np 
from ultralytics import YOLO
import imutils 

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
writer = cv2.VideoWriter("result/video.avi", fourcc, 20.0, (width, height))


polygon_1_dict = {}
polygon_2_dict = {}
polygon_3_dict = {}
polygon_4_dict = {}
polygon_5_dict = {}
polygon_6_dict = {}
polygon_7_dict = {}
polygon_8_dict = {}
polygon_9_dict = {}
polygon_10_dict = {}
polygon_11_dict = {}
polygon_12_dict = {}
polygon_13_dict = {}
polygon_14_dict = {}





vehicle_ids = [2, 3, 5, 7]

video_path = "inferance/test.mp4"
cap = cv2.VideoCapture(video_path)

model_path = "models/yolo12n.pt"
model = YOLO(model_path)

polygon_1 = np.array([[355,519],[421,464],
                        [474,466],[427,527]],np.int32)

polygon_2 = np.array([[446,532],[489,467],
                        [555,470],[525,536]],np.int32)


polygon_3 = np.array([[630,528],[635,468],
                        [752,464],[782,524]],np.int32)

polygon_4 = np.array([[794,526],[765,460],
                        [825,451],[880,519]],np.int32)

polygon_5 = np.array([[893,516],[843,456],
                        [902,454],[970,508]],np.int32)

polygon_6 = np.array([[1184,474],[1121,426],
                        [1065,436],[1127,487]],np.int32)

polygon_7 = np.array([[1057,495],[999,438],
                        [1055,429],[1121,492]],np.int32)

polygon_8 = np.array([[983,507],[916,456],
                        [982,436],[1052,500]],np.int32)

polygon_9 = np.array([[249,421],[294,399],
                        [352,400],[311,422]],np.int32)

polygon_10 = np.array([[319,424],[359,400],
                        [394,400],[352,425]],np.int32)

polygon_11 = np.array([[361,426],[402,400],
                        [439,398],[402,424]],np.int32)

polygon_12 = np.array([[411,424],[446,399],
                        [486,400],[451,425]],np.int32)

polygon_13 = np.array([[461,428],[495,400],
                        [533,399],[510,425]],np.int32)

polygon_14 = np.array([[524,424],[546,391],
                        [584,392],[568,422]],np.int32)

free_space_counter = []


while True:
    ret, frame = cap.read()
    if ret==False:
        break
    
    frame = imutils.resize(frame, width=1280)
    frame_copy = frame.copy()
    cv2.polylines(frame_copy, [polygon_1], isClosed=True, color=green, thickness=thickness)
    cv2.polylines(frame_copy, [polygon_2], isClosed=True, color=green, thickness=thickness)
    cv2.polylines(frame_copy, [polygon_3], isClosed=True, color=green, thickness=thickness)
    cv2.polylines(frame_copy, [polygon_4], isClosed=True, color=green, thickness=thickness)
    cv2.polylines(frame_copy, [polygon_5], isClosed=True, color=green, thickness=thickness)
    cv2.polylines(frame_copy, [polygon_6], isClosed=True, color=green, thickness=thickness)
    cv2.polylines(frame_copy, [polygon_7], isClosed=True, color=green, thickness=thickness)
    cv2.polylines(frame_copy, [polygon_8], isClosed=True, color=green, thickness=thickness)
    cv2.polylines(frame_copy, [polygon_9], isClosed=True, color=green, thickness=thickness)
    cv2.polylines(frame_copy, [polygon_10], isClosed=True, color=green, thickness=thickness)
    cv2.polylines(frame_copy, [polygon_11], isClosed=True, color=green, thickness=thickness)
    cv2.polylines(frame_copy, [polygon_12], isClosed=True, color=green, thickness=thickness)
    cv2.polylines(frame_copy, [polygon_13], isClosed=True, color=green, thickness=thickness)
    cv2.polylines(frame_copy, [polygon_14], isClosed=True, color=green, thickness=thickness)


    results = model.track(frame, persist=True, verbose=False)[0]
    
    bboxes = np.array(results.boxes.data.tolist(), dtype="int")
    
    for box in bboxes:
        
        if len(box) == 7:  # Eğer 7 değer varsa normal şekilde al
            x1, y1, x2, y2, track_id, score, class_id = box
        elif len(box) == 6:  # Eğer sadece 6 değer varsa, track_id yoktur, varsayılan olarak None ata
            x1, y1, x2, y2, score, class_id = box
            track_id = None 
        
        cx = int((x1 + x2)/2)
        cy = int((y1 + y2)/2)
        if class_id in vehicle_ids:
            cv2.circle(frame_copy, (cx,cy), 3, blue, -1)
            cv2.rectangle(frame_copy, (x1,y1),(x2,y2), blue,thickness=1)
            
            polygon_1_result = cv2.pointPolygonTest(polygon_1, (cx,cy), measureDist=False)
            polygon_2_result = cv2.pointPolygonTest(polygon_2, (cx,cy), measureDist=False)
            polygon_3_result = cv2.pointPolygonTest(polygon_3, (cx,cy), measureDist=False)
            polygon_4_result = cv2.pointPolygonTest(polygon_4, (cx,cy), measureDist=False)
            polygon_5_result = cv2.pointPolygonTest(polygon_5, (cx,cy), measureDist=False)
            polygon_6_result = cv2.pointPolygonTest(polygon_6, (cx,cy), measureDist=False)
            polygon_7_result = cv2.pointPolygonTest(polygon_7, (cx,cy), measureDist=False)
            polygon_8_result = cv2.pointPolygonTest(polygon_8, (cx,cy), measureDist=False)
            polygon_9_result = cv2.pointPolygonTest(polygon_9, (cx,cy), measureDist=False)
            polygon_10_result = cv2.pointPolygonTest(polygon_10, (cx,cy), measureDist=False)
            polygon_11_result = cv2.pointPolygonTest(polygon_11, (cx,cy), measureDist=False)
            polygon_12_result = cv2.pointPolygonTest(polygon_12, (cx,cy), measureDist=False)
            polygon_13_result = cv2.pointPolygonTest(polygon_13, (cx,cy), measureDist=False)
            polygon_14_result = cv2.pointPolygonTest(polygon_14, (cx,cy), measureDist=False)


            if polygon_1_result >= 0:
                print("Polygon 1!")
                polygon_1_dict[track_id] = x1, y1, x2, y2
                free_space_counter.append(polygon_1_dict)
                #cv2.putText(frame_copy, "Not Free", (cx, cy), font, 0.8, red, thickness)
                
            if polygon_2_result >= 0:
                print("Polygon 2!")
                polygon_2_dict[track_id] = x1, y1, x2, y2
                free_space_counter.append(polygon_2_dict)
                #cv2.putText(frame_copy, "Not Free", (cx, cy), font, 0.8, red, thickness)

                
            if polygon_3_result >= 0:
                print("Polygon 3!"),
                polygon_3_dict[track_id] = x1, y1, x2, y2
                free_space_counter.append(polygon_3_dict)
                #cv2.putText(frame_copy, "Not Free", (cx, cy), font, 0.8, red, thickness)
                
            if polygon_4_result >= 0:
                print("Polygon 4!")
                polygon_4_dict[track_id] = x1, y1, x2, y2
                free_space_counter.append(polygon_4_dict)
                #cv2.putText(frame_copy, "Not Free", (cx, cy), font, 0.8, red, thickness)

            if polygon_5_result >= 0:
                print("Polygon 5!")
                polygon_5_dict[track_id] = x1, y1, x2, y2
                free_space_counter.append(polygon_5_dict)
                #cv2.putText(frame_copy, "Not Free", (cx, cy), font, 0.8, red, thickness)

            if polygon_6_result >= 0:
                print("Polygon 6!")
                polygon_6_dict[track_id] = x1, y1, x2, y2
                free_space_counter.append(polygon_6_dict)
                #cv2.putText(frame_copy, "Not Free", (cx, cy), font, 0.8, red, thickness)

            if polygon_7_result >= 0:
                print("Polygon 7!")
                polygon_7_dict[track_id] = x1, y1, x2, y2
                free_space_counter.append(polygon_7_dict)
                #cv2.putText(frame_copy, "Not Free", (cx, cy), font, 0.8, red, thickness)

            if polygon_8_result >= 0:
                print("Polygon 8!")
                polygon_8_dict[track_id] = x1, y1, x2, y2
                free_space_counter.append(polygon_8_dict)
                #cv2.putText(frame_copy, "Not Free", (cx, cy), font, 0.8, red, thickness)

            if polygon_9_result >= 0:
                print("Polygon 9!")
                polygon_9_dict[track_id] = x1, y1, x2, y2
                free_space_counter.append(polygon_9_dict)
                #cv2.putText(frame_copy, "Not Free", (cx, cy), font, 0.8, red, thickness)

            if polygon_10_result >= 0:
                print("Polygon 10!")
                polygon_10_dict[track_id] = x1, y1, x2, y2
                free_space_counter.append(polygon_10_dict)
                #cv2.putText(frame_copy, "Not Free", (cx, cy), font, 0.8, red, thickness)

            if polygon_11_result >= 0:
                print("Polygon 11!")
                polygon_11_dict[track_id] = x1, y1, x2, y2
                free_space_counter.append(polygon_11_dict)
                #cv2.putText(frame_copy, "Not Free", (cx, cy), font, 0.8, red, thickness)

            if polygon_12_result >= 0:
                print("Polygon 12!")
                polygon_12_dict[track_id] = x1, y1, x2, y2
                free_space_counter.append(polygon_12_dict)
                #cv2.putText(frame_copy, "Not Free", (cx, cy), font, 0.8, red, thickness)

            if polygon_13_result >= 0:
                print("Polygon 13!")
                polygon_13_dict[track_id] = x1, y1, x2, y2
                free_space_counter.append(polygon_13_dict)
                #cv2.putText(frame_copy, "Not Free", (cx, cy), font, 0.8, red, thickness)

            if polygon_14_result >= 0:
                print("Polygon 14!")
                polygon_14_dict[track_id] = x1, y1, x2, y2
                free_space_counter.append(polygon_14_dict)
                #cv2.putText(frame_copy, "Not Free", (cx, cy), font, 0.8, red, thickness)

    free_lot = 14 - len(free_space_counter)
    free_space_counter.clear()
    
    text = "Free Parking Lot: {}".format(free_lot)
    cv2.putText(frame_copy, text, (10,50), font, 0.8, black, thickness)
    
    
    
    writer.write(frame_copy)
    cv2.imshow("Test", frame_copy)
    
    if cv2.waitKey(10) & 0xFF==ord("q"):
        break
    
cap.release()
writer.release()
cv2.destroyAllWindows()
