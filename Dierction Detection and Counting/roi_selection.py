import cv2 
import numpy as np
import imutils 


points = []

font = cv2.FONT_HERSHEY_SIMPLEX


video_path = "inferance/intersection.mp4"
cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()
frame = imutils.resize(frame, width=1280)

window_name = "ROI"
cv2.namedWindow(window_name)



def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print("x:{} y:{}".format(x,y))
        
        for i in range(len(points)-1):
            cv2.circle(frame, points[i], 5, (0,0,255), -1)
            cv2.putText(frame, str(points[i]), points[i], font, 0.5, (0,0,255), 2)
            cv2.putText(frame, str(points[i+1]), points[i+1], font, 0.5, (0,0,255), 2)
            

        cv2.imshow(window_name, frame)

cv2.setMouseCallback(window_name, mouse_callback)

while True:
    cv2.imshow(window_name, frame)
    key = cv2.waitKey(1)
    
    if key == 27:
        cv2.imwrite("roi.png",frame)
        break


cv2.destroyAllWindows()


