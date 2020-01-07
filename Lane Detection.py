import math
import cv2
import time
import numpy as np

def getROI(frame):
    upper_left  = (50, 50)
    bottom_right = (720, 480)
    rect_frame = frame[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]
    return rect_frame

cap=cv2.VideoCapture(0)

while True:
    theta = 0 
    success, frame= cap.read()
    time.sleep(0.3)
    image  = getROI(frame)
    gray=cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blurred=cv2.GaussianBlur(gray,(15,15),0)
    edged = cv2.Canny(blurred,40,120)
    lines = cv2.HoughLinesP(edges, 10, np.pi/180, 15, 5, 10)
    
    if lines is not None:
        for x in range(0, len(lines)):
            for x1, y1, x2, y2 in lines[x]:
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                theta = theta + math.atan2((y2-y1), (x2-x1))
            
    threshold = 5
    print("theta: ", theta)
    print("threshold: ", threshold)

    cv2.imshow('Input', blurred)
    cv2.imshow('Canny detected', edged)
    cv2.imshow('Output', frame)
    
    if(theta > threshold):
        print("left")
    if(theta < -threshold):
        print("right")
    if(abs(theta) < threshold):
        print("straight")

    k=cv2.waitKey(5) & 0xFF
    if exit == ord('q')
        break

cv2.destroyAllWindows()
