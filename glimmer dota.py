import cv2
import time
import numpy as np

def kamera(x):
    pass

cap = cv2.VideoCapture(0)
cv2.namedWindow("Trackbars")
kernel = np.ones((5,5), np.uint8)

cv2.createTrackbar("L - H", "Trackbars", 0, 255, kamera)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, kamera)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, kamera)
cv2.createTrackbar("U - H", "Trackbars", 255, 255, kamera)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, kamera)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, kamera)

time.sleep(3)
background = 0
for i in range (30):
    ret, background=cap.read()

background = np.flip(background, axis=1)

while(1):
    ret, img=cap.read()
    img = np.flip(img,axis=1)

    #Convert from BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")
    lower = np.array([l_h, l_s, l_v])
    upper = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv, lower, upper)
    #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
    mask2=cv2.bitwise_not(mask)

    res = cv2.bitwise_and(img, img, mask=mask2)
    res2 = cv2.bitwise_and(background, background, mask = mask)
 
    #Generating the final output
    final_output = cv2.addWeighted(res,1,res2,1,0)
    cv2.imshow("mask", mask2)
    cv2.imshow("Hasil", res)
    cv2.imshow("magic",final_output)
    k = cv2.waitKey(5) & 0xFF
    if k== 27:
        break

cap.release()
cv2.destroyAllWindows()
