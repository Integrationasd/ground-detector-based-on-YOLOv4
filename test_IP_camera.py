import cv2
import time

video = "http://admin:admin@192.168.137.199:8081/"
capture = cv2.VideoCapture(video)

num = 0;
while True:

    ret,img = capture.read()
    
    print("")
    if not ret:
        print("An Error in reading frame")
        break
    cv2.imshow("AIRCRAFT_camera", img)
    
    key =cv2.waitKey(1)
    if key == 27:
        print("esc break...")
        break

capture.release()
cv2.destroyWindow("camera")
