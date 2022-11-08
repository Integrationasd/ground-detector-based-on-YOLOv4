from itertools import product
import cv2
import time
from numpy import round
from object_detection import ObjectDetection

od = ObjectDetection()

# video path, from the camera
input_path = 'http://admin:admin@192.168.137.199:8081/'
# output path
output_path = 'results/result.mp4'

#load video flow
cap = cv2.VideoCapture(input_path)

#load save video
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width=int(cap.get(3))
frame_height=int(cap.get(4))
output_video = cv2.VideoWriter(output_path, 0x00000021, fps, (frame_width, frame_height))

number = 0 #this counts of the number of frames, is used at naming outputs 
frame_num = 1 # the number of frame
while True:
    #load video flow
    cap = cv2.VideoCapture(input_path)
    ret, frame = cap.read()
        
    # run the network, get the outputs 
    (class_ids, scores, boxes) = od.detect(frame)

    #Print the boxes n ids
    # for box, ids in product(boxes, class_ids):
    for box in boxes:
        
         #1.get datas to put
        (x, y, w, h) = box

        #2.plot them on figure
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
    #show output
    cv2.imshow("output", frame)
    
    #save frame
    output_video.write(frame)

    #end loop,press esc to exit
    k=cv2.waitKey(1)
    if k==27:
        break
    

#end program
cap.release()
output_video.release()
cv2.destroyAllWindows()
