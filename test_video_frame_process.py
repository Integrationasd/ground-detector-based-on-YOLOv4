#This is a program based on a trained YOLOv4(by the author) network, and DNN function in CV2

##################### N.B. #####################
# recomended to clear "JPGresults" before test #
# to avoid potential errors in real time       #
################################################

from itertools import product
import cv2
import time
from numpy import round
from object_detection import ObjectDetection

od = ObjectDetection()

# video path, edit the input path here
input_path='test.mp4'
# output path
output_path='results/result_car.mp4'

#load videos
cap = cv2.VideoCapture(input_path)

#load save video
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width=int(cap.get(3))
frame_height=int(cap.get(4))
output_video = cv2.VideoWriter(output_path, 0x00000021, fps, (frame_width, frame_height))

while True:
    
    # capture a picture form video
    ret, frame = cap.read()
    
    if not ret:
        print("An Error in reading frame")
        break

    # run the network, get the outputs 
    (class_ids, scores, boxes) = od.detect(frame)
    scores=round(scores,2)

    #Print the boxes n ids
    for box, ids in product(boxes, class_ids):
        
         #1.get datas to put
        (x, y, w, h) = box
        # text='code: %s' %(str(ids))

        #2.plot them on figure
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.putText(frame,text,(x, y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
            
    #save frame
    output_video.write(frame)
            
    #show output
    cv2.imshow("output", frame)

    #end loop,press esc to exit
    k=cv2.waitKey(1)
    if k==27:
        break
    

#end program
output_video.release()
cap.release()
cv2.destroyAllWindows()