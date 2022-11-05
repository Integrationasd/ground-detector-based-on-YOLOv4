#This is a program based on a trained YOLOv4(by the author of it) network, and DNN function in CV2

##################### N.B. #####################
# recomended to clear "PNGresults" before test #
# to avoid potential errors in real time       #
################################################

from itertools import product
import cv2
from numpy import round
from object_detection import ObjectDetection

od = ObjectDetection()

# video path, edit the input path here
input_path='test.mp4'
# output path
output_path='PNGresults'

#load videos
cap = cv2.VideoCapture(input_path)

number=0 #this counts of the number of frames, is used at naming outputs 
while True:
    # capture a picture form video
    ret, frame = cap.read()
    
    # run the network, get the outputs 
    (class_ids, scores, boxes) = od.detect(frame)
    scores=round(scores,2)

    #Print the boxes n ids
    for box, ids in product(boxes, class_ids):
         #1.get datas to put
        (x, y, w, h) = box
        text='code: %s' %(str(ids))

        #2.plot them on figure
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame,text,(x, y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
            
    #save images
    number=number+1
    savePath='%s\%s.jpg' %(output_path, str(number))

    cv2.imwrite(savePath,frame)
    print('figure No. %s saved' %(str(number)))
            
    #show output
    cv2.imshow("output", frame)

    #end loop,press esc to exit
    k=cv2.waitKey(1)
    if k==27:
        break
    

#end program
cap.release()
cv2.destroyAllWindows()
