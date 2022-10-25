#This is a program based on a trained YOLOv4(by the author of it) network, and DNN function in CV2

##################### N.B. #####################
# recomended to clear "PNGresults" before test #
# to avoid potential errors in real time       #
################################################

import cv2
from object_detection import ObjectDetection

od = ObjectDetection()

#load videos
cap = cv2.VideoCapture("test.mp4")

number=0 #this counts of the number of frames, is used at naming outputs 
while True:
    # capture a picture form video
    _, frame = cap.read()

    # run the network, get the outputs 
    (class_ids, scores, boxes) = od.detect(frame)

    #open classes name file
    file=open("classes.txt")
    classes=file.readlines()
    file.close()

    #Print the boxes
    for box in boxes:
        for id in class_ids:

            #1.get datas to put
            (x, y, w, h) = box
            text=classes[id]

            #2.plot them on figure
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame,text,(x, y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
            

    #save images
    number=number+1
    savePath='PNGresults/'+'%s.jpg' %(str(number))
    cv2.imwrite(savePath,frame)
    print('%s saved' %(str(number)))
            
    #show output
    cv2.imshow("output", frame)
    

    #end loop
    k=cv2.waitKey(1)
    if k==27:
        break

#end program
cap.release()
cv2.destroyAllWindows()