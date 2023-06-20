#import opencv library
import cv2
# Import YOLO
from ultralytics import YOLO
#Import Numpy
import numpy as np

#To access the gpu, we need to enable the mps backend from pytorch


# Now to load the video, we need a capture object. So use cv2.videoCapture("path of video")
cap = cv2.VideoCapture("Bugatti.mp4")

#Now downlaod the object detection model from yolo
model = YOLO("yolov8m.pt ")

while True : 
    #Now lets take frames from this video. We use the .read function 
    ret, frame = cap.read()
    #ret : the read function returns true or false, wether the frame exists or not.
    #frame : if frame exists it returns the frame 
    
    #break the loop if no frame exists
    if not ret:
        break
    
    #To detect the model, pass the frame inside the model
    #Also tell the ultralytics that mps is available so use it
    results = model(frame,device="mps")
    #Extract the results
    result = results[0]
    
    # From result access the Bounding boxes
    bboxes  = np.array(result.boxes.xyxy.cpu(),dtype="int")
    #We get the x1,y1 and x2,y2 coordinates which we can use to construct the bounding boxes
    #But we observe that this values are tensors from pytorch which we cannot use  it in opencv. so we use the .cpu abaove and convert into numpy array
   
    #Lets extract the coordinates from Bounding boxes
    #We alaos add tehcalss of the objects
    classes = np.array(result.boxes.cls.cpu(),dtype="int")
    #Extracting 2 arrays at the same time so .zip method
    for cls,bboox in zip(classes,bboxes):
        (x,y,x2,y2) = bboox
        
        #Now create a Cv rectangle using these coordinates
        cv2.rectangle(frame,(x,y),(x2,y2),(0,0,225),2)
        #Also put class name on the bounding boxes
        cv2.putText(frame,str(cls),(x,y-5),cv2.FONT_HERSHEY_PLAIN,1,(0,0,225),2)
        
       
    
    
    
    
    # To display the frame on the screen use cv2.imshow("window",frame)
    cv2.imshow("Img",frame)
    # But thsi executes pretty quickly and we cannot see. So set a key wait time i.e. display till key is pressed
    #waitkey value is set to 1 i.e. it will wait for 1 milli sec and go to next frame
    key = cv2.waitKey(1)
    #Again we observe that we only get one frame here. But a video is nothing but a collection of frames. So we need to take many frames in a loop. So a while loop in line 7
    
    #to stop video on pressing esc
    if key==27:
        break
    
# Once we used the videos, once we are done using the video, we need to release the cap object, so that it doesnt hol dthe video anymore
cap.release()
#Close any windows that might be open
cv2.destroyAllWindows()


 
    