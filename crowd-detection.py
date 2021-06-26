import numpy as np
import cv2
import os
import time
from scipy.spatial import distance as distance
import cmath


labelpath=r'darknet/data/coco.names'
file=open(labelpath)
label=file.read().strip().split("\n")

weightspath=r'darknet/cfg/yolov3.weights'
configpath=r'darknet/cfg/yolov3.cfg'

net=cv2.dnn.readNetFromDarknet(configpath,weightspath)

ln=net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

vcap = cv2.VideoCapture("rtsp://admin:SGFMCU@10.50.0.28:554/H.264")
init=time.time()
while(1):
    ret, frame = vcap.read()
    frame = cv2.resize(frame, (640,440), interpolation =cv2.INTER_AREA)
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    r = blob[0, 0, :, :]
    net.setInput(blob)
    t0 = time.time()
    outputs = net.forward(ln)
    t = time.time()

    boxes = []
    confidences = []
    classIDs = []
    center=[]
    output=[]
    count=0
    results=[]
    breach=set()
    
    h, w = frame.shape[:2]
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
           
            confidence = scores[classID]
           
            if confidence > 0.5:
                box = detection[0:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                center.append((centerX,centerY))
                box = [x, y, int(width), int(height)]
                boxes.append(box)
                confidences.append(float(confidence))
                classIDs.append(classID)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            #color = [int(c) for c in colors[classIDs[i]]]
            if(label[classIDs[i]]=='person'):
                #people()
                cX=(int)(x+(y/2))
                cY=(int)(w+(h/2))
                center.append((cX,cY))
                res=((x,y,x+w,y+h),center[i])
                results.append(res) 
                dist=cmath.sqrt(((center[i][0]-center[i+1][0])**2)+((center[i][1]-center[i+1][1])**2))
                if(dist.real <50):
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255), 2)
                    cv2.circle(frame,center[i],4,(0,0,255),-1)
                    cv2.line(frame, (center[i][0], center[i][1]), (center[i+1][0], center[i+1][1]), (0,0, 255), thickness=3, lineType=8)
                    count=count+1
                else:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0), 2)
                    cv2.circle(frame,center[i],4,(0,255,0),-1)
        #cv2.rectangle(frame,(startX, startY), (endX, endY),color, 2)
        #cv2.circle(frame,(cX,cY),4,color,-1)
        #cv2.putText(frame,"Violation: {}".format(count), (20, frame.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (23,255,255), 1) 
    cv2.imshow('VIDEO', frame)
    cv2.waitKey(1)