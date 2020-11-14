import numpy as np
import cv2
import pickle
import urllib.request
import urllib
from urllib.request import urlopen
url='http://192.168.43.1:8080/video'

face_cascade =cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")
labels = {"person_name" : 1}
with open("labels.pickle",'rb') as f:
    og_labels = pickle.load(f)
    labels ={v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(url)

while(True):
    
    ret,frame = cap.read()
    frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
    gray =cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces =face_cascade.detectMultiScale(gray,scaleFactor = 1.5,minNeighbors=5)

    for(x,y,w,h) in faces:
        print(x,y,w,h)
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]

        id_,conf =recognizer.predict(roi_gray)
        if conf>=45:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name= labels[id_]
            color =(255,255,255)
            stroke =2
            cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)

        color =(255,0,0)
        stroke = 2
        width=x+w
        height = y+h
        cv2.rectangle(frame,(x,y),(width,height),color,stroke)
        
    cv2.imshow('frame',frame)
    
    if cv2.waitKey(20) &0xff ==ord('q'):
        break
cv2.destroyAllWindows()
