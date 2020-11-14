import numpy as np
import cv2


face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)
face_id = input('Enter User Name ')
count = 0
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,0), 2)     
        count += 1
        if(count%2==0):
            cv2.imwrite("dataset/" + str(face_id) +str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('image', img)
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    elif count >= 400: 
         break
cam.release()
cv2.destroyAllWindows()
