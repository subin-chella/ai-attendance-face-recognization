from sklearn.neighbors import KNeighborsClassifier
import cv2
import numpy as np
import pickle
import os

video = cv2.VideoCapture(1)
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
with open( 'data/names.pkl', 'rb') as f:
    LABEL = pickle.load(f)
with open( 'data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

knn = KNeighborsClassifier(n_neighbors=5)        
knn.fit(FACES, LABEL)


while True:
    ret, frame =  video.read()
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(grey, 1.3, 5 )
    for (x, y, w, h) in faces:
        crop_image = frame[y:y+h, x:x+h, :]
        resize_image = cv2.resize(crop_image, (50,50)).flatten().reshape(1,-1)
        output = knn.predict(resize_image)
        cv2.putText(frame, str(output[0]), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(50,50,255), 1 )    
        cv2.rectangle(frame,(x,y), (x+w, y+h),(50,50,255),1 )
    cv2.imshow("Attendance", frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
video.release()
cv2.destroyAllWindows    




