import cv2
import numpy as np
import pickle
import os

video = cv2.VideoCapture(1)
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces_data =[]
name =input("Enter user Name:")
i=0;
while True:
    ret, frame =  video.read()
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(grey, 1.3, 5 )
    for (x, y, w, h) in faces:
        crop_image = frame[y:y+h, x:x+h, :]
        resize_image = cv2.resize(crop_image, (50,50))
        if len(faces_data) <=100: and i%10==0:
            faces_data.append(resize_image)
        i = i+1
        cv2.putText(frame, str(len(faces_data)), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(50,50,255), 1 )    
        cv2.rectangle(frame,(x,y), (x+w, y+h),(50,50,255),1 )
    cv2.imshow("Attendance", frame)
    k = cv2.waitKey(1)
    if k == ord('q') or len(faces_data)==100:
        break
video.release()
cv2.destroyAllWindows    

faces_data = np.asarray(faces_data)
faces_data  = faces_data.reshape(100, -1)

if 'names.pkl' not in os.listdir('data/'):
    names=[name]*100
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    with open('data/names.pkl', 'rb') as f:
        names=pickle.load(f)
    names=names+[name]*100
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)

if 'faces_data.pkl' not in os.listdir('data/'):
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)
else:
    with open('data/faces_data.pkl', 'rb') as f:
        faces=pickle.load(f)
    faces=np.append(faces, faces_data, axis=0)
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces, f)


