import cv2
import numpy as np 
import os
from tensorflow.keras.models import load_model
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input
import base64
from io import BytesIO
import json
import random
import numpy as np

face_detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def extractor(img):
	faces=face_detector.detectMultiScale(img,1.3,5)
	if len(faces)==0:
		return None

	for (x,y,w,h) in faces:
		x=x-10
		y=y-10
		cropped_face=img[y:y+h+50,x:x+w+50]
		cv2.rectangle(frame,(x,y),(x+w+50,y+h+50),(0,255,255),2)

	return cropped_face

cap=cv2.VideoCapture(0)
model=load_model('facefeatures_new_model.h5')

path='./Images/Train'
os.chdir(path)
print(os.listdir())

i=0
names={}
for name in os.listdir():
	names[i]=name
	i+=1

while True:
	ret,frame=cap.read()
	face=extractor(frame)
	if type(face) is np.ndarray:
	    face = cv2.resize(face, (224, 224))
	    im = Image.fromarray(face, 'RGB')
	       #Resizing into 128x128 because we trained the model with this image size.
	    img_array = np.array(im)
	                #Our keras model used a 4D tensor, (images x height x width x channel)
	                #So changing dimension 128x128x3 into 1x128x128x3 
	    img_array = np.expand_dims(img_array, axis=0)
	    pred = model.predict(img_array)
	    print(pred)
	    num=np.argmax(pred[0])
	    cv2.putText(frame,names[num], (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
	else:
	    cv2.putText(frame,"No face found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
	cv2.imshow('Video', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
	    break

cap.release()
cv2.destroyAllWindows()
print('Completed Face Image Collection')

