import cv2
import numpy as np 
import os

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

parent_dir="./Images/"
mode=int(input("Train Images(0) or Test Images(1)?"))
subdir=''
if mode==0:
	subdir='Train'
else:
	subdir='Test'
path=os.path.join(parent_dir,subdir)
os.chdir(path)
print(os.getcwd())
name=input("Enter Name : ")
folder_path=os.getcwd()+'/'+name
os.mkdir(folder_path)
os.chdir(folder_path)
print(os.getcwd())


cap=cv2.VideoCapture(0)
count=0

while True:
	ret,frame=cap.read()
	face=extractor(frame)
	if face is not None:
		count+=1
		face=cv2.resize(face,(400,400))

		file_path=name+str(count)+'.jpg'
		cv2.imwrite(file_path,face)
	else:
		print("Face Not Detected")
		pass
	cv2.imshow("Frame",frame)
	#cv2.imshow("Face Section",face)
	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q') or count==50:
		break

cap.release()
cv2.destroyAllWindows()
print('Completed Face Image Collection')

