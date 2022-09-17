import cv2
import numpy as np
import face_recognition

#import images

imgElon = face_recognition.load_image_file('images/Elon_Musk 1.jpg')
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('images/ELON MUSK.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

#rectangle line in the face

faceLoc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

#rectangle line in the face

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

#checking if both the images scale are same

results = face_recognition.compare_faces([encodeElon],encodeTest)
faceDis=face_recognition.face_distance([encodeElon],encodeTest)
print(results,faceDis)

#font size and color

cv2.putText(imgTest,f'{results}{round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,225),2)

cv2.imshow('Elon_Musk 1',imgElon)
cv2.imshow('ELON MUSK',imgTest)
cv2.waitKey(0)