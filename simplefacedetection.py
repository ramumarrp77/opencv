# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 16:44:25 2019

@author: Ram Kumar R P
"""
#importing library
import cv2

#loading cascades

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

#defining a function that will do detections

def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
         cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
         roi_gray = gray[y:y+h, x:x+w]
         roi_color= frame[y:y+h,x:x+w]
         eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
         for (x1,y1,w1,h1) in eyes:
             cv2.rectangle(roi_color, (x1,y1), (x1+w1,y1+h1), (255,255,0), 2)
    return frame

#doing some Face Recognition with webcam

video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    gray =cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray,frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()