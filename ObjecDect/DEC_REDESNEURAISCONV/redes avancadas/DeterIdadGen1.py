#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pathlib import Path
import cv2
import dlib
import sys
import numpy as np
import argparse
from contextlib import contextmanager
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file

# Load our cassade classifier for faces
face_classifier = cv2.CascadeClassifier('./Haarcascades/haarcascade_frontalface_default.xml')

# Load our pretrained model for Gender and Age Detection
pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.28-3.73.hdf5"
modhash = 'fbe63257a054c1c5466cfd7bf14646d6'

# Face Detection function
def face_detector(img):
    # Convert image to grayscale for faster detection
    gray = cv2.cvtColor(img.copy(),cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return False ,(0,0,0,0), np.zeros((1,48,48,3), np.uint8), img
    
    allfaces = []   
    rects = []
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi = img[y:y+h, x:x+w]
        roi_groiray = cv2.resize(roi, (64, 64), interpolation = cv2.INTER_AREA)
        allfaces.append(roi)
        rects.append((x,w,y,h))
    return True, rects, allfaces, img

# Define our model parameters
depth = 16
k = 8
weight_file = None
margin = 0.4
image_dir = None

# Get our weight file 
if not weight_file:
    weight_file = get_file("weights.28-3.73.hdf5", pretrained_model, cache_subdir="pretrained_models",
                           file_hash=modhash, cache_dir=Path(sys.argv[0]).resolve().parent)

# load model and weights
img_size = 64
model = WideResNet(img_size, depth=depth, k=k)()
model.load_weights(weight_file)

# Initialize Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    ret, rects, faces, image = face_detector(frame)
    preprocessed_faces = []
    i = 0
    if ret:
        for (i,face) in enumerate(faces):
            face = cv2.resize(face, (64, 64), interpolation = cv2.INTER_AREA)
            preprocessed_faces.append(face)

        # make a prediction on the faces detected
        results = model.predict(np.array(preprocessed_faces))
        predicted_genders = results[0]
        ages = np.arange(0, 101).reshape(101, 1)
        predicted_ages = results[1].dot(ages).flatten()

        # draw results
        for (i, f) in enumerate(faces):
            label = "{}, {}".format(int(predicted_ages[i]),
                                        "F" if predicted_genders[i][0] > 0.5 else "M")

        #Overlay our detected emotion on our pic
        label_position = (rects[i][0] + int((rects[i][1]/2)), abs(rects[i][2] - 10))
        i =+ 1
        cv2.putText(image, label, label_position , cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 2)

    cv2.imshow("Emotion Detector", image)
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break

cap.release()
cv2.destroyAllWindows()  


# In[ ]:




