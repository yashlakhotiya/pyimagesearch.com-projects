#!/usr/bin/env python
# coding: utf-8

# In[27]:


import numpy as np
import cv2
import time
import imutils
#to install imutils
#pip install imutils


# In[28]:


prototext_path = "deploy.prototxt.txt"
model_path = "res10_300x300_ssd_iter_140000.caffemodel"
conf_threshold = 0.5


# In[35]:


detector = cv2.dnn.readNetFromCaffe(prototext_path, model_path)


# In[ ]:


cap = cv2.VideoCapture(0)
time.sleep(2.0)
while(True):
    ret,image = cap.read()
    frame = imutils.resize(image,width=400)
    (h,w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
    detector.setInput(blob)
    detections = detector.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if(confidence > conf_threshold):
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            (startX, startY, endX, endY) = box.astype("int")
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY-10 > 10 else staryY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY),(0, 0, 255), 2)
            cv2.putText(image, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    cv2.imshow("Output", image)
    if(cv2.waitKey(1) == 13):
        cap.release()
        break
cv2.destroyAllWindows()


# In[ ]:




