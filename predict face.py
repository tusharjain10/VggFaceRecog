#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.models import load_model
import cv2
import numpy as np


# In[2]:


face = load_model('myvgg.h5')


# In[3]:


cap = cv2.VideoCapture(0)


# In[6]:


window_name = 'Image'
font = cv2.FONT_HERSHEY_SIMPLEX 
org = (50, 50)
fontScale = 1
color = (255, 255, 255)
thickness = 2
shotNumber1 = 0
shotNumber0 = 0
while True:
    status, img = cap.read()
    dim = (224,224)
    img1 = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    img1 = img1.reshape(1, 224, 224, 3)
    person = face.predict(img1)
    if person[0][0] ==  0:
        img = cv2.putText(img, 'anima', org, font,  
                   fontScale, color, thickness, cv2.LINE_AA)
        save_name = "data/anima"+str(shotNumber0)+".jpg"
        cv2.imwrite(save_name, img)
        shotNumber +=1
    elif person[0][0] ==  1:
        img = cv2.putText(img, 'tushar', org, font,  
                   fontScale, color, thickness, cv2.LINE_AA)
        save_name = "data/tushar"+str(shotNumber1)+".jpg"
        cv2.imwrite(save_name, img)
        shotNumber +=1
    cv2.imshow(window_name,img)
    if cv2.waitKey(1) >= 0:
        break
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:




