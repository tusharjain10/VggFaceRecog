#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2


# In[2]:


cap = cv2.VideoCapture(0)


# In[3]:


for i in range (0,400):
    status , im = cap.read()
    cv2.imshow('take pics', im)
    cv2.imwrite("data/train/anima/anima" +str(i)+".jpg", im)
    if cv2.waitKey(10) >= 0:
        break
cv2.destroyAllWindows()
cap.release()


# In[6]:


cap.release()
cap = cv2.VideoCapture(0)


# In[5]:


for i in range (0,50):
    status , img = cap.read()
    cv2.imshow('take pics', im)
    if cv2.waitKey(10) >= 0:
        break
    cv2.imwrite("data/test/anima/anima" +str(i)+".jpg", img)
cv2.destroyAllWindows()
cap.release()

