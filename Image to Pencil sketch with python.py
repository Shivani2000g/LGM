#!/usr/bin/env python
# coding: utf-8

# ## Lets Grow More LGM VIP Internship August(2022)
# ### Task-4: Image to Pencil sketch with python
# ### Author: Gore Shivani Kailas¶

# ### Importing library

# In[2]:


import cv2
import matplotlib.pyplot as plt
import numpy as np


# ### Read the Image

# In[3]:


Image = cv2.imread('C:/Users/Hp/Downloads/37-370883_wallpaper-download-lovely-two-little-blue-birds-on.jpg')
plt.imshow(np.real(Image))
plt.show()


# In[4]:


Image= cv2.cvtColor(Image, cv2.COLOR_BGR2RGB)
plt.imshow(Image)
plt.show()


# ### Convert Image to grayscale image

# In[5]:


gray_Image=cv2.cvtColor(Image,cv2.COLOR_BGR2GRAY)
plt.imshow(gray_Image,cmap='gray')
plt.show()


# ### Invert grayscale image to inverted image

# In[6]:


inverted_Image=255- gray_Image
plt.imshow(inverted_Image,cmap='gray')
plt.show()


# ### Inverted image to blurry image

# In[7]:


blurred=cv2.GaussianBlur(inverted_Image, (21,21),0)
plt.imshow(blurred,cmap="gray")


# ### Create the pencil sketch by mixing grayscale image with blurry image

# In[8]:


inverted_blurred=255-blurred
pencil_sketch=cv2.divide(gray_Image,inverted_blurred,scale=256.0)
plt.imshow(pencil_sketch,cmap="gray")


# ### Thank You
