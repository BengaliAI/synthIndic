#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
#---------------------------------------------------------------
# imports
#---------------------------------------------------------------
import imgaug.augmenters as iaa
import random
import numpy as np
import cv2 


def additiveGaussian(img):
    scale=np.random.uniform(8,32)
    per_channel=np.random.rand() < 0.5
    aug = iaa.AdditiveGaussianNoise(scale=scale, per_channel=per_channel)
    img = aug(image=img)
    return img

def coarseDropout(img):
    p =np.random.uniform(0.05, 0.25)
    per_channel=np.random.rand() < 0.5
    size_px = None
    size_percent = None
    aug = iaa.CoarseDropout(p=p, size_px=size_px, size_percent=size_percent, per_channel=per_channel)
    img = aug(image=img)
    return img
    
def elasticDistortion(img):
    alpha = np.random.uniform(10,15)
    sigma = np.random.uniform(3,3)
    aug = iaa.ElasticTransformation(alpha=alpha, sigma=sigma, mode="nearest")
    img = aug(image=img)
    return img

def gaussianBlur(img):
    sigma = np.random.randint(1, 3)
    aug = iaa.GaussianBlur(sigma=sigma)
    img = aug(image=img)
    return img

def medianBlur(img):
    k = np.random.choice([1, 3])
    aug = iaa.MedianBlur(k=k)
    img = aug(image=img)
    return img

def motionBlur(img):
    k = np.random.choice([3,5,7])
    angle = np.random.uniform(0,360)
    aug = iaa.MotionBlur(k=k, angle=angle)
    img = aug(image=img)
    return img

def addBrightness(image):    
    ## Conversion to HLSmask
    image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)     
    image_HLS = np.array(image_HLS, dtype = np.float64)
    ## generates value between 0.5 and 1.5       
    random_brightness_coefficient = np.random.uniform()+0.5  
    ## scale pixel values up or down for channel 1(Lightness) 
    image_HLS[:,:,1] = image_HLS[:,:,1]*random_brightness_coefficient
    ##Sets all values above 255 to 255    
    image_HLS[:,:,1][image_HLS[:,:,1]>255]  = 255     
    image_HLS = np.array(image_HLS, dtype = np.uint8)    
    ## Conversion to RGB
    image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB)     
    return image_RGB

AUGMENTATIONS=[additiveGaussian,
              gaussianBlur,
              medianBlur,
              motionBlur]

def augment(img):
    op=random.choice(AUGMENTATIONS)
    return op(img)