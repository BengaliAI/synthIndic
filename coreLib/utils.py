#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
#---------------------------------------------------------------
# imports
#---------------------------------------------------------------
from termcolor import colored
import os 
import cv2 
import numpy as np
from tqdm import tqdm
import random
from PIL import Image
import math
from wand.image import Image as WImage
#---------------------------------------------------------------
def LOG_INFO(msg,mcolor='blue'):
    '''
        prints a msg/ logs an update
        args:
            msg     =   message to print
            mcolor  =   color of the msg    
    '''
    print(colored("#LOG     :",'green')+colored(msg,mcolor))
#---------------------------------------------------------------
def create_dir(base,ext):
    '''
        creates a directory extending base
        args:
            base    =   base path 
            ext     =   the folder to create
    '''
    _path=os.path.join(base,ext)
    if not os.path.exists(_path):
        os.mkdir(_path)
    return _path

def randColor(col=True,depth=64):
    '''
        generates random color
    '''
    if col:
        return (random.randint(0,255),random.randint(0,255),random.randint(0,255))
    else:
        d=random.randint(0,depth)
        return (d,d,d)

def random_exec(poplutation=[0,1],weights=[0.7,0.3],match=0):
    return random.choices(population=poplutation,weights=weights,k=1)[0]==match
#--------------------
# processing 
#--------------------
def get_warped_image(img,warp_vec,coord,max_warp_perc):
    '''
        returns warped image and new coords
        args:
            img          : image to warp
            warp_vec     : which vector to warp
            coord        : list of current coords
            max_warp_perc: maximum lenght for warping data
              
    '''
    height,width=img.shape
 
    # construct dict warp
    x1,y1=coord[0]
    x2,y2=coord[1]
    x3,y3=coord[2]
    x4,y4=coord[3]
    # warping calculation
    xwarp=random.randint(0,max_warp_perc)/100
    ywarp=random.randint(0,max_warp_perc)/100
    # construct destination
    dx=int(width*xwarp)
    dy=int(height*ywarp)
    # const
    if warp_vec=="p1":
        dst= [[dx,dy], [x2,y2],[x3,y3],[x4,y4]]
    elif warp_vec=="p2":
        dst=[[x1,y1],[x2-dx,dy],[x3,y3],[x4,y4]]
    elif warp_vec=="p3":
        dst= [[x1,y1],[x2,y2],[x3-dx,y3-dy],[x4,y4]]
    else:
        dst= [[x1,y1],[x2,y2],[x3,y3],[dx,y4-dy]]
    M   = cv2.getPerspectiveTransform(np.float32(coord),np.float32(dst))
    img = cv2.warpPerspective(img, M, (width,height),flags=cv2.INTER_NEAREST)
    return img,dst

def warp_data(img,max_warp_perc):
    warp_types=["p1","p2","p3","p4"]
    height,width=img.shape

    coord=[[0,0], 
        [width-1,0], 
        [width-1,height-1], 
        [0,height-1]]

    # warp
    for i in range(2):
        if i==0:
            idxs=[0,2]
        else:
            idxs=[1,3]
        if random_exec():    
            idx=random.choice(idxs)
            img,coord=get_warped_image(img,warp_types[idx],coord,max_warp_perc)
    return img


def rotate_image(mat, angle_max=5):
    """
        Rotates an image (angle in degrees) and expands image to avoid cropping
    """
    angle=random.randint(-angle_max,angle_max)
    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h),flags=cv2.INTER_NEAREST)
    return rotated_mat

def curve_data(img,angle):
    with WImage.from_array(img) as wimg:
        wimg.virtual_pixel = 'black'
        wimg.distort('arc',(angle,))
        wimg=np.array(wimg)
    return wimg

def post_process_word_image(img,config):
    # curve
    if random_exec(weights=config.curve_exec_weights):
        angle=random.randint(config.curve_angle_min,config.curve_angle_max)
        img=curve_data(img,angle)
        return img
    # warp 
    if random_exec(weights=config.warping_exec_weights):
        img=warp_data(img,config.warping_len_max_perc)
        
    # rotate
    if random_exec(weights=config.rotation_exec_weights):
        img=rotate_image(img,angle_max=config.rotation_angle_max)
        
    return img

#---------------------------------------------------------------
# image utils
#---------------------------------------------------------------
def stripPads(arr,
              val):
    '''
        strip specific value
        args:
            arr :   the numpy array (2d)
            val :   the value to strip
        returns:
            the clean array
    '''
    # x-axis
    arr=arr[~np.all(arr == val, axis=1)]
    # y-axis
    arr=arr[:, ~np.all(arr == val, axis=0)]
    return arr

def padAllAround(img,pad_dim,pad_val,pad_single=None):
    '''
        pads all around the image
    '''
    if pad_single is None:
        h,w=img.shape
        # pads
        left_pad =np.ones((h,pad_dim))*pad_val
        right_pad=np.ones((h,pad_dim))*pad_val
        # pad
        img =np.concatenate([left_pad,img,right_pad],axis=1)
        # shape
        h,w=img.shape
        top_pad =np.ones((pad_dim,w))*pad_val
        bot_pad=np.ones((pad_dim,w))*pad_val
        # pad
        img =np.concatenate([top_pad,img,bot_pad],axis=0)
    elif pad_single=="tb":
        # shape
        h,w=img.shape
        top_pad =np.ones((pad_dim,w))*pad_val
        bot_pad=np.ones((pad_dim,w))*pad_val
        # pad
        img =np.concatenate([top_pad,img,bot_pad],axis=0)
    else:
        h,w=img.shape
        # pads
        left_pad =np.ones((h,pad_dim))*pad_val
        right_pad=np.ones((h,pad_dim))*pad_val
        # pad
        img =np.concatenate([left_pad,img,right_pad],axis=1)
    return img

def padWordImage(img,pad_loc,pad_dim,pad_type,pad_val):
    '''
        pads an image with white value
        args:
            img     :       the image to pad
            pad_loc :       (lr/tb) lr: left-right pad , tb=top_bottom pad
            pad_dim :       the dimension to pad upto
            pad_type:       central or left aligned pad
            pad_val :       the value to pad 
    '''
    
    if pad_loc=="lr":
        # shape
        h,w,d=img.shape
        if pad_type=="central":
            # pad widths
            left_pad_width =(pad_dim-w)//2
            # print(left_pad_width)
            right_pad_width=pad_dim-w-left_pad_width
            # pads
            left_pad =np.ones((h,left_pad_width,3))*pad_val
            right_pad=np.ones((h,right_pad_width,3))*pad_val
            # pad
            img =np.concatenate([left_pad,img,right_pad],axis=1)
        else:
            # pad widths
            pad_width =pad_dim-w
            # pads
            pad =np.ones((h,pad_width,3))*pad_val
            # pad
            img =np.concatenate([img,pad],axis=1)
    else:
        # shape
        h,w,d=img.shape
        # pad heights
        if h>= pad_dim:
            return img 
        else:
            pad_height =pad_dim-h
            # pads
            pad =np.ones((pad_height,w,3))*pad_val
            # pad
            img =np.concatenate([img,pad],axis=0)
    return img.astype("uint8")    

def correctPadding(img,dim,ptype="left",pvalue=255):
    '''
        corrects an image padding 
        args:
            img     :       numpy array of single channel image
            dim     :       tuple of desired img_height,img_width
            ptype   :       type of padding (central,left)
            pvalue  :       the value to pad
        returns:
            correctly padded image

    '''
    img_height,img_width=dim
    mask=0
    # check for pad
    h,w,d=img.shape
    
    w_new=int(img_height* w/h) 
    img=cv2.resize(img,(w_new,img_height),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    h,w,d=img.shape
    if w > img_width:
        # for larger width
        h_new= int(img_width* h/w) 
        img=cv2.resize(img,(img_width,h_new),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
        # pad
        img=padWordImage(img,
                     pad_loc="tb",
                     pad_dim=img_height,
                     pad_type=ptype,
                     pad_val=pvalue)
        mask=img_width

    elif w < img_width:
        # pad
        img=padWordImage(img,
                    pad_loc="lr",
                    pad_dim=img_width,
                    pad_type=ptype,
                    pad_val=pvalue)
        mask=w
    
    # error avoid
    img=cv2.resize(img,(img_width,img_height),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    return img,mask 
#---------------------------------------------------------------
# parsing utils
#---------------------------------------------------------------
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

